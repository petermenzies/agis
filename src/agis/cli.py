from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from prompt_toolkit.completion import Completer, Completion
from rich.console import Console
from rich.markup import escape

from agis import __version__
from agis import config as config_mod
from agis.errors import AgisError, PlanError, StepError

# A file token, so "roads.gpkg" is one word and Tab can finish it.
_FILE_WORD = re.compile(r"[^\s\"']+")

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="Natural-language GIS on your own machine.\n\n"
    "Run [bold]agis[/bold] for an interactive prompt, or pass one request:\n"
    '[bold]agis "clip roads.gpkg to county.shp and reproject to 3857"[/bold]',
)

ModelOpt = Annotated[str | None, typer.Option("--model", "-m", help="Ollama model to use.")]
HostOpt = Annotated[str | None, typer.Option("--host", help="Ollama URL.", show_default=False)]
DirOpt = Annotated[Path, typer.Option("--dir", "-C", help="Working directory.", file_okay=False,
                                      exists=True)]


def _version(value: bool):
    if value:
        console.print(f"agis {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[bool, typer.Option("--version", callback=_version, is_eager=True,
                                          help="Show the version and exit.")] = False,
):
    pass


def _session(model: str | None, host: str | None, directory: Path):
    from agis.llm import make_backend
    from agis.session import Session

    cfg = config_mod.load()
    backend = make_backend(model or cfg.model, host or cfg.host)
    return Session(backend=backend, cwd=directory.resolve(), console=console)


@app.command("run")
def run_cmd(
    request: Annotated[list[str], typer.Argument(help="What you want done, in plain English.")],
    dry_run: Annotated[bool, typer.Option("--dry-run", "-n", help="Show the plan only.")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask before running.")] = False,
    force: Annotated[bool, typer.Option("--force", "-f", help="Overwrite existing outputs.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show context and raw plans.")] = False,
    model: ModelOpt = None,
    host: HostOpt = None,
    directory: DirOpt = Path("."),
):
    """Plan and run a request. This is the default command, so `agis run` can be omitted."""
    from agis.session import Options

    session = _session(model, host, directory)
    with _errors():
        session.handle(" ".join(request), Options(dry_run=dry_run, yes=yes, force=force,
                                                  verbose=verbose))


@app.command()
def repl(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask before running.")] = False,
    force: Annotated[bool, typer.Option("--force", "-f", help="Overwrite existing outputs.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    model: ModelOpt = None,
    host: HostOpt = None,
    directory: DirOpt = Path("."),
):
    """Interactive mode. Later requests can refer to files made earlier ("now buffer that")."""
    from agis.session import Options

    session = _session(model, host, directory)
    options = Options(yes=yes, force=force, verbose=verbose)
    completer = DatasetCompleter(session.cwd)
    prompt = _prompt_session(completer)
    console.print(f"agis {__version__} using [bold]{session.backend.model}[/bold] in "
                  f"{session.cwd}. Type a request, or 'exit' to quit. "
                  "ls, cd, and pwd run locally.")
    while True:
        try:
            request = _read_request(prompt)
        except EOFError:
            console.print()
            return
        except KeyboardInterrupt:
            # At the prompt, cancel this line and keep the session. A Ctrl-C during a
            # run is handled inside the loop below and also returns here.
            if prompt is None:
                console.print()
                return
            console.print()
            continue
        if request.lower() in ("exit", "quit", ":q"):
            return
        if not request:
            continue
        if try_builtin(request, session, completer, console):
            continue
        with _errors(exit_on_error=False):
            session.handle(request, options)


@app.command()
def setup(
    model: ModelOpt = None,
    host: HostOpt = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Accept the recommended model.")] = False,
):
    """Check Ollama, download the recommended model, and save the choice."""
    from agis.setup import run_setup

    with _errors():
        run_setup(console, model=model, host=host, assume_yes=yes)


@app.command()
def doctor(host: HostOpt = None):
    """Print environment details that are useful for troubleshooting and bug reports."""
    from agis.setup import run_doctor

    with _errors():
        if not run_doctor(console, host=host):
            raise typer.Exit(1)


@app.command("eval")
def eval_cmd(
    model: ModelOpt = None,
    host: HostOpt = None,
    cases: Annotated[Path, typer.Option("--cases", help="YAML file of eval cases.")] = Path(
        "tests/evals/cases.yaml"),
    only: Annotated[str | None, typer.Option("--only", help="Run cases whose id contains this.")] = None,
    show: Annotated[bool, typer.Option("--show", help="Print every plan, not only failures.")] = False,
):
    """Score how well a model turns the eval prompts into the expected plans."""
    from agis.evals import run_evals

    cfg = config_mod.load()
    with _errors():
        score = run_evals(console, cases, model or cfg.model, host or cfg.host, only=only,
                          show=show)
    raise typer.Exit(0 if score == 1.0 else 1)


class _errors:
    """Print AgisErrors nicely instead of a traceback."""

    def __init__(self, exit_on_error: bool = True):
        self.exit_on_error = exit_on_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None or not issubclass(exc_type, (AgisError, KeyboardInterrupt)):
            return False
        if exc_type is KeyboardInterrupt:
            err_console.print("\nInterrupted.")
        elif isinstance(exc, PlanError) and len(exc.problems) > 1:
            err_console.print("[red]The plan can't run:[/red]")
            for problem in exc.problems:
                err_console.print(f"  - {escape(problem)}")
        elif isinstance(exc, PlanError):
            err_console.print(f"[red]The plan can't run:[/red] {escape(exc.problems[0])}")
        elif isinstance(exc, StepError):
            err_console.print(f"[red]Step {exc.index} ({exc.op}) failed:[/red] "
                              f"{escape(str(exc.cause))}")
        else:
            err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
        if self.exit_on_error:
            raise typer.Exit(130 if exc_type is KeyboardInterrupt else 1)
        return True


SUBCOMMANDS = {"run", "repl", "setup", "doctor", "eval"}
ROOT_FLAGS = {"--help", "-h", "--version"}
# Flags accepted by both `run` and `repl`. True when the next argv token is the value,
# unless the flag is written as --name=value.
_SHARED_FLAGS = {
    "-y": False, "--yes": False,
    "-f": False, "--force": False,
    "-v": False, "--verbose": False,
    "-m": True, "--model": True,
    "--host": True,
    "-C": True, "--dir": True,
}


def argv_for(args: list[str]) -> list[str]:
    """Turn a bare invocation into the repl, and a request into `run`.

    Flags with no request (`agis -C ~/data`) also open the repl. A run-only flag such as
    `--dry-run` still goes to `run`, so a missing request is reported there.
    """
    if not args:
        return ["repl"]
    if args[0] in SUBCOMMANDS or args[0] in ROOT_FLAGS:
        return list(args)
    if _only_shared_flags(args):
        return ["repl", *args]
    return ["run", *args]


def _only_shared_flags(args: list[str]) -> bool:
    index = 0
    while index < len(args):
        name, separator, _value = args[index].partition("=")
        if name not in _SHARED_FLAGS:
            return False
        takes_value = _SHARED_FLAGS[name] and not separator
        index += 2 if takes_value else 1
        if takes_value and index > len(args):
            return False
    return True


def _prompt_session(completer: DatasetCompleter):
    """Line editor with history and file completion, or None when stdin is piped."""
    if not sys.stdin.isatty():
        return None
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory

    path = config_mod.config_path().parent / "history"
    path.parent.mkdir(parents=True, exist_ok=True)
    return PromptSession(history=FileHistory(str(path)), completer=completer,
                         complete_while_typing=False)


_BUILTINS = {"ls", "cd", "pwd"}


def try_builtin(line: str, session, completer: DatasetCompleter | None, console: Console) -> bool:
    """Run ls, cd, or pwd locally. Anything else, including a pipe, returns False."""
    if any(character in line for character in "|;&><"):
        return False
    try:
        argv = shlex.split(line)
    except ValueError:
        return False
    if not argv or argv[0] not in _BUILTINS:
        return False
    command, args = argv[0], argv[1:]
    if command == "ls":
        _ls(args, session, console)
    elif command == "cd":
        _cd(args, session, completer, console)
    else:
        _pwd(args, session, console)
    return True


def _ls(args: list[str], session, console: Console) -> None:
    binary = shutil.which("ls")
    if binary is None:
        console.print("ls is not available on this system.", style="red")
        return
    # ls writes the listing itself. Its own message covers a bad path.
    subprocess.run([binary, *args], cwd=session.cwd, check=False)


def _cd(args: list[str], session, completer: DatasetCompleter | None, console: Console) -> None:
    if len(args) > 1:
        console.print("cd takes one directory.", style="red")
        return
    raw = args[0] if args else "~"
    target = Path(raw).expanduser()
    if not target.is_absolute():
        target = session.cwd / target
    try:
        target = target.resolve()
    except OSError as exc:
        console.print(str(exc), style="red", markup=False)
        return
    if not target.is_dir():
        console.print(f"not a directory: {raw}", style="red", markup=False)
        return
    session.cwd = target
    if completer is not None:
        completer.cwd = target
    console.print(str(target), markup=False)


def _pwd(args: list[str], session, console: Console) -> None:
    if args:
        console.print("pwd takes no arguments.", style="red")
        return
    console.print(str(session.cwd), markup=False)


class DatasetCompleter(Completer):
    """Tab-completes dataset names in one directory. Listing them does not open the files."""

    def __init__(self, cwd: Path):
        self.cwd = cwd

    def get_completions(self, document, complete_event):
        from agis.context import find_datasets

        word = document.get_word_before_cursor(pattern=_FILE_WORD)
        if not word:
            return
        prefix = word.casefold()
        for path in find_datasets(self.cwd):
            name = path.name
            if name.casefold().startswith(prefix):
                insert = f'"{name}"' if " " in name else name
                yield Completion(insert, start_position=-len(word), display=name)


def _read_request(prompt) -> str:
    if prompt is None:
        return input("agis> ").strip()
    from prompt_toolkit.formatted_text import HTML

    return prompt.prompt(HTML("<b><ansicyan>agis></ansicyan></b> ")).strip()


def main() -> None:
    """Entry point. A bare `agis` opens the repl; other text is a one-shot request."""
    app(args=argv_for(sys.argv[1:]), prog_name="agis")


if __name__ == "__main__":
    main()
