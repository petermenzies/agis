from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

from agis import __version__
from agis import config as config_mod
from agis.errors import AgisError, PlanError, StepError

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="Natural-language GIS on your own machine.\n\n"
    'Example: [bold]agis "clip roads.gpkg to county.shp and reproject to 3857"[/bold]',
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
    console.print(f"agis {__version__} using [bold]{session.backend.model}[/bold] in "
                  f"{session.cwd}. Type a request, or 'exit' to quit.")
    while True:
        try:
            request = console.input("[bold cyan]agis>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if request.lower() in ("exit", "quit", ":q"):
            return
        if not request:
            continue
        with _errors(exit_on_error=False):
            session.handle(request, options)


@app.command()
def setup(
    model: ModelOpt = None,
    host: HostOpt = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Accept the recommended model.")] = False,
):
    """Check Ollama, pick a model that fits this Mac, download it, and save the choice."""
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


def main() -> None:
    """Entry point. Anything that isn't a subcommand is treated as a request for `run`."""
    args = sys.argv[1:]
    if args and args[0] not in SUBCOMMANDS and args[0] not in ROOT_FLAGS:
        args = ["run", *args]
    app(args=args, prog_name="agis")


if __name__ == "__main__":
    main()
