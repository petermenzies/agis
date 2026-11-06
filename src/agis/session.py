"""One request end to end: plan, validate, preview, confirm, run, and re-plan on failure."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm
from rich.table import Table

from agis.context import build_context
from agis.errors import AgisError, PlanError, StepError
from agis.executor import ResolvedStep, cleanup, make_workdir, resolve, run
from agis.llm import Backend
from agis.ops.geoprocessing import Info
from agis.plan import Plan
from agis.planner import InvalidPlanOutput, Planner, compact_json


@dataclass
class Options:
    dry_run: bool = False
    yes: bool = False
    force: bool = False
    verbose: bool = False
    retries: int = 1


@dataclass
class Session:
    backend: Backend
    cwd: Path
    console: Console = field(default_factory=Console)
    recent: list[Path] = field(default_factory=list)
    summaries: dict = field(default_factory=dict)

    def handle(self, request: str, options: Options) -> list[Path]:
        """Run one request and return the files it created."""
        planner = Planner(self.backend)
        context = build_context(request, self.cwd, self.recent, self.summaries)
        if options.verbose:
            self.console.print(context.render() + "\n", style="dim", markup=False)

        feedback: list[tuple[str, list[str]]] = []
        workdir = make_workdir()
        try:
            for attempt in range(options.retries + 1):
                last = attempt == options.retries
                with self.console.status(f"Planning with {self.backend.model}..."):
                    try:
                        plan = planner.plan(request, context, feedback)
                    except InvalidPlanOutput as exc:
                        if last:
                            raise
                        feedback.append((exc.raw, exc.problems))
                        self._retrying(exc.problems, options)
                        continue
                raw = compact_json(plan)
                if options.verbose:
                    self.console.print(raw, style="dim", markup=False)
                if not plan.steps:
                    self.console.print(plan.message or "Nothing to do for that request.",
                                       markup=False)
                    return []

                try:
                    steps = resolve(plan, self.cwd, workdir, force=options.force)
                except PlanError as exc:
                    if last or not exc.retryable:
                        raise
                    feedback.append((raw, exc.problems))
                    self._retrying(exc.problems, options)
                    continue

                if feedback:
                    self.console.print("[yellow]Revised plan:[/yellow]")
                self.preview(plan, steps)
                if options.dry_run:
                    self.console.print("[dim]Dry run: nothing was written.[/dim]")
                    return []
                if not self._confirm(steps, options):
                    self.console.print("Cancelled.")
                    return []

                try:
                    results = run(steps, on_step=self._announce)
                except StepError as exc:
                    if last:
                        raise
                    self.console.print(str(exc), style="red", markup=False)
                    feedback.append((raw, [str(exc)]))
                    self._retrying([], options)
                    continue

                created = self._report(steps, results)
                self.recent = [*self.recent, *created][-10:]
                return created
        finally:
            cleanup(workdir)
        return []

    def preview(self, plan: Plan, steps: list[ResolvedStep]) -> None:
        if plan.message:
            self.console.print(plan.message, style="bold", markup=False)
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2, 0, 0))
        table.add_column("#", justify="right", no_wrap=True)
        table.add_column("operation", no_wrap=True)
        # Fold instead of ellipsis so a narrow terminal wraps the arguments
        # rather than hiding the rest of the step.
        table.add_column("details", overflow="fold", ratio=1)
        table.add_column("output", overflow="fold")
        for step in steps:
            details = []
            for name, value in step.args.model_dump(exclude={"op", "output"}).items():
                default = type(step.args).model_fields[name].default
                if value is not None and (value != default or name in step.args.input_fields):
                    details.append(f"{name}={_detail(name, value)}")
            if step.output is None:
                output = "[dim]-[/dim]"
            elif step.temporary:
                output = "[dim](intermediate)[/dim]"
            else:
                output = escape(self._rel(step.output))
            table.add_row(str(step.number), step.args.op, escape(" ".join(details)), output)
        self.console.print(table)

    def _confirm(self, steps: list[ResolvedStep], options: Options) -> bool:
        if options.yes or all(isinstance(s.args, Info) for s in steps):
            return True
        if not sys.stdin.isatty():
            raise AgisError("not running in a terminal; pass --yes to run without confirming")
        return Confirm.ask("Run this plan?", default=True, console=self.console)

    def _announce(self, step: ResolvedStep) -> None:
        if not isinstance(step.args, Info):
            self.console.print(f"[dim]step {step.number}: {step.args.op}...[/dim]")

    def _report(self, steps: list[ResolvedStep], results) -> list[Path]:
        created = []
        for step, result in zip(steps, results):
            if isinstance(step.args, Info):
                self.console.print(result.message, markup=False)
                continue
            if step.temporary or step.output is None:
                continue
            created.append(step.output)
            self.console.print(f"[green]Wrote[/green] {escape(self._rel(step.output))}"
                               + (f"  [dim]{escape(result.message)}[/dim]"
                                  if result.message else ""))
        return created

    def _retrying(self, problems: list[str], options: Options) -> None:
        if options.verbose and problems:
            self.console.print("[yellow]Plan problems:[/yellow] " + "; ".join(problems))
        self.console.print("[dim]Asking the model to fix the plan...[/dim]")

    def _rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.cwd.resolve()))
        except ValueError:
            return str(path)


def _detail(name: str, value) -> str:
    """How one argument is shown in the plan preview."""
    if name == "where":
        clauses = [_condition(item) for item in value]
        return str(clauses[0] if len(clauses) == 1 else clauses)
    return str(value)


def _condition(item: dict) -> list:
    """A filter test as [field, op, value], dropping a null value."""
    clause = [item["field"], item["op"]]
    if item.get("value") is not None:
        clause.append(item["value"])
    return clause

