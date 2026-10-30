"""`agis eval`: score a model on prompt -> expected plan cases, against sample data."""

from __future__ import annotations

import math
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from agis.context import build_context
from agis.crs import is_auto_utm, resolve_crs
from agis.errors import AgisError, PlanError
from agis.executor import resolve
from agis.llm import make_backend
from agis.ops.geoprocessing import UNIT_METERS
from agis.plan import Plan
from agis.planner import InvalidPlanOutput, Planner, compact_json
from agis.sample_data import write_sample_data

PATH_FIELDS = {"input", "mask", "other", "output"}
PATH_LIST_FIELDS = {"others"}
CRS_FIELDS = {"crs", "bbox_crs"}


@dataclass
class CaseResult:
    id: str
    passed: bool
    seconds: float
    retried: bool
    problems: list[str]
    plan_json: str


def load_cases(path: Path) -> list[dict]:
    if not path.exists():
        raise AgisError(f"eval cases not found at {path}; pass --cases")
    cases = yaml.safe_load(path.read_text()) or []
    for case in cases:
        if not {"id", "prompt", "expect"} <= set(case):
            raise AgisError(f"case {case.get('id', '?')} needs id, prompt and expect")
    return cases


def run_evals(console: Console, cases_path: Path, model: str, host: str | None,
              only: str | None = None, show: bool = False) -> float:
    cases = [c for c in load_cases(cases_path) if not only or only in c["id"]]
    if not cases:
        raise AgisError("no cases matched")
    planner = Planner(make_backend(model, host))

    results: list[CaseResult] = []
    with tempfile.TemporaryDirectory(prefix="agis-eval-") as tmp:
        data_dir = Path(tmp) / "data"
        write_sample_data(data_dir)
        with console.status(f"Running {len(cases)} cases with {model}...") as status:
            for i, case in enumerate(cases, start=1):
                status.update(f"[{i}/{len(cases)}] {case['id']}")
                results.append(run_case(planner, case, data_dir, Path(tmp) / "work"))

    table = Table(box=None, header_style="bold", padding=(0, 2, 0, 0))
    for column in ("case", "result", "time", "notes"):
        table.add_column(column)
    for r in results:
        mark = "[green]pass[/green]" if r.passed else "[red]FAIL[/red]"
        notes = "; ".join(r.problems) + (" (after re-plan)" if r.retried else "")
        table.add_row(r.id, mark, f"{r.seconds:.1f}s", escape(notes))
        if show or not r.passed:
            table.add_row("", "", "", f"[dim]{escape(r.plan_json)}[/dim]")
    console.print(table)

    passed = sum(r.passed for r in results)
    mean = sum(r.seconds for r in results) / len(results)
    console.print(f"\n[bold]{model}: {passed}/{len(results)} passed[/bold] "
                  f"({passed / len(results):.0%}), mean {mean:.1f}s per case")
    return passed / len(results)


def run_case(planner: Planner, case: dict, data_dir: Path, workdir: Path) -> CaseResult:
    """Plan like a real session does: one re-plan is allowed if the first plan can't run."""
    context = build_context(case["prompt"], data_dir)
    start = time.perf_counter()
    feedback: list[tuple[str, list[str]]] = []
    plan, raw, retried = None, "", False
    for attempt in range(2):
        try:
            plan = planner.plan(case["prompt"], context, feedback)
        except InvalidPlanOutput as exc:
            raw, plan = exc.raw, None
            feedback.append((exc.raw, exc.problems))
            retried = attempt == 0
            continue
        except AgisError as exc:
            return CaseResult(case["id"], False, time.perf_counter() - start, retried,
                              [str(exc)], "")
        raw = compact_json(plan)
        if not plan.steps:
            break
        try:
            resolve(plan, data_dir, workdir)
            break
        except PlanError as exc:
            if attempt == 0:
                feedback.append((raw, exc.problems))
                retried = True
    seconds = time.perf_counter() - start

    if plan is None:
        return CaseResult(case["id"], False, seconds, retried, ["output never parsed"], raw)
    problems = compare(plan, case["expect"], data_dir)
    if not problems and plan.steps:
        try:
            resolve(plan, data_dir, workdir)
        except PlanError as exc:
            problems = exc.problems
    return CaseResult(case["id"], not problems, seconds, retried, problems, raw)


def compare(plan: Plan, expected: list[dict], data_dir: Path) -> list[str]:
    if len(plan.steps) != len(expected):
        got = ", ".join(s.op for s in plan.steps) or "nothing"
        want = ", ".join(e["op"] for e in expected) or "nothing"
        return [f"expected {want}, got {got}"]
    problems = []
    for n, (step, want) in enumerate(zip(plan.steps, expected), start=1):
        if step.op != want["op"]:
            problems.append(f"step {n}: expected {want['op']}, got {step.op}")
            continue
        got = step.model_dump()
        if step.op == "buffer" and "distance" in want:
            meters = want["distance"] * UNIT_METERS[want.get("units", "meters")]
            actual = step.distance * UNIT_METERS[step.units]
            if not math.isclose(meters, actual, rel_tol=0.01):
                problems.append(f"step {n}: buffer of {step.distance:g} {step.units}, expected "
                                f"{want['distance']:g} {want.get('units', 'meters')}")
            want = {k: v for k, v in want.items() if k not in ("distance", "units")}
        if (step.op == "reproject" and is_auto_utm(str(want.get("crs", "")))
                and not is_auto_utm(step.crs)):
            # Naming the concrete UTM zone is as good as asking for "utm".
            want = {**want, "crs": _utm_for(want.get("input", step.input), data_dir)}
        for key, value in want.items():
            actual = got.get(key)
            if key == "where" and isinstance(actual, list) and isinstance(value, list):
                actual, value = sorted(actual, key=_where_key), sorted(value, key=_where_key)
            if key == "columns" and isinstance(actual, list) and isinstance(value, list):
                actual = sorted(actual, key=lambda item: str(item).lower())
                value = sorted(value, key=lambda item: str(item).lower())
            if key != "op" and not _same(key, actual, value, data_dir):
                problems.append(f"step {n}: {key}={got.get(key)!r}, expected {value!r}")
    return problems


def _same(key: str, got, want, data_dir: Path) -> bool:
    if want is None or got is None:
        return want is got
    if key in PATH_FIELDS:
        return _norm_path(str(got)) == _norm_path(str(want))
    if key in CRS_FIELDS:
        if is_auto_utm(str(want)) or is_auto_utm(str(got)):
            return is_auto_utm(str(want)) and is_auto_utm(str(got))
        try:
            return resolve_crs(str(got), data_dir).equals(resolve_crs(str(want), data_dir))
        except AgisError:
            return False
    if isinstance(want, list):
        if not isinstance(got, list) or len(got) != len(want):
            return False
        if not want:
            return True
        if key in PATH_LIST_FIELDS:
            return [_norm_path(str(item)) for item in got] == [_norm_path(w) for w in want]
        if isinstance(want[0], dict):
            return all(_same_dict(item, expected, data_dir) for item, expected in zip(got, want))
        if isinstance(want[0], str):
            return [str(item).lower() for item in got] == [w.lower() for w in want]
        return all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(got, want))
    got_num, want_num = _as_float(got), _as_float(want)
    if got_num is not None and want_num is not None:
        return math.isclose(got_num, want_num, rel_tol=1e-6)
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        return math.isclose(got, want, rel_tol=1e-6)
    return str(got).lower() == str(want).lower()


def _same_dict(got, want: dict, data_dir: Path) -> bool:
    if not isinstance(got, dict):
        return False
    return all(_same(key, got.get(key), value, data_dir) for key, value in want.items())


def _where_key(condition) -> str:
    if not isinstance(condition, dict):
        return str(condition)
    return "|".join(str(condition.get(key, "")).lower() for key in ("field", "op", "value"))


def _as_float(value) -> float | None:
    if value is None or isinstance(value, (bool, list, dict)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)", str(value).strip()):
        return float(value)
    return None


def _utm_for(name: str, data_dir: Path) -> str:
    from agis.ops.io import read_vector

    utm = read_vector(data_dir / name).estimate_utm_crs()
    return ":".join(utm.to_authority())


def _norm_path(value: str) -> str:
    if value.startswith("$"):
        return value.strip()
    return Path(value).name.lower()
