"""Validates a plan against the real files, then runs it step by step."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agis.crs import is_auto_utm, resolve_crs
from agis.errors import AgisError, PlanError, StepError
from agis.ops import OpArgs, StepResult
from agis.ops.geoprocessing import Convert, Reproject
from agis.ops.io import Kind, data_kind, is_dataset
from agis.plan import Plan, step_ref

TEMP_EXT: dict[Kind, str] = {"vector": ".gpkg", "raster": ".tif", "tiles": ".mbtiles"}
SHAPEFILE_PARTS = (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx", ".shp.xml")


@dataclass
class ResolvedStep:
    number: int
    args: OpArgs
    inputs: dict[str, Path | list[Path]]
    output: Path | None
    kind: Kind | None
    temporary: bool
    stem: str


def resolve(plan: Plan, cwd: Path, workdir: Path, force: bool = False) -> list[ResolvedStep]:
    """Check every step against the filesystem and decide output paths.

    Raises PlanError listing every problem found, so the model can fix them all at once.
    """
    cwd = cwd.resolve()
    problems: list[str] = []
    resolved: list[ResolvedStep] = []
    referenced = {
        n for step in plan.steps for name in step.input_fields
        for value in _texts(getattr(step, name, None))
        if (n := step_ref(value)) is not None
    }
    final_paths: set[Path] = set()
    existing: list[str] = []

    if not plan.steps:
        raise PlanError([plan.message or "the plan has no steps"])

    for number, args in enumerate(plan.steps, start=1):
        inputs: dict[str, Path | list[Path]] = {}
        kinds: dict[str, Kind | list[Kind]] = {}
        stem = None
        for name in args.input_fields:
            raw = getattr(args, name, None)
            if raw is None:
                continue
            values = _texts(raw)
            if not values:
                problems.append(f"step {number}: {name} has no files")
                continue
            paths: list[Path] = []
            field_kinds: list[Kind] = []
            ok = True
            for value in values:
                ref = step_ref(value)
                if ref is not None:
                    if not 1 <= ref < number:
                        problems.append(f"step {number}: {name} refers to $step{ref}, "
                                        "which isn't an earlier step")
                        ok = False
                        continue
                    source = resolved[ref - 1] if ref - 1 < len(resolved) else None
                    if source is None or source.output is None or source.kind is None:
                        problems.append(
                            f"step {number}: $step{ref} doesn't produce a file to use"
                        )
                        ok = False
                        continue
                    paths.append(source.output)
                    field_kinds.append(source.kind)
                    stem = stem or source.stem
                else:
                    path = find_input(value, cwd)
                    if path is None:
                        problems.append(f"step {number}: file {value!r} not found")
                        ok = False
                        continue
                    paths.append(path)
                    field_kinds.append(data_kind(path))
                    stem = stem or path.stem
            if not ok:
                continue
            many = isinstance(raw, list)
            inputs[name] = paths if many else paths[0]
            kinds[name] = field_kinds if many else field_kinds[0]

        if len(inputs) < sum(getattr(args, n, None) is not None for n in args.input_fields):
            resolved.append(ResolvedStep(number, args, inputs, None, None, False, stem or "out"))
            continue

        problems += [f"step {number}: {p}" for p in args.check(kinds)]
        problems += [f"step {number}: {p}" for p in _check_crs(args, cwd)]

        kind = args.output_kind(kinds["input"]) if "input" in kinds else None
        suffix = args.suffix()
        new_stem = f"{stem}_{suffix}" if suffix else stem
        output, temporary = None, False
        if kind is not None:
            if args.output:
                output = _absolute(Path(args.output).expanduser(), cwd)
            elif number in referenced:
                ext = TEMP_EXT[kind] if not isinstance(args, Convert) else args.output_ext(
                    inputs["input"])
                output, temporary = workdir / f"step{number}_{new_stem}{ext}", True
            else:
                output = cwd / f"{new_stem}{args.output_ext(inputs['input'])}"

        if output is not None and not temporary:
            if output in _flat_paths(inputs):
                problems.append(f"step {number}: output {output.name} would overwrite its input")
            elif output in final_paths:
                problems.append(f"step {number}: two steps write {output.name}")
            elif output.exists() and not force:
                existing.append(f"{output.name} already exists (use --force to overwrite)")
            final_paths.add(output)

        resolved.append(ResolvedStep(number, args, inputs, output, kind, temporary, new_stem))

    if problems:
        raise PlanError(problems)
    if existing:
        raise PlanError(existing, retryable=False)
    return resolved


def _texts(value: object) -> list[str]:
    """File names or "$stepN" references stored in one input field."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _flat_paths(inputs: dict[str, Path | list[Path]]) -> list[Path]:
    paths: list[Path] = []
    for value in inputs.values():
        if isinstance(value, list):
            paths.extend(value)
        else:
            paths.append(value)
    return paths


def find_input(value: str, cwd: Path) -> Path | None:
    """Resolve a file name from the model, tolerating a missing extension."""
    path = Path(value).expanduser()
    path = path if path.is_absolute() else cwd / path
    if is_dataset(path) or data_kind(path) == "tiles":
        return path.resolve()
    if not path.suffix and path.parent.is_dir():
        matches = [p for p in path.parent.glob(f"{path.name}.*") if is_dataset(p)]
        if len(matches) == 1:
            return matches[0].resolve()
    return None


def run(steps: list[ResolvedStep], on_step=None) -> list[StepResult]:
    """Run resolved steps in order. On failure, delete the files this run created."""
    results: list[StepResult] = []
    created: list[Path] = []
    for step in steps:
        if on_step:
            on_step(step)
        if step.output is not None and step.output.exists():
            _remove(step.output)
        try:
            result = step.args.run(step.inputs, step.output)
        except Exception as exc:
            for path in [*created, *([step.output] if step.output else [])]:
                _remove(path)
            raise StepError(step.number, step.args.op, exc) from exc
        if step.output is not None and not step.temporary:
            created.append(step.output)
        results.append(result)
    return results


def make_workdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="agis-"))


def cleanup(workdir: Path) -> None:
    shutil.rmtree(workdir, ignore_errors=True)


def _check_crs(args: OpArgs, cwd: Path) -> list[str]:
    values = []
    if isinstance(args, Reproject) and not is_auto_utm(args.crs):
        values.append(args.crs)
    bbox_crs = getattr(args, "bbox_crs", None)
    if getattr(args, "bbox", None) is not None and bbox_crs:
        values.append(bbox_crs)
    problems = []
    for value in values:
        try:
            resolve_crs(value, cwd)
        except AgisError as exc:
            problems.append(str(exc))
    return problems


def _absolute(path: Path, cwd: Path) -> Path:
    return (path if path.is_absolute() else cwd / path).resolve()


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        return
    if path.suffix.lower() == ".shp":
        for ext in SHAPEFILE_PARTS:
            path.with_suffix(ext).unlink(missing_ok=True)
    path.unlink(missing_ok=True)
