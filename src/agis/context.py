"""Summarize the files a request refers to, so the model doesn't have to guess."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pyogrio
import rasterio
from pyproj import CRS

from agis.crs import crs_label
from agis.ops.io import SIDECAR_EXTS, data_kind, is_dataset

MAX_LISTED = 40
MAX_DESCRIBED = 12
MAX_FIELDS = 15


@dataclass
class Context:
    cwd: Path
    datasets: list[Path]
    mentioned: list[Path]
    recent: list[Path] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"Working directory: {self.cwd}"]
        described = _unique([*self.mentioned, *self.recent, *self.datasets])[:MAX_DESCRIBED]
        if described:
            lines.append("Files:")
            lines += [f"- {describe(p, base=self.cwd)}" for p in described]
        others = [p for p in self.datasets if p not in described][: MAX_LISTED - len(described)]
        if others:
            lines.append("Other files: " + ", ".join(_rel(p, self.cwd) for p in others))
        if not self.datasets and not self.recent:
            lines.append("Files: none found in the working directory.")
        if self.recent:
            lines.append("Created earlier in this session: "
                         + ", ".join(_rel(p, self.cwd) for p in self.recent))
        return "\n".join(lines)


def build_context(prompt: str, cwd: Path, recent: list[Path] | None = None) -> Context:
    datasets = find_datasets(cwd)
    recent = [p for p in (recent or []) if p.exists()]
    return Context(cwd=cwd, datasets=datasets,
                   mentioned=mentioned_files(prompt, cwd, [*datasets, *recent]), recent=recent)


def find_datasets(cwd: Path) -> list[Path]:
    found = [p for p in sorted(cwd.iterdir()) if is_dataset(p) or data_kind(p) == "tiles"]
    return [p for p in found if p.suffix.lower() not in SIDECAR_EXTS]


def mentioned_files(prompt: str, cwd: Path, datasets: list[Path]) -> list[Path]:
    """Files named in the prompt, by path ("data/roads.shp") or by bare name ("roads")."""
    found: list[Path] = []
    for token in re.findall(r"[\w~./\\-]+\.[A-Za-z0-9]+", prompt):
        path = Path(token).expanduser()
        path = path if path.is_absolute() else cwd / path
        if is_dataset(path):
            found.append(path.resolve())
    words = {w.lower() for w in re.findall(r"[\w-]+", prompt)}
    found += [p for p in datasets if p.stem.lower() in words or p.name.lower() in words]
    return _unique(found)


def describe(path: Path, base: Path | None = None, detailed: bool = False) -> str:
    name = _rel(path, base) if base else path.name
    kind = data_kind(path)
    try:
        if kind == "raster":
            return f"{name}: {_describe_raster(path, detailed)}"
        if kind == "vector":
            return f"{name}: {_describe_vector(path, detailed)}"
        if kind == "table":
            return f"{name}: {_describe_table(path, detailed)}"
        if kind == "tiles":
            zooms = sorted(int(p.name) for p in path.iterdir() if p.name.isdigit())
            return f"{name}: XYZ tile folder, zoom {zooms[0]}-{zooms[-1]}"
    except Exception as exc:  # noqa: BLE001 - one unreadable file shouldn't break planning
        return f"{name}: unreadable ({exc})"
    return f"{name}: unsupported file"


def _describe_vector(path: Path, detailed: bool) -> str:
    info = pyogrio.read_info(path, force_total_bounds=True)
    crs = CRS.from_user_input(info["crs"]) if info.get("crs") else None
    fields = [f"{n}:{_short_dtype(t)}" for n, t in zip(info["fields"], info["dtypes"])]
    shown = fields if detailed else fields[:MAX_FIELDS]
    more = "" if len(shown) == len(fields) else f", +{len(fields) - len(shown)} more"
    text = (f"vector, {info['geometry_type']}, {info['features']} features, {crs_label(crs)}, "
            f"fields [{', '.join(shown)}{more}]")
    if detailed:
        text += f", bounds {_bounds(info.get('total_bounds'))}"
        if info.get("layer_name"):
            text += f", layer {info['layer_name']}"
    return text


def _describe_table(path: Path, detailed: bool) -> str:
    from agis.ops.io import read_table

    table = read_table(path)
    fields = [f"{name}:{_pandas_dtype(table[name])}" for name in table.columns]
    shown = fields if detailed else fields[:MAX_FIELDS]
    more = "" if len(shown) == len(fields) else f", +{len(fields) - len(shown)} more"
    return f"table, {len(table)} rows, fields [{', '.join(shown)}{more}]"


def _pandas_dtype(series) -> str:
    import pandas as pd

    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"
    return "str"


def _describe_raster(path: Path, detailed: bool) -> str:
    with rasterio.open(path) as src:
        crs = CRS.from_user_input(src.crs) if src.crs else None
        unit = "deg" if crs is not None and crs.is_geographic else "units"
        if crs is not None and crs.is_projected:
            unit = crs.axis_info[0].unit_name or "units"
        text = (f"raster, {src.count} band{'s' if src.count > 1 else ''} {src.dtypes[0]}, "
                f"{src.width}x{src.height} px, {src.res[0]:g} {unit} pixels, {crs_label(crs)}")
        if src.nodata is not None:
            text += f", nodata {src.nodata:g}"
        if detailed:
            text += f", bounds {_bounds(src.bounds)}"
    return text


def _short_dtype(dtype: str) -> str:
    if dtype.startswith("int"):
        return "int"
    if dtype.startswith("float"):
        return "float"
    if dtype in ("object", "str"):
        return "str"
    return dtype


def _bounds(b) -> str:
    if b is None:
        return "unknown"
    return "[" + ", ".join(f"{v:.6g}" for v in b) + "]"


def _rel(path: Path, base: Path | None) -> str:
    if base is None:
        return path.name
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def _unique(paths: list[Path]) -> list[Path]:
    seen, out = set(), []
    for p in paths:
        key = p.resolve()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out
