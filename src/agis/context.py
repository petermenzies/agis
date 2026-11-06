"""Summarize the files a request refers to, so the model doesn't have to guess."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pyogrio
import rasterio
from pyproj import CRS
from pyproj.exceptions import CRSError

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
    cache: dict | None = None

    def render(self) -> str:
        lines = [f"Working directory: {self.cwd}"]
        # A request that names files only opens those, plus anything this session wrote.
        # A request that names nothing ("what is in this folder?") describes the first few.
        if self.mentioned or self.recent:
            described = _unique([*self.mentioned, *self.recent])
        else:
            described = self.datasets[:MAX_DESCRIBED]
        if described:
            lines.append("Files:")
            lines += [f"- {cached_describe(p, self.cwd, cache=self.cache)}" for p in described]
        described_keys = {p.resolve() for p in described}
        others = [p for p in self.datasets if p.resolve() not in described_keys]
        shown, extra = others[:MAX_LISTED], len(others) - MAX_LISTED
        if shown:
            names = ", ".join(_rel(p, self.cwd) for p in shown)
            if extra > 0:
                names += f", and {extra} more"
            lines.append("Other files: " + names)
        if not self.datasets and not self.recent:
            lines.append("Files: none found in the working directory.")
        if self.recent:
            lines.append("Created earlier in this session: "
                         + ", ".join(_rel(p, self.cwd) for p in self.recent))
        return "\n".join(lines)


def build_context(prompt: str, cwd: Path, recent: list[Path] | None = None,
                  cache: dict | None = None) -> Context:
    datasets = find_datasets(cwd)
    recent = [p for p in (recent or []) if p.exists()]
    return Context(cwd=cwd, datasets=datasets,
                   mentioned=mentioned_files(prompt, cwd, [*datasets, *recent]),
                   recent=recent, cache=cache)


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


def cached_describe(path: Path, base: Path | None = None, detailed: bool = False,
                    cache: dict | None = None) -> str:
    """describe(), reused while the file's size and mtime stay the same."""
    if cache is None:
        return describe(path, base=base, detailed=detailed)
    try:
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, detailed)
    except OSError:
        return describe(path, base=base, detailed=detailed)
    text = cache.get(key)
    if text is None:
        text = describe(path, base=base, detailed=detailed)
        cache[key] = text
    return text


def describe(path: Path, base: Path | None = None, detailed: bool = False) -> str:
    name = _rel(path, base) if base else path.name
    kind = data_kind(path)
    try:
        if detailed:
            return _describe_detailed(path, name, kind)
        if kind == "raster":
            return f"{name}: {_describe_raster(path)}"
        if kind == "vector":
            return f"{name}: {_describe_vector(path)}"
        if kind == "table":
            return f"{name}: {_describe_table(path)}"
        if kind == "tiles":
            return f"{name}: {_tiles_compact(path)}"
    except Exception as exc:  # noqa: BLE001 - one unreadable file shouldn't break planning
        return f"{name}: unreadable ({exc})"
    return f"{name}: unsupported file"


def _describe_detailed(path: Path, name: str, kind: str | None) -> str:
    """One fact per line, for the info command. The planning summary stays a single line."""
    if kind == "vector":
        info = _load_vector_info(path, force_total_bounds=True)
        items = [
            ("kind", "vector"),
            ("geometry", str(info["geometry_type"] or "unknown")),
            ("features", "not read" if info["partial"] else str(info["features"])),
            ("crs", _crs_text(info)),
            ("bounds", "not read" if info["partial"] else _bounds(info.get("total_bounds"))),
        ]
        if info.get("layer_name"):
            items.append(("layer", str(info["layer_name"])))
        return _lines(name, items, _attribute_fields(info, spaced=True))
    if kind == "raster":
        with rasterio.open(path) as src:
            crs = CRS.from_user_input(src.crs) if src.crs else None
            unit = _resolution_unit(crs)
            items = [
                ("kind", "raster"),
                ("bands", f"{src.count} {src.dtypes[0]}"),
                ("size", f"{src.width}x{src.height} px"),
                ("resolution", f"{src.res[0]:g} {unit}"),
                ("crs", crs_label(crs)),
            ]
            if src.nodata is not None:
                items.append(("nodata", f"{src.nodata:g}"))
            items.append(("bounds", _bounds(src.bounds)))
        return _lines(name, items)
    if kind == "table":
        rows, fields = _table_fields(path)
        return _lines(name, [("kind", "table"), ("rows", str(rows))], fields)
    if kind == "tiles":
        return _tiles_detailed(path, name)
    return f"{name}: unsupported file"


def _lines(name: str, items: list[tuple[str, str]], fields: list[str] | None = None) -> str:
    lines = [name, *[f"{label}: {value}" for label, value in items]]
    if fields is not None:
        if fields:
            lines.append("fields:")
            lines.extend(f"  {field}" for field in fields)
        else:
            lines.append("fields: none")
    return "\n".join(lines)


def _tiles_compact(path: Path) -> str:
    if path.suffix.lower() == ".mbtiles":
        meta, _count = _read_mbtiles(path)
        return (f"MBTiles {meta.get('format', 'png')}, "
                f"zoom {meta.get('minzoom', '?')}-{meta.get('maxzoom', '?')}")
    zooms = _xyz_zooms(path)
    return f"XYZ tile folder, zoom {zooms[0]}-{zooms[-1]}"


def _tiles_detailed(path: Path, name: str) -> str:
    if path.suffix.lower() == ".mbtiles":
        meta, count = _read_mbtiles(path)
        items = [
            ("kind", "mbtiles"),
            ("format", meta.get("format", "png")),
            ("zoom", f"{meta.get('minzoom', '?')}-{meta.get('maxzoom', '?')}"),
            ("tiles", str(count)),
        ]
        if meta.get("bounds"):
            items.append(("bounds", meta["bounds"]))
        return _lines(name, items)
    zooms = _xyz_zooms(path)
    return _lines(name, [("kind", "XYZ tiles"), ("zoom", f"{zooms[0]}-{zooms[-1]}")])


def _xyz_zooms(path: Path) -> list[int]:
    return sorted(int(child.name) for child in path.iterdir() if child.name.isdigit())


def _read_mbtiles(path: Path) -> tuple[dict[str, str], int]:
    import sqlite3

    con = sqlite3.connect(path)
    try:
        meta = dict(con.execute("select name, value from metadata"))
        count = con.execute("select count(*) from tiles").fetchone()[0]
        return meta, count
    finally:
        con.close()


def _describe_vector(path: Path) -> str:
    # Header metadata is enough for planning. Scanning every feature for bounds is reserved
    # for `info`, which asks for the detailed summary.
    info = _load_vector_info(path, force_total_bounds=False)
    fields = _attribute_fields(info, spaced=False)
    shown = fields[:MAX_FIELDS]
    if not fields:
        field_text = "none"
    else:
        more = "" if len(shown) == len(fields) else f", +{len(fields) - len(shown)} more"
        field_text = ", ".join(shown) + more
    count = "features not read" if info["partial"] else f"{info['features']} features"
    return (f"vector, {info['geometry_type'] or 'unknown'}, {count}, {_crs_text(info)}, "
            f"fields [{field_text}]")


def _load_vector_info(path: Path, force_total_bounds: bool) -> dict:
    try:
        info = pyogrio.read_info(path, force_total_bounds=force_total_bounds)
    except Exception as exc:
        if not _geojson_too_large(path, exc):
            raise
        info = _peek_geojson(path)
        info["partial"] = True
        return info
    info["partial"] = False
    return info


def _geojson_too_large(path: Path, exc: Exception) -> bool:
    if path.suffix.lower() not in {".geojson", ".json"}:
        return False
    text = str(exc)
    return "OGR_GEOJSON_MAX_OBJ_SIZE" in text or "too complex/large" in text


def _attribute_fields(info: dict, spaced: bool) -> list[str]:
    """Attribute columns plus the feature id, which pyogrio reports separately."""
    sep = ": " if spaced else ":"
    raw_fields = info.get("fields")
    raw_dtypes = info.get("dtypes")
    names = [] if raw_fields is None else [str(name) for name in raw_fields]
    dtypes = [] if raw_dtypes is None else [str(dtype) for dtype in raw_dtypes]
    fields = [f"{name}{sep}{_short_dtype(dtype)}" for name, dtype in zip(names, dtypes)]
    fid = info.get("fid_column")
    if fid is not None and str(fid) not in names:
        fields.insert(0, f"{fid}{sep}int")
    return fields


def _crs_text(info: dict) -> str:
    raw = info.get("crs")
    if not raw:
        return "not in header" if info.get("partial") else crs_label(None)
    try:
        return crs_label(CRS.from_user_input(raw))
    except CRSError:
        return str(raw)


_GEOM_TYPES = (
    "Point", "LineString", "Polygon", "MultiPoint", "MultiLineString", "MultiPolygon",
    "GeometryCollection",
)
_GEOM_TYPE = re.compile(r'"type"\s*:\s*"(' + "|".join(_GEOM_TYPES) + ')"')


def _peek_geojson(path: Path) -> dict:
    """CRS, geometry type, and first-feature fields, without reading coordinates."""
    decoder = json.JSONDecoder()
    buf = ""
    info = _empty_peek()
    with path.open(encoding="utf-8", errors="replace") as handle:
        while len(buf) < 8_000_000:
            block = handle.read(65536)
            if not block:
                break
            buf += block
            info, done = _parse_geojson_prefix(buf, decoder)
            if done:
                break
    return info


def _empty_peek() -> dict:
    return {
        "geometry_type": None, "features": None, "crs": None, "total_bounds": None,
        "layer_name": None, "fields": [], "dtypes": [], "fid_column": None,
    }


def _parse_geojson_prefix(text: str, decoder: json.JSONDecoder) -> tuple[dict, bool]:
    info = _empty_peek()
    feature = re.search(r'"type"\s*:\s*"Feature"', text)
    if feature is None:
        return info, False
    crs_at = _json_key_at(text, "crs", 0, feature.start())
    if crs_at is not None:
        try:
            obj, _end = decoder.raw_decode(text, crs_at)
        except json.JSONDecodeError:
            obj = None
        name = _crs_name(obj)
        if name:
            info["crs"] = name
    props_at = _json_key_at(text, "properties", feature.end(), len(text))
    if props_at is None:
        return info, False
    try:
        value, end = decoder.raw_decode(text, props_at)
    except json.JSONDecodeError:
        return info, False
    if isinstance(value, dict):
        info["fields"] = list(value)
        info["dtypes"] = [_json_type(item) for item in value.values()]
    geom = _GEOM_TYPE.search(text, end)
    if geom is None:
        return info, False
    info["geometry_type"] = geom.group(1)
    return info, True


def _json_key_at(text: str, key: str, start: int, end: int) -> int | None:
    """Index of the JSON value for key, searching only within text[start:end]."""
    window = text[start:end]
    match = re.search(rf'"{re.escape(key)}"\s*:', window)
    if match is None:
        return None
    value_at = start + match.end()
    while value_at < len(text) and text[value_at].isspace():
        value_at += 1
    return value_at


def _crs_name(value) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    if not isinstance(value, dict):
        return None
    props = value.get("properties")
    if isinstance(props, dict) and props.get("name"):
        return str(props["name"])
    if value.get("name"):
        return str(value["name"])
    return None


def _json_type(value) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"


def _describe_table(path: Path) -> str:
    """Column types from a short sample. The row count is a line count, not a full parse."""
    rows, fields = _table_fields(path)
    shown = fields[:MAX_FIELDS]
    more = "" if len(shown) == len(fields) else f", +{len(fields) - len(shown)} more"
    compact = [field.replace(": ", ":") for field in shown]
    return f"table, {rows} rows, fields [{', '.join(compact)}{more}]"


def _table_fields(path: Path) -> tuple[int, list[str]]:
    import pandas as pd

    from agis.ops.io import TABLE_SEPARATORS

    separator = TABLE_SEPARATORS.get(path.suffix.lower(), ",")
    rows = _count_data_rows(path)
    sample = pd.read_csv(path, sep=separator, nrows=100)
    fields = [f"{name}: {_pandas_dtype(sample[name])}" for name in sample.columns]
    return rows, fields


def _count_data_rows(path: Path) -> int:
    with path.open(encoding="utf-8", errors="replace") as handle:
        next(handle, None)
        return sum(1 for line in handle if line.strip())


def _pandas_dtype(series) -> str:
    import pandas as pd

    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"
    return "str"


def _describe_raster(path: Path) -> str:
    with rasterio.open(path) as src:
        crs = CRS.from_user_input(src.crs) if src.crs else None
        unit = _resolution_unit(crs)
        text = (f"raster, {src.count} band{'s' if src.count > 1 else ''} {src.dtypes[0]}, "
                f"{src.width}x{src.height} px, {src.res[0]:g} {unit} pixels, {crs_label(crs)}")
        if src.nodata is not None:
            text += f", nodata {src.nodata:g}"
    return text


def _resolution_unit(crs: CRS | None) -> str:
    if crs is None or crs.is_geographic:
        return "deg" if crs is not None else "units"
    return crs.axis_info[0].unit_name or "units"


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
