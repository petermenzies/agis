from __future__ import annotations

import re
from pathlib import Path

from pyproj import CRS
from pyproj.exceptions import CRSError

from agis.errors import AgisError

# Names people commonly use for a CRS, mapped to something pyproj understands.
ALIASES: dict[str, str] = {
    "web mercator": "EPSG:3857",
    "webmercator": "EPSG:3857",
    "pseudo mercator": "EPSG:3857",
    "google mercator": "EPSG:3857",
    "wgs84": "EPSG:4326",
    "wgs 84": "EPSG:4326",
    "lat/lon": "EPSG:4326",
    "lat lon": "EPSG:4326",
    "latlon": "EPSG:4326",
    "lat long": "EPSG:4326",
    "lon/lat": "EPSG:4326",
    "geographic": "EPSG:4326",
    "nad83": "EPSG:4269",
    "nad27": "EPSG:4267",
    "etrs89": "EPSG:4258",
    "conus albers": "EPSG:5070",
    "us albers": "EPSG:5070",
    "albers equal area": "EPSG:5070",
    "california albers": "EPSG:3310",
    "british national grid": "EPSG:27700",
    "osgb": "EPSG:27700",
    "lambert 93": "EPSG:2154",
    "equal earth": "EPSG:8857",
    "robinson": "ESRI:54030",
    "mollweide": "ESRI:54009",
}

UTM_KEYWORDS = {"utm", "local utm", "auto utm", "utm zone"}


def is_auto_utm(value: str) -> bool:
    return value.strip().lower() in UTM_KEYWORDS


def resolve_crs(value: str, cwd: Path | None = None) -> CRS:
    """Turn user or model input like "3857", "web mercator" or "roads.gpkg" into a CRS.

    "utm" is not handled here because it depends on the data; see `is_auto_utm`.
    """
    text = value.strip()
    key = text.lower()
    if key in ALIASES:
        return CRS.from_user_input(ALIASES[key])
    if re.fullmatch(r"\d{4,6}", text):
        return CRS.from_epsg(int(text))
    m = re.fullmatch(r"(?i)(epsg|esri)\s*[: ]\s*(\d+)", text)
    if m:
        return CRS.from_user_input(f"{m.group(1).upper()}:{m.group(2)}")

    if cwd is not None:
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = cwd / candidate
        if candidate.exists():
            from agis.ops.io import read_crs

            crs = read_crs(candidate)
            if crs is None:
                raise AgisError(f"{value} has no CRS to match")
            return crs

    try:
        return CRS.from_user_input(text)
    except CRSError as exc:
        raise AgisError(f"unrecognized CRS {value!r}") from exc


def crs_label(crs: CRS | None) -> str:
    if crs is None:
        return "no CRS"
    auth = crs.to_authority()
    code = f"{auth[0]}:{auth[1]}" if auth else "custom"
    return f"{code} ({crs.name})"


def crs_suffix(crs: CRS) -> str:
    auth = crs.to_authority()
    if auth:
        return auth[1]
    return "reprojected"
