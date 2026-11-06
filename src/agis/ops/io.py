from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import geopandas as gpd
import pandas as pd
import pyogrio
import rasterio
from pyproj import CRS

from agis.errors import AgisError

Kind = Literal["vector", "raster", "tiles", "table"]

VECTOR_DRIVERS: dict[str, str] = {
    ".gpkg": "GPKG",
    ".shp": "ESRI Shapefile",
    ".geojson": "GeoJSON",
    ".json": "GeoJSON",
    ".fgb": "FlatGeobuf",
    ".kml": "KML",
}
VECTOR_READ_ONLY = {".zip", ".gml"}
RASTER_EXTS = {".tif", ".tiff", ".vrt", ".img", ".jp2", ".asc", ".nc"}
TABLE_SEPARATORS = {".csv": ",", ".tsv": "\t"}

# Shapefile sidecars and other files that should never be listed as datasets.
SIDECAR_EXTS = {".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx", ".xml", ".aux", ".ovr"}


def data_kind(path: Path) -> Kind | None:
    if path.is_dir():
        return "tiles" if any(p.is_dir() and p.name.isdigit() for p in path.iterdir()) else None
    ext = path.suffix.lower()
    if ext == ".mbtiles":
        return "tiles"
    if ext in VECTOR_DRIVERS or ext in VECTOR_READ_ONLY:
        return "vector"
    if ext in RASTER_EXTS:
        return "raster"
    if ext in TABLE_SEPARATORS:
        return "table"
    return None


def is_dataset(path: Path) -> bool:
    return path.is_file() and data_kind(path) in ("vector", "raster", "table")


def require_kind(path: Path) -> Kind:
    kind = data_kind(path)
    if kind is None:
        raise AgisError(f"{path.name} is not a supported vector, raster, or table file")
    return kind


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or TSV. A vector file contributes its attribute columns only."""
    separator = TABLE_SEPARATORS.get(path.suffix.lower())
    if separator is None:
        if require_kind(path) != "vector":
            raise AgisError(f"{path.name} is not a table")
        gdf = read_vector(path)
        return pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
    try:
        table = pd.read_csv(path, sep=separator)
    except pd.errors.EmptyDataError as exc:
        raise AgisError(f"{path.name} is empty") from exc
    return table


def read_vector(path: Path) -> gpd.GeoDataFrame:
    if path.suffix.lower() not in {".geojson", ".json"}:
        return gpd.read_file(path)
    # A FeatureCollection is one JSON object, so GDAL's default size cap rejects large
    # files before any features are read. Lift it only for this read.
    previous = pyogrio.get_gdal_config_option("OGR_GEOJSON_MAX_OBJ_SIZE")
    pyogrio.set_gdal_config_options({"OGR_GEOJSON_MAX_OBJ_SIZE": "0"})
    try:
        return gpd.read_file(path)
    finally:
        pyogrio.set_gdal_config_options({"OGR_GEOJSON_MAX_OBJ_SIZE": previous})


def write_vector(gdf: gpd.GeoDataFrame, path: Path) -> None:
    driver = VECTOR_DRIVERS.get(path.suffix.lower())
    if driver is None:
        raise AgisError(f"can't write vector data as {path.suffix}")
    if driver == "KML" and gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gdf.to_file(path, driver=driver)


def read_crs(path: Path) -> CRS | None:
    kind = require_kind(path)
    if kind == "raster":
        with rasterio.open(path) as src:
            return CRS.from_user_input(src.crs) if src.crs else None
    info = pyogrio.read_info(path)
    return CRS.from_user_input(info["crs"]) if info.get("crs") else None


def vector_mask(path: Path, target_crs: CRS | None) -> gpd.GeoDataFrame:
    if require_kind(path) != "vector":
        raise AgisError(f"{path.name} must be a vector file to use as a mask or overlay")
    gdf = read_vector(path)
    if target_crs is not None and gdf.crs is not None and not gdf.crs.equals(target_crs):
        gdf = gdf.to_crs(target_crs)
    return gdf
