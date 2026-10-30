from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

import geopandas as gpd
import rasterio
import rasterio.mask
import rasterio.shutil
from pydantic import Field, model_validator
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import box, mapping

from agis.crs import crs_label, crs_suffix, is_auto_utm, resolve_crs
from agis.errors import AgisError
from agis.ops.base import OpArgs, StepResult
from agis.ops.io import Kind, read_vector, require_kind, vector_mask, write_vector

INPUT_DESC = 'Input file name, or "$stepN" to use the output of step N.'
OUTPUT_DESC = "Output file name. Leave null unless the user names one."


def _ref_stem(value: str | None, fallback: str) -> str:
    if not value or value.startswith("$"):
        return fallback
    return Path(value).stem


class Reproject(OpArgs):
    op: Literal["reproject"] = "reproject"
    input: str = Field(description=INPUT_DESC)
    crs: str = Field(
        description='Target CRS, e.g. "EPSG:3857", "EPSG:4326", "utm" for the local UTM zone, '
        "or a file name to match that file's CRS."
    )
    resampling: Literal["nearest", "bilinear", "cubic"] = Field(
        "nearest", description="Raster resampling method. Ignored for vector data."
    )
    output: str | None = Field(None, description=OUTPUT_DESC)

    accepts: ClassVar = frozenset({"vector", "raster"})
    summary: ClassVar = "Reproject vector or raster data to another coordinate reference system."

    def output_ext(self, input_path: Path) -> str:
        if require_kind(input_path) == "raster":
            return ".tif"
        # KML is always WGS 84, so a reprojected copy has to be another format.
        return ".gpkg" if input_path.suffix.lower() == ".kml" else input_path.suffix.lower()

    def suffix(self) -> str:
        if is_auto_utm(self.crs):
            return "utm"
        try:
            return crs_suffix(resolve_crs(self.crs))
        except AgisError:
            return "reprojected"

    def _target(self, sample: gpd.GeoDataFrame | None):
        if is_auto_utm(self.crs):
            if sample is None:
                raise AgisError("auto UTM is only supported for vector data; give an EPSG code")
            return sample.estimate_utm_crs()
        return resolve_crs(self.crs, Path.cwd())

    def run(self, inputs: dict[str, Path], output: Path | None) -> StepResult:
        src_path = inputs["input"]
        if require_kind(src_path) == "vector":
            gdf = read_vector(src_path)
            if gdf.crs is None:
                raise AgisError(f"{src_path.name} has no CRS, so it can't be reprojected")
            target = self._target(gdf)
            write_vector(gdf.to_crs(target), output)
            return StepResult(output, "vector", f"reprojected to {crs_label(target)}")

        target = self._target(None)
        with rasterio.open(src_path) as src:
            if src.crs is None:
                raise AgisError(f"{src_path.name} has no CRS, so it can't be reprojected")
            transform, width, height = calculate_default_transform(
                src.crs, target, src.width, src.height, *src.bounds
            )
            profile = src.profile.copy()
            profile.update(driver="GTiff", crs=target, transform=transform, width=width,
                           height=height)
            with rasterio.open(output, "w", **profile) as dst:
                for band in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band),
                        destination=rasterio.band(dst, band),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target,
                        src_nodata=src.nodata,
                        dst_nodata=src.nodata,
                        resampling=Resampling[self.resampling],
                    )
        return StepResult(output, "raster", f"reprojected to {crs_label(target)}")


class Clip(OpArgs):
    op: Literal["clip"] = "clip"
    input: str = Field(description=INPUT_DESC)
    mask: str | None = Field(
        None, description='Vector file (or "$stepN") whose shapes define the clip area.'
    )
    bbox: list[float] | None = Field(
        None, description="Clip box as [minx, miny, maxx, maxy], used when there is no mask file."
    )
    bbox_crs: str = Field("EPSG:4326", description="CRS of bbox coordinates.")
    output: str | None = Field(None, description=OUTPUT_DESC)

    input_fields: ClassVar = ("input", "mask")
    vector_fields: ClassVar = ("mask",)
    accepts: ClassVar = frozenset({"vector", "raster"})
    summary: ClassVar = "Clip vector or raster data to a mask layer or a bounding box."

    @model_validator(mode="after")
    def _one_area(self):
        if (self.mask is None) == (self.bbox is None):
            raise ValueError("clip needs exactly one of mask or bbox")
        if self.bbox is not None and len(self.bbox) != 4:
            raise ValueError("bbox must be [minx, miny, maxx, maxy]")
        return self

    def output_ext(self, input_path: Path) -> str:
        return ".tif" if require_kind(input_path) == "raster" else input_path.suffix.lower()

    def suffix(self) -> str:
        return "clipped"

    def _mask(self, inputs: dict[str, Path], target_crs) -> gpd.GeoDataFrame:
        if self.bbox is not None:
            area = gpd.GeoDataFrame(geometry=[box(*self.bbox)], crs=resolve_crs(self.bbox_crs))
            return area.to_crs(target_crs) if target_crs is not None else area
        return vector_mask(inputs["mask"], target_crs)

    def run(self, inputs: dict[str, Path], output: Path | None) -> StepResult:
        src_path = inputs["input"]
        if require_kind(src_path) == "vector":
            gdf = read_vector(src_path)
            clipped = gpd.clip(gdf, self._mask(inputs, gdf.crs))
            write_vector(clipped, output)
            return StepResult(output, "vector", f"kept {len(clipped)} of {len(gdf)} features")

        with rasterio.open(src_path) as src:
            mask = self._mask(inputs, src.crs)
            shapes = [mapping(g) for g in mask.geometry if g is not None and not g.is_empty]
            try:
                data, transform = rasterio.mask.mask(src, shapes, crop=True)
            except ValueError as exc:
                raise AgisError("the clip area doesn't overlap the raster") from exc
            profile = src.profile.copy()
            profile.update(driver="GTiff", height=data.shape[1], width=data.shape[2],
                           transform=transform)
            with rasterio.open(output, "w", **profile) as dst:
                dst.write(data)
        return StepResult(output, "raster", f"clipped to {data.shape[2]}x{data.shape[1]} pixels")


class Difference(OpArgs):
    op: Literal["difference"] = "difference"
    input: str = Field(description=INPUT_DESC)
    other: str = Field(description='Vector file (or "$stepN") to subtract from the input.')
    output: str | None = Field(None, description=OUTPUT_DESC)

    input_fields: ClassVar = ("input", "other")
    vector_fields: ClassVar = ("other",)
    summary: ClassVar = "Remove the parts of the input that overlap another vector layer."

    def suffix(self) -> str:
        stem = _ref_stem(self.other, "")
        return f"minus_{stem}" if stem else "difference"

    def run(self, inputs: dict[str, Path], output: Path | None) -> StepResult:
        gdf = read_vector(inputs["input"])
        other = vector_mask(inputs["other"], gdf.crs)
        cutter = other.geometry.union_all()
        result = gdf.copy()
        result["geometry"] = gdf.geometry.difference(cutter)
        result = result[~result.geometry.is_empty & result.geometry.notna()]
        write_vector(result, output)
        return StepResult(output, "vector", f"{len(result)} features remain")


class Union(OpArgs):
    op: Literal["union"] = "union"
    input: str = Field(description=INPUT_DESC)
    other: str = Field(description='Polygon vector file (or "$stepN") to union with the input.')
    output: str | None = Field(None, description=OUTPUT_DESC)

    input_fields: ClassVar = ("input", "other")
    vector_fields: ClassVar = ("other",)
    summary: ClassVar = (
        "Overlay two polygon layers, keeping all areas and the attributes of both."
    )

    def suffix(self) -> str:
        stem = _ref_stem(self.other, "")
        return f"union_{stem}" if stem else "union"

    def run(self, inputs: dict[str, Path], output: Path | None) -> StepResult:
        gdf = read_vector(inputs["input"])
        other = vector_mask(inputs["other"], gdf.crs)
        for layer, path in ((gdf, inputs["input"]), (other, inputs["other"])):
            if not layer.geom_type.isin(["Polygon", "MultiPolygon"]).all():
                raise AgisError(f"union needs polygon layers, but {path.name} isn't polygons")
        result = gpd.overlay(gdf, other, how="union", keep_geom_type=True)
        write_vector(result, output)
        return StepResult(output, "vector", f"{len(result)} features")


UNIT_METERS = {"meters": 1.0, "kilometers": 1000.0, "feet": 0.3048, "miles": 1609.344}
UNIT_ABBR = {"meters": "m", "kilometers": "km", "feet": "ft", "miles": "mi"}


class Buffer(OpArgs):
    op: Literal["buffer"] = "buffer"
    input: str = Field(description=INPUT_DESC)
    distance: float = Field(description="Buffer distance.")
    units: Literal["meters", "kilometers", "feet", "miles"] = Field(
        "meters", description="Units of distance."
    )
    output: str | None = Field(None, description=OUTPUT_DESC)

    summary: ClassVar = "Buffer vector features by a distance. Works in any CRS."

    def suffix(self) -> str:
        return f"buffer{self.distance:g}{UNIT_ABBR[self.units]}"

    def run(self, inputs: dict[str, Path], output: Path | None) -> StepResult:
        gdf = read_vector(inputs["input"])
        if gdf.crs is None:
            raise AgisError(f"{inputs['input'].name} has no CRS, so distances are undefined")
        meters = self.distance * UNIT_METERS[self.units]
        if gdf.crs.is_geographic:
            utm = gdf.estimate_utm_crs()
            buffered = gdf.to_crs(utm)
            buffered["geometry"] = buffered.geometry.buffer(meters)
            buffered = buffered.to_crs(gdf.crs)
        else:
            per_unit = gdf.crs.axis_info[0].unit_conversion_factor or 1.0
            buffered = gdf.copy()
            buffered["geometry"] = gdf.geometry.buffer(meters / per_unit)
        write_vector(buffered, output)
        return StepResult(output, "vector", f"buffered {len(buffered)} features by "
                          f"{self.distance:g} {self.units}")


class Dissolve(OpArgs):
    op: Literal["dissolve"] = "dissolve"
    input: str = Field(description=INPUT_DESC)
    by: str | None = Field(
        None, description="Field to group by. Null merges everything into one feature."
    )
    output: str | None = Field(None, description=OUTPUT_DESC)

    summary: ClassVar = (
        "Combine geometries into one feature, optionally grouped by a field. "
        "This is not for stacking files; use merge to append features."
    )

    def suffix(self) -> str:
        return f"dissolved_{self.by}" if self.by else "dissolved"

    def run(self, inputs: dict[str, Path], output: Path | None) -> StepResult:
        gdf = read_vector(inputs["input"])
        if self.by and self.by not in gdf.columns:
            fields = ", ".join(c for c in gdf.columns if c != gdf.geometry.name)
            raise AgisError(f"no field named {self.by!r}; available fields: {fields}")
        result = gdf.dissolve(by=self.by).reset_index() if self.by else gdf.dissolve()
        write_vector(result, output)
        return StepResult(output, "vector", f"{len(gdf)} features merged into {len(result)}")


VECTOR_FORMATS = {"gpkg": ".gpkg", "shp": ".shp", "geojson": ".geojson", "fgb": ".fgb",
                  "kml": ".kml"}
RASTER_FORMATS = {"tif": ".tif", "cog": ".tif"}


class Convert(OpArgs):
    op: Literal["convert"] = "convert"
    input: str = Field(description=INPUT_DESC)
    format: Literal["gpkg", "shp", "geojson", "fgb", "kml", "tif", "cog"] = Field(
        description="Target format. Vector: gpkg, shp, geojson, fgb, kml. "
        "Raster: tif, or cog for a Cloud Optimized GeoTIFF."
    )
    output: str | None = Field(None, description=OUTPUT_DESC)

    accepts: ClassVar = frozenset({"vector", "raster"})
    summary: ClassVar = "Convert a file to another format."

    def check(self, kinds: dict[str, Kind]) -> list[str]:
        problems = super().check(kinds)
        kind = kinds.get("input")
        if kind == "vector" and self.format not in VECTOR_FORMATS:
            problems.append(f"{self.format} is a raster format but the input is vector")
        if kind == "raster" and self.format not in RASTER_FORMATS:
            problems.append(f"{self.format} is a vector format but the input is raster")
        return problems

    def output_ext(self, input_path: Path) -> str:
        return {**VECTOR_FORMATS, **RASTER_FORMATS}[self.format]

    def suffix(self) -> str:
        return "cog" if self.format == "cog" else ""

    def run(self, inputs: dict[str, Path], output: Path | None) -> StepResult:
        src_path = inputs["input"]
        if require_kind(src_path) == "vector":
            gdf = read_vector(src_path)
            write_vector(gdf, output)
            return StepResult(output, "vector", f"wrote {len(gdf)} features")
        driver = "COG" if self.format == "cog" else "GTiff"
        rasterio.shutil.copy(src_path, output, driver=driver)
        return StepResult(output, "raster", f"wrote {driver}")


class Info(OpArgs):
    op: Literal["info"] = "info"
    input: str = Field(description=INPUT_DESC)

    accepts: ClassVar = frozenset({"vector", "raster", "table"})
    summary: ClassVar = "Describe a file: CRS, extent, feature count, fields, bands, or table columns."

    def output_kind(self, input_kind: Kind) -> Kind | None:
        return None

    def run(self, inputs: dict[str, Path], output: Path | None) -> StepResult:
        from agis.context import describe

        return StepResult(None, None, describe(inputs["input"], detailed=True))

