from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import ClassVar, Literal

import morecantile
import numpy as np
import rasterio
from pydantic import Field, model_validator
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, transform_bounds

from agis.errors import AgisError
from agis.ops.base import OpArgs, StepResult
from agis.ops.geoprocessing import INPUT_DESC
from agis.ops.io import Kind

TILE_SIZE = 256
MAX_TILES = 20_000
TMS = morecantile.tms.get("WebMercatorQuad")


class TileRaster(OpArgs):
    op: Literal["tile_raster"] = "tile_raster"
    input: str = Field(description=INPUT_DESC)
    min_zoom: int | None = Field(None, description="Lowest zoom level. Null picks automatically.")
    max_zoom: int | None = Field(None, description="Highest zoom level. Null picks automatically.")
    output: str | None = Field(
        None, description="Output folder name. Leave null unless the user names one."
    )

    accepts: ClassVar = frozenset({"raster"})
    summary: ClassVar = "Cut a raster into an XYZ folder of 256px PNG web map tiles."

    @model_validator(mode="after")
    def _zoom_range(self):
        for z in (self.min_zoom, self.max_zoom):
            if z is not None and not 0 <= z <= 24:
                raise ValueError("zoom levels must be between 0 and 24")
        if None not in (self.min_zoom, self.max_zoom) and self.min_zoom > self.max_zoom:
            raise ValueError("min_zoom must not be greater than max_zoom")
        return self

    def output_kind(self, input_kind: Kind) -> Kind | None:
        return "tiles"

    def output_ext(self, input_path: Path) -> str:
        return ""

    def suffix(self) -> str:
        return "tiles"

    def run(self, inputs: dict[str, Path], output: Path | None) -> StepResult:
        with rasterio.open(inputs["input"]) as src:
            if src.crs is None:
                raise AgisError(f"{inputs['input'].name} has no CRS, so it can't be tiled")
            minz, maxz = self._zooms(src)
            west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            north, south = min(north, 85.0511), max(south, -85.0511)
            tiles = list(TMS.tiles(west, south, east, north, zooms=list(range(minz, maxz + 1))))
            if len(tiles) > MAX_TILES:
                raise AgisError(
                    f"zooms {minz}-{maxz} would make {len(tiles)} tiles (limit {MAX_TILES}); "
                    "lower max_zoom"
                )
            bands = [1, 2, 3] if src.count >= 3 else [1]
            scale = None if all(src.dtypes[b - 1] == "uint8" for b in bands) else _stats(src, bands)

            written = 0
            for tile in tiles:
                if _write_tile(src, tile, bands, scale, output):
                    written += 1
        return StepResult(
            output, "tiles",
            f"wrote {written} tiles, zoom {minz}-{maxz}; URL template {output.name}/{{z}}/{{x}}/{{y}}.png",
        )

    def _zooms(self, src) -> tuple[int, int]:
        maxz = self.max_zoom
        if maxz is None:
            # Pick the zoom whose pixel size is closest to the raster's native resolution.
            with WarpedVRT(src, crs="EPSG:3857") as vrt:
                res = vrt.res[0]
            world = 2 * math.pi * 6378137
            maxz = max(0, min(20, round(math.log2(world / (TILE_SIZE * res)))))
        minz = self.min_zoom if self.min_zoom is not None else max(0, maxz - 4)
        return min(minz, maxz), maxz


def _stats(src, bands: list[int]) -> list[tuple[float, float]]:
    """Approximate 2nd/98th percentile per band, from a decimated read."""
    factor = max(1, max(src.width, src.height) // 1024)
    shape = (len(bands), max(1, src.height // factor), max(1, src.width // factor))
    data = src.read(bands, out_shape=shape, masked=True)
    out = []
    for band in data:
        values = band.compressed()
        if values.size == 0:
            out.append((0.0, 1.0))
            continue
        lo, hi = np.percentile(values, [2, 98])
        out.append((float(lo), float(hi) if hi > lo else float(lo) + 1.0))
    return out


def _write_tile(src, tile, bands, scale, root: Path) -> bool:
    b = TMS.xy_bounds(tile)
    transform = from_bounds(b.left, b.bottom, b.right, b.top, TILE_SIZE, TILE_SIZE)
    with WarpedVRT(src, crs="EPSG:3857", transform=transform, width=TILE_SIZE,
                   height=TILE_SIZE, resampling=Resampling.bilinear) as vrt:
        data = vrt.read(bands, masked=True)
        alpha = vrt.dataset_mask()
    if not alpha.any():
        return False

    if scale is None:
        rgb = data.filled(0).astype("uint8")
    else:
        rgb = np.empty(data.shape, dtype="uint8")
        for i, (lo, hi) in enumerate(scale):
            stretched = (data[i].filled(lo).astype("float64") - lo) / (hi - lo) * 255
            rgb[i] = np.clip(stretched, 0, 255).astype("uint8")
    if rgb.shape[0] == 1:
        rgb = np.repeat(rgb, 3, axis=0)

    path = root / str(tile.z) / str(tile.x) / f"{tile.y}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path, "w", driver="PNG", width=TILE_SIZE, height=TILE_SIZE, count=4,
                           dtype="uint8") as dst:
            dst.write(rgb, [1, 2, 3])
            dst.write(alpha.astype("uint8"), 4)
    return True
