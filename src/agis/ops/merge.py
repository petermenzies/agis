"""Append features from several vector files into one layer."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

import geopandas as gpd
import pandas as pd
from pydantic import Field
from pyproj import CRS

from agis.crs import crs_label
from agis.errors import AgisError
from agis.ops.base import OpArgs, StepResult
from agis.ops.geoprocessing import INPUT_DESC, OUTPUT_DESC
from agis.ops.io import read_vector, write_vector


class Merge(OpArgs):
    op: Literal["merge"] = "merge"
    input: str = Field(description=INPUT_DESC)
    others: list[str] = Field(
        min_length=1,
        description="Vector files whose features are appended after the input, "
        "in this order.",
    )
    output: str | None = Field(None, description=OUTPUT_DESC)

    input_fields: ClassVar = ("input", "others")
    vector_fields: ClassVar = ("others",)
    summary: ClassVar = (
        "Append the features of other vector files onto a layer. Columns line up by name. "
        "A file in a different CRS is reprojected to match the input."
    )

    def suffix(self) -> str:
        if len(self.others) == 1 and not self.others[0].startswith("$"):
            return f"merged_{Path(self.others[0]).stem}"
        return "merged"

    def run(self, inputs: dict[str, Path | list[Path]], output: Path | None) -> StepResult:
        base = _prepare(read_vector(inputs["input"]))
        frames = [base]
        aligned: list[str] = []
        others = inputs["others"]
        if isinstance(others, Path):
            others = [others]
        for path in others:
            frame, changed = _align(path, base.crs)
            frames.append(frame)
            if changed:
                aligned.append(path.name)
        result = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True), geometry="geometry", crs=base.crs
        )
        write_vector(result, output)
        counts = " + ".join(str(len(frame)) for frame in frames)
        message = f"{len(result)} features ({counts})"
        if aligned and base.crs is not None:
            message += f"; reprojected {', '.join(aligned)} to {crs_label(base.crs)}"
        return StepResult(output, "vector", message)


def _prepare(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.geometry.name != "geometry":
        gdf = gdf.rename_geometry("geometry")
    return gdf


def _align(path: Path, target: CRS | None) -> tuple[gpd.GeoDataFrame, bool]:
    gdf = _prepare(read_vector(path))
    if target is None:
        if gdf.crs is not None:
            raise AgisError(
                f"{path.name} has a CRS but the first file does not, so they can't be aligned"
            )
        return gdf, False
    if gdf.crs is None:
        raise AgisError(f"{path.name} has no CRS, so it can't be aligned with the first file")
    if gdf.crs.equals(target):
        return gdf, False
    return gdf.to_crs(target), True
