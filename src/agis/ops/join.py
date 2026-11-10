"""Attach a CSV or another layer's attributes to a vector layer."""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar, Literal

import geopandas as gpd
import pandas as pd
from pydantic import Field

from agis.errors import AgisError
from agis.ops.base import OpArgs, StepResult
from agis.ops.geoprocessing import INPUT_DESC, OUTPUT_DESC
from agis.ops.io import Kind, read_table, read_vector, write_vector

_KEY = "_agis_join_key"


class Join(OpArgs):
    op: Literal["join"] = "join"
    input: str = Field(description=INPUT_DESC)
    table: str = Field(
        description="CSV, TSV, or vector file whose attributes are joined on. "
        "A vector file contributes columns, not geometry."
    )
    left_on: str | None = Field(
        None,
        description="Field on the vector layer. Leave null to use the single column name the "
        "two files share.",
    )
    right_on: str | None = Field(
        None,
        description="Field on the table. Leave null when it has the same name as left_on, or "
        "when the shared column should be detected.",
    )
    how: Literal["left", "inner"] = Field(
        "left",
        description='"left" keeps every feature; "inner" keeps only features that match a row.',
    )
    output: str | None = Field(None, description=OUTPUT_DESC)

    input_fields: ClassVar = ("input", "table")
    summary: ClassVar = (
        "Join a table, such as a CSV, onto a vector layer using a shared attribute."
    )

    def check(self, kinds: dict[str, Kind]) -> list[str]:
        problems = super().check(kinds)
        kind = kinds.get("table")
        if kind is not None and kind not in ("table", "vector"):
            problems.append(f"join needs a table or vector for 'table', got {kind}")
        return problems

    def suffix(self) -> str:
        if not self.table or self.table.startswith("$"):
            return "joined"
        stem = re.sub(r"[^A-Za-z0-9]+", "_", Path(self.table).stem).strip("_")
        return f"join_{stem}" if stem else "joined"

    def run(self, inputs: dict[str, Path], output: Path | None) -> StepResult:
        gdf = read_vector(inputs["input"])
        table = read_table(inputs["table"])
        left_key, right_key = _keys(gdf, table, self.left_on, self.right_on)

        left = gdf.copy()
        right = table.copy()
        right = right.rename(columns=_renames(left, right, right_key))
        left[_KEY] = left[left_key].map(_norm_key)
        right[_KEY] = right[right_key].map(_norm_key)
        right = right.drop(columns=[right_key])

        merged = left.merge(right, on=_KEY, how=self.how, indicator=True)
        matched = int((merged["_merge"] != "left_only").sum())
        merged = merged.drop(columns=[_KEY, "_merge"])
        result = gpd.GeoDataFrame(merged, geometry=gdf.geometry.name, crs=gdf.crs)
        write_vector(result, output)

        added = [c for c in result.columns if c not in gdf.columns]
        key = left_key if left_key == right_key else f"{left_key} = {right_key}"
        message = f"joined on {key}; {matched} of {len(gdf)} features matched"
        if len(result) != len(gdf):
            message += f"; {len(gdf)} features became {len(result)}"
        if added:
            message += f"; added {', '.join(added)}"
        return StepResult(output, "vector", message)


def _keys(gdf: gpd.GeoDataFrame, table: pd.DataFrame, left_on: str | None,
          right_on: str | None) -> tuple[str, str]:
    left_fields = _attributes(gdf)
    right_fields = list(table.columns)
    if left_on or right_on:
        left_name = left_on or right_on
        right_name = right_on or left_on
        return (_resolve(left_fields, left_name), _resolve(right_fields, right_name))

    shared = [(left, right) for left in left_fields
              if (right := _match(right_fields, left)) is not None]
    if len(shared) == 1:
        return shared[0]
    if not shared:
        raise AgisError(
            "no column name in common. "
            f"vector fields: {', '.join(left_fields) or 'none'}; "
            f"table fields: {', '.join(right_fields) or 'none'}"
        )
    names = ", ".join(sorted({left for left, _ in shared}))
    raise AgisError(f"several shared columns ({names}); set left_on and right_on")


def _attributes(gdf: gpd.GeoDataFrame) -> list[str]:
    return [column for column in gdf.columns if column != gdf.geometry.name]


def _resolve(columns: list[str], name: str) -> str:
    found = _match(columns, name)
    if found is None:
        raise AgisError(f"no field named {name!r}; available fields: {', '.join(columns)}")
    return found


def _match(columns: list[str], name: str) -> str | None:
    if name in columns:
        return name
    matches = [column for column in columns if column.casefold() == name.casefold()]
    return matches[0] if len(matches) == 1 else None


def _renames(left: gpd.GeoDataFrame, right: pd.DataFrame, right_key: str) -> dict[str, str]:
    taken = {column.casefold() for column in left.columns}
    renames = {}
    for column in right.columns:
        if column == right_key:
            continue
        if column.casefold() in taken:
            renamed = f"{column}_table"
            renames[column] = renamed
            taken.add(renamed.casefold())
        taken.add(column.casefold())
    return renames


def _norm_key(value):
    """Comparable join key: case-insensitive text, and 6, 6.0, and "6" all match."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return pd.NA
    if isinstance(value, bool):
        return str(value).casefold()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return str(int(float(text)))
    return text.casefold()
