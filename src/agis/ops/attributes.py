"""Row filtering and column selection. These don't change geometry or CRS."""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agis.errors import AgisError
from agis.ops.base import OpArgs, StepResult
from agis.ops.geoprocessing import INPUT_DESC, OUTPUT_DESC
from agis.ops.io import read_vector, write_vector

CompareOp = Literal["==", "!=", ">", ">=", "<", "<=", "contains", "in", "is_null", "not_null"]
_OP_WORD = {"==": "eq", "!=": "ne", ">": "gt", ">=": "gte", "<": "lt", "<=": "lte",
            "contains": "has", "in": "in", "is_null": "null", "not_null": "present"}


class Condition(BaseModel):
    """One attribute test. `op` here is the comparison, not the step name."""

    model_config = ConfigDict(extra="forbid")

    field: str
    op: CompareOp
    value: str | None = None

    @model_validator(mode="after")
    def _value_required(self):
        if self.op not in ("is_null", "not_null") and not (self.value and self.value.strip()):
            raise ValueError(f"{self.op} needs a value")
        return self


class Filter(OpArgs):
    op: Literal["filter"] = "filter"
    input: str = Field(description=INPUT_DESC)
    where: list[Condition] = Field(
        min_length=1,
        description="Conditions, each with field, op and value. op is one of ==, !=, >, >=, "
        "<, <=, contains, in, is_null, not_null. value is text; for in, separate values with "
        "commas. Leave value null for is_null and not_null.",
    )
    match: Literal["all", "any"] = Field(
        "all", description='"all" keeps rows matching every condition; "any" keeps rows matching one.'
    )
    output: str | None = Field(None, description=OUTPUT_DESC)

    summary: ClassVar = (
        "Keep vector features whose attributes match the conditions. Does not change geometry."
    )

    def suffix(self) -> str:
        if len(self.where) != 1:
            return "filtered"
        cond = self.where[0]
        parts = [cond.field, _OP_WORD[cond.op]]
        if cond.value:
            parts.append(cond.value.split(",")[0])
        return _safe("_".join(parts))

    def run(self, inputs: dict[str, Path], output: Path | None) -> StepResult:
        gdf = read_vector(inputs["input"])
        masks = [_mask(gdf, cond) for cond in self.where]
        keep = masks[0]
        for mask in masks[1:]:
            keep = keep & mask if self.match == "all" else keep | mask
        result = gdf.loc[keep].reset_index(drop=True)
        write_vector(result, output)
        return StepResult(output, "vector", f"kept {len(result)} of {len(gdf)} features")


class Select(OpArgs):
    op: Literal["select"] = "select"
    input: str = Field(description=INPUT_DESC)
    columns: list[str] = Field(
        min_length=1, description="Attribute fields to keep. The geometry column is always kept."
    )
    output: str | None = Field(None, description=OUTPUT_DESC)

    summary: ClassVar = (
        "Keep only the named attribute columns of a vector file, plus the geometry."
    )

    def suffix(self) -> str:
        return _safe("_".join(self.columns[:3])) or "columns"

    def run(self, inputs: dict[str, Path], output: Path | None) -> StepResult:
        gdf = read_vector(inputs["input"])
        geom = gdf.geometry.name
        chosen: list[str] = []
        for name in self.columns:
            if name.casefold() == geom.casefold():
                continue
            field = resolve_field(gdf, name)
            if field not in chosen:
                chosen.append(field)
        if not chosen:
            raise AgisError("select needs at least one attribute column")
        result = gdf[[*chosen, geom]]
        write_vector(result, output)
        dropped = [c for c in gdf.columns if c not in result.columns]
        return StepResult(output, "vector",
                          f"kept {', '.join(chosen)}; dropped {', '.join(dropped) or 'nothing'}")


def resolve_field(gdf, name: str) -> str:
    """Match a field name, ignoring case. Geometry is not an attribute field."""
    geom = gdf.geometry.name
    attributes = [c for c in gdf.columns if c != geom]
    if name in attributes:
        return name
    matches = [c for c in attributes if c.casefold() == name.casefold()]
    if len(matches) == 1:
        return matches[0]
    if name.casefold() == geom.casefold():
        raise AgisError(f"{name!r} is the geometry column, not an attribute")
    raise AgisError(f"no field named {name!r}; available fields: {', '.join(attributes)}")


def _mask(gdf, cond: Condition) -> pd.Series:
    field = resolve_field(gdf, cond.field)
    series = gdf[field]
    if cond.op == "is_null":
        return series.isna()
    if cond.op == "not_null":
        return series.notna()

    value = _unquote(cond.value or "")
    text = pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)
    if cond.op == "contains":
        if not text:
            raise AgisError(f"contains needs a text field, and {field} is numeric")
        return series.astype("string").str.contains(re.escape(value), case=False, na=False)
    if cond.op == "in":
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if not parts:
            raise AgisError("in needs at least one value")
        if text:
            wanted = {p.casefold() for p in parts}
            return series.astype("string").str.casefold().isin(wanted).fillna(False)
        return series.isin([_number(p, field) for p in parts]).fillna(False)
    if text:
        if cond.op not in ("==", "!="):
            raise AgisError(f"{cond.op} needs a numeric field, and {field} is text")
        same = series.astype("string").str.casefold().eq(value.casefold())
        return same.fillna(False) if cond.op == "==" else ~same & series.notna()

    number = _number(value, field)
    compared = {"==": series.eq, "!=": series.ne, ">": series.gt, ">=": series.ge,
                "<": series.lt, "<=": series.le}[cond.op](number)
    if cond.op == "!=":
        compared = compared & series.notna()
    return compared.fillna(False)


def _number(value: str, field: str) -> float:
    try:
        return float(value.replace(",", ""))
    except ValueError:
        raise AgisError(f"can't compare {field} to {value!r}; it isn't a number") from None


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1].strip()
    return value


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:40]
