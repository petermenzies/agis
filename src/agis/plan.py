"""The plan the model produces: an ordered list of operations."""

from __future__ import annotations

import re
import typing
from functools import cache
from typing import Annotated

from pydantic import BaseModel, Field, TypeAdapter

from agis.ops import REGISTRY, OpArgs

REF = re.compile(r"^\$step(\d+)$")

Step = Annotated[typing.Union[tuple(REGISTRY.values())], Field(discriminator="op")]  # noqa: UP007


class Plan(BaseModel):
    steps: list[Step] = Field(
        description="The one operation to run, or none if the request can't be done in one "
        "operation."
    )
    message: str = Field(
        "",
        description="One short sentence for the user. If the request can't be done with the "
        "available operations, or needs more than one operation, leave steps empty and "
        "explain why here.",
    )


def step_ref(value: str | None) -> int | None:
    """The 1-based step number if value is a "$stepN" reference."""
    if value is None:
        return None
    m = REF.match(value.strip())
    return int(m.group(1)) if m else None


@cache
def plan_schema() -> dict:
    """JSON schema used for constrained decoding.

    "op" and "message" have defaults so plans can be built directly in Python, but the model
    must always emit them, so they are forced into the required lists here.
    """
    schema = Plan.model_json_schema()
    schema["required"] = ["steps", "message"]
    schema["properties"]["steps"]["maxItems"] = 1
    for definition in schema.get("$defs", {}).values():
        if "op" in definition.get("properties", {}):
            required = [r for r in definition.get("required", []) if r != "op"]
            definition["required"] = ["op", *required]
    return schema


def parse_step(data: dict) -> OpArgs:
    return TypeAdapter(Step).validate_python(data)


def describe_ops() -> str:
    """A compact signature per operation for the system prompt."""
    lines = []
    for name, cls in REGISTRY.items():
        params = []
        for field_name, info in cls.model_fields.items():
            if field_name == "op":
                continue
            params.append(field_name if info.is_required() else f"{field_name}?")
        lines.append(f"- {name}({', '.join(params)}): {cls.summary}")
        for field_name, info in cls.model_fields.items():
            if field_name not in ("op", "input", "output") and info.description:
                lines.append(f"    {field_name}: {info.description}")
    return "\n".join(lines)
