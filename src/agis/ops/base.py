from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from agis.ops.io import Kind

REGISTRY: dict[str, type[OpArgs]] = {}


@dataclass
class StepResult:
    output: Path | None
    kind: Kind | None
    message: str = ""


class OpArgs(BaseModel):
    """Base class for an operation. Subclasses are both the JSON schema the model fills in
    and the implementation that runs it."""

    model_config = ConfigDict(extra="forbid")

    # Fields that hold an input path or a "$stepN" reference.
    input_fields: ClassVar[tuple[str, ...]] = ("input",)
    # Input fields (other than "input") that must be vector data.
    vector_fields: ClassVar[tuple[str, ...]] = ()
    # Kinds accepted by the main "input" field.
    accepts: ClassVar[frozenset[Kind]] = frozenset({"vector"})
    # One-line description shown to the model.
    summary: ClassVar[str] = ""

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        super().__pydantic_init_subclass__(**kwargs)
        field = cls.model_fields.get("op")
        if field is not None and field.default is not None:
            REGISTRY[field.default] = cls

    def check(self, kinds: dict[str, Kind | list[Kind]]) -> list[str]:
        """Problems with running this op on inputs of the given kinds."""
        problems = []
        for kind in _kind_list(kinds.get("input")):
            if kind not in self.accepts:
                allowed = " or ".join(sorted(self.accepts))
                problems.append(f"{self.op} needs {allowed} input, got {kind}")
        for name in self.vector_fields:
            bad = [kind for kind in _kind_list(kinds.get(name)) if kind != "vector"]
            if bad:
                problems.append(f"{self.op} needs vector data for {name!r}, got {bad[0]}")
        return problems

    def output_kind(self, input_kind: Kind) -> Kind | None:
        """Kind of data this step writes, or None if it writes nothing."""
        return input_kind

    def output_ext(self, input_path: Path) -> str:
        """File extension (or "" for a directory) of the auto-generated output name."""
        return input_path.suffix.lower()

    def suffix(self) -> str:
        """Text appended to the input's name when auto-naming the output."""
        return getattr(self, "op", "out")

    def run(self, inputs: dict[str, Path | list[Path]], output: Path | None) -> StepResult:
        raise NotImplementedError


def _kind_list(kind: Kind | list[Kind] | None) -> list[Kind]:
    if kind is None:
        return []
    return list(kind) if isinstance(kind, list) else [kind]
