"""Operation registry. Importing this package registers every operation."""

from agis.ops import attributes, geoprocessing, join, merge, tiling  # noqa: F401
from agis.ops.base import REGISTRY, OpArgs, StepResult

__all__ = ["REGISTRY", "OpArgs", "StepResult"]
