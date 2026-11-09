"""Turns a request plus file context into a validated Plan with a single model call."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from agis.context import Context
from agis.errors import LLMError, PlanError
from agis.llm import Backend, Message
from agis.ops.attributes import NUMERIC_OPS, Condition, Filter, Select, is_number
from agis.ops.geoprocessing import Buffer, Clip, Difference, Info, Reproject
from agis.ops.join import Join
from agis.ops.merge import Merge
from agis.ops.tiling import TileRaster
from agis.plan import Plan, describe_ops, plan_schema, step_ref

SYSTEM_PROMPT = """You are agis, a planner for GIS tasks. Turn the user's request into a JSON plan \
that uses only these operations:

{ops}

Rules:
- Use file names exactly as they appear in the Files list.
- To use the result of an earlier step, set the file argument (input, mask, other, or any \
other) to "$stepN", where N is the 1-based position of that step.
- For attribute queries use one filter even when there are several tests: match "all" means \
and, "any" means or. Use unique to list the distinct values of one attribute; it writes no \
file. Do not add a select unless the user asks to keep columns. Don't reproject for these.
- Use join to attach a CSV or another layer's attributes to a vector layer. Leave left_on and \
right_on empty when the files share one column name.
- Use merge to append the rows of vector files into one layer. The first file is input; every \
other file goes in others, in the order given. Columns line up by name. Use dissolve only when \
geometries should be combined into fewer shapes, and union for a polygon overlay.
- Only set "output" when the user names an output file, and use the extension they named. \
Do not change .fgb, .shp, or .geojson to .gpkg. Otherwise leave it out; the name and the \
input's format are kept automatically. Use convert when the user asks for a different format.
- Only add the steps the user asked for. Clip, difference, union, buffer and merge handle \
different CRSs themselves, so don't add reproject steps for them. Buffer accepts meters, \
kilometers, feet or miles, so keep the user's distance and units as given.
- Write CRSs as EPSG codes when you know them, e.g. "EPSG:3857" for web mercator.
- If the request can't be done with these operations, or refers to a file that isn't listed, \
return an empty steps list and explain why in message. Never substitute a different operation \
for one that isn't available."""


@dataclass
class Example:
    files: str
    request: str
    plan: Plan


EXAMPLES = [
    Example(
        files="- parcels.shp: vector, Polygon, 5210 features, EPSG:2227, fields [apn:str, zoning:str]\n"
        "- streams.gpkg: vector, LineString, 88 features, EPSG:4326, fields [name:str]",
        request="buffer the streams by 30 m and cut those areas out of the parcels, "
        "save it as buildable.gpkg",
        plan=Plan(
            steps=[
                Buffer(input="streams.gpkg", distance=30, units="meters"),
                Difference(input="parcels.shp", other="$step1", output="buildable.gpkg"),
            ],
            message="Buffer streams by 30 m, then remove the buffers from the parcels.",
        ),
    ),
    Example(
        files="- elevation.tif: raster, 1 band float32, 4000x3000 px, 10 metre pixels, EPSG:26910\n"
        "- city.geojson: vector, Polygon, 1 features, EPSG:4326, fields [name:str]",
        request="clip elevation to the city and make an MBTiles file up to zoom 14",
        plan=Plan(
            steps=[
                Clip(input="elevation.tif", mask="city.geojson"),
                TileRaster(input="$step1", max_zoom=14),
            ],
            message="Clip the elevation raster to the city, then write an MBTiles file up to zoom 14.",
        ),
    ),
    Example(
        files="- parcels.shp: vector, Polygon, 5210 features, EPSG:2227, "
        "fields [apn:str, zoning:str, acres:float]",
        request="residential parcels over 1 acre, keeping only apn and zoning, "
        "save as residential.gpkg",
        plan=Plan(
            steps=[
                Filter(
                    input="parcels.shp",
                    where=[
                        Condition(field="zoning", op="==", value="residential"),
                        Condition(field="acres", op=">", value="1"),
                    ],
                ),
                Select(input="$step1", columns=["apn", "zoning"], output="residential.gpkg"),
            ],
            message="Keep residential parcels over 1 acre, with the apn and zoning columns.",
        ),
    ),
    Example(
        files="- shapes.fgb: vector, Polygon, 40 features, EPSG:4326, "
        "fields [country_of_residence:str, sector:str, beds:int, name:str]",
        request="keep shapes in Spain, then keep only the name column",
        plan=Plan(
            steps=[
                Filter(input="shapes.fgb", where=[
                    Condition(field="country_of_residence", op="==", value="Spain"),
                ]),
                Select(input="$step1", columns=["name"]),
            ],
            message="Keep shapes in Spain, then keep only the name column.",
        ),
    ),
    Example(
        files="- shapes.fgb: vector, Polygon, 40 features, EPSG:4326, "
        "fields [country_of_residence:str, sector:str, beds:int]",
        request="filter shapes.fgb where the country of residence is Spain",
        plan=Plan(
            steps=[Filter(input="shapes.fgb", where=[
                Condition(field="country_of_residence", op="==", value="Spain"),
            ])],
            message="Keep shapes whose country of residence is Spain.",
        ),
    ),
    Example(
        files="- shapes.fgb: vector, Polygon, 40 features, EPSG:4326, "
        "fields [country_of_residence:str, sector:str, beds:int]",
        request="shapes where the country of residence is not Spain, or there are at least 10 beds",
        plan=Plan(
            steps=[Filter(
                input="shapes.fgb",
                match="any",
                where=[
                    Condition(field="country_of_residence", op="!=", value="Spain"),
                    Condition(field="beds", op=">=", value="10"),
                ],
            )],
            message="Keep shapes that are outside Spain, or that have at least 10 beds.",
        ),
    ),
    Example(
        files="- shapes.fgb: vector, Polygon, 40 features, EPSG:4326, "
        "fields [country_of_residence:str, sector:str, beds:int, name:str]",
        request="where country of residence is filled in",
        plan=Plan(
            steps=[Filter(input="shapes.fgb", where=[
                Condition(field="country_of_residence", op="not_null"),
            ])],
            message="Keep shapes that have a country of residence.",
        ),
    ),
    Example(
        files="- parcels.shp: vector, Polygon, 5210 features, EPSG:2227, fields [apn:str, zoning:str]\n"
        "- owners.csv: table, 4800 rows, fields [apn:str, owner:str]",
        request="add the owner names from owners.csv onto the parcels",
        plan=Plan(
            steps=[Join(input="parcels.shp", table="owners.csv")],
            message="Join owners.csv onto the parcels using the shared apn column.",
        ),
    ),
    Example(
        files="- zones.gpkg: vector, Polygon, 12 features, EPSG:4326, fields [zone:str]\n"
        "- parks.geojson: vector, Polygon, 2 features, EPSG:4326, fields [name:str, acres:float]",
        request="append the parks onto the zones",
        plan=Plan(
            steps=[Merge(input="zones.gpkg", others=["parks.geojson"])],
            message="Append the park features onto the zones.",
        ),
    ),
    Example(
        files="- trails.kml: vector, LineString, 40 features, EPSG:4326, fields [name:str]",
        request="put trails in web mercator",
        plan=Plan(
            steps=[Reproject(input="trails.kml", crs="EPSG:3857")],
            message="Reproject trails to Web Mercator (EPSG:3857).",
        ),
    ),
    Example(
        files="- elevation.tif: raster, 1 band float32, 4000x3000 px, 10 metre pixels, EPSG:26910",
        request="calculate contour lines every 10 m from elevation.tif",
        plan=Plan(
            steps=[],
            message="Contour lines aren't one of the available operations yet.",
        ),
    ),
    Example(
        files="- trails.kml: vector, LineString, 40 features, EPSG:4326, fields [name:str]\n"
        "- elevation.tif: raster, 1 band float32, 4000x3000 px, 10 metre pixels, EPSG:26910",
        request="what's in elevation.tif? also compute the slope",
        plan=Plan(
            steps=[Info(input="elevation.tif")],
            message="I can describe elevation.tif, but slope isn't one of the available "
            "operations yet.",
        ),
    ),
]


def compact_json(plan: Plan) -> str:
    """Plan JSON with defaults removed (except "op"), matching what the model should write."""
    steps = []
    for step in plan.steps:
        full = step.model_dump()
        defaults = {name: f.default for name, f in type(step).model_fields.items()}
        steps.append({k: v for k, v in full.items() if k == "op" or v != defaults.get(k)})
    return json.dumps({"steps": steps, "message": plan.message})


def keep_named_outputs(plan: Plan, request: str) -> Plan:
    """Clear output names the request doesn't mention, so they are generated instead.

    A name given without an extension keeps the stem; the executor adds the input's format.
    """
    plan = link_steps(plan)
    inputs = {
        Path(value).stem.casefold()
        for step in plan.steps for name in step.input_fields
        for value in _file_values(getattr(step, name, None))
    }
    steps = []
    for step in plan.steps:
        output = getattr(step, "output", None)
        if output:
            path = Path(output)
            if _mentions(request, output) or _mentions(request, path.name):
                pass
            elif path.stem.casefold() not in inputs and _mentions(request, path.stem):
                output = str(path.with_suffix(""))
            else:
                output = None
        if output != getattr(step, "output", None):
            step = step.model_copy(update={"output": output})
        steps.append(step)
    named = _save_as(request)
    last = steps[-1] if steps else None
    if (named and last is not None and "output" in type(last).model_fields
            and not any(getattr(s, "output", None) for s in steps)):
        steps[-1] = last.model_copy(update={"output": named})
    return plan.model_copy(update={"steps": steps})


def link_steps(plan: Plan) -> Plan:
    """Turn a later step's use of an earlier step's output name into "$stepN".

    The model often names an intermediate file and then reads it by that name. A step that
    refers to itself means the step before it.
    """
    written: dict[str, int] = {}
    steps = []
    for number, step in enumerate(plan.steps, start=1):
        update = {}
        for name in step.input_fields:
            raw = getattr(step, name, None)
            values = raw if isinstance(raw, list) else [raw]
            linked = [_link(v, number, written) for v in values]
            if linked != values:
                update[name] = linked if isinstance(raw, list) else linked[0]
        if update:
            step = step.model_copy(update=update)
        output = getattr(step, "output", None)
        if output:
            written.setdefault(Path(output).name.casefold(), number)
        steps.append(step)
    return plan.model_copy(update={"steps": steps})


def _link(value, number: int, written: dict[str, int]):
    if not isinstance(value, str):
        return value
    ref = step_ref(value)
    if ref is not None:
        return f"$step{number - 1}" if ref == number and number > 1 else value
    source = written.get(Path(value).name.casefold())
    return f"$step{source}" if source is not None else value


def fill_numbers(plan: Plan, request: str) -> Plan:
    """Take a numeric comparison's value from the request when the model's isn't a number.

    The number has to follow the field name, so "buffer by 1km then filter shape_id < 1000"
    gives 1000.
    """
    steps = []
    for step in plan.steps:
        if isinstance(step, Filter) and any(
            c.op in NUMERIC_OPS and not is_number(c.value) for c in step.where
        ):
            where = []
            for cond in step.where:
                found = None
                if cond.op in NUMERIC_OPS and not is_number(cond.value):
                    found = _number_after(request, cond.field)
                where.append(cond if found is None else cond.model_copy(update={"value": found}))
            step = step.model_copy(update={"where": where})
        steps.append(step)
    return plan.model_copy(update={"steps": steps})


def _number_after(request: str, field: str) -> str | None:
    name = r"[_\s]+".join(re.escape(part) for part in re.split(r"[_\s]+", field.strip()) if part)
    if not name:
        return None
    pattern = rf"(?<![\w]){name}(?!\w)[^\d\n]{{0,40}}?(-?\d[\d,]*(?:\.\d+)?)(?![\w.])"
    m = re.search(pattern, request, re.IGNORECASE)
    return m.group(1) if m else None


def _save_as(request: str) -> str | None:
    """The file in "save as X" or "write it to X", which the model sometimes leaves out."""
    m = re.search(r"\b(?:save|write|export)\b(?:\s+\w+){0,3}?\s+(?:as|to)\s+"
                  r"[\"']?([\w~./\\-]+\.[A-Za-z0-9]+)", request, re.IGNORECASE)
    return m.group(1).rstrip(".") if m else None


def _mentions(request: str, name: str) -> bool:
    # Whole name only: "roads" is not mentioned by "roads.fgb", and "a.gpkg" not by "data.gpkg".
    pattern = rf"(?<![\w.\-/]){re.escape(name)}(?![\w\-]|\.\w)"
    return re.search(pattern, request, re.IGNORECASE) is not None


def _file_values(value: object) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [v for v in values if isinstance(v, str) and step_ref(v) is None]


def user_message(context_text: str, request: str) -> str:
    return f"{context_text}\n\nRequest: {request}"


class Planner:
    def __init__(self, backend: Backend):
        self.backend = backend

    def messages(self, request: str, context: Context) -> list[Message]:
        messages: list[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(ops=describe_ops())}
        ]
        for ex in EXAMPLES:
            messages.append({"role": "user", "content": user_message(f"Files:\n{ex.files}",
                                                                     ex.request)})
            messages.append({"role": "assistant", "content": compact_json(ex.plan)})
        messages.append({"role": "user", "content": user_message(context.render(), request)})
        return messages

    def plan(self, request: str, context: Context,
             feedback: list[tuple[str, list[str]]] | None = None) -> Plan:
        """Ask the model for a plan.

        feedback holds earlier (raw plan JSON, problems) attempts so the model can fix them.
        Raises InvalidPlanOutput if the response doesn't parse.
        """
        messages = self.messages(request, context)
        for raw, problems in feedback or []:
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": "That plan can't run:\n- " + "\n- ".join(problems)
                + "\nReturn a corrected plan for the same request.",
            })
        raw = self.backend.generate_json(messages, plan_schema())
        if not raw.strip():
            raise LLMError("the model returned an empty response")
        try:
            return Plan.model_validate_json(raw)
        except ValidationError as exc:
            raise InvalidPlanOutput(raw, _validation_problems(exc)) from exc


class InvalidPlanOutput(PlanError):
    def __init__(self, raw: str, problems: list[str]):
        self.raw = raw
        super().__init__(problems)


def _validation_problems(exc: ValidationError) -> list[str]:
    problems = []
    for err in exc.errors()[:8]:
        loc = ".".join(str(p) for p in err["loc"])
        problems.append(f"{loc}: {err['msg']}" if loc else err["msg"])
    return problems
