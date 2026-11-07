"""Turns a request plus file context into a validated Plan with a single model call."""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import ValidationError

from agis.context import Context
from agis.errors import LLMError, PlanError
from agis.llm import Backend, Message
from agis.ops.attributes import Condition, Filter, Select
from agis.ops.geoprocessing import Buffer, Clip, Difference, Info, Reproject
from agis.ops.join import Join
from agis.ops.merge import Merge
from agis.ops.tiling import TileRaster
from agis.plan import Plan, describe_ops, plan_schema

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
- Only set "output" when the user names an output file. Otherwise leave it out; names are \
generated automatically.
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
