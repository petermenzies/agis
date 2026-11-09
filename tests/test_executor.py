import json

import geopandas as gpd
import pytest
from rich.console import Console

from agis.crs import resolve_crs
from agis.errors import PlanError, StepError
from agis.executor import resolve, run
from agis.ops.geoprocessing import Buffer, Clip, Convert, Difference, Dissolve, Reproject
from agis.ops.merge import Merge
from agis.plan import Plan, plan_schema
from agis.session import Options, Session


def test_chain_uses_intermediate_and_lineage_name(data, tmp_path):
    plan = Plan(steps=[Clip(input="roads.gpkg", mask="county.shp"),
                       Reproject(input="$step1", crs="web mercator")])
    work = tmp_path / "work"
    steps = resolve(plan, tmp_path, work)
    assert steps[0].temporary and steps[0].output.parent == work
    assert steps[1].output == (tmp_path / "roads_clipped_3857.gpkg").resolve()
    run(steps)
    assert gpd.read_file(steps[1].output).crs.to_epsg() == 3857


def test_chain_keeps_the_input_format(data, tmp_path):
    plan = Plan(steps=[Dissolve(input="parks.geojson"),
                       Buffer(input="$step1", distance=10)])
    work = tmp_path / "work"
    steps = resolve(plan, tmp_path, work)
    assert steps[0].temporary and steps[0].output.suffix == ".gpkg"
    assert steps[1].output == (tmp_path / "parks_dissolved_buffer10m.geojson").resolve()


def test_chain_keeps_a_converted_format(data, tmp_path):
    plan = Plan(steps=[Convert(input="zones.gpkg", format="geojson"),
                       Dissolve(input="$step1")])
    steps = resolve(plan, tmp_path, tmp_path / "w")
    assert steps[0].output.suffix == ".geojson"
    assert steps[1].output == (tmp_path / "zones_dissolved.geojson").resolve()


def test_named_output_keeps_its_extension(data, tmp_path):
    plan = Plan(steps=[Dissolve(input="parks.geojson"),
                       Buffer(input="$step1", distance=10, output="kept.gpkg")])
    steps = resolve(plan, tmp_path, tmp_path / "w")
    assert steps[1].output == (tmp_path / "kept.gpkg").resolve()


def test_named_output_without_extension_gets_the_input_format(data, tmp_path):
    plan = Plan(steps=[Dissolve(input="parks.geojson"),
                       Buffer(input="$step1", distance=10, output="buildable")])
    steps = resolve(plan, tmp_path, tmp_path / "w")
    assert steps[1].output == (tmp_path / "buildable.geojson").resolve()


def test_session_ignores_invented_gpkg_outputs(data, tmp_path):
    backend = FakeBackend([{"steps": [
        {"op": "reproject", "input": "parks.geojson", "crs": "EPSG:3857",
         "output": "parks_reproj.gpkg"},
        {"op": "buffer", "input": "$step1", "distance": 10, "output": "parks_buffer.gpkg"},
    ]}])
    session = Session(backend, tmp_path, console=Console(quiet=True))
    created = session.handle("reproj parks.geojson to 3857 and then buffer by 10 m",
                             Options(yes=True))
    assert [p.name for p in created] == ["parks_3857_buffer10m.geojson"]


def test_missing_file_and_bad_ref_reported_together(data, tmp_path):
    plan = Plan(steps=[Buffer(input="rivers.shp", distance=5),
                       Dissolve(input="$step3")])
    with pytest.raises(PlanError) as exc:
        resolve(plan, tmp_path, tmp_path / "w")
    assert len(exc.value.problems) == 2 and exc.value.retryable


def test_extension_is_optional(data, tmp_path):
    steps = resolve(Plan(steps=[Dissolve(input="zones")]), tmp_path, tmp_path / "w")
    assert steps[0].inputs["input"].name == "zones.gpkg"


def test_existing_output_gets_a_number(data, tmp_path):
    (tmp_path / "zones_dissolved.gpkg").write_text("x")
    (tmp_path / "zones_dissolved_2.gpkg").write_text("x")
    steps = resolve(Plan(steps=[Dissolve(input="zones.gpkg")]), tmp_path, tmp_path / "w")
    assert steps[0].output == (tmp_path / "zones_dissolved_3.gpkg").resolve()


def test_named_output_gets_a_number(data, tmp_path):
    (tmp_path / "custom.gpkg").write_text("x")
    steps = resolve(Plan(steps=[Dissolve(input="zones.gpkg", output="custom.gpkg")]),
                    tmp_path, tmp_path / "w")
    assert steps[0].output == (tmp_path / "custom_2.gpkg").resolve()


def test_force_overwrites_an_existing_output(data, tmp_path):
    (tmp_path / "zones_dissolved.gpkg").write_text("x")
    steps = resolve(Plan(steps=[Dissolve(input="zones.gpkg")]), tmp_path, tmp_path / "w",
                    force=True)
    assert steps[0].output == (tmp_path / "zones_dissolved.gpkg").resolve()


def test_output_cannot_overwrite_input(data, tmp_path):
    plan = Plan(steps=[Dissolve(input="zones.gpkg", output="zones.gpkg")])
    with pytest.raises(PlanError, match="overwrite its input"):
        resolve(plan, tmp_path, tmp_path / "w", force=True)


def test_kind_mismatch(data, tmp_path):
    plan = Plan(steps=[Difference(input="dem.tif", other="parks.geojson")])
    with pytest.raises(PlanError, match="needs vector"):
        resolve(plan, tmp_path, tmp_path / "w")


def test_merge_resolves_a_step_and_several_files(data, tmp_path):
    plan = Plan(steps=[
        Buffer(input="parks.geojson", distance=10),
        Merge(input="zones.gpkg", others=["$step1", "county.shp"]),
    ])
    steps = resolve(plan, tmp_path, tmp_path / "w")
    assert steps[1].inputs["others"] == [steps[0].output, data["county"].resolve()]
    assert steps[1].output == (tmp_path / "zones_merged.gpkg").resolve()
    run(steps)
    assert len(gpd.read_file(steps[1].output)) == 15


def test_merge_rejects_a_raster(data, tmp_path):
    plan = Plan(steps=[Merge(input="zones.gpkg", others=["dem.tif"])])
    with pytest.raises(PlanError, match="needs vector"):
        resolve(plan, tmp_path, tmp_path / "w")


def test_merge_reports_a_missing_file(data, tmp_path):
    plan = Plan(steps=[Merge(input="zones.gpkg", others=["parks.geojson", "missing.geojson"])])
    with pytest.raises(PlanError, match="missing.geojson"):
        resolve(plan, tmp_path, tmp_path / "w")


def test_merge_output_cannot_overwrite_an_appended_file(data, tmp_path):
    plan = Plan(steps=[Merge(input="zones.gpkg", others=["parks.geojson"],
                             output="parks.geojson")])
    with pytest.raises(PlanError, match="overwrite its input"):
        resolve(plan, tmp_path, tmp_path / "w", force=True)


def test_failed_run_removes_its_outputs(data, tmp_path):
    plan = Plan(steps=[Dissolve(input="zones.gpkg", output="a.gpkg"),
                       Dissolve(input="zones.gpkg", by="nope", output="b.gpkg")])
    with pytest.raises(StepError):
        run(resolve(plan, tmp_path, tmp_path / "w"))
    assert not (tmp_path / "a.gpkg").exists()


def test_schema_requires_op():
    for definition in plan_schema()["$defs"].values():
        if "op" in definition.get("properties", {}):
            assert definition["required"][0] == "op"


@pytest.mark.parametrize("text,code", [("3857", 3857), ("EPSG:4326", 4326), ("epsg 2227", 2227),
                                       ("web mercator", 3857), ("WGS84", 4326)])
def test_resolve_crs(text, code):
    assert resolve_crs(text).to_epsg() == code


class FakeBackend:
    model = "fake"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_json(self, messages, schema):
        self.calls.append(messages)
        return json.dumps(self.responses.pop(0))


def test_session_replans_after_step_failure(data, tmp_path):
    backend = FakeBackend([
        {"steps": [{"op": "dissolve", "input": "zones.gpkg", "by": "landuse"}]},
        {"steps": [{"op": "dissolve", "input": "zones.gpkg", "by": "zone"}]},
    ])
    session = Session(backend, tmp_path, console=Console(quiet=True))
    created = session.handle("dissolve zones by land use", Options(yes=True))
    assert [p.name for p in created] == ["zones_dissolved_zone.gpkg"]
    assert "no field named 'landuse'" in backend.calls[1][-1]["content"]
    assert session.recent == created


def test_session_replans_after_invalid_json(data, tmp_path):
    backend = FakeBackend([
        {"steps": [{"op": "clip", "input": "roads.gpkg"}]},
        {"steps": [{"op": "clip", "input": "roads.gpkg", "mask": "county.shp"}]},
    ])
    session = Session(backend, tmp_path, console=Console(quiet=True))
    assert session.handle("clip roads", Options(yes=True))


def test_session_dry_run_writes_nothing(data, tmp_path):
    backend = FakeBackend([{"steps": [{"op": "dissolve", "input": "zones.gpkg"}]}])
    session = Session(backend, tmp_path, console=Console(quiet=True))
    assert session.handle("dissolve zones", Options(dry_run=True)) == []
    assert not (tmp_path / "zones_dissolved.gpkg").exists()


def test_session_empty_plan(data, tmp_path):
    backend = FakeBackend([{"steps": [], "message": "Slope isn't supported."}])
    session = Session(backend, tmp_path, console=Console(quiet=True))
    assert session.handle("slope of dem", Options(yes=True)) == []
