import pytest

from agis.ops.attributes import Condition, Filter
from agis.ops.geoprocessing import Buffer, Clip, Difference, Info, Reproject
from agis.ops.merge import Merge
from agis.plan import Plan
from agis.planner import fill_numbers, keep_named_outputs, use_named_files


def _chain(value):
    return Plan(steps=[
        Reproject(input="fishing.fgb", crs="EPSG:3857"),
        Buffer(input="$step1", distance=1, units="kilometers"),
        Filter(input="$step2", where=[Condition(field="shape_id", op="<", value=value)]),
    ])


@pytest.mark.parametrize("request_text", [
    "reproj fishing.fgb to 3857 and then buffer by 1km then filter where shape_id is less than 1000",
    "reproj fishing.fgb to 3857 and then buffer by 1km then filter for shape_id less than 1000",
    "filter fishing.fgb for shape id less than 1,000",
])
@pytest.mark.parametrize("value", ["testing", ""])
def test_fill_numbers_takes_the_number_after_the_field(request_text, value):
    filled = fill_numbers(_chain(value), request_text)
    expected = "1,000" if "1,000" in request_text else "1000"
    assert filled.steps[2].where[0].value == expected


def test_fill_numbers_leaves_a_condition_without_a_number_in_the_request():
    filled = fill_numbers(_chain("testing"), "buffer fishing by 1km then filter shape_id")
    assert filled.steps[2].where[0].value == "testing"


def test_fill_numbers_keeps_a_good_value():
    filled = fill_numbers(_chain("500"), "filter shape_id less than 1000")
    assert filled.steps[2].where[0].value == "500"


def _outputs(plan: Plan) -> list:
    return [getattr(step, "output", None) for step in plan.steps]


def test_invented_outputs_are_cleared():
    plan = Plan(steps=[
        Reproject(input="fishing.fgb", crs="EPSG:3857", output="fishing_reproj.gpkg"),
        Buffer(input="$step1", distance=1, units="kilometers", output="fishing_buffer.gpkg"),
    ])
    cleaned = keep_named_outputs(plan, "reproj fishing.fgb to 3857 and then buffer by 1km")
    assert _outputs(cleaned) == [None, None]


def test_a_step_reading_an_invented_name_uses_the_step_reference():
    plan = Plan(steps=[
        Reproject(input="fishing.fgb", crs="EPSG:3857", output="fishing_reproj.gpkg"),
        Buffer(input="fishing_reproj.gpkg", distance=1, units="kilometers",
               output="fishing_buffer.gpkg"),
        Filter(input="fishing_buffer.gpkg",
               where=[Condition(field="shape_id", op="<", value="1000")]),
    ])
    cleaned = keep_named_outputs(plan, "reproj fishing.fgb to 3857, buffer 1km, shape_id < 1000")
    assert [s.input for s in cleaned.steps] == ["fishing.fgb", "$step1", "$step2"]
    assert _outputs(cleaned) == [None, None, None]


def test_a_step_that_refers_to_itself_means_the_previous_step():
    plan = Plan(steps=[
        Reproject(input="fishing.fgb", crs="EPSG:3857"),
        Buffer(input="$step2", distance=1),
    ])
    assert keep_named_outputs(plan, "x").steps[1].input == "$step1"


def test_a_named_output_is_kept():
    plan = Plan(steps=[
        Buffer(input="streams.gpkg", distance=30),
        Difference(input="parcels.shp", other="$step1", output="buildable.gpkg"),
    ])
    request = "buffer the streams by 30 m and cut them out of the parcels, save it as buildable.gpkg"
    assert _outputs(keep_named_outputs(plan, request)) == [None, "buildable.gpkg"]


def test_a_named_stem_keeps_no_extension():
    plan = Plan(steps=[Buffer(input="parks.geojson", distance=30, output="buildable.gpkg")])
    cleaned = keep_named_outputs(plan, "buffer parks by 30 m and save it as buildable.")
    assert _outputs(cleaned) == ["buildable"]


def test_an_input_name_is_not_an_output_name():
    plan = Plan(steps=[Buffer(input="fishing.fgb", distance=30, output="fishing.gpkg")])
    assert _outputs(keep_named_outputs(plan, "buffer fishing by 30 m")) == [None]


def test_a_save_as_name_the_model_left_out_goes_on_the_last_step():
    plan = Plan(steps=[
        Reproject(input="zones.gpkg", crs="EPSG:3857"),
        Buffer(input="$step1", distance=10),
    ])
    cleaned = keep_named_outputs(plan, "reproject zones to 3857, buffer 10 m, save as out.fgb.")
    assert _outputs(cleaned) == [None, "out.fgb"]


def test_steps_without_output_are_untouched():
    plan = Plan(steps=[Info(input="roads.gpkg")])
    assert keep_named_outputs(plan, "what is in roads.gpkg").steps == plan.steps


def test_a_named_extension_replaces_the_same_stem_sibling():
    plan = Plan(steps=[Reproject(
        input="osm_land_polygons.gpkg", crs="EPSG:3857", output="osm_land_polygons.geojson",
    )])
    request = "make osm_land_polygons.geojson crs 3857"
    fixed = keep_named_outputs(use_named_files(plan, request), request)
    assert fixed.steps[0].input == "osm_land_polygons.geojson"
    assert fixed.steps[0].output is None


def test_two_named_extensions_are_left_alone():
    plan = Plan(steps=[Reproject(input="osm_land_polygons.gpkg", crs="EPSG:3857")])
    request = "reproject osm_land_polygons.gpkg, not osm_land_polygons.geojson"
    assert use_named_files(plan, request).steps[0].input == "osm_land_polygons.gpkg"


def test_use_named_files_leaves_step_references():
    plan = Plan(steps=[
        Reproject(input="fishing.fgb", crs="EPSG:3857"),
        Buffer(input="$step1", distance=1),
    ])
    assert use_named_files(plan, "reproject fishing.fgb").steps[1].input == "$step1"


def test_a_named_extension_replaces_another_input_field():
    plan = Plan(steps=[Clip(input="roads.gpkg", mask="county.gpkg")])
    fixed = use_named_files(plan, "clip roads.gpkg to county.geojson")
    assert fixed.steps[0].input == "roads.gpkg"
    assert fixed.steps[0].mask == "county.geojson"


def test_a_named_extension_replaces_a_file_in_a_list():
    plan = Plan(steps=[Merge(input="zones.gpkg", others=["parks.gpkg"])])
    fixed = use_named_files(plan, "append parks.geojson onto zones.gpkg")
    assert fixed.steps[0].others == ["parks.geojson"]


def test_a_different_output_name_is_kept():
    plan = Plan(steps=[Reproject(input="parks.geojson", crs="EPSG:3857", output="parks.gpkg")])
    request = "reproject parks.geojson to parks.gpkg"
    cleaned = keep_named_outputs(use_named_files(plan, request), request)
    assert cleaned.steps[0].input == "parks.geojson"
    assert cleaned.steps[0].output == "parks.gpkg"


def test_save_as_the_input_name_is_kept():
    plan = Plan(steps=[Reproject(
        input="osm_land_polygons.geojson", crs="EPSG:3857", output="osm_land_polygons.geojson",
    )])
    request = "reproject osm_land_polygons.geojson save as osm_land_polygons.geojson"
    assert keep_named_outputs(plan, request).steps[0].output == "osm_land_polygons.geojson"
