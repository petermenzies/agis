from pathlib import Path

from agis.evals import compare, load_cases
from agis.ops.geoprocessing import Buffer, Clip, Reproject
from agis.plan import Plan

CASES = Path(__file__).parent / "evals" / "cases.yaml"


def test_cases_file_parses_and_ops_exist():
    from agis.ops import REGISTRY

    cases = load_cases(CASES)
    assert len({c["id"] for c in cases}) == len(cases)
    assert all(step["op"] in REGISTRY for c in cases for step in c["expect"])


def test_buffer_units_are_compared_by_distance(data, tmp_path):
    plan = Plan(steps=[Buffer(input="parks.geojson", distance=1609.344, units="meters")])
    assert compare(plan, [{"op": "buffer", "input": "parks.geojson", "distance": 1,
                           "units": "miles"}], tmp_path) == []


def test_crs_and_paths_are_normalized(data, tmp_path):
    plan = Plan(steps=[Reproject(input="./roads.gpkg", crs="3857")])
    assert compare(plan, [{"op": "reproject", "input": "roads.gpkg", "crs": "EPSG:3857"}],
                   tmp_path) == []


def test_concrete_utm_zone_counts_as_utm(data, tmp_path):
    expect = [{"op": "reproject", "input": "zones.gpkg", "crs": "utm"}]
    for crs in ("EPSG:32610", "utm"):
        assert compare(Plan(steps=[Reproject(input="zones.gpkg", crs=crs)]), expect,
                       tmp_path) == []
    assert compare(Plan(steps=[Reproject(input="zones.gpkg", crs="EPSG:32611")]), expect,
                   tmp_path)


def test_filter_conditions_ignore_order_and_number_spelling(data, tmp_path):
    from agis.ops.attributes import Condition, Filter

    plan = Plan(steps=[Filter(input="roads.gpkg", where=[
        Condition(field="lanes", op=">=", value="6.0"),
        Condition(field="name", op="==", value="Market St"),
    ])])
    expect = [{"op": "filter", "input": "roads.gpkg", "where": [
        {"field": "name", "op": "==", "value": "market st"},
        {"field": "lanes", "op": ">=", "value": "6"},
    ]}]
    assert compare(plan, expect, tmp_path) == []


def test_mismatches_are_reported(data, tmp_path):
    plan = Plan(steps=[Clip(input="roads.gpkg", mask="parks.geojson")])
    assert compare(plan, [{"op": "clip", "input": "roads.gpkg", "mask": "county.shp"}], tmp_path)
    assert compare(plan, [], tmp_path)
