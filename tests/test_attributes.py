import geopandas as gpd
import pandas as pd
import pytest
from pydantic import ValidationError
from shapely.geometry import Point

from agis.errors import AgisError
from agis.ops.attributes import Condition, Filter, Select, Unique


@pytest.mark.parametrize("value,expected", [
    ("net_1000", "1000"),
    ("/Users/me/shapes/province/1000", "1000"),
    ("/Users/me/agis/data/340524.0", "340524.0"),
    ("1,000", "1,000"),
    ("12.5", "12.5"),
    ("-3", "-3"),
])
def test_numeric_comparison_keeps_only_the_number(value, expected):
    assert Condition(field="shape_id", op="<", value=value).value == expected


@pytest.mark.parametrize("value,expected", [(1000, "1000"), (1000.0, "1000"), (2.5, "2.5")])
def test_a_json_number_becomes_text(value, expected):
    assert Condition(field="shape_id", op="<", value=value).value == expected


@pytest.mark.parametrize("value", ["", "testing", None])
def test_a_missing_number_is_reported_by_check(value):
    filt = Filter(input="fishing.fgb", where=[Condition(field="shape_id", op="<", value=value)])
    problems = filt.check({"input": "vector"})
    assert len(problems) == 1 and "< needs a number for shape_id" in problems[0]


def test_text_ops_still_need_a_value():
    with pytest.raises(ValidationError, match="needs a value"):
        Condition(field="name", op="==", value="")


def test_text_value_drops_an_absolute_path():
    assert Condition(field="country", op="==", value="/Users/me/data/Spain").value == "Spain"
    assert Condition(field="status", op="==", value="N/A").value == "N/A"
    assert Condition(field="day", op="==", value="2024/01/01").value == "2024/01/01"


def _points(tmp_path):
    path = tmp_path / "places.gpkg"
    gpd.GeoDataFrame(
        {"name": ["Market", "mission", None], "lanes": [4, 6, None], "zone": ["residential",
                                                                               "commercial",
                                                                               "residential"]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs="EPSG:4326",
    ).to_file(path)
    return path


def test_filter_equals_is_case_insensitive(tmp_path):
    out = tmp_path / "out.gpkg"
    src = _points(tmp_path)
    result = Filter(input="places.gpkg", where=[Condition(field="Zone", op="==",
                                                          value="Residential")]).run(
        {"input": src}, out)
    kept = gpd.read_file(out)
    assert kept["name"].iloc[0] == "Market" and pd.isna(kept["name"].iloc[1])
    assert "kept 2 of 3" in result.message


def test_filter_numeric_and_contains(tmp_path):
    src = _points(tmp_path)
    out = tmp_path / "lanes.gpkg"
    Filter(input="places.gpkg", where=[Condition(field="lanes", op=">=", value="6")]).run(
        {"input": src}, out)
    assert list(gpd.read_file(out)["name"]) == ["mission"]

    named = tmp_path / "named.gpkg"
    Filter(input="places.gpkg", where=[Condition(field="name", op="contains", value="MARK")]).run(
        {"input": src}, named)
    assert list(gpd.read_file(named)["name"]) == ["Market"]


def test_filter_in_and_any(tmp_path):
    src = _points(tmp_path)
    out = tmp_path / "either.gpkg"
    Filter(
        input="places.gpkg",
        match="any",
        where=[
            Condition(field="lanes", op="==", value="4"),
            Condition(field="name", op="in", value="mission, Geary"),
        ],
    ).run({"input": src}, out)
    assert sorted(v for v in gpd.read_file(out)["name"] if v) == ["Market", "mission"]


def test_filter_null_and_missing_field(tmp_path):
    src = _points(tmp_path)
    out = tmp_path / "nulls.gpkg"
    Filter(input="places.gpkg", where=[Condition(field="lanes", op="is_null")]).run(
        {"input": src}, out)
    assert len(gpd.read_file(out)) == 1
    with pytest.raises(AgisError, match="available fields"):
        Filter(input="places.gpkg", where=[Condition(field="nope", op="==", value="x")]).run(
            {"input": src}, tmp_path / "bad.gpkg")


def test_filter_rejects_a_missing_value():
    with pytest.raises(ValueError, match="needs a value"):
        Condition(field="name", op="contains")


def test_filter_rejects_a_value_on_a_null_check():
    with pytest.raises(ValueError, match="use == or !="):
        Condition(field="country_of_residence", op="not_null", value="Spain")
    Condition(field="country_of_residence", op="is_null")


def test_select_keeps_geometry_and_drops_the_rest(data, tmp_path):
    out = tmp_path / "names.geojson"
    result = Select(input="parks.geojson", columns=["name", "geometry"]).run(
        {"input": data["parks"]}, out)
    kept = gpd.read_file(out)
    assert list(kept.columns) == ["name", "geometry"]
    assert "acres" in result.message
    assert kept.crs.to_epsg() == 4326


def test_unique_lists_sorted_values(data):
    result = Unique(input="roads.gpkg", field="Name").run({"input": data["roads"]}, None)
    assert result.output is None
    assert result.message.splitlines() == [
        "name: 4 values", "Geary Blvd", "Market St", "Mission St", "Van Ness Ave",
    ]


def test_unique_unknown_field(tmp_path):
    src = _points(tmp_path)
    with pytest.raises(AgisError, match="available fields"):
        Unique(input="places.gpkg", field="nope").run({"input": src}, None)


def test_unique_counts_empty_and_omits_them_from_the_list(tmp_path):
    src = _points(tmp_path)
    lines = Unique(input="places.gpkg", field="name").run(
        {"input": src}, None).message.splitlines()
    assert lines == ["name: 2 values", "Market", "mission", "1 empty"]


def test_select_unknown_column(data, tmp_path):
    with pytest.raises(AgisError, match="available fields"):
        Select(input="zones.gpkg", columns=["landuse"]).run(
            {"input": data["zones"]}, tmp_path / "z.gpkg")
