import geopandas as gpd
import pytest
from shapely.geometry import Point

from agis.context import describe
from agis.errors import AgisError
from agis.ops.join import Join


def _layer(tmp_path, name, rows):
    path = tmp_path / name
    gpd.GeoDataFrame(rows, geometry=[Point(i, i) for i in range(len(rows[next(iter(rows))]))],
                     crs="EPSG:4326").to_file(path)
    return path


def test_join_detects_the_shared_column(data, tmp_path):
    out = tmp_path / "joined.gpkg"
    result = Join(input="zones.gpkg", table="zone_stats.csv").run(
        {"input": data["zones"], "table": data["zone_stats"]}, out)
    joined = gpd.read_file(out)
    assert set(joined.columns) >= {"zone", "households", "geometry"}
    assert len(joined) == 12
    assert set(joined["households"]) == {1200, 450}
    assert "joined on zone" in result.message and "12 of 12" in result.message
    assert "table, 2 rows" in describe(data["zone_stats"])


def test_join_matches_text_to_numbers_and_ignores_case(tmp_path):
    layer = _layer(tmp_path, "roads.gpkg", {"code": [6, 4], "name": ["Market", "Mission"]})
    table = tmp_path / "codes.csv"
    table.write_text("Code,label\n6,arterial\n4,local\n")
    out = tmp_path / "out.gpkg"
    Join(input="roads.gpkg", table="codes.csv").run({"input": layer, "table": table}, out)
    assert list(gpd.read_file(out)["label"]) == ["arterial", "local"]


def test_join_uses_explicit_names_and_keeps_unmatched_rows(tmp_path):
    layer = _layer(tmp_path, "zones.gpkg", {"zone": ["residential", "commercial", "industrial"]})
    table = tmp_path / "stats.csv"
    table.write_text("ZoneType,households\nResidential,10\nCommercial,3\n")
    out = tmp_path / "out.gpkg"
    Join(input="zones.gpkg", table="stats.csv", left_on="zone", right_on="ZoneType").run(
        {"input": layer, "table": table}, out)
    joined = gpd.read_file(out)
    assert list(joined["households"].fillna(-1)) == [10, 3, -1]
    assert "ZoneType" not in joined.columns

    inner = tmp_path / "inner.gpkg"
    Join(input="zones.gpkg", table="stats.csv", left_on="zone", right_on="zonetype",
         how="inner").run({"input": layer, "table": table}, inner)
    assert len(gpd.read_file(inner)) == 2


def test_join_renames_colliding_columns_and_repeats_duplicate_keys(tmp_path):
    layer = _layer(tmp_path, "places.gpkg", {"id": ["a", "b"], "name": ["one", "two"]})
    table = tmp_path / "names.csv"
    table.write_text("id,name\na,alpha\na,alto\n")
    out = tmp_path / "out.gpkg"
    result = Join(input="places.gpkg", table="names.csv", left_on="id").run(
        {"input": layer, "table": table}, out)
    joined = gpd.read_file(out)
    assert list(joined["name"]) == ["one", "one", "two"]
    assert list(joined["name_table"].dropna()) == ["alpha", "alto"]
    assert "became 3" in result.message


def test_join_reports_missing_and_ambiguous_keys(tmp_path):
    layer = _layer(tmp_path, "places.gpkg", {"id": ["a"], "name": ["one"]})
    (tmp_path / "other.csv").write_text("code,value\n1,x\n")
    with pytest.raises(AgisError, match="no column name in common"):
        Join(input="places.gpkg", table="other.csv").run(
            {"input": layer, "table": tmp_path / "other.csv"}, tmp_path / "out.gpkg")

    (tmp_path / "both.csv").write_text("id,name,extra\na,one,9\n")
    with pytest.raises(AgisError, match="several shared columns"):
        Join(input="places.gpkg", table="both.csv").run(
            {"input": layer, "table": tmp_path / "both.csv"}, tmp_path / "out.gpkg")


def test_join_can_use_another_vector_layer(data, tmp_path):
    out = tmp_path / "out.gpkg"
    Join(input="zones.gpkg", table="parks.geojson", left_on="zone", right_on="name").run(
        {"input": data["zones"], "table": data["parks"]}, out)
    assert gpd.read_file(out).crs.to_epsg() == 4326


def test_join_rejects_a_raster_table():
    assert Join(input="zones.gpkg", table="dem.tif").check({"input": "vector", "table": "raster"})
