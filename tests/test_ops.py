import geopandas as gpd
import pytest
import rasterio

from agis.errors import AgisError
from agis.ops import REGISTRY
from agis.ops.geoprocessing import (
    Buffer,
    Clip,
    Convert,
    Difference,
    Dissolve,
    Info,
    Reproject,
    Union,
)
from agis.ops.merge import Merge
from agis.ops.tiling import TileRaster


def test_registry_has_all_ops():
    assert set(REGISTRY) == {
        "reproject", "clip", "difference", "union", "buffer", "dissolve", "convert", "info",
        "tile_raster", "filter", "select", "join", "merge",
    }


def test_reproject_vector(data, tmp_path):
    out = tmp_path / "out.gpkg"
    Reproject(input="roads.gpkg", crs="web mercator").run({"input": data["roads"]}, out)
    assert gpd.read_file(out).crs.to_epsg() == 3857


def test_reproject_auto_utm(data, tmp_path):
    out = tmp_path / "out.gpkg"
    Reproject(input="roads.gpkg", crs="utm").run({"input": data["roads"]}, out)
    assert gpd.read_file(out).crs.to_epsg() == 32610


def test_reproject_raster(data, tmp_path):
    out = tmp_path / "dem_4326.tif"
    Reproject(input="dem.tif", crs="4326").run({"input": data["dem"]}, out)
    with rasterio.open(out) as src:
        assert src.crs.to_epsg() == 4326
        assert src.nodata == -9999


def test_clip_vector_with_mask_in_other_crs(data, tmp_path):
    out = tmp_path / "clipped.gpkg"
    Clip(input="roads.gpkg", mask="county.shp").run(
        {"input": data["roads"], "mask": data["county"]}, out
    )
    clipped = gpd.read_file(out)
    assert len(clipped) == 4
    assert clipped.total_bounds[0] >= -122.4701


def test_clip_raster_with_bbox(data, tmp_path):
    out = tmp_path / "clipped.tif"
    Clip(input="dem.tif", bbox=[-122.45, 37.75, -122.43, 37.77]).run({"input": data["dem"]}, out)
    with rasterio.open(out) as src, rasterio.open(data["dem"]) as orig:
        assert src.width < orig.width


def test_clip_requires_exactly_one_area():
    with pytest.raises(ValueError):
        Clip(input="roads.gpkg")


def test_difference(data, tmp_path):
    out = tmp_path / "diff.gpkg"
    Difference(input="zones.gpkg", other="parks.geojson").run(
        {"input": data["zones"], "other": data["parks"]}, out
    )
    before = gpd.read_file(data["zones"]).to_crs(3310).area.sum()
    after = gpd.read_file(out).to_crs(3310).area.sum()
    assert after < before


def test_union(data, tmp_path):
    out = tmp_path / "union.gpkg"
    Union(input="zones.gpkg", other="parks.geojson").run(
        {"input": data["zones"], "other": data["parks"]}, out
    )
    assert {"zone", "name"} <= set(gpd.read_file(out).columns)


def test_union_rejects_lines(data, tmp_path):
    with pytest.raises(AgisError):
        Union(input="roads.gpkg", other="parks.geojson").run(
            {"input": data["roads"], "other": data["parks"]}, tmp_path / "u.gpkg"
        )


def test_merge_appends_rows_and_lines_up_columns(data, tmp_path):
    out = tmp_path / "merged.gpkg"
    result = Merge(input="zones.gpkg", others=["parks.geojson"]).run(
        {"input": data["zones"], "others": [data["parks"]]}, out
    )
    merged = gpd.read_file(out)
    assert len(merged) == 14
    assert merged.crs.to_epsg() == 4326
    assert {"zone", "name", "acres"} <= set(merged.columns)
    assert merged["zone"].isna().sum() == 2
    assert result.message.startswith("14 features (12 + 2)")


def test_merge_reprojects_to_the_input_crs(data, tmp_path):
    out = tmp_path / "merged.gpkg"
    result = Merge(input="zones.gpkg", others=["county.shp"]).run(
        {"input": data["zones"], "others": [data["county"]]}, out
    )
    merged = gpd.read_file(out)
    study = merged[merged["name"] == "Study Area"].geometry.iloc[0]
    assert len(merged) == 13
    assert merged.crs.to_epsg() == 4326
    assert study.centroid.x == pytest.approx(-122.435, abs=0.02)
    assert "reprojected county.shp to EPSG:4326" in result.message


def test_merge_appends_several_files(data, tmp_path):
    out = tmp_path / "all.gpkg"
    Merge(input="roads.gpkg", others=["parks.geojson", "zones.gpkg"]).run(
        {"input": data["roads"], "others": [data["parks"], data["zones"]]}, out
    )
    merged = gpd.read_file(out)
    assert len(merged) == 18
    assert set(merged.geom_type) == {"LineString", "Polygon"}


def test_merge_requires_a_crs_to_match_the_input(data, tmp_path):
    import warnings

    from shapely.geometry import box

    bare = tmp_path / "bare.gpkg"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        gpd.GeoDataFrame({"name": ["a"]}, geometry=[box(0, 0, 1, 1)]).to_file(bare)
    with pytest.raises(AgisError, match="no CRS"):
        Merge(input="zones.gpkg", others=["bare.gpkg"]).run(
            {"input": data["zones"], "others": [bare]}, tmp_path / "out.gpkg"
        )


def test_buffer_geographic_uses_meters(data, tmp_path):
    out = tmp_path / "buf.gpkg"
    Buffer(input="roads.gpkg", distance=100).run({"input": data["roads"]}, out)
    buffered = gpd.read_file(out)
    assert buffered.crs.to_epsg() == 4326
    edge = buffered.to_crs(32610).geometry.iloc[3].boundary
    line = gpd.read_file(data["roads"]).to_crs(32610).geometry.iloc[3]
    assert edge.distance(line) == pytest.approx(100, abs=1)


def test_dissolve_by_field(data, tmp_path):
    out = tmp_path / "dis.gpkg"
    Dissolve(input="zones.gpkg", by="zone").run({"input": data["zones"]}, out)
    assert sorted(gpd.read_file(out)["zone"]) == ["commercial", "residential"]


def test_dissolve_unknown_field(data, tmp_path):
    with pytest.raises(AgisError, match="available fields"):
        Dissolve(input="zones.gpkg", by="nope").run({"input": data["zones"]}, tmp_path / "d.gpkg")


def test_convert_vector_and_cog(data, tmp_path):
    out = tmp_path / "roads.geojson"
    Convert(input="roads.gpkg", format="geojson").run({"input": data["roads"]}, out)
    assert len(gpd.read_file(out)) == 4
    cog = tmp_path / "dem_cog.tif"
    Convert(input="dem.tif", format="cog").run({"input": data["dem"]}, cog)
    with rasterio.open(cog) as src:
        assert src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"


def test_convert_kind_check():
    assert Convert(input="dem.tif", format="shp").check({"input": "raster"})


def test_info(data):
    result = Info(input="roads.gpkg").run({"input": data["roads"]}, None)
    lines = result.message.splitlines()
    assert lines[0] == "roads.gpkg"
    assert "crs: EPSG:4326 (WGS 84)" in lines
    assert "features: 4" in lines
    assert "  lanes: int" in lines
    assert "bounds:" in result.message


@pytest.mark.parametrize("name", ["dem", "ortho"])
def test_tile_raster(data, tmp_path, name):
    import sqlite3
    import warnings

    import morecantile
    from rasterio.io import MemoryFile
    from rasterio.warp import transform_bounds

    out = tmp_path / f"{name}.mbtiles"
    result = TileRaster(input=f"{name}.tif", min_zoom=10, max_zoom=12).run(
        {"input": data[name]}, out
    )
    assert out.is_file() and result.kind == "tiles"
    con = sqlite3.connect(out)
    meta = dict(con.execute("select name, value from metadata"))
    stored = set(con.execute("select zoom_level, tile_column, tile_row from tiles"))
    blob = con.execute("select tile_data from tiles limit 1").fetchone()[0]
    con.close()
    assert meta["format"] == "png"
    assert meta["minzoom"] == "10" and meta["maxzoom"] == "12"
    assert {z for z, _x, _y in stored} == {10, 11, 12}
    assert blob.startswith(b"\x89PNG\r\n\x1a\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with MemoryFile(blob) as mem, mem.open() as png:
            assert (png.width, png.height, png.count) == (256, 256, 4)
    with rasterio.open(data[name]) as src:
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    north, south = min(north, 85.0511), max(south, -85.0511)
    tms = morecantile.tms.get("WebMercatorQuad")
    expected = {
        (tile.z, tile.x, (1 << tile.z) - 1 - tile.y)
        for tile in tms.tiles(west, south, east, north, zooms=[10, 11, 12])
    }
    assert stored <= expected


def test_tile_raster_auto_zoom(data, tmp_path):
    out = tmp_path / "t.mbtiles"
    result = TileRaster(input="dem.tif").run({"input": data["dem"]}, out)
    assert out.is_file() and "zoom" in result.message
    from agis.context import describe
    assert describe(out).startswith("t.mbtiles: MBTiles png, zoom ")
