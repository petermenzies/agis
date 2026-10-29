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
from agis.ops.tiling import TileRaster


def test_registry_has_all_ops():
    assert set(REGISTRY) == {
        "reproject", "clip", "difference", "union", "buffer", "dissolve", "convert", "info",
        "tile_raster", "filter", "select", "join",
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
    assert "EPSG:4326" in result.message and "lanes" in result.message


@pytest.mark.parametrize("name", ["dem", "ortho"])
def test_tile_raster(data, tmp_path, name):
    out = tmp_path / f"{name}_tiles"
    result = TileRaster(input=f"{name}.tif", min_zoom=10, max_zoom=12).run(
        {"input": data[name]}, out
    )
    pngs = list(out.rglob("*.png"))
    assert pngs and result.kind == "tiles"
    assert {p.parts[-3] for p in pngs} == {"10", "11", "12"}


def test_tile_raster_auto_zoom(data, tmp_path):
    result = TileRaster(input="dem.tif").run({"input": data["dem"]}, tmp_path / "t")
    assert "zoom" in result.message
