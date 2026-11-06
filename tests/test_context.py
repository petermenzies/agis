import geopandas as gpd
import pyogrio
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from shapely.geometry import Point

from agis.cli import DatasetCompleter
from agis.context import (
    MAX_DESCRIBED,
    MAX_LISTED,
    _peek_geojson,
    build_context,
    cached_describe,
    describe,
)
from agis.ops.io import read_vector


def test_named_file_is_described_and_the_rest_are_listed(data, tmp_path):
    text = build_context("reproject roads", tmp_path).render()
    assert "roads.gpkg: vector" in text
    assert "parks.geojson: vector" not in text
    assert "parks.geojson" in text.split("Other files: ", 1)[1]


def test_unnamed_request_describes_only_up_to_the_cap(tmp_path):
    for i in range(MAX_DESCRIBED + 3):
        (tmp_path / f"layer{i:02d}.csv").write_text("id,value\n1,a\n")
    text = build_context("what is in this folder?", tmp_path).render()
    described = [line for line in text.splitlines() if line.startswith("- layer")]
    assert len(described) == MAX_DESCRIBED
    others = text.split("Other files: ", 1)[1]
    assert "layer12.csv" in others
    assert "layer12.csv:" not in text


def test_long_directory_reports_how_many_names_were_cut(tmp_path):
    for i in range(MAX_LISTED + 5):
        (tmp_path / f"layer{i:03d}.csv").write_text("id\n")
    text = build_context("reproject layer000", tmp_path).render()
    assert "and 4 more" in text


def test_planning_summary_does_not_scan_bounds(data, monkeypatch):
    calls = []
    real = pyogrio.read_info

    def wrapped(*args, **kwargs):
        calls.append(kwargs.get("force_total_bounds"))
        return real(*args, **kwargs)

    monkeypatch.setattr("agis.context.pyogrio.read_info", wrapped)
    summary = describe(data["roads"])
    detailed = describe(data["roads"], detailed=True)
    assert calls == [False, True]
    assert "EPSG:4326" in summary and "lanes" in summary and "bounds" not in summary
    assert "bounds" in detailed


def test_table_summary_does_not_load_the_whole_file(tmp_path, monkeypatch):
    path = tmp_path / "counts.csv"
    path.write_text("zone,households\n" + "".join(f"r{i},{i}\n" for i in range(250)))

    def fail_full_read(*_args, **_kwargs):
        raise AssertionError("summary read the whole table")

    monkeypatch.setattr("agis.ops.io.read_table", fail_full_read)
    summary = describe(path)
    assert "250 rows" in summary
    assert "zone:str" in summary and "households:int" in summary


def test_summary_cache_reuses_an_unchanged_file_and_rereads_a_changed_one(data, monkeypatch):
    calls = {"n": 0}
    real = pyogrio.read_info

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr("agis.context.pyogrio.read_info", wrapped)
    cache = {}
    path = data["roads"]
    first = cached_describe(path, cache=cache)
    second = cached_describe(path, cache=cache)
    assert first == second and calls["n"] == 1
    path.touch()
    cached_describe(path, cache=cache)
    assert calls["n"] == 2


def test_geometry_only_gpkg_reports_its_fid(tmp_path):
    path = tmp_path / "only.gpkg"
    gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326").to_file(path)
    detailed = describe(path, detailed=True)
    summary = describe(path)
    assert "fid: int" in detailed
    assert "fid:int" in summary


def test_geojson_peek_reads_the_first_feature_and_not_its_coordinates(tmp_path, monkeypatch):
    path = tmp_path / "land.geojson"
    path.write_text(
        '{"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"EPSG:4326"}},'
        '"features":[{"type":"Feature","properties":{"name":"Park","beds":4,"open":true},'
        '"geometry":{"type":"Polygon","coordinates":['
    )
    info = _peek_geojson(path)
    assert info["geometry_type"] == "Polygon"
    assert info["fields"] == ["name", "beds", "open"]
    assert info["dtypes"] == ["str", "int", "bool"]
    assert info["crs"] == "EPSG:4326"
    assert info["features"] is None

    def too_large(*_args, **_kwargs):
        raise RuntimeError(
            "GeoJSON object too complex/large. Define OGR_GEOJSON_MAX_OBJ_SIZE to allow it."
        )

    # The file is not valid GeoJSON past the first type, so the fallback must not parse it.
    monkeypatch.setattr("agis.context.pyogrio.read_info", too_large)
    text = describe(path, detailed=True)
    assert "features: not read" in text
    assert "bounds: not read" in text
    assert "name: str" in text and "beds: int" in text and "open: bool" in text
    assert "EPSG:4326" in text


def test_geojson_read_lifts_the_size_limit_and_restores_it(tmp_path, monkeypatch):
    path = tmp_path / "a.geojson"
    path.write_text('{"type":"FeatureCollection","features":[]}')
    seen = {}

    def get_option(name):
        assert name == "OGR_GEOJSON_MAX_OBJ_SIZE"
        return "200"

    def set_options(options):
        seen.update(options)

    def read_file(_path):
        assert seen["OGR_GEOJSON_MAX_OBJ_SIZE"] == "0"
        return "frame"

    monkeypatch.setattr("agis.ops.io.pyogrio.get_gdal_config_option", get_option)
    monkeypatch.setattr("agis.ops.io.pyogrio.set_gdal_config_options", set_options)
    monkeypatch.setattr("agis.ops.io.gpd.read_file", read_file)
    assert read_vector(path) == "frame"
    assert seen["OGR_GEOJSON_MAX_OBJ_SIZE"] == "200"


def test_tab_completes_dataset_names_and_skips_sidecars(data, tmp_path):
    (tmp_path / "roads.shx").write_text("not a dataset")
    (tmp_path / "my roads.gpkg").write_bytes(b"")
    completer = DatasetCompleter(tmp_path)
    document = Document("clip roa", cursor_position=len("clip roa"))
    texts = [item.text for item in completer.get_completions(
        document, CompleteEvent(completion_requested=True))]
    assert "roads.gpkg" in texts
    assert "roads.shx" not in texts
    assert all(not name.endswith(".shx") for name in texts)

    spaced = Document("clip my", cursor_position=len("clip my"))
    spaced_texts = [item.text for item in completer.get_completions(
        spaced, CompleteEvent(completion_requested=True))]
    assert '"my roads.gpkg"' in spaced_texts
