"""Post-install check that the geospatial stack works without any system GDAL.

Run with the Python from a fresh install, e.g. the one `uv tool install agis` created.
It doesn't need Ollama: it exercises the operations directly.
"""

import sys
import tempfile
from pathlib import Path

from agis.executor import resolve, run
from agis.ops.geoprocessing import Buffer, Clip, Convert, Reproject
from agis.ops.tiling import TileRaster
from agis.plan import Plan, plan_schema
from agis.sample_data import write_sample_data


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_sample_data(root)
        plan = Plan(steps=[
            Clip(input="roads.gpkg", mask="county.shp"),
            Buffer(input="$step1", distance=50),
            Reproject(input="$step2", crs="EPSG:3857"),
            Convert(input="dem.tif", format="cog"),
            TileRaster(input="ortho.tif", min_zoom=11, max_zoom=12),
        ])
        steps = resolve(plan, root, root / "work")
        results = run(steps)
        for step, result in zip(steps, results):
            print(f"{step.args.op}: {result.message}")
        assert (root / "roads_clipped_buffer50m_3857.gpkg").exists()
        assert (root / "ortho_tiles.mbtiles").read_bytes().startswith(b"SQLite format 3")
    assert plan_schema()["$defs"]
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
