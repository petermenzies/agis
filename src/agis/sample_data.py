"""Small synthetic datasets around San Francisco, used by tests and `agis eval`."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Polygon, box


def write_sample_data(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "roads": root / "roads.gpkg",
        "county": root / "county.shp",
        "parks": root / "parks.geojson",
        "zones": root / "zones.gpkg",
        "dem": root / "dem.tif",
        "ortho": root / "ortho.tif",
    }

    roads = gpd.GeoDataFrame(
        {
            "name": ["Market St", "Mission St", "Geary Blvd", "Van Ness Ave"],
            "lanes": [4, 4, 6, 6],
        },
        geometry=[
            LineString([(-122.50, 37.77), (-122.39, 37.79)]),
            LineString([(-122.48, 37.74), (-122.39, 37.78)]),
            LineString([(-122.51, 37.78), (-122.40, 37.785)]),
            LineString([(-122.424, 37.72), (-122.424, 37.81)]),
        ],
        crs="EPSG:4326",
    )
    roads.to_file(paths["roads"])

    county = gpd.GeoDataFrame(
        {"name": ["Study Area"]},
        geometry=[box(-122.47, 37.74, -122.40, 37.80)],
        crs="EPSG:4326",
    ).to_crs("EPSG:3310")
    county.to_file(paths["county"])

    parks = gpd.GeoDataFrame(
        {"name": ["Golden Gate Park", "Dolores Park"], "acres": [1017.0, 16.0]},
        geometry=[
            box(-122.511, 37.765, -122.453, 37.774),
            Polygon([(-122.428, 37.758), (-122.425, 37.758), (-122.425, 37.762),
                     (-122.428, 37.762)]),
        ],
        crs="EPSG:4326",
    )
    parks.to_file(paths["parks"], driver="GeoJSON")

    zone_geoms, zone_names = [], []
    for i in range(4):
        for j in range(3):
            zone_geoms.append(box(-122.50 + i * 0.025, 37.73 + j * 0.025,
                                  -122.475 + i * 0.025, 37.755 + j * 0.025))
            zone_names.append("residential" if (i + j) % 2 else "commercial")
    zones = gpd.GeoDataFrame({"zone": zone_names}, geometry=zone_geoms, crs="EPSG:4326")
    zones.to_file(paths["zones"])

    paths["zone_stats"] = root / "zone_stats.csv"
    paths["zone_stats"].write_text("zone,households\nresidential,1200\ncommercial,450\n")

    # 30 m DEM in UTM 10N covering roughly the same area.
    transform = from_origin(545_000, 4_185_000, 30, 30)
    y, x = np.mgrid[0:200, 0:200]
    dem = (50 + 200 * np.sin(x / 40) * np.cos(y / 30)).astype("float32")
    dem[:10, :10] = -9999
    with rasterio.open(paths["dem"], "w", driver="GTiff", width=200, height=200, count=1,
                       dtype="float32", crs="EPSG:32610", transform=transform,
                       nodata=-9999) as dst:
        dst.write(dem, 1)

    rgb = np.stack([(x * 1.2) % 256, (y * 1.2) % 256, ((x + y) * 0.6) % 256]).astype("uint8")
    with rasterio.open(paths["ortho"], "w", driver="GTiff", width=200, height=200, count=3,
                       dtype="uint8", crs="EPSG:32610", transform=transform) as dst:
        dst.write(rgb)

    return paths
