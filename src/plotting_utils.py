
"""Utility helpers for consistent mapping and visual styling across notebooks."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID_PATH = PROJECT_ROOT / "data/utils/grids/all_cities_30m_grid.parquet"


def load_grid_geodataframe(path: Path | None = None) -> gpd.GeoDataFrame:
    """Load the standard 30 m grid for all cities as a GeoDataFrame."""
    grid_path = path or GRID_PATH
    gdf = gpd.read_parquet(grid_path)
    if str(gdf.geometry.dtype) == "object":
        gdf["geometry"] = gpd.GeoSeries.from_wkb(gdf["geometry"])
    return gdf.set_geometry("geometry")
