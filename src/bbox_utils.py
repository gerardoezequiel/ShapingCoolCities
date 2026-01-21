
import math
import json
import geopandas as gpd
from shapely.geometry import box
from typing import Dict, Tuple, Optional
from pathlib import Path
import pandas as pd
import warnings

# Global Streetscapes cities data URL
GS_CITIES_URL = 'https://huggingface.co/datasets/NUS-UAL/global-streetscapes/resolve/main/cities688.csv'

# Cache for cities dataframe
_cities_df_cache: Optional[pd.DataFrame] = None

# Default zoom level for consistent bbox size across all datasets
BBOX_ZOOM = 14  # Default zoom level

# Our study cities
STUDY_CITIES = ['Amsterdam', 'Athens', 'Barcelona', 'Berlin', 'Madrid', 'Paris']


def get_cities_df() -> pd.DataFrame:
    """Load and cache cities data from Global Streetscapes."""
    global _cities_df_cache
    if _cities_df_cache is None:
        try:
            _cities_df_cache = pd.read_csv(GS_CITIES_URL)
        except Exception as e:
            raise RuntimeError(f"Failed to load cities from Global Streetscapes: {e}")
    return _cities_df_cache


def get_study_cities_df() -> pd.DataFrame:
    """Get filtered dataframe with only our study cities."""
    cities_df = get_cities_df()
    
    # Filter for our study cities
    # IMPORTANT: Filter for Madrid in Spain specifically (not Colombia)
    study_df = cities_df[
        (cities_df['city'].isin(STUDY_CITIES)) & 
        ((cities_df['city'] != 'Madrid') | (cities_df['country'] == 'Spain'))
    ].copy()
    
    if len(study_df) != len(STUDY_CITIES):
        found_cities = study_df['city'].tolist()
        missing = set(STUDY_CITIES) - set(found_cities)
        warnings.warn(f"Missing cities in dataset: {missing}")
    
    # Ensure we return a DataFrame, not a Series
    if isinstance(study_df, pd.Series):
        study_df = study_df.to_frame().T
    
    return study_df


def get_tile_bounds(lat: float, lon: float, zoom: int = BBOX_ZOOM) -> Tuple[int, int, int]:
    """Convert lat/lon to tile coordinates."""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    xtile = int(n * ((lon + 180) / 360))
    ytile = int(n * (1 - (math.log(math.tan(lat_rad) + 1/math.cos(lat_rad)) / math.pi)) / 2)
    return zoom, xtile, ytile


def tile_to_bbox(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    """Get bounding box (west, south, east, north) of a tile."""
    n = 2 ** z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return west, south, east, north


def get_city_bbox(city_name: str, zoom_level: Optional[int] = None) -> Dict:
    """Get standardized bounding box for a city using cities688.csv data.
    
    Args:
        city_name: Name of the city
        zoom_level: Optional zoom level override (default: BBOX_ZOOM)
    
    Returns:
        Dict with bbox coordinates and metadata
    """
    # Get study cities data
    study_df = get_study_cities_df()
    
    # Find city (case-insensitive)
    city_row = study_df[study_df['city'].str.lower() == city_name.lower()]
    
    if city_row.empty:
        raise ValueError(f"City '{city_name}' not found in study cities")
    
    # Get coordinates from cities688.csv
    city_data = city_row.iloc[0]
    lat = float(city_data['city_lat'])
    lon = float(city_data['city_lon'])
    country = city_data['country']
    
    # Use provided zoom level or default
    zoom = zoom_level if zoom_level is not None else BBOX_ZOOM
    
    # Get tile coordinates
    z, x, y = get_tile_bounds(lat, lon, zoom)
    
    # Get bbox
    west, south, east, north = tile_to_bbox(z, x, y)
    
    # Calculate dimensions
    width_km = (east - west) * 111.32 * math.cos(math.radians(lat))
    height_km = (north - south) * 111.32
    
    return {
        'city': city_name,
        'country': country,
        'center_lat': lat,
        'center_lon': lon,
        'west': west,
        'south': south,
        'east': east,
        'north': north,
        'width_km': round(width_km, 2),
        'height_km': round(height_km, 2),
        'area_km2': round(width_km * height_km, 2),
        'zoom_level': zoom,
        'data_source': 'Global Streetscapes cities688.csv'
    }


def get_city_bbox_gdf(city_name: str) -> gpd.GeoDataFrame:
    """Get city bbox as GeoDataFrame."""
    bbox_info = get_city_bbox(city_name)
    geom = box(bbox_info['west'], bbox_info['south'], 
               bbox_info['east'], bbox_info['north'])
    return gpd.GeoDataFrame([bbox_info], geometry=[geom], crs='EPSG:4326')



def get_aligned_grid_bbox(city_name: str, grid_size: int = 30, cells_per_side: int = 82) -> Dict:
    """Get an aligned 82x82 grid bbox centred on the zoom 14 tile centre.
    
    This function creates consistent, grid-aligned bounding boxes for spatial analysis
    across all study cities, ensuring uniform data resolution and spatial coverage.
    
    Args:
        city_name: Name of the city
        grid_size: Grid cell size in metres (default: 30)
        cells_per_side: Number of cells per side (default: 82)
    
    Returns:
        Dict with aligned grid bbox coordinates centred on tile
    """
    import numpy as np
    from pyproj import Transformer
    
    # Get city coordinates to find the zoom 14 tile
    study_df = get_study_cities_df()
    city_row = study_df[study_df['city'].str.lower() == city_name.lower()]
    
    if city_row.empty:
        raise ValueError(f"City '{city_name}' not found in study cities")
    
    city_data = city_row.iloc[0]
    city_lat = float(city_data['city_lat'])
    city_lon = float(city_data['city_lon'])
    country = city_data['country']
    
    # Get the zoom 14 tile that contains this city
    z, x, y = get_tile_bounds(city_lat, city_lon, 14)
    west, south, east, north = tile_to_bbox(z, x, y)
    
    # Calculate the TRUE CENTER of the zoom 14 tile (not the city point)
    tile_center_lon = (west + east) / 2
    tile_center_lat = (south + north) / 2
    
    # Calculate exact extent needed
    extent_m = cells_per_side * grid_size  # 82 * 30 = 2460m
    half_extent = extent_m / 2
    
    # Transform TILE CENTER to Web Mercator
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
    center_x, center_y = transformer.transform(tile_center_lon, tile_center_lat)
    
    # Create bbox with exact size, snapped to grid
    minx_3857 = np.floor((center_x - half_extent) / grid_size) * grid_size
    miny_3857 = np.floor((center_y - half_extent) / grid_size) * grid_size
    maxx_3857 = minx_3857 + extent_m
    maxy_3857 = miny_3857 + extent_m
    
    # Transform corners back to EPSG:4326
    transformer_back = Transformer.from_crs('EPSG:3857', 'EPSG:4326', always_xy=True)
    west_aligned, south_aligned = transformer_back.transform(minx_3857, miny_3857)
    east_aligned, north_aligned = transformer_back.transform(maxx_3857, maxy_3857)
    
    # Calculate dimensions
    width_km = (east_aligned - west_aligned) * 111.32 * math.cos(math.radians(tile_center_lat))
    height_km = (north_aligned - south_aligned) * 111.32
    
    return {
        'city': city_name,
        'country': country,
        'city_lat': city_lat,  # Original city coordinates
        'city_lon': city_lon,
        'center_lat': tile_center_lat,  # Tile center (where grid is centered)
        'center_lon': tile_center_lon,
        'west': west_aligned,
        'south': south_aligned,
        'east': east_aligned,
        'north': north_aligned,
        'minx_3857': minx_3857,
        'miny_3857': miny_3857,
        'maxx_3857': maxx_3857,
        'maxy_3857': maxy_3857,
        'width_km': round(width_km, 3),
        'height_km': round(height_km, 3),
        'area_km2': round(width_km * height_km, 3),
        'grid_size_m': grid_size,
        'cells_per_side': cells_per_side,
        'total_cells': cells_per_side ** 2,
        'extent_m': extent_m,
        'zoom_level': 14,
        'tile_x': x,
        'tile_y': y,
        'grid_aligned': True,
        'data_source': f'Aligned {cells_per_side}x{cells_per_side} grid for spatial analysis'
    }


def get_all_study_cities_bbox(align_to_grid: bool = True, grid_size: int = 30, cells_per_side: int = 82) -> gpd.GeoDataFrame:
    """Get bounding boxes for all study cities.
    
    NOTE: Returns pre-calculated EPSG:3857 aligned bounds in minx_3857, miny_3857, etc.
    These should be used directly for grid creation, NOT the bounds from .to_crs() transformation.
    
    Args:
        align_to_grid: If True, create 82x82 grid centered on zoom 14 tile
        grid_size: Grid cell size in meters (default: 30)
        cells_per_side: Number of cells per side (default: 82)
    
    Returns:
        GeoDataFrame with city bounding boxes
    """
    bboxes = []
    geometries = []
    
    for city in STUDY_CITIES:
        try:
            if align_to_grid:
                # Create aligned 82x82 grid for spatial analysis
                bbox_info = get_aligned_grid_bbox(city, grid_size, cells_per_side)
            else:
                # Use traditional zoom-14 tile approach (unaligned)
                bbox_info = get_city_bbox(city)
            
            geom = box(bbox_info['west'], bbox_info['south'], 
                      bbox_info['east'], bbox_info['north'])
            bboxes.append(bbox_info)
            geometries.append(geom)
        except Exception as e:
            warnings.warn(f"Failed to get bbox for {city}: {e}")
    
    return gpd.GeoDataFrame(bboxes, geometry=geometries, crs='EPSG:4326')


def clip_to_city_bbox(gdf: gpd.GeoDataFrame, city_name: str, target_crs: str = 'EPSG:4326') -> gpd.GeoDataFrame:
    """Clip GeoDataFrame to city bbox and optionally reproject.
    
    Args:
        gdf: Input GeoDataFrame to clip
        city_name: Name of the city for bbox
        target_crs: Target CRS for output (default: EPSG:4326)
    
    Returns:
        Clipped and reprojected GeoDataFrame
    """
    bbox_gdf = get_city_bbox_gdf(city_name)
    
    # Ensure both are in the same CRS for clipping
    if gdf.crs is not None and bbox_gdf.crs is not None and gdf.crs != bbox_gdf.crs:
        gdf = gdf.to_crs(bbox_gdf.crs)
    
    # Clip to bbox
    clipped = gpd.clip(gdf, bbox_gdf)
    
    # Reproject to target CRS if needed
    if clipped.crs != target_crs:
        clipped = clipped.to_crs(target_crs)
    
    return clipped


def verify_city_coordinates():
    """Verify and display coordinates for all study cities."""
    print("Study Cities Coordinates (from cities688.csv):")
    print("=" * 60)
    
    study_df = get_study_cities_df()
    
    for _, row in study_df.iterrows():
        print(f"{row['city']:12} ({row['country']:12}): "
              f"{row['city_lat']:8.4f}, {row['city_lon']:9.4f}")
    
    return study_df


if __name__ == "__main__":
    # Test the utilities
    print("Testing bbox utilities with cities688.csv data...\n")
    
    # Verify coordinates
    verify_city_coordinates()
    
    # Test bbox generation
    print("\n\nBounding Box Generation:")
    print("=" * 60)
    
    for city in STUDY_CITIES:
        try:
            bbox = get_city_bbox(city)
            print(f"\n{city}:")
            print(f"  Center: ({bbox['center_lat']:.4f}, {bbox['center_lon']:.4f})")
            print(f"  Bounds: [{bbox['west']:.4f}, {bbox['south']:.4f}, "
                  f"{bbox['east']:.4f}, {bbox['north']:.4f}]")
            print(f"  Size: {bbox['width_km']} × {bbox['height_km']} km")
        except Exception as e:
            print(f"\n{city}: ERROR - {e}")