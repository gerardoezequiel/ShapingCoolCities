from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import pytz

from voxcity.simulator.solar import (
    get_global_solar_irradiance_using_epw,
    get_direct_solar_irradiance_map,
    get_diffuse_solar_irradiance_map,
    get_cumulative_global_solar_irradiance,
)

try:
    from voxcity.simulator.view import get_view_index
except ImportError:
    get_view_index = None

try:
    from src.bbox_utils import STUDY_CITIES
except ModuleNotFoundError:  # pragma: no cover - support direct src execution
    from bbox_utils import STUDY_CITIES

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:  # pragma: no cover - SciPy should be available but guard regardless
    distance_transform_edt = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - Progress tracking is optional
    # Fallback if tqdm not available
    def tqdm(iterable, desc=None, **kwargs):
        if desc:
            print(f"{desc}...")
        return iterable


LAND_COVER_KEYWORDS: Dict[str, Sequence[str]] = {
    "impervious": [
        "Developed",
        "Building",
        "Road",
        "Rail",
        "Commercial",
        "Industrial",
        "Residential",
        "Urban",
        "Construction",
    ],
    "vegetation": ["Tree", "Vegetation", "Grass", "Shrub", "Forest", "Park", "Wetland"],
    "water": ["Water", "Sea", "Lake", "River", "Reservoir", "Canal", "Ocean"],
    "bare": ["Bare", "Rock", "Sand", "Desert", "Barren"],
    "agriculture": ["Agriculture", "Crop", "Farmland", "Meadow", "Orchard", "Vineyard", "Pasture"],
}


def build_land_cover_indices(
    land_cover_source: str,
    keywords: Mapping[str, Sequence[str]] = LAND_COVER_KEYWORDS,
) -> Dict[str, list[int]]:
    """Map canonical land-cover groups to indices for the chosen VoxCity source."""

    # For OpenStreetMap land cover, use known mapping based on actual data analysis
    if land_cover_source == "OpenStreetMap":
        return {
            "impervious": [1, 10, 11, 12],  # Bareland, developed space, roads, buildings
            "vegetation": [8],  # Trees/vegetation (value 8 in urban context)
        }

    # Fallback: try to use VoxCity's class mapping (may return colors instead of indices)
    try:
        from voxcity.utils.visualization import get_land_cover_classes
        classes = get_land_cover_classes(land_cover_source)
        indices: Dict[str, list[int]] = {}
        for group, terms in keywords.items():
            group_indices: list[int] = []
            for idx, name in classes.items():
                # Skip if idx is not an integer (e.g., RGB tuple)
                if not isinstance(idx, int):
                    continue
                name_lower = str(name).lower()
                if any(term.lower() in name_lower for term in terms):
                    group_indices.append(idx)
            indices[group] = group_indices
        return indices
    except Exception:
        # Ultimate fallback - return empty mapping
        return {group: [] for group in keywords}


def compute_canyon_width(voxel_grid: np.ndarray, meshsize: float) -> np.ndarray:
    """Estimate horizontal canyon width (metres) from the voxel stack."""

    if distance_transform_edt is None:
        return np.full(voxel_grid.shape[:2], meshsize, dtype=float)

    building_presence = (voxel_grid == -3).any(axis=2)
    walkable = ~building_presence
    dist = distance_transform_edt(walkable) * meshsize
    return dist * 2.0



def diagnose_land_cover_issue(land_cover: np.ndarray, land_cover_indices: Dict) -> Dict:
    """Diagnostic function to identify land cover classification issues"""
    unique_values = np.unique(land_cover[~np.isnan(land_cover)])
    diagnostics = {
        'unique_land_cover_values': unique_values.tolist(),
        'land_cover_shape': land_cover.shape,
        'expected_indices': land_cover_indices,
        'value_coverage': {}
    }

    for group, indices in land_cover_indices.items():
        if indices:
            matches = np.isin(land_cover, indices)
            coverage = np.sum(matches) / land_cover.size
            diagnostics['value_coverage'][group] = {
                'indices': indices,
                'pixel_matches': int(np.sum(matches)),
                'coverage_fraction': float(coverage)
            }

    return diagnostics

VOXCITY_DEFAULT_CONFIG: Dict[str, object] = {
    "building_source": "Local file",
    "land_cover_source": "OpenStreetMap",
    "canopy_height_source": "High Resolution 1m Global Canopy Height Maps",
    "dem_source": "FABDEM",
    "meshsize": 5,
    "dem_interpolation": True,
    "gridvis": False,
    "static_tree_height": 8.0,
}


@dataclass
class VoxCityPaths:
    """Convenience container with all filesystem locations used by the workflow."""

    base_dir: Path
    data_dir: Path
    processed_dir: Path
    cache_dir: Path
    features_dir: Path
    visuals_dir: Path
    views_dir: Path
    grids_path: Path
    combined_features_path: Path
    epw_dir: Path

    def city_cache(self, city: str) -> Path:
        """Return the cache directory for ``city`` creating it when missing."""

        path = self.cache_dir / city.lower()
        path.mkdir(parents=True, exist_ok=True)
        return path


def ensure_workflow_paths(base_dir: Optional[Path] = None) -> VoxCityPaths:
    """Ensure every directory used by the VoxCity notebooks exists.

    The helper also configures environment variables so that NumPy/Numba and
    Matplotlib can write caches in the repository workspace (avoids failures in
    read-only home directories).
    """

    base_dir = Path(base_dir) if base_dir else Path.cwd()
    data_dir = base_dir / "data"
    processed_dir = data_dir / "1-processed" / "VoxCity"
    cache_dir = processed_dir / "cache"
    features_dir = processed_dir / "features"
    visuals_dir = processed_dir / "visualisations"
    views_dir = processed_dir / "views_3d"
    epw_dir = processed_dir / "epw"
    grids_path = data_dir / "utils" / "grids" / "all_cities_30m_grid.parquet"
    combined_features_path = processed_dir / "VoxCity_grid_30m.parquet"

    for folder in [processed_dir, cache_dir, features_dir, visuals_dir, views_dir, epw_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    numba_cache = processed_dir / ".numba_cache"
    numba_cache.mkdir(exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache))

    mpl_cache = processed_dir / ".mpl_cache"
    mpl_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    font_cache = processed_dir / ".fontconfig"
    font_cache.mkdir(exist_ok=True)
    os.environ.setdefault("FONTCONFIG_PATH", str(font_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(processed_dir / ".cache"))

    return VoxCityPaths(
        base_dir=base_dir,
        data_dir=data_dir,
        processed_dir=processed_dir,
        cache_dir=cache_dir,
        features_dir=features_dir,
        visuals_dir=visuals_dir,
        views_dir=views_dir,
        grids_path=grids_path,
        combined_features_path=combined_features_path,
        epw_dir=epw_dir,
    )


def load_city_context(city: str, paths: VoxCityPaths) -> Dict[str, object]:
    """Prepare lightweight metadata required to process a city."""

    if city not in STUDY_CITIES:
        raise ValueError(f"City '{city}' is not part of STUDY_CITIES: {STUDY_CITIES}")

    if not paths.grids_path.exists():
        raise FileNotFoundError(f"Master 30 m grid is missing at {paths.grids_path}")

    master_grid = gpd.read_parquet(paths.grids_path)
    city_grid = master_grid[master_grid["city"] == city].copy()
    if city_grid.empty:
        raise ValueError(f"No grid cells found for {city} in master grid")

    rectangle_vertices = _grid_bounds_to_vertices(city_grid)

    building_path = paths.data_dir / "1-processed" / "Buildings" / f"{city.lower()}_buildings.gpkg"
    if not building_path.exists():
        raise FileNotFoundError(f"Building file missing for {city}: {building_path}")

    return {
        "city": city,
        "rectangle_vertices": rectangle_vertices,
        "grid": city_grid,
        "building_path": building_path,
    }


def _grid_bounds_to_vertices(grid: gpd.GeoDataFrame) -> list[tuple[float, float]]:
    grid_ll = grid.to_crs(4326)
    minx, miny, maxx, maxy = grid_ll.total_bounds
    return [
        (minx, miny),
        (minx, maxy),
        (maxx, maxy),
        (maxx, miny),
    ]


def _bundle_file_map() -> Dict[str, str]:
    return {
        "voxcity": "voxcity.npy",
        "building_height": "building_height.npy",
        "building_min_height": "building_min_height.npy",
        "building_id": "building_id.npy",
        "canopy_height": "canopy_height.npy",
        "canopy_bottom": "canopy_bottom.npy",
        "land_cover": "land_cover.npy",
        "dem": "dem.npy",
        "solar_peak": "solar_peak.npy",
        "solar_daily_summer": "solar_daily_summer.npy",
        "solar_winter_peak": "solar_winter_peak.npy",
        "solar_direct": "solar_direct.npy",
        "solar_diffuse": "solar_diffuse.npy",
        "solar_diffuse_ratio": "solar_diffuse_ratio.npy",
        "solar_seasonal_ratio": "solar_seasonal_ratio.npy",
        "solar_direct_ratio": "solar_direct_ratio.npy",
        "solar_availability": "solar_availability.npy",
        "solar_annual_kwh": "solar_annual_kwh.npy",
    }


def _bundle_complete(cache_dir: Path) -> bool:
    return all((cache_dir / filename).exists() for filename in _bundle_file_map().values())


def load_cached_bundle(paths: VoxCityPaths, city: str) -> Dict[str, np.ndarray]:
    """Load cached VoxCity arrays for ``city``."""

    cache_dir = paths.city_cache(city)
    data: Dict[str, np.ndarray] = {}
    for key, filename in _bundle_file_map().items():
        file_path = cache_dir / filename
        if file_path.exists():
            data[key] = np.load(file_path, allow_pickle=True)
    return data


def save_bundle_to_cache(paths: VoxCityPaths, city: str, bundle: Dict[str, np.ndarray]) -> None:
    """Persist voxel bundle arrays to the city cache directory."""

    cache_dir = paths.city_cache(city)
    for key, array in bundle.items():
        if isinstance(array, np.ndarray):
            np.save(cache_dir / _bundle_file_map()[key], array)


def derive_surface_layers(
    bundle: Dict[str, np.ndarray],
    view_maps: Dict[str, np.ndarray],
    *,
    config: Optional[Dict[str, object]] = None,
    land_cover_indices: Optional[Dict[str, Sequence[int]]] = None,
) -> Dict[str, np.ndarray]:
    """Derive 2D rasters for aggregation from the 3D voxel bundle."""

    voxel_grid = bundle["voxcity"]
    meshsize = float((config or {}).get("meshsize", 5))
    cell_area = meshsize ** 2

    svf_raw = view_maps.get("svf")
    if svf_raw is None:
        raise ValueError("SVF map is required to build surface layers")
    svf_map = np.array(svf_raw, copy=False)
    expected_shape = svf_map.shape

    def _align_layer(arr: Optional[np.ndarray], name: str, *, allow_none: bool = True) -> Optional[np.ndarray]:
        if arr is None:
            if allow_none:
                return None
            raise ValueError(f"Layer '{name}' is required but missing")
        arr_np = np.array(arr, copy=False)
        if arr_np.shape == expected_shape:
            return arr_np
        if arr_np.ndim == 2 and arr_np.shape[::-1] == expected_shape:
            return arr_np.T
        raise ValueError(
            f"Layer '{name}' has shape {arr_np.shape}, expected {expected_shape} (or transposed)"
        )

    gvi_map = _align_layer(view_maps.get("gvi"), "gvi", allow_none=False)

    building_mask = _align_layer((voxel_grid == -3).any(axis=2).astype(float), "building_fraction", allow_none=False)
    tree_mask = _align_layer((voxel_grid == -2).any(axis=2).astype(float), "tree_fraction", allow_none=False)

    canopy_height = _align_layer(bundle.get("canopy_height", np.zeros(expected_shape)), "canopy_height", allow_none=False)
    canopy_base = _align_layer(bundle.get("canopy_bottom", np.zeros(expected_shape)), "canopy_base", allow_none=False)
    canopy_thickness = np.clip(canopy_height - canopy_base, 0.0, None)

    building_height_mean = _align_layer(bundle.get("building_height", np.zeros(expected_shape)), "building_height_mean", allow_none=False)

    building_min_raw = bundle.get("building_min_height")
    building_height_min = np.zeros(expected_shape, dtype=float)
    if building_min_raw is not None:
        building_min_aligned = _align_layer(building_min_raw, "building_min_height", allow_none=False)
        total_cells = building_min_aligned.shape[0] * building_min_aligned.shape[1]
        print(f"Processing building minimum heights for {total_cells:,} cells...")
        cell_count = 0
        for i in tqdm(range(building_min_aligned.shape[0]), desc="Extracting building min heights", leave=False):
            for j in range(building_min_aligned.shape[1]):
                heights_data = building_min_aligned[i, j]
                if heights_data is None:
                    building_height_min[i, j] = 0.0
                    continue
                all_heights: List[float] = []
                if isinstance(heights_data, (list, np.ndarray)):
                    for item in heights_data:
                        if isinstance(item, (list, np.ndarray)):
                            all_heights.extend([float(h) for h in item if np.isfinite(h)])
                        elif np.isfinite(item):
                            all_heights.append(float(item))
                elif np.isfinite(heights_data):
                    all_heights.append(float(heights_data))

                valid_heights = [h for h in all_heights if h > 0]
                if valid_heights:
                    building_height_min[i, j] = min(valid_heights)
                    cell_count += 1
                else:
                    building_height_min[i, j] = 0.0
        print(f"  Extracted minimum heights for {cell_count:,} cells with buildings")

    building_volume = building_height_mean * building_mask * cell_area
    canopy_volume = canopy_thickness * tree_mask * cell_area

    canyon_width = _align_layer(compute_canyon_width(voxel_grid, meshsize), "canyon_width", allow_none=False)
    canyon_aspect_ratio = np.zeros(expected_shape, dtype=float)

    valid_width = canyon_width > 1e-6
    non_building_areas = ~building_mask.astype(bool)
    valid_canyons = valid_width & non_building_areas
    if np.any(valid_canyons):
        from scipy.ndimage import uniform_filter

        smoothed_heights = uniform_filter(building_height_mean, size=3, mode="nearest")
        canyon_aspect_ratio[valid_canyons] = smoothed_heights[valid_canyons] / canyon_width[valid_canyons]

    building_areas = building_mask.astype(bool)
    if np.any(building_areas):
        local_canyon_width = np.maximum(canyon_width[building_areas], meshsize * 2)
        canyon_aspect_ratio[building_areas] = building_height_mean[building_areas] / local_canyon_width

    thermal_cfg = (config or {}).get("thermal", {}) if config else {}
    density = float(thermal_cfg.get("density", 2400.0))
    specific_heat = float(thermal_cfg.get("specific_heat", 880.0))
    thermal_mass_index = np.zeros(expected_shape, dtype=float)
    if density > 0 and specific_heat > 0:
        thermal_mass_index = (building_volume * density * specific_heat) / max(cell_area, 1e-6)

    land_cover = _align_layer(bundle.get("land_cover"), "land_cover")
    if land_cover_indices is None and land_cover is not None:
        source = (config or {}).get("land_cover_source", "OpenStreetMap")
        try:
            land_cover_indices = build_land_cover_indices(source)
        except Exception:  # pragma: no cover
            land_cover_indices = {}
    elif land_cover_indices is None:
        land_cover_indices = {}

    if land_cover is not None and land_cover_indices:
        diagnostics = diagnose_land_cover_issue(land_cover, land_cover_indices)
        unique_values = diagnostics["unique_land_cover_values"]
        print(f"Land cover validation: {len(unique_values)} unique values - {unique_values}")
        valid_range = [v for v in unique_values if 1 <= v <= 14]
        if valid_range:
            print(f"  ✓ Valid land cover indices found: {valid_range}")
        else:
            print(f"  ⚠️  No valid indices in range [1-14]: {unique_values}")
        for group, info in diagnostics["value_coverage"].items():
            if info["pixel_matches"] > 0:
                print(f"  {group}: {info['pixel_matches']:,} pixels ({info['coverage_fraction']:.1%})")

    landcover_masks: Dict[str, np.ndarray] = {}
    if land_cover is not None and land_cover_indices:
        for group, indices in land_cover_indices.items():
            if indices:
                mask = np.isin(land_cover, indices).astype(float)
                landcover_masks[group] = mask

    solar_peak = _align_layer(bundle.get("solar_peak"), "solar_peak", allow_none=False)
    solar_daily = _align_layer(bundle.get("solar_daily_summer"), "solar_daily_summer")
    if solar_daily is None:
        solar_daily = solar_peak * 8.0
    solar_winter_peak = _align_layer(bundle.get("solar_winter_peak"), "solar_winter_peak")
    solar_direct = _align_layer(bundle.get("solar_direct"), "solar_direct")
    solar_diffuse = _align_layer(bundle.get("solar_diffuse"), "solar_diffuse")
    solar_diffuse_ratio = _align_layer(bundle.get("solar_diffuse_ratio"), "solar_diffuse_ratio")
    solar_seasonal_ratio = _align_layer(bundle.get("solar_seasonal_ratio"), "solar_seasonal_ratio")
    solar_direct_ratio = _align_layer(bundle.get("solar_direct_ratio"), "solar_direct_ratio")
    solar_availability = _align_layer(bundle.get("solar_availability"), "solar_availability")
    solar_annual_kwh = _align_layer(bundle.get("solar_annual_kwh"), "solar_annual_kwh")

    if solar_winter_peak is None:
        solar_winter_peak = solar_peak * 0.3
    if solar_direct is None:
        solar_direct = solar_peak * 0.8
    if solar_diffuse is None:
        solar_diffuse = solar_peak * 0.2
    if solar_diffuse_ratio is None:
        default_diffuse = 200.0
        solar_diffuse_ratio = np.divide(
            solar_diffuse,
            default_diffuse if default_diffuse else 1.0,
            out=np.zeros_like(solar_diffuse, dtype=float),
            where=np.isfinite(solar_diffuse),
        )
    if solar_seasonal_ratio is None:
        solar_seasonal_ratio = np.divide(
            solar_winter_peak,
            solar_peak,
            out=np.zeros_like(solar_peak),
            where=solar_peak != 0,
        )
    if solar_direct_ratio is None:
        solar_direct_ratio = np.divide(
            solar_direct,
            solar_peak,
            out=np.zeros_like(solar_peak),
            where=solar_peak != 0,
        )
    if solar_availability is None:
        solar_availability = np.divide(
            solar_peak,
            np.nanmax(solar_peak) if np.isfinite(solar_peak).any() else 1.0,
            out=np.zeros_like(solar_peak),
            where=solar_peak != 0,
        )
    if solar_annual_kwh is None:
        solar_annual_kwh = solar_daily * 365 / 1000 * 0.3

    from scipy.ndimage import maximum_filter

    try:
        solar_local_max = maximum_filter(solar_peak, size=3)
        solar_contrast = solar_local_max - solar_peak
    except Exception:
        solar_contrast = np.zeros_like(solar_peak)

    urban_heat_island = solar_daily * building_mask * (1 - svf_map)
    evapotranspiration = gvi_map * tree_mask * solar_daily * 0.001
    canyon_heating = np.where(
        canyon_aspect_ratio > 1.0,
        solar_peak * (1 - svf_map) * canyon_aspect_ratio,
        0.0,
    )
    building_solar_exposure = solar_peak * building_mask * np.exp(-svf_map)
    green_cooling_potential = -(gvi_map * svf_map * solar_peak * 0.001)
    shadow_cooling = -(1 - solar_availability) * svf_map * tree_mask

    building_id = _align_layer(bundle.get("building_id"), "building_id")

    layers: Dict[str, np.ndarray] = {
        "svf": svf_map,
        "gvi": gvi_map,
        "solar": solar_peak,
        "solar_daily_summer": solar_daily,
        "solar_winter_peak": solar_winter_peak,
        "solar_direct": solar_direct,
        "solar_diffuse": solar_diffuse,
        "solar_diffuse_ratio": solar_diffuse_ratio,
        "solar_seasonal_ratio": solar_seasonal_ratio,
        "solar_direct_ratio": solar_direct_ratio,
        "solar_availability": solar_availability,
        "solar_annual_kwh": solar_annual_kwh,
        "urban_heat_island": urban_heat_island,
        "evapotranspiration": evapotranspiration,
        "canyon_heating": canyon_heating,
        "building_solar_exposure": building_solar_exposure,
        "green_cooling_potential": green_cooling_potential,
        "shadow_cooling": shadow_cooling,
        "solar_contrast": solar_contrast,
        "building_height_mean": building_height_mean,
        "building_height_min": building_height_min,
        "canopy_height": canopy_height,
        "canopy_base": canopy_base,
        "canopy_thickness": canopy_thickness,
        "building_fraction": building_mask,
        "tree_fraction": tree_mask,
        "building_volume": building_volume,
        "canopy_volume": canopy_volume,
        "thermal_mass_index": thermal_mass_index,
        "canyon_aspect_ratio": canyon_aspect_ratio,
    }

    if building_id is not None:
        layers["building_id"] = building_id

    for group, mask in landcover_masks.items():
        layers[f"landcover_{group}"] = mask

    return layers


def mesh_layers_to_geodataframe(
    city: str,
    layers: Dict[str, np.ndarray],
    rectangle_vertices: Iterable[Iterable[float]],
    meshsize: float,
) -> gpd.GeoDataFrame:
    """Convert per-voxel rasters into a GeoDataFrame with 5 m cells."""

    from voxcity.geoprocessor.grid import grid_to_geodataframe

    first_key = next(iter(layers))
    base_gdf = grid_to_geodataframe(layers[first_key], rectangle_vertices, meshsize).rename(
        columns={"value": first_key}
    )

    for key, array in layers.items():
        if key == first_key:
            continue
        base_gdf[key] = np.flipud(array).ravel()

    base_gdf["city"] = city
    return base_gdf


def aggregate_city_features(
    city_ctx: Dict[str, object],
    mesh_gdf: gpd.GeoDataFrame,
    aggregation: Optional[Dict[str, str]] = None,
) -> gpd.GeoDataFrame:
    """Aggregate 5 m voxcity rasters to the 30 m dissertation grid."""

    aggregation = aggregation or {
        # Enhanced aggregation strategies for better signal preservation
        "svf": "mean",  # Mean SVF better represents typical openness (min too extreme)
        "gvi": "mean",  # Mean GVI better captures overall greenery level
        "solar": "max",  # Peak solar matters more than mean for heat accumulation
        "solar_daily_summer": "max",  # Daily energy hotspots drive UHI
        "solar_winter_peak": "max",
        "solar_direct": "max",  # Direct radiation creates hotspots
        "solar_diffuse": "mean",  # Diffuse is more uniform
        "solar_seasonal_ratio": "mean",
        "solar_direct_ratio": "max",  # High direct ratio = strong heating
        "solar_availability": "max",  # Maximum exposure per cell
        "solar_annual_kwh": "max",
        # Enhanced physics-based features
        "urban_heat_island": "max",  # Heat island hotspots are key
        "evapotranspiration": "mean",  # Average cooling effect
        "canyon_heating": "max",  # Maximum canyon heating effect
        "building_solar_exposure": "max",  # Peak building exposure
        "green_cooling_potential": "min",  # Most cooling (negative values)
        "shadow_cooling": "min",  # Most cooling (negative values)
        "solar_contrast": "max",  # Maximum thermal stress
        # Building features
        "building_height_mean": "mean",
        "building_height_min": "mean",
        "canopy_height": "mean",
        "canopy_base": "mean",
        "canopy_thickness": "mean",
        "building_fraction": "mean",
        "tree_fraction": "mean",
        "building_volume": "sum",  # Total volume per 30m cell
        "canopy_volume": "sum",   # Total canopy volume
        "thermal_mass_index": "sum",  # Total thermal mass (accumulates heat)
        "canyon_aspect_ratio": "max",  # Deepest canyon effects
        "landcover_impervious": "mean",
        "landcover_vegetation": "mean",
    }

    master_grid = city_ctx["grid"].copy()
    grid_ll = master_grid.to_crs(4326)[["global_grid_id", "geometry"]]

    joined = gpd.sjoin(mesh_gdf, grid_ll, how="inner", predicate="intersects")
    if "index_right" in joined.columns:
        joined = joined.drop(columns="index_right")

    value_cols = [col for col in aggregation if col in joined.columns]
    joined_numeric = joined[value_cols + ["global_grid_id"]].copy()

    def _coerce_scalar(value):
        if isinstance(value, (list, tuple)):
            return np.nan
        if hasattr(value, "__array__") and not np.isscalar(value):
            arr = np.asarray(value).ravel()
            return float(arr[0]) if arr.size else np.nan
        return value

    for col in value_cols:
        joined_numeric[col] = joined_numeric[col].map(_coerce_scalar)
        joined_numeric[col] = pd.to_numeric(joined_numeric[col], errors="coerce")

    aggregated = joined_numeric.groupby("global_grid_id").agg({col: aggregation[col] for col in value_cols})
    rename_map = {
        "building_fraction": "built_up_ratio",
        "tree_fraction": "tree_canopy_ratio",
        "canopy_height": "canopy_height_mean",
        "canopy_base": "canopy_base_mean",
        "canopy_thickness": "canopy_thickness_mean",
        "building_volume": "building_volume_sum",
        "canopy_volume": "canopy_volume_sum",
        "landcover_impervious": "landcover_impervious_ratio",
        "landcover_vegetation": "landcover_vegetation_ratio",
    }
    aggregated = aggregated.rename(columns={k: v for k, v in rename_map.items() if k in aggregated.columns})

    enriched = master_grid.merge(aggregated, on="global_grid_id", how="left")
    return enriched


def summarise_city_layers(enriched_grid: gpd.GeoDataFrame) -> pd.Series:
    """Quick summary statistics for diagnostic display in notebooks."""

    stats = {"cells": len(enriched_grid)}
    coverage_cols = [col for col in ["svf", "gvi", "solar", "built_up_ratio"] if col in enriched_grid]
    if coverage_cols:
        stats["coverage_pct"] = (
            enriched_grid[coverage_cols].dropna().shape[0] / len(enriched_grid) * 100
        )

    for field in [
        "svf",
        "gvi",
        "solar",
        "solar_daily_summer",
        "solar_winter_peak",
        "solar_direct",
        "solar_diffuse",
        "solar_seasonal_ratio",
        "solar_direct_ratio",
        "solar_availability",
        "solar_annual_kwh",
        "built_up_ratio",
        "tree_canopy_ratio",
        "canopy_height_mean",
        "canopy_base_mean",
        "canopy_thickness_mean",
        "building_volume_sum",
        "canopy_volume_sum",
        "thermal_mass_index",
        "canyon_aspect_ratio",
        "landcover_impervious_ratio",
        "landcover_vegetation_ratio",
    ]:
        if field in enriched_grid:
            stats[f"{field}_mean"] = float(enriched_grid[field].mean(skipna=True))

    return pd.Series(stats)


def crop_background(
    img: np.ndarray,
    base_tol: int = 8,
    pad_ratio: float = 0.02
) -> np.ndarray:
    """
    Automatically crop whitespace/background from an image while preserving content.

    Useful for tightening visualization layouts by removing excess padding around
    rendered images (e.g., isometric views).

    Args:
        img: Input image array (H, W, C) with values in [0, 1] or [0, 255]
        base_tol: Tolerance for background detection (higher = more aggressive crop)
        pad_ratio: Padding to add back after crop as ratio of cropped dimensions

    Returns:
        Cropped image with minimal background padding

    Example:
        >>> img = mpimg.imread('city_view.png')
        >>> cropped = crop_background(img, base_tol=10, pad_ratio=0.015)
        >>> plt.imshow(cropped)
    """
    # Normalize to uint8
    arr = (img * 255).clip(0, 255).astype(np.uint8) if img.dtype.kind == 'f' else img.copy()

    # Ensure RGB (handle grayscale and RGBA)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    rgb = arr[..., :3] if arr.shape[-1] == 4 else arr

    h, w = rgb.shape[:2]

    # Sample border pixels to estimate background color
    b = max(1, min(h, w) // 25)
    border = np.concatenate([
        rgb[:b].reshape(-1, 3),
        rgb[-b:].reshape(-1, 3),
        rgb[:, :b].reshape(-1, 3),
        rgb[:, -b:].reshape(-1, 3)
    ], axis=0)
    bg = np.median(border, axis=0)

    # Create mask of non-background pixels
    mask = np.any(np.abs(rgb.astype(np.int16) - bg.astype(np.int16)) > base_tol, axis=-1)
    coords = np.argwhere(mask)

    if coords.size == 0:
        return img  # Nothing to crop

    # Find bounding box
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # Add padding
    pad_y = max(1, int((y1 - y0) * pad_ratio))
    pad_x = max(1, int((x1 - x0) * pad_ratio))

    return img[max(0, y0-pad_y):min(h, y1+pad_y), max(0, x0-pad_x):min(w, x1+pad_x)]


def create_metric_panel_with_basemap(
    city_gdfs: Mapping[str, gpd.GeoDataFrame],
    column: str,
    *,
    city_order: Optional[Sequence[str]] = None,
    cmap: str = "viridis",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (24, 18),
    save_dir: Optional[Path] = None,
    filename: Optional[str] = None,
    normalize_mode: str = "per_city",
    value_range: Optional[Tuple[float, float]] = None,
    percentile_bounds: Tuple[float, float] = (5.0, 95.0),
    colorbar_label: Optional[str] = None,
    basemap_alpha: float = 0.5,
    show: bool = True,
) -> Tuple[plt.Figure, Optional[Path]]:
    """Create a 3×2 panel using GeoDataFrames and add the dual basemap."""

    if not city_gdfs:
        raise ValueError("city_gdfs cannot be empty")

    cities = list(city_order) if city_order else sorted(city_gdfs)
    if len(cities) > 6:
        raise ValueError("create_metric_panel_with_basemap supports up to 6 cities")

    mode = normalize_mode.lower()
    if mode not in {"per_city", "global", "percentile"}:
        raise ValueError("normalize_mode must be 'per_city', 'global', or 'percentile'")

    if value_range is not None and mode != "global":
        raise ValueError("value_range is only valid when normalize_mode='global'")

    # Get colormap and configure NaN handling (matching VoxCity style)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='lightgray')  # NaN values shown in light gray

    if mode == "global" and value_range is None:
        mins, maxs = [], []
        for city in cities:
            gdf = city_gdfs[city]
            if column not in gdf:
                continue
            arr = gdf[column].to_numpy(dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                mins.append(float(finite.min()))
                maxs.append(float(finite.max()))
        if mins and maxs:
            value_range = (min(mins), max(maxs))
        else:
            value_range = (0.0, 1.0)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    for idx, city in enumerate(cities):
        ax = axes[idx]
        gdf = city_gdfs[city]
        if column not in gdf:
            ax.text(0.5, 0.5, f"{column} missing", ha="center", va="center")
            ax.set_axis_off()
            continue

        plot_gdf = gdf[[column, "geometry"]].dropna(subset=[column]).copy()
        if plot_gdf.empty:
            ax.text(0.5, 0.5, f"No data for {city}", ha="center", va="center")
            ax.set_axis_off()
            continue

        if plot_gdf.crs is None:
            plot_gdf = plot_gdf.set_crs(4326)
        plot_gdf = plot_gdf.to_crs("EPSG:3857")

        series = plot_gdf[column].to_numpy(dtype=float)
        finite = series[np.isfinite(series)]
        if finite.size:
            vmin_local = float(finite.min())
            vmax_local = float(finite.max())
        else:
            vmin_local = 0.0
            vmax_local = 1.0

        if mode == "per_city":
            if vmax_local == vmin_local:
                vmax_local = vmin_local + 1e-9
            norm = Normalize(vmin=vmin_local, vmax=vmax_local)
        elif mode == "percentile":
            lower, upper = percentile_bounds
            if finite.size:
                bounds = np.percentile(finite, [lower, upper])
            else:
                bounds = (0.0, 1.0)
            if bounds[1] == bounds[0]:
                bounds = (bounds[0], bounds[0] + 1e-9)
            norm = Normalize(vmin=float(bounds[0]), vmax=float(bounds[1]))
        else:
            norm = Normalize(vmin=value_range[0], vmax=value_range[1])

        plot_gdf.plot(
            column=column,
            ax=ax,
            cmap=cmap_obj,
            norm=norm,
            alpha=0.85,
            linewidth=0,
        )

        _add_dual_basemap(ax, plot_gdf.crs, alpha=basemap_alpha)

        ax.set_axis_off()
        ax.set_title(city, fontsize=16, fontweight="bold")

        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj),
            ax=ax,
            fraction=0.046,
            pad=0.02,
        )
        if colorbar_label:
            cbar.set_label(colorbar_label, fontsize=12)
        cbar.locator = MaxNLocator(4)
        cbar.update_ticks()

    for idx in range(len(cities), 6):
        axes[idx].set_axis_off()

    if title:
        fig.suptitle(title, fontsize=20, fontweight="bold", y=0.985)

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.975])

    saved_path = None
    if save_dir and filename:
        save_dir.mkdir(parents=True, exist_ok=True)
        saved_path = save_dir / filename
        fig.savefig(saved_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, saved_path


def _crop_background(img: np.ndarray, base_tol: int = 8, pad_ratio: float = 0.02) -> np.ndarray:
    """Crop white/transparent borders from rendered images for cleaner panels."""

    arr = img.copy()
    if arr.dtype.kind == "f":
        arr = (arr * 255).clip(0, 255).astype(np.uint8)

    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)

    rgb = arr[..., :3] if arr.shape[-1] == 4 else arr
    h, w = rgb.shape[:2]
    border = np.concatenate(
        [
            rgb[: max(1, h // 25)].reshape(-1, 3),
            rgb[-max(1, h // 25):].reshape(-1, 3),
            rgb[:, : max(1, w // 25)].reshape(-1, 3),
            rgb[:, -max(1, w // 25):].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(border, axis=0)
    mask = np.any(np.abs(rgb.astype(np.int16) - bg.astype(np.int16)) > base_tol, axis=-1)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    pad_y = max(1, int((y1 - y0) * pad_ratio))
    pad_x = max(1, int((x1 - x0) * pad_ratio))
    return img[max(0, y0 - pad_y): min(h, y1 + pad_y), max(0, x0 - pad_x): min(w, x1 + pad_x)]


def crop_render_background(img: np.ndarray, base_tol: int = 8, pad_ratio: float = 0.02) -> np.ndarray:
    """Convenience wrapper exposing background cropping for notebooks."""

    return _crop_background(img, base_tol=base_tol, pad_ratio=pad_ratio)


def render_voxel_iso(
    city: str,
    bundle: Dict[str, np.ndarray],
    paths: VoxCityPaths,
    *,
    meshsize: float,
    overlay: Optional[object] = None,
    show: bool = False,
    distance_factor: float = 1.2,
    crop: bool = True,
    overlay_label: Optional[str] = None,
    overlay_clim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (10, 10),
    dpi: int = 200,
    output_filename: Optional[str] = None,
) -> Path:
    """Generate a back-right isometric render and return the saved PNG path.

    Rendering is optional in notebooks: set ``show=False`` to skip inline display.
    The helper relies on PyVista and may take ~30 s on the first call due to
    shader compilation and font cache creation.

    Parameters
    ----------
    city:
        City name (used for output directory naming).
    bundle:
        VoxCity bundle containing at least the ``voxcity`` array. When an overlay is
        requested the bundle's ``dem`` grid is used as the default elevation surface.
    overlay:
        When ``None`` (default) renders the base voxels. Provide either the name of a
        bundle field (e.g. ``"solar_peak"``) or a configuration dictionary with keys:

        ``grid`` / ``grid_key``:
            Either a 2D numpy array or the name of a bundle field holding the values to
            overlay as an elevated surface.
        ``dem``:
            Optional DEM array matching the grid shape. Defaults to ``bundle["dem"]`` or
            an all-zero plane when missing.
        ``colormap`` / ``vmin`` / ``vmax`` / ``view_point_height``:
            Styling overrides forwarded to :func:`voxcity.geoprocessor.mesh.create_sim_surface_mesh`.
    output_filename:
        Optional override for the saved PNG name inside the city view directory.
    """

    from voxcity.utils.visualization import (
        create_city_meshes,
        create_multi_view_scene,
        get_voxel_color_map,
    )

    import pyvista as pv

    pv.set_plot_theme("document")
    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1280, 960]
    pv.global_theme.jupyter_backend = "static"

    output_dir = paths.views_dir / city
    output_dir.mkdir(parents=True, exist_ok=True)
    target_filename = output_filename or f"{city}_iso.png"

    vox_dict = get_voxel_color_map("default")
    # Boost chroma for hydrological ground classes so coastlines and rivers pop.
    water_overrides = {
        7: [52, 140, 168],   # wetlands → teal
        8: [32, 110, 170],   # mangrove / tidal edges
        9: [12, 94, 200],    # open water
    }
    for class_id, rgb in water_overrides.items():
        if class_id in vox_dict:
            vox_dict[class_id] = rgb
    meshes = create_city_meshes(bundle["voxcity"], vox_dict, meshsize=meshsize)

    building_mesh = meshes.get(-3)

    cmap_name = "magma"
    overlay_vmin: Optional[float] = None
    overlay_vmax: Optional[float] = None

    show_overlay_colorbar = True
    if overlay is not None:
        overlay_grid: Optional[np.ndarray]
        overlay_dem: Optional[np.ndarray]
        overlay_cfg: Dict[str, object]

        if isinstance(overlay, str):
            overlay_grid = bundle.get(overlay)
            overlay_dem = bundle.get("dem")
            overlay_cfg = {}
        elif isinstance(overlay, dict):
            overlay_cfg = dict(overlay)
            show_overlay_colorbar = overlay_cfg.pop("show_colorbar", True)
            key = overlay_cfg.pop("grid_key", None)
            overlay_grid = overlay_cfg.pop("grid", None)
            if overlay_grid is None and key is not None:
                overlay_grid = bundle.get(key)
            overlay_dem = overlay_cfg.pop("dem", None)
            if overlay_dem is None and overlay_cfg.get("use_bundle_dem", True):
                overlay_dem = bundle.get("dem")
            cmap_name = overlay_cfg.get("colormap", cmap_name)
            overlay_vmin = overlay_cfg.get("vmin")
            overlay_vmax = overlay_cfg.get("vmax")
        else:
            raise TypeError("overlay must be a string key or a configuration dictionary")

        if overlay_grid is not None:
            overlay_dem = overlay_dem if overlay_dem is not None else np.zeros_like(overlay_grid)
            try:
                dem_array = np.asarray(overlay_dem, dtype=float)
                if np.isfinite(dem_array).any():
                    dem_array = dem_array - float(np.nanmin(dem_array[np.isfinite(dem_array)]))
                else:
                    dem_array = np.zeros_like(dem_array)
            except Exception:
                dem_array = np.zeros_like(overlay_grid, dtype=float)

            sim_params = {
                "meshsize": meshsize,
                "z_offset": float(
                    overlay_cfg.get("view_point_height", 1.5) if isinstance(overlay_cfg, dict) else 1.5
                ),
                "cmap_name": cmap_name,
                "vmin": overlay_vmin,
                "vmax": overlay_vmax,
            }

            from voxcity.geoprocessor.mesh import create_sim_surface_mesh

            sim_mesh = create_sim_surface_mesh(
                np.asarray(overlay_grid, dtype=float),
                dem_array,
                **sim_params,
            )

            if sim_mesh is not None:
                meshes["sim_surface"] = sim_mesh

        if building_mesh is not None:
            try:
                face_cols = np.asarray(building_mesh.visual.face_colors, dtype=np.float32)
                if face_cols.size:
                    face_cols[..., :3] = 0.55 * face_cols[..., :3] + 115.0
                    face_cols[..., 3] = np.clip(face_cols[..., 3] * 0.6, 60, 255)
                    building_mesh.visual.face_colors = face_cols.astype(np.uint8)
            except Exception:
                pass

    views = create_multi_view_scene(
        meshes,
        output_directory=str(output_dir),
        projection_type="perspective",
        distance_factor=distance_factor,
    )

    iso_path: Optional[Path] = None
    for name, filename in views:
        if "iso_back_right" in name:
            iso_path = Path(filename)
            break

    if iso_path is None and views:
        iso_path = Path(views[0][1])

    if not show:
        import matplotlib.pyplot as plt

        plt.close("all")

    if iso_path is None:
        raise RuntimeError("Failed to render isometric view")

    if iso_path is None:
        raise RuntimeError("Failed to render isometric view")

    import matplotlib.pyplot as plt

    img = plt.imread(str(iso_path))
    if crop:
        img = _crop_background(img)

    output_dir = iso_path.parent
    final_path = output_dir / target_filename

    if overlay is not None:
        if overlay_clim is None and overlay_vmin is not None and overlay_vmax is not None:
            overlay_clim = (float(overlay_vmin), float(overlay_vmax))

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)
        ax.axis("off")

        if overlay_clim is not None and show_overlay_colorbar:
            from matplotlib.cm import ScalarMappable

            norm = Normalize(vmin=overlay_clim[0], vmax=overlay_clim[1])
            sm = ScalarMappable(norm=norm, cmap=cmap_name)
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
            if overlay_label:
                cbar.set_label(overlay_label)

        fig.savefig(final_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        if not show:
            plt.close(fig)
    else:
        plt.imsave(final_path, img)

    if iso_path != final_path and iso_path.exists():
        try:
            iso_path.unlink()
        except OSError:
            pass

    if not show:
        plt.close("all")

    return final_path


# ============================================================================
# Lightweight Caching Helpers for Notebook Transparency
# ============================================================================


def render_isometric_panel(
    *,
    bundles: Mapping[str, Mapping[str, np.ndarray]],
    paths: VoxCityPaths,
    meshsize: float,
    cities: Sequence[str],
    output_path: Path,
    title: Optional[str] = None,
    overlay_key: Optional[str] = None,
    overlay_label: Optional[str] = None,
    overlay_percentiles: Tuple[float, float] = (5.0, 95.0),
    overlay_cmap: str = "BuPu_r",
    distance_factor: float = 1.15,
    figsize: Tuple[int, int] = (18, 10),
    row_gap: float = -0.4,
    col_gap: float = 0.02,
    title_offset: float = 0.85,
    title_kwargs: Optional[Dict[str, object]] = None,
    force: bool = False,
) -> Path:
    """Render a 3×2 panel of isometric views with consistent styling."""

    from matplotlib import image as mpimg
    from matplotlib.cm import ScalarMappable

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes_grid = plt.subplots(2, 3, figsize=figsize)
    axes = axes_grid.flatten().tolist()
    overlay_dict: Optional[Dict[str, object]] = None
    overlay_clim: Optional[Tuple[float, float]] = None

    if overlay_key is not None:
        lower, upper = overlay_percentiles
        if not 0 <= lower < upper <= 100:
            raise ValueError("overlay_percentiles must be an increasing pair within [0, 100]")

        value_chunks: list[np.ndarray] = []
        for city in cities:
            bundle = bundles.get(city)
            if bundle is None:
                continue
            grid = bundle.get(overlay_key)
            if grid is None:
                continue
            finite = np.asarray(grid, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                value_chunks.append(finite)

        if not value_chunks:
            raise KeyError(f"Overlay key '{overlay_key}' is missing for the requested cities")

        stacked = np.concatenate(value_chunks)
        vmin = float(np.nanpercentile(stacked, lower))
        vmax = float(np.nanpercentile(stacked, upper))
        overlay_clim = (vmin, vmax)
        overlay_dict = {
            "grid_key": overlay_key,
            "colormap": overlay_cmap,
            "vmin": vmin,
            "vmax": vmax,
            "show_colorbar": False,
        }

    for idx, city in enumerate(cities):
        ax = axes[idx]
        bundle = bundles.get(city)
        if bundle is None:
            ax.text(0.5, 0.5, f"No bundle for {city}", ha="center", va="center")
            ax.set_axis_off()
            continue

        cache_dir = paths.views_dir / city
        cache_dir.mkdir(parents=True, exist_ok=True)
        suffix = "iso" if overlay_key is None else f"iso_{overlay_key}"
        cached_image = cache_dir / f"{city}_{suffix}.png"

        if not force and cached_image.exists():
            render_path = cached_image
        else:
            render_path = render_voxel_iso(
                city,
                dict(bundle),
                paths,
                meshsize=meshsize,
                overlay=overlay_dict,
                overlay_label=None,
                overlay_clim=overlay_clim,
                distance_factor=distance_factor,
                show=False,
                output_filename=f"{city}_{suffix}.png",
            )
            render_path = cached_image if cached_image.exists() else render_path

        img = mpimg.imread(str(render_path))
        ax.imshow(crop_render_background(img))
        ax.set_title(city, fontsize=12, fontweight="bold")
        ax.set_axis_off()

    for idx in range(len(cities), 6):
        axes[idx].set_axis_off()

    final_title = title if title is not None else "Six European cores (3D voxel inputs)"
    if final_title:
        subtitle_opts = {"fontsize": 18, "fontweight": "bold", "y": title_offset}
        if title_kwargs:
            subtitle_opts.update(title_kwargs)
        fig.suptitle(final_title, **subtitle_opts)

    layout_rect = [0.015, 0.01, 0.985, 0.97]
    if overlay_clim is not None:
        norm = Normalize(vmin=overlay_clim[0], vmax=overlay_clim[1])
        sm = ScalarMappable(norm=norm, cmap=overlay_cmap)
        sm.set_array([])
        cax = fig.add_axes([0.905, 0.2, 0.012, 0.6])
        cbar = fig.colorbar(sm, cax=cax)
        if overlay_label:
            cbar.set_label(overlay_label, fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        layout_rect = [0.015, 0.01, 0.88, 0.97]

    fig.tight_layout(rect=layout_rect)
    fig.subplots_adjust(hspace=row_gap, wspace=col_gap)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path


def create_voxcity_plot_panel(
    *,
    bundles: Mapping[str, Mapping[str, np.ndarray]],
    paths: VoxCityPaths,
    meshsize: float,
    cities: Sequence[str],
    metric_type: str,
    output_path: Path,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (30, 20),
    **voxcity_kwargs,
) -> Path:
    """
    Create a 3×2 panel by arranging VoxCity's built-in plots.

    This function calls VoxCity functions with obj_export=True to generate
    individual city PNGs with VoxCity's native styling (colormaps, alpha, etc.),
    then arranges them into a multi-city comparison panel.

    Args:
        bundles: Dict mapping city names to voxel bundles
        paths: Workflow paths configuration
        meshsize: Voxel size in meters
        cities: List of city names to include
        metric_type: Type of metric - 'gvi', 'svi', 'solar_instantaneous', 'solar_diffuse'
        output_path: Where to save the final panel
        title: Panel title (auto-generated if None)
        figsize: Figure size in inches
        **voxcity_kwargs: VoxCity-specific parameters
            For GVI/SVI: colormap='viridis', view_point_height=1.5, etc.
            For solar: colormap='magma', calc_time='06-15 14:00:00', etc.

    Returns:
        Path to saved panel PNG

    Example:
        >>> # Green View Index with VoxCity's viridis colormap
        >>> create_voxcity_plot_panel(
        ...     bundles=voxel_bundles,
        ...     paths=paths,
        ...     meshsize=5.0,
        ...     cities=SELECTED_CITIES,
        ...     metric_type='gvi',
        ...     output_path=paths.visuals_dir / 'panels' / 'gvi_voxcity_panel.png',
        ...     colormap='viridis',
        ...     view_point_height=1.5,
        ... )
        >>>
        >>> # Solar diffuse with VoxCity's magma colormap
        >>> create_voxcity_plot_panel(
        ...     bundles=voxel_bundles,
        ...     paths=paths,
        ...     meshsize=5.0,
        ...     cities=SELECTED_CITIES,
        ...     metric_type='solar_diffuse',
        ...     output_path=paths.visuals_dir / 'panels' / 'solar_diffuse_voxcity_panel.png',
        ...     colormap='magma',
        ...     calc_time='06-15 14:00:00',
        ... )
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from pathlib import Path

    # VoxCity output directory
    voxcity_output_dir = paths.cache_dir / 'voxcity_plots'
    voxcity_output_dir.mkdir(parents=True, exist_ok=True)

    # Default VoxCity parameters
    vox_defaults = {
        'view_point_height': 1.5,
        'tree_k': 0.6,
        'tree_lad': 0.5,
        'obj_export': True,  # Enable VoxCity's PNG export
        'colormap': 'viridis',
        'output_directory': str(voxcity_output_dir),
        'alpha': 1.0,
        'vmin': 0,
    }
    vox_defaults.update(voxcity_kwargs)

    # Metric-specific defaults
    if metric_type in ['solar_instantaneous', 'solar_diffuse']:
        vox_defaults.setdefault('colormap', 'magma')
    elif metric_type == 'svi':
        vox_defaults.setdefault('colormap', 'BuPu_r')

    plot_paths = {}

    # Generate VoxCity plots for each city
    for city in cities:
        print(f"Generating VoxCity {metric_type} plot for {city}...")
        bundle = bundles[city]
        voxcity_grid = bundle.get('voxcity_grid')

        if voxcity_grid is None:
            print(f"  Warning: No voxcity_grid for {city}, skipping")
            continue

        dem_grid = bundle.get('dem', np.zeros_like(voxcity_grid[0]))

        # Set output filename
        output_filename = f"{city}_{metric_type}"
        vox_defaults['output_file_name'] = output_filename

        try:
            if metric_type == 'gvi':
                if get_view_index is None:
                    raise ImportError("VoxCity view module not available")
                get_view_index(
                    voxcity_grid, meshsize,
                    mode='green',
                    dem_grid=dem_grid,
                    **vox_defaults
                )
                plot_paths[city] = voxcity_output_dir / f"{output_filename}.png"

            elif metric_type == 'svi':
                if get_view_index is None:
                    raise ImportError("VoxCity view module not available")
                get_view_index(
                    voxcity_grid, meshsize,
                    mode='sky',
                    dem_grid=dem_grid,
                    elevation_min_degrees=0,
                    **vox_defaults
                )
                plot_paths[city] = voxcity_output_dir / f"{output_filename}.png"

            elif metric_type == 'solar_instantaneous':
                solar_kwargs = {
                    **vox_defaults,
                    'calc_type': 'instantaneous',
                    'calc_time': vox_defaults.get('calc_time', '06-15 14:00:00'),
                    'dem_grid': dem_grid,
                }
                get_global_solar_irradiance_using_epw(
                    voxcity_grid, meshsize,
                    **solar_kwargs
                )
                plot_paths[city] = voxcity_output_dir / f"{output_filename}.png"

            elif metric_type == 'solar_diffuse':
                get_diffuse_solar_irradiance_map(
                    voxcity_grid, meshsize,
                    dem_grid=dem_grid,
                    **vox_defaults
                )
                plot_paths[city] = voxcity_output_dir / f"{output_filename}.png"

            else:
                raise ValueError(f"Unknown metric_type: {metric_type}")

            print(f"  ✓ Plot saved to: {plot_paths[city]}")

        except Exception as e:
            print(f"  Error generating {metric_type} for {city}: {e}")
            continue

    if not plot_paths:
        raise ValueError(f"No {metric_type} plots generated for any city")

    # Create 3x2 panel by arranging VoxCity plots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for idx, city in enumerate(cities):
        ax = axes[idx]

        if city in plot_paths and plot_paths[city].exists():
            img = mpimg.imread(str(plot_paths[city]))
            # Apply crop_background if available
            try:
                img = crop_background(img, base_tol=10, pad_ratio=0.015)
            except:
                pass  # Use uncropped if crop fails

            ax.imshow(img)
            ax.set_title(city, fontsize=28, fontweight='bold', pad=15)
        else:
            ax.text(0.5, 0.5, f"No plot for {city}",
                   ha='center', va='center', fontsize=20)

        ax.axis('off')

    # Hide unused subplots
    for idx in range(len(cities), 6):
        axes[idx].axis('off')

    # Auto-generate title if not provided
    if title is None:
        titles = {
            'gvi': 'Green View Index (VoxCity)',
            'svi': 'Sky View Index (VoxCity)',
            'solar_instantaneous': 'Instantaneous Solar Irradiance (VoxCity)',
            'solar_diffuse': 'Diffuse Solar Irradiance (VoxCity)',
        }
        title = titles.get(metric_type, metric_type)

    fig.suptitle(title, fontsize=36, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n✅ {metric_type} panel saved to: {output_path}")
    return output_path


def try_load_cached_bundle(city: str, paths: VoxCityPaths) -> Optional[Dict[str, np.ndarray]]:
    """Try to load a cached voxel bundle. Returns None if not found or incomplete.

    This is a lightweight helper for notebooks that want to show explicit
    VoxCity function calls while still benefiting from caching.
    """
    cache_dir = paths.city_cache(city)
    if _bundle_complete(cache_dir):
        return load_cached_bundle(paths, city)
    return None


def save_to_cache(city: str, bundle: Dict[str, np.ndarray], paths: VoxCityPaths) -> None:
    """Save a voxel bundle to cache.

    Lightweight helper for notebooks showing explicit VoxCity calls.
    """
    save_bundle_to_cache(paths, city, bundle)


def get_city_epw(city: str, paths: Optional[VoxCityPaths] = None) -> Optional[Path]:
    """Get EPW file path for a city, with special handling for Berlin.

    Berlin requires a custom EPW file due to data quality issues in the default
    TMY file. For other cities, returns None to let VoxCity auto-download.

    Args:
        city: City name
        paths: Optional VoxCityPaths instance for Berlin's custom EPW

    Returns:
        Path to custom EPW if Berlin and file exists, otherwise None (auto-download)
    """
    if city.lower() == 'berlin' and paths is not None:
        epw_dir = paths.epw_dir
        candidates = sorted(epw_dir.glob('DEU_BE_Berlin-Tempelhof.AP.103840_TMYx*.epw'))
        if not candidates:
            candidates = sorted(epw_dir.glob('berlin-tempelhof.ap_be_deu.epw'))
        if candidates:
            # Use the longest filename first (tends to include year range)
            candidates.sort(key=lambda p: len(p.name), reverse=True)
            return candidates[0]
    return None  # Let VoxCity auto-download


def _parse_calc_time(calc_time: str) -> tuple[int, int, int, int]:
    dt = datetime.strptime(calc_time, "%m-%d %H:%M:%S")
    return dt.month, dt.day, dt.hour, dt.minute


def _extract_epw_instant_inputs(
    epw_path: Optional[Path],
    calc_time: str,
    solar_params: Dict[str, object],
) -> Optional[Dict[str, float]]:
    """Return DNI, DHI, azimuth, elevation from EPW for ``calc_time`` or ``None`` on failure."""

    if epw_path is None or not Path(epw_path).exists():
        return None

    try:
        df, lon, lat, tz_offset, _ = read_epw_for_solar_simulation(epw_path)
        month, day, hour, minute = _parse_calc_time(calc_time)

        rows = df[(df.index.month == month) & (df.index.day == day) & (df.index.hour == hour)]
        if minute:
            rows = rows[rows.index.minute == minute]
        if rows.empty:
            return None

        row = rows.iloc[0]
        tz_minutes = int(tz_offset * 60)
        local_tz = pytz.FixedOffset(tz_minutes)
        timestamp_local = rows.index[0].tz_localize(local_tz)
        solar_positions = get_solar_positions_astral(pd.DatetimeIndex([timestamp_local]), lon, lat)
        if solar_positions.empty:
            return None
        solar_pos = solar_positions.iloc[0]

        return {
            'dni': float(row['DNI']),
            'dhi': float(row['DHI']),
            'azimuth': float(solar_pos['azimuth']),
            'elevation': float(solar_pos['elevation']),
        }
    except Exception as exc:  # pragma: no cover - resilience for malformed EPW
        print(f"Warning: could not extract EPW solar inputs ({exc}). Using fallback parameters.")
        return None


def compute_secondary_solar_metrics(
    city: str,
    bundle: Dict[str, np.ndarray],
    solar_params: Dict[str, object],
    paths: Optional[VoxCityPaths] = None,
) -> None:
    """Compute additional solar metrics beyond the primary peak irradiance.

    Calculates and adds to bundle:
    - solar_daily_summer: Daily cumulative energy (Wh/m²)
    - solar_winter_peak: Winter solstice peak (W/m²)
    - solar_direct: Direct irradiance component (W/m²)
    - solar_diffuse: Diffuse irradiance component (W/m²)
    - solar_annual_kwh: Estimated annual energy (kWh/m²/year)
    - Derived ratios: seasonal, direct, availability

    This helper keeps the main notebook focused on primary metrics while still
    computing comprehensive solar data for advanced analysis.

    Args:
        city: City name
        bundle: Voxel bundle dict (modified in place)
        solar_params: Solar simulation parameters
        paths: Optional VoxCityPaths for EPW handling
    """

    voxcity_grid = bundle['voxcity']
    meshsize = bundle.get('meshsize', 5)

    # Get EPW for this city
    epw_path = get_city_epw(city, paths)

    # Common arguments for EPW-based calculations
    solar_base_args = {"voxel_data": voxcity_grid, "meshsize": meshsize}
    if epw_path is None:
        if paths is None:
            raise ValueError("paths is required when EPW must be downloaded")
        context = load_city_context(city, paths)
        epw_args = {
            "download_nearest_epw": True,
            "rectangle_vertices": context["rectangle_vertices"],
            "output_dir": str(paths.epw_dir),
        }
    else:
        epw_args = {"epw_file_path": str(epw_path)}

    # Daily cumulative for summer solstice
    print(f"  Computing daily cumulative solar for {city}...")
    try:
        bundle["solar_daily_summer"] = get_global_solar_irradiance_using_epw(
            **solar_base_args, **epw_args,
            calc_type="daily",
            calc_time="06-21",
            obj_export=False
        )
        # Close plots (VoxCity hardcodes show_plot=True internally)
        import matplotlib.pyplot as plt
        plt.close('all')
    except Exception as e:
        print(f"    Warning: Daily calculation failed, using estimate: {e}")
        bundle["solar_daily_summer"] = bundle["solar_peak"] * 8.0

    # Winter peak
    print(f"  Computing winter peak solar for {city}...")
    try:
        bundle["solar_winter_peak"] = get_global_solar_irradiance_using_epw(
            **solar_base_args, **epw_args,
            calc_type="instantaneous",
            calc_time=solar_params['winter_calc_time'],
            obj_export=False
        )
        # Close plots (VoxCity hardcodes show_plot=True internally)
        import matplotlib.pyplot as plt
        plt.close('all')
    except Exception as e:
        print(f"    Warning: Winter calculation failed, using estimate: {e}")
        bundle["solar_winter_peak"] = bundle["solar_peak"] * 0.3

    # Direct and diffuse components (EPW-aware fallback)
    print(f"  Computing direct/diffuse solar components for {city}...")
    epw_path = get_city_epw(city, paths)
    epw_inputs = _extract_epw_instant_inputs(epw_path, solar_params.get('peak_calc_time', '07-15 14:00:00'), solar_params)

    if epw_inputs is None:
        print("    EPW inputs unavailable; falling back to configured solar parameters.")
        dni = float(solar_params.get('direct_irradiance', 800))
        dhi = float(solar_params.get('diffuse_irradiance', 200))
        azimuth = float(solar_params.get('sun_azimuth', 180))
        elevation = float(solar_params.get('sun_elevation', 60))
    else:
        dni = max(float(epw_inputs['dni']), 0.0)
        dhi = max(float(epw_inputs['dhi']), 0.0)
        azimuth = float(epw_inputs['azimuth'])
        elevation = float(epw_inputs['elevation'])

    try:
        bundle["solar_direct"] = get_direct_solar_irradiance_map(
            voxel_data=voxcity_grid,
            meshsize=meshsize,
            azimuth_degrees_ori=azimuth,
            elevation_degrees=elevation,
            direct_normal_irradiance=dni,
            view_point_height=solar_params.get('view_point_height', 1.5),
            tree_k=solar_params.get('tree_k', 0.6),
            tree_lad=solar_params.get('tree_lad', 1.0),
            show_plot=False,
            obj_export=False,
        )

        bundle["solar_diffuse"] = get_diffuse_solar_irradiance_map(
            voxel_data=voxcity_grid,
            meshsize=meshsize,
            diffuse_irradiance=dhi,
            view_point_height=solar_params.get('view_point_height', 1.5),
            tree_k=solar_params.get('tree_k', 0.6),
            tree_lad=solar_params.get('tree_lad', 1.0),
            show_plot=False,
            obj_export=False,
        )

        diffuse_scale = dhi if dhi else float(solar_params.get('diffuse_irradiance', 200) or 1.0)
        bundle["solar_diffuse_ratio"] = np.divide(
            bundle["solar_diffuse"],
            diffuse_scale,
            out=np.zeros_like(bundle["solar_diffuse"], dtype=float),
            where=np.isfinite(bundle["solar_diffuse"]),
        )

        # Close any plots created during direct/diffuse calculations
        import matplotlib.pyplot as plt
        plt.close('all')
    except Exception as e:
        print(f"    Warning: Component calculation failed, using fallback estimates: {e}")
        bundle["solar_direct"] = bundle["solar_peak"] * 0.8
        bundle["solar_diffuse"] = bundle["solar_peak"] * 0.2
        fallback_scale = float(solar_params.get('diffuse_irradiance', 200) or 1.0)
        bundle["solar_diffuse_ratio"] = np.divide(
            bundle["solar_diffuse"],
            fallback_scale,
            out=np.zeros_like(bundle["solar_diffuse"], dtype=float),
            where=np.isfinite(bundle["solar_diffuse"]),
        )

    # Annual estimate
    bundle["solar_annual_kwh"] = bundle["solar_daily_summer"] * 365 / 1000 * 0.3

    # Derived ratios
    bundle["solar_seasonal_ratio"] = np.divide(
        bundle["solar_winter_peak"],
        bundle["solar_peak"],
        out=np.zeros_like(bundle["solar_peak"]),
        where=bundle["solar_peak"] != 0
    )

    bundle["solar_direct_ratio"] = np.divide(
        bundle["solar_direct"],
        bundle["solar_peak"],
        out=np.zeros_like(bundle["solar_peak"]),
        where=bundle["solar_peak"] != 0
    )

    bundle["solar_availability"] = np.divide(
        bundle["solar_peak"],
        np.nanmax(bundle["solar_peak"]),
        out=np.zeros_like(bundle["solar_peak"]),
        where=bundle["solar_peak"] != 0
    )


def compute_instantaneous_solar_snapshot(
    calc_time: str,
    *,
    cities: Optional[Sequence[str]] = None,
    paths: Optional[VoxCityPaths] = None,
    tree_k: float = 0.6,
    tree_lad: float = 1.0,
    view_point_height: float = 1.5,
    output_summary: Optional[Path] = None,
    show_plots: bool = False,
) -> pd.DataFrame:
    """Generate instantaneous global and diffuse solar maps for each city.

    The helper pulls the requested timestamp from the local EPW archives,
    computes global (direct + diffuse), direct-only, and diffuse-only grids,
    updates the cached voxel bundle, and returns per-city summary statistics.

    Args:
        calc_time: Time for calculation in format "MM-DD HH:MM:SS"
        cities: List of cities to process
        paths: VoxCity paths configuration
        tree_k: Tree extinction coefficient
        tree_lad: Leaf area density
        view_point_height: Observation height in meters
        output_summary: Path to save summary CSV
        show_plots: If False, suppress individual city plots (default: False)
    """

    if paths is None:
        paths = ensure_workflow_paths(Path.cwd())

    selected_cities = list(cities) if cities is not None else list(STUDY_CITIES)
    summary_rows: List[Dict[str, float]] = []

    for city in selected_cities:
        bundle = try_load_cached_bundle(city, paths)
        if bundle is None:
            raise RuntimeError(f"No cached VoxCity bundle available for {city}.")

        meshsize = float(bundle.get("meshsize", VOXCITY_DEFAULT_CONFIG.get("meshsize", 5)))

        epw_path = get_city_epw(city, paths)
        if epw_path is None:
            context = load_city_context(city, paths)
            epw_args = {
                "download_nearest_epw": True,
                "rectangle_vertices": context["rectangle_vertices"],
                "output_dir": str(paths.epw_dir),
            }
        else:
            epw_args = {"epw_file_path": str(epw_path)}

        # Global irradiance (direct + diffuse)
        solar_global = get_global_solar_irradiance_using_epw(
            voxel_data=bundle["voxcity"],
            meshsize=meshsize,
            calc_type="instantaneous",
            calc_time=calc_time,
            obj_export=show_plots,  # Suppress 3D exports unless requested
            **epw_args,
        )

        # Close plots if suppression is requested (VoxCity hardcodes show_plot=True internally)
        if not show_plots:
            import matplotlib.pyplot as plt
            plt.close('all')

        # Retrieve instantaneous DNI/DHI/angles for diffuse & direct decomposition
        epw_inputs = _extract_epw_instant_inputs(
            epw_path if epw_path is not None else epw_args.get("epw_file_path"),
            calc_time,
            {
                "direct_irradiance": 800,
                "diffuse_irradiance": 200,
                "sun_azimuth": 180,
                "sun_elevation": 60,
            },
        )

        if epw_inputs is None:
            # Fall back to simple clear-sky constants
            dni = 800.0
            dhi = 200.0
            azimuth = 180.0
            elevation = 60.0
        else:
            dni = max(float(epw_inputs["dni"]), 0.0)
            dhi = max(float(epw_inputs["dhi"]), 0.0)
            azimuth = float(epw_inputs["azimuth"])
            elevation = float(epw_inputs["elevation"])

        solar_diffuse = get_diffuse_solar_irradiance_map(
            voxel_data=bundle["voxcity"],
            meshsize=meshsize,
            diffuse_irradiance=dhi,
            view_point_height=view_point_height,
            tree_k=tree_k,
            tree_lad=tree_lad,
            show_plot=show_plots,  # Suppress individual plots unless requested
            obj_export=show_plots,  # Suppress 3D exports unless requested
        )

        solar_direct = get_direct_solar_irradiance_map(
            voxel_data=bundle["voxcity"],
            meshsize=meshsize,
            azimuth_degrees_ori=azimuth,
            elevation_degrees=elevation,
            direct_normal_irradiance=dni,
            tree_k=tree_k,
            tree_lad=tree_lad,
            view_point_height=view_point_height,
            show_plot=show_plots,  # Suppress individual plots unless requested
            obj_export=show_plots,  # Suppress 3D exports unless requested
        )

        # Close any remaining plots from direct/diffuse calculations
        if not show_plots:
            import matplotlib.pyplot as plt
            plt.close('all')

        # Persist grids into the bundle cache
        bundle["solar_peak"] = solar_global
        bundle["solar_diffuse"] = solar_diffuse
        bundle["solar_direct"] = solar_direct
        bundle["solar_diffuse_ratio"] = np.divide(
            solar_diffuse,
            dhi if dhi else 1.0,
            out=np.zeros_like(solar_diffuse, dtype=float),
            where=np.isfinite(solar_diffuse),
        )
        bundle["solar_direct_ratio"] = np.divide(
            solar_direct,
            np.maximum(solar_global, 1e-9),
            out=np.zeros_like(solar_direct, dtype=float),
            where=np.isfinite(solar_direct),
        )
        bundle["solar_snapshot_meta"] = {
            "calc_time": calc_time,
            "dni": float(dni),
            "dhi": float(dhi),
            "azimuth": float(azimuth),
            "elevation": float(elevation),
        }
        save_to_cache(city, bundle, paths)

        global_flat = np.asarray(solar_global, dtype=float)
        finite = global_flat[np.isfinite(global_flat)]
        diffuse_flat = np.asarray(solar_diffuse, dtype=float)
        diffuse_finite = diffuse_flat[np.isfinite(diffuse_flat)]

        summary_rows.append({
            "city": city,
            "calc_time": calc_time,
            "dni_wm2": dni,
            "dhi_wm2": dhi,
            "global_min_wm2": float(np.min(finite)) if finite.size else float("nan"),
            "global_p10_wm2": float(np.percentile(finite, 10)) if finite.size else float("nan"),
            "global_median_wm2": float(np.median(finite)) if finite.size else float("nan"),
            "global_p90_wm2": float(np.percentile(finite, 90)) if finite.size else float("nan"),
            "global_max_wm2": float(np.max(finite)) if finite.size else float("nan"),
            "diffuse_median_wm2": float(np.median(diffuse_finite)) if diffuse_finite.size else float("nan"),
        })

    summary_df = pd.DataFrame(summary_rows)
    if output_summary is not None:
        output_summary.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_summary, index=False)

    return summary_df
BASEMAP_WARNING_EMITTED = False


def _add_dual_basemap(ax, crs, *, alpha: float = 0.5) -> None:
    """Add dark-matter + imagery basemap combination matching modelling notebook style."""

    global BASEMAP_WARNING_EMITTED
    try:
        import contextily as ctx

        ctx.add_basemap(ax, crs=crs, source=ctx.providers.CartoDB.DarkMatter, attribution_size=6)
        ctx.add_basemap(ax, crs=crs, source=ctx.providers.Esri.WorldImagery, alpha=alpha, attribution_size=6)
    except Exception as exc:  # pragma: no cover - network/tiles may be unavailable
        if not BASEMAP_WARNING_EMITTED:
            BASEMAP_WARNING_EMITTED = True
            print(f"Basemap unavailable ({exc}); continuing without background imagery.")
