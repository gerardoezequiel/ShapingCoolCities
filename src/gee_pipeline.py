from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import ee  # type: ignore
import geopandas as gpd  # type: ignore
import geemap  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, uniform_filter  # type: ignore

import contextily as ctx

from .bbox_utils import STUDY_CITIES
from .plotting_utils import GRID_PATH as DEFAULT_GRID_PATH



@dataclass
class PipelineConfig:
    base_config: Dict[str, object]
    processing_config: Dict[str, object]
    grid_paths: Dict[int, Path]
    grid_resolutions: Sequence[int]
    reference_resolution: int
    output_dir: Path
    study_cities: Sequence[str]
    feature_type_mapping: Dict[str, List[str]]
    missing_value_strategy: Dict[str, object]


_CONFIG: PipelineConfig | None = None


def configure_pipeline(
    *,
    base_config: Dict[str, object],
    processing_config: Dict[str, object],
    grid_paths: Dict[int, Path],
    grid_resolutions: Sequence[int],
    reference_resolution: int,
    output_dir: Path,
    study_cities: Sequence[str],
    feature_type_mapping: Dict[str, List[str]],
    missing_value_strategy: Dict[str, object],
) -> None:
    """Store shared configuration so helper functions can run outside the notebook."""
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_grid_paths = {k: Path(v) for k, v in grid_paths.items()}
    global _CONFIG
    _CONFIG = PipelineConfig(
        base_config=dict(base_config),
        processing_config=dict(processing_config),
        grid_paths=resolved_grid_paths,
        grid_resolutions=list(grid_resolutions),
        reference_resolution=reference_resolution,
        output_dir=output_dir,
        study_cities=list(study_cities),
        feature_type_mapping={k: list(v) for k, v in feature_type_mapping.items()},
        missing_value_strategy=dict(missing_value_strategy),
    )


def _config() -> PipelineConfig:
    if _CONFIG is None:
        raise RuntimeError("gee_pipeline.configure_pipeline must be called before use.")
    return _CONFIG


def load_city_grid(city_name: str, grid_size: int, grid_path: Path) -> ee.FeatureCollection | None:
    """Load a standardised grid for a city using local parquet data."""
    try:
        if grid_path is None or not grid_path.exists():
            print(f"Grid path not found for {city_name}: {grid_path}")
            return None

        grid_gdf = gpd.read_parquet(grid_path)
        city_grid = grid_gdf[grid_gdf["city"] == city_name].copy()

        if city_grid.empty:
            print(f"No grid found for {city_name}")
            return None

        if city_grid.crs and city_grid.crs.to_string() != "EPSG:4326":
            city_grid = city_grid.to_crs("EPSG:4326")

        grid_fc = geemap.gdf_to_ee(city_grid)
        grid_count = grid_fc.size().getInfo()
        print(f"Loaded {grid_count:,} grid cells for {city_name} ({grid_size} m) from {grid_path}")
        return grid_fc

    except Exception as exc:  # pragma: no cover - pass through for notebook visibility
        print(f"Error loading grid for {city_name}: {exc}")
        return None


def check_existing_files(
    city_name: Optional[str] = None,
    grid_size: Optional[int] = None,
) -> Dict[str, Dict[str, float]] | Tuple[bool, float]:
    """Check if GEE feature files already exist and satisfy minimum size."""
    config = _config()
    min_size_mb = float(config.processing_config.get("min_file_size_mb", 0.0))
    reference_resolution = config.reference_resolution
    grid_size = grid_size or reference_resolution
    output_dir = config.output_dir

    def _valid_file(path: Path) -> Tuple[bool, float]:
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            if size_mb >= min_size_mb:
                return True, size_mb
        return False, 0.0

    if city_name:
        candidates = [output_dir / f"{city_name}_GEE_features_{grid_size}m.parquet"]
        if grid_size == reference_resolution:
            candidates.append(output_dir / f"{city_name}_GEE_features.parquet")
        for candidate in candidates:
            exists, size = _valid_file(candidate)
            if exists:
                return True, size
        return False, 0.0

    existing: Dict[str, Dict[str, float]] = {}
    for city in config.study_cities:
        exists, size = check_existing_files(city, grid_size)
        existing[city] = {"exists": bool(exists), "size_mb": float(size)}
    return existing


def get_cities_to_process(grid_size: int) -> Tuple[List[str], List[str]]:
    """Determine which cities need processing for a specific grid size."""
    config = _config()
    cities_to_process: List[str] = []
    cities_skipped: List[str] = []

    check_existing = bool(config.processing_config.get("check_existing", False))
    force_download = bool(config.processing_config.get("force_download", False))
    verbose = bool(config.processing_config.get("verbose", False))

    if check_existing and not force_download:
        for city in config.study_cities:
            exists, size = check_existing_files(city, grid_size)
            if exists:
                if verbose:
                    print(f"✓ {city}: Found existing {grid_size} m file ({size:.1f} MB) - skipping")
                cities_skipped.append(city)
            else:
                cities_to_process.append(city)
    else:
        cities_to_process = list(config.study_cities)
        if force_download and verbose:
            print(f"Force download enabled - reprocessing all cities for {grid_size} m grid")

    return cities_to_process, cities_skipped


def export_city_features_batch(
    city_name: str,
    features_image: ee.Image,
    grid_fc: ee.FeatureCollection,
    config: Dict[str, object],
    grid_size: int,
) -> pd.DataFrame | None:
    """Export city features in batches with proper missing value handling."""
    pipeline_config = _config()
    processing_config = pipeline_config.processing_config
    feature_type_mapping = pipeline_config.feature_type_mapping
    missing_value_strategy = pipeline_config.missing_value_strategy
    output_dir = pipeline_config.output_dir
    reference_resolution = pipeline_config.reference_resolution

    print(f"Starting batch export for {city_name} ({grid_size} m grid)")
    total_features = int(grid_fc.size().getInfo())
    batch_size = int(config["batch_size"])
    num_batches = (total_features + batch_size - 1) // batch_size
    print(f"Processing {total_features:,} cells in {num_batches} batches")

    grid_list = ee.List(grid_fc.toList(total_features))

    feature_bands = config.get("feature_bands")
    if feature_bands:
        features_image = features_image.select(feature_bands)
    else:
        feature_bands = features_image.bandNames().getInfo()

    target_properties = ["global_grid_id", "city"] + [band for band in feature_bands if band]
    all_data: List[Dict[str, object]] = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_features)

        try:
            batch_list = grid_list.slice(start_idx, end_idx)
            batch_grid = ee.FeatureCollection(batch_list)

            reduced = features_image.reduceRegions(
                collection=batch_grid,
                reducer=ee.Reducer.mean(),
                scale=config["scale"],
                tileScale=4,
            ).map(lambda f: f.select(target_properties))

            batch_features = reduced.toList(end_idx - start_idx).getInfo()
            for feature in batch_features:
                all_data.append(feature.get("properties", {}))
        except Exception as exc:
            print(f"Error processing batch {batch_idx + 1}: {exc}")
            continue

    if not all_data:
        print(f"No data exported for {city_name}")
        return None

    df = pd.DataFrame(all_data)

    if "global_grid_id" not in df.columns:
        raise KeyError(
            "global_grid_id missing from exported data. Ensure grid assets retain this property."
        )

    missing_ids = df["global_grid_id"].isna() | (df["global_grid_id"] == "")
    if missing_ids.any():
        raise ValueError(
            f"{missing_ids.sum()} rows missing global_grid_id for {city_name}. Verify grid asset properties."
        )

    df["global_grid_id"] = df["global_grid_id"].astype(str)
    df["city"] = df["city"].astype(str)

    fill_values: Dict[str, object] = {}
    for col in df.columns:
        if col in ["global_grid_id", "city"]:
            continue
        feature_type = next(
            (ftype for ftype, members in feature_type_mapping.items() if col in members),
            None,
        )
        fill_values[col] = (
            missing_value_strategy.get(feature_type, np.nan) if feature_type else np.nan
        )
        if feature_type is None and processing_config.get("verbose", False):
            print(f"Warning: Feature '{col}' not in mapping, using NaN default")

    non_nan_fill = {k: v for k, v in fill_values.items() if pd.notna(v)}
    if non_nan_fill:
        df = df.fillna(non_nan_fill)

    output_file = output_dir / f"{city_name}_GEE_features_{grid_size}m.parquet"
    df.to_parquet(output_file, index=False)

    print(f"Exported {len(df):,} features to {output_file}")

    total_cells = len(df)
    valid_lst = df["LST_mean"].notna().sum() if "LST_mean" in df.columns else 0
    valid_ndvi = df["NDVI"].notna().sum() if "NDVI" in df.columns else 0

    if total_cells:
        print(
            f"Data quality: LST coverage {valid_lst:,}/{total_cells:,} ({100 * valid_lst / total_cells:.1f}%)"
        )
        print(
            f"Data quality: NDVI coverage {valid_ndvi:,}/{total_cells:,} ({100 * valid_ndvi / total_cells:.1f}%)"
        )

    return df


def process_city(
    city_name: str,
    config: Dict[str, object],
    grid_size: int,
    grid_path: Path,
    prepare_landsat_collection: Callable[[str, str, ee.Geometry, int], ee.ImageCollection],
    calculate_all_features: Callable[[ee.ImageCollection, ee.Geometry, Optional[str], Optional[str]], ee.Image],
    get_bbox: Callable[[str, int], Dict[str, float]],
) -> pd.DataFrame | None:
    """Process a single city: load data, calculate features, export."""
    print(f"Processing {city_name.upper()} ({grid_size} m)")
    print("=" * 50)

    try:
        grid_fc = load_city_grid(city_name, grid_size, grid_path)
        if grid_fc is None:
            return None

        bbox_info = get_bbox(city_name, grid_size)
        bounds = ee.Geometry.Rectangle(
            [
                bbox_info["west"],
                bbox_info["south"],
                bbox_info["east"],
                bbox_info["north"],
            ]
        )

        collection = prepare_landsat_collection(
            str(config["start_date"]),
            str(config["end_date"]),
            bounds,
            config["cloud_threshold"],
        )

        image_count = collection.size().getInfo()
        print(f"Found {image_count} cloud-free images")

        if image_count == 0:
            print(f"No suitable images found for {city_name}")
            return None

        print("Calculating GEE features…")
        features_image = calculate_all_features(
            collection,
            bounds,
            str(config["start_date"]),
            str(config["end_date"]),
        )

        result_df = export_city_features_batch(
            city_name,
            features_image,
            grid_fc,
            config,
            grid_size,
        )

        if result_df is not None:
            print(f"{city_name} processing completed successfully")
            print(f"Dataset shape: {result_df.shape}")
            return result_df

        print(f"{city_name} processing failed")
        return None

    except Exception as exc:
        print(f"Error processing {city_name}: {exc}")
        return None


def main_processing_with_cache(
    prepare_landsat_collection: Callable[[str, str, ee.Geometry, int], ee.ImageCollection],
    calculate_all_features: Callable[[ee.ImageCollection, ee.Geometry, Optional[str], Optional[str]], ee.Image],
    get_bbox: Callable[[str, int], Dict[str, float]],
) -> Dict[int, Dict[str, Optional[pd.DataFrame]]]:
    """Run the full processing loop across grid resolutions with caching."""
    config = _config()
    print("GEE FEATURE EXTRACTION")
    print("=" * 80)

    results_by_resolution: Dict[int, Dict[str, Optional[pd.DataFrame]]] = {}
    combined_stats: Dict[int, Dict[str, int]] = {}

    overall_start = time.time()

    for grid_size in config.grid_resolutions:
        grid_path = config.grid_paths.get(grid_size)
        if grid_path is None:
            continue

        print("=" * 80)
        print(f"GRID SIZE: {grid_size} m")
        print("=" * 80)
        if not grid_path.exists():
            print(f"Grid file missing: {grid_path}")
            results_by_resolution[grid_size] = {}
            continue

        cities_to_process, cities_skipped = get_cities_to_process(grid_size)

        print(f"Cities to process ({grid_size} m): {cities_to_process}")
        if cities_skipped:
            print(f"Cities skipped ({grid_size} m, cached): {cities_skipped}")

        config_per_run = dict(config.base_config)
        config_per_run["scale"] = grid_size

        grid_results: Dict[str, Optional[pd.DataFrame]] = {}
        start_time = time.time()

        for idx, city_name in enumerate(cities_to_process, 1):
            city_start = time.time()
            print(f"[{idx}/{len(cities_to_process)}] PROCESSING {city_name.upper()}")

            result = process_city(
                city_name,
                config_per_run,
                grid_size,
                grid_path,
                prepare_landsat_collection,
                calculate_all_features,
                get_bbox,
            )
            grid_results[city_name] = result

            city_duration = time.time() - city_start
            print(f"Processing time: {city_duration/60:.1f} minutes")

        for city_name in cities_skipped:
            candidate_paths = [config.output_dir / f"{city_name}_GEE_features_{grid_size}m.parquet"]
            if grid_size == config.reference_resolution:
                candidate_paths.append(config.output_dir / f"{city_name}_GEE_features.parquet")

            loaded = None
            for candidate in candidate_paths:
                if candidate.exists():
                    try:
                        loaded = pd.read_parquet(candidate)
                        if config.processing_config.get("verbose", False):
                            print(f"Loaded existing data for {city_name} ({grid_size} m): {loaded.shape}")
                        break
                    except Exception as exc:
                        print(f"Error loading {candidate}: {exc}")
            grid_results[city_name] = loaded

        duration = time.time() - start_time

        successful = [city for city, result in grid_results.items() if result is not None]
        failed = [city for city, result in grid_results.items() if result is None]

        print("-" * 80)
        print(f"SUMMARY ({grid_size} m)")
        print(f"Total processing time: {duration/60:.1f} minutes")
        print(f"Cities processed: {len(cities_to_process)}")
        print(f"Cities loaded from cache: {len(cities_skipped)}")
        print(f"Successful datasets ({len(successful)}): {successful}")
        if failed:
            print(f"Failed datasets ({len(failed)}): {failed}")

        results_by_resolution[grid_size] = grid_results
        combined_stats[grid_size] = {
            "processed": len(cities_to_process),
            "cached": len(cities_skipped),
            "successful": len(successful),
        }

    total_duration = time.time() - overall_start

    print("=" * 80)
    print("MULTI-RESOLUTION SUMMARY")
    print("=" * 80)
    print(f"Total processing time across resolutions: {total_duration/60:.1f} minutes")
    for grid_size in config.grid_resolutions:
        stats = combined_stats.get(grid_size, {})
        print(
            f"  {grid_size} m → processed {stats.get('processed', 0)}, "
            f"cached {stats.get('cached', 0)}, successful {stats.get('successful', 0)}"
        )

    print("GEE feature extraction complete - ready for export and visualisation")
    return results_by_resolution


def add_neighbourhood_context_features(dataset: pd.DataFrame, grid_size: int) -> pd.DataFrame:
    """Append canopy distance/proximity plus neighbourhood fractions for a grid size."""
    if dataset is None or dataset.empty:
        return dataset

    config = _config()
    grid_path = config.grid_paths.get(grid_size)
    if grid_path is None or not grid_path.exists():
        print(f"Grid lookup missing for {grid_size} m; skipping neighbourhood metrics.")
        return dataset

    dataset = dataset.copy()
    grid_gdf = gpd.read_parquet(grid_path)
    city_groups = grid_gdf.groupby("city")

    legacy_columns = [
        "tree_canopy_density_3",
        "tree_canopy_density_5",
        "impervious_density_3",
        "impervious_density_5",
    ]
    dataset = dataset.drop(columns=[c for c in legacy_columns if c in dataset.columns], errors="ignore")

    required_columns = [
        "tree_canopy_distance",
        "tree_canopy_proximity",
        "dense_vegetation_distance",
        "dense_vegetation_proximity",
        "tree_canopy_fraction_90m",
        "tree_canopy_fraction_150m",
        "tree_canopy_fraction_300m",
        "impervious_fraction_90m",
        "impervious_fraction_150m",
        "impervious_fraction_300m",
        "vegetation_fraction_90m",
        "vegetation_fraction_150m",
        "vegetation_fraction_300m",
    ]
    for column in required_columns:
        if column not in dataset.columns:
            dataset[column] = np.nan

    kernels = _resolve_kernel_sizes(grid_size)
    cell_size = float(grid_size)

    for city, city_df in dataset.groupby("city"):
        if city not in city_groups.groups:
            print(f"Warning: no grid lookup found for {city}; skipping neighbourhood metrics.")
            continue
        grid_city = city_groups.get_group(city).copy()
        if grid_city.empty:
            print(f"Warning: grid lookup empty for {city}; skipping neighbourhood metrics.")
            continue

        lookup = _prepare_city_lookup(grid_city)
        rows = city_df["global_grid_id"].map(lookup["row"])
        cols = city_df["global_grid_id"].map(lookup["col"])
        if rows.isna().any() or cols.isna().any():
            missing = city_df.loc[rows.isna() | cols.isna(), "global_grid_id"].tolist()[:5]
            raise ValueError(f"Missing grid indices for city {city}: {missing}")

        rows_array = rows.to_numpy(dtype=int)
        cols_array = cols.to_numpy(dtype=int)
        n_rows = rows_array.max() + 1
        n_cols = cols_array.max() + 1
        indexer = (rows_array, cols_array)

        canopy_mask = city_df["tree_canopy_cover"].fillna(0.0).to_numpy() >= CANOPY_DISTANCE_THRESHOLD
        canopy_grid = np.zeros((n_rows, n_cols), dtype=bool)
        canopy_grid[indexer] = canopy_mask
        canopy_distance = _distance_series(canopy_grid, rows_array, cols_array, cell_size)
        dataset.loc[city_df.index, "tree_canopy_distance"] = canopy_distance
        dataset.loc[city_df.index, "tree_canopy_proximity"] = 1.0 / (1.0 + canopy_distance)

        if "vegetation_fraction" in city_df.columns:
            dense_mask_values = city_df["vegetation_fraction"].fillna(0.0).to_numpy() >= DENSE_VEGETATION_THRESHOLD
            dense_grid = np.zeros((n_rows, n_cols), dtype=bool)
            dense_grid[indexer] = dense_mask_values
            dense_distance = _distance_series(dense_grid, rows_array, cols_array, cell_size)
            dataset.loc[city_df.index, "dense_vegetation_distance"] = dense_distance
            dataset.loc[city_df.index, "dense_vegetation_proximity"] = 1.0 / (1.0 + dense_distance)
        else:
            dataset.loc[city_df.index, ["dense_vegetation_distance", "dense_vegetation_proximity"]] = np.nan
            print(f"Warning: vegetation_fraction missing for {city}; dense vegetation metrics left NaN.")

        canopy_array = np.full((n_rows, n_cols), np.nan, dtype=float)
        canopy_array[indexer] = city_df["tree_canopy_cover"].to_numpy(dtype=float)

        if "impervious_fraction" in city_df.columns:
            impervious_array = np.full((n_rows, n_cols), np.nan, dtype=float)
            impervious_array[indexer] = city_df["impervious_fraction"].to_numpy(dtype=float)
        else:
            impervious_array = None
            dataset.loc[
                city_df.index,
                ["impervious_fraction_90m", "impervious_fraction_150m", "impervious_fraction_300m"],
            ] = np.nan
            print(f"Warning: impervious_fraction missing for {city}; impervious context metrics left NaN.")

        if "vegetation_fraction" in city_df.columns:
            vegetation_array = np.full((n_rows, n_cols), np.nan, dtype=float)
            vegetation_array[indexer] = city_df["vegetation_fraction"].to_numpy(dtype=float)
        else:
            vegetation_array = None
            dataset.loc[
                city_df.index,
                ["vegetation_fraction_90m", "vegetation_fraction_150m", "vegetation_fraction_300m"],
            ] = np.nan
            print(f"Warning: vegetation_fraction missing for {city}; neighbourhood vegetation metrics left NaN.")

        for size, label in kernels.items():
            canopy_density = _rolling_density(canopy_array, size=size)
            dataset.loc[city_df.index, f"tree_canopy_fraction_{label}"] = canopy_density[indexer]

            if impervious_array is not None:
                impervious_density = _rolling_density(impervious_array, size=size)
                dataset.loc[city_df.index, f"impervious_fraction_{label}"] = impervious_density[indexer]

            if vegetation_array is not None:
                vegetation_density = _rolling_density(vegetation_array, size=size)
                dataset.loc[city_df.index, f"vegetation_fraction_{label}"] = vegetation_density[indexer]

    return dataset


def export_combined_dataset(
    results: Dict[str, Optional[pd.DataFrame]],
    grid_size: int,
    expected_features: Optional[Iterable[str]] = None,
) -> Optional[pd.DataFrame]:
    """Export combined dataset with all cities for ML pipeline for a grid size."""
    if not results:
        print(f"No data available for export at {grid_size} m")
        return None

    config = _config()
    print("EXPORTING COMBINED DATASET")
    print("=" * 50)

    combined_data: List[pd.DataFrame] = []
    export_summary: Dict[str, Dict[str, float]] = {}

    for city_name, result_df in results.items():
        if result_df is None or result_df.empty:
            print(f"✗ {city_name}: no data to export")
            continue

        if "city" not in result_df.columns:
            result_df = result_df.copy()
            result_df["city"] = city_name

        combined_data.append(result_df)

        file_path = config.output_dir / f"{city_name}_GEE_features_{grid_size}m.parquet"
        file_size_mb = file_path.stat().st_size / 1024 / 1024 if file_path.exists() else 0.0

        export_summary[city_name] = {
            "cells": float(len(result_df)),
            "features": float(len(result_df.columns)),
            "file_size_mb": file_size_mb,
        }
        print(f"✓ {city_name}: {len(result_df):,} cells × {len(result_df.columns)} features")

    if not combined_data:
        print("No valid city datasets – combined export skipped")
        return None

    combined_df = pd.concat(combined_data, ignore_index=True)
    combined_df = add_neighbourhood_context_features(combined_df, grid_size)

    stats = combined_df.groupby("city")["LST_mean"].agg(["median", "std"]).reset_index()
    stats["std"] = stats["std"].fillna(0.0).replace(0.0, 1.0)
    combined_df = combined_df.merge(stats, on="city", how="left", suffixes=("", "_city"))
    combined_df.rename(columns={"median": "LST_city_median", "std": "LST_city_std"}, inplace=True)
    combined_df["UHI_intensity_sigma"] = (
        combined_df["LST_mean"] - combined_df["LST_city_median"]
    ) / combined_df["LST_city_std"]
    combined_df["UHI_sigma_threshold"] = combined_df["LST_city_median"] + combined_df["LST_city_std"]
    combined_df["UHI_hotspot_flag"] = (combined_df["LST_mean"] >= combined_df["UHI_sigma_threshold"]).astype(int)

    combined_df.attrs["grid_size"] = grid_size

    combined_export_path = config.grid_paths.get(grid_size)
    combined_path = config.output_dir / f"All_cities_GEE_features_{grid_size}m.parquet"
    combined_df.to_parquet(combined_path, index=False)

    combined_size_mb = combined_path.stat().st_size / 1024 / 1024

    print(f"Combined dataset exported to: {combined_path}")
    print(f"Combined dataset: {len(combined_df):,} cells × {len(combined_df.columns)} features ({combined_size_mb:.1f} MB)")

    expected_features = list(expected_features) if expected_features is not None else []

    present_features = [f for f in expected_features if f in combined_df.columns]
    missing_features = [f for f in expected_features if f not in combined_df.columns]

    if expected_features:
        print(f"Feature validation: {len(present_features)}/{len(expected_features)} expected features present")
        if missing_features:
            print(f"Missing features: {missing_features}")
        else:
            print("All expected features present ✓")

    total_mb = sum(info["file_size_mb"] for info in export_summary.values())
    print("Export Summary:")
    print("-" * 50)
    print(f"Individual files: {len(export_summary)} cities ({total_mb:.1f} MB total)")
    print(f"Combined file: {combined_size_mb:.1f} MB")
    print(f"Total export: {total_mb + combined_size_mb:.1f} MB")

    print("Dataset ready for ML pipeline integration")
    return combined_df


def create_city_visualisation(
    combined_dataset: pd.DataFrame,
    feature_name: str,
    title: str,
    cmap,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    basemap_source=None,
    use_global_scale: bool = True,
    figsize: Tuple[int, int] = (18, 12),
    grid_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """Render a 2x3 grid visualisation showing a feature across all study cities."""
    if combined_dataset is None or feature_name not in combined_dataset.columns:
        print(f"Feature '{feature_name}' not available for visualisation")
        return None

    try:
        config = _config()
        study_cities = list(config.study_cities)
        default_grid_path = config.grid_paths.get(config.reference_resolution)
    except RuntimeError:
        config = None
        study_cities = list(STUDY_CITIES)
        default_grid_path = DEFAULT_GRID_PATH

    effective_grid_path = grid_path or default_grid_path
    if effective_grid_path is None or not Path(effective_grid_path).exists():
        print(f"Grid path missing for visualisation: {effective_grid_path}")
        return None

    grid = gpd.read_parquet(effective_grid_path)

    merge_cols = ["global_grid_id", feature_name]
    join_df = combined_dataset[merge_cols]
    grid_with_features = grid.merge(join_df, on="global_grid_id", how="left")

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    global_vmin, global_vmax = vmin, vmax
    if use_global_scale and vmin is None and vmax is None:
        feature_values = grid_with_features[feature_name].dropna()
        if not feature_values.empty:
            global_vmin, global_vmax = feature_values.quantile([0.02, 0.98])

    for i, city in enumerate(study_cities):
        ax = axes[i]
        city_grid = grid_with_features[grid_with_features["city"] == city].copy()

        if not city_grid.empty:
            city_grid = city_grid.to_crs("EPSG:3857")

            if not use_global_scale:
                city_feature = city_grid[feature_name].dropna()
                if not city_feature.empty:
                    city_vmin, city_vmax = city_feature.min(), city_feature.max()
                else:
                    city_vmin, city_vmax = 0.0, 1.0
            else:
                city_vmin, city_vmax = global_vmin, global_vmax

            norm = plt.Normalize(vmin=city_vmin, vmax=city_vmax)

            city_grid.plot(
                column=feature_name,
                ax=ax,
                cmap=cmap,
                norm=norm,
                legend=False,
                edgecolor="none",
                alpha=0.3,
                missing_kwds={"color": "lightgrey", "alpha": 0.3},
            )

            if basemap_source:
                try:
                    ctx.add_basemap(
                        ax,
                        crs=city_grid.crs,
                        source=basemap_source,
                        attribution_size=6,
                        alpha=0.6,
                    )
                except Exception as exc:
                    if config is not None and config.processing_config.get("verbose", False):
                        print(f"Basemap error for {city}: {exc}")

            if not use_global_scale:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.01)
                cbar.ax.tick_params(labelsize=8)
                city_title = f"{city}\n({feature_name}: {city_vmin:.2f} - {city_vmax:.2f})"
            else:
                city_title = city
        else:
            city_title = f"{city}\n(No data)"

        ax.set_title(city_title, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

    if use_global_scale and global_vmin is not None and global_vmax is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=global_vmin, vmax=global_vmax))
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(feature_name, fontsize=12)
        plt.subplots_adjust(right=0.9)

    return fig


# --- Internal helpers for neighbourhood metrics -------------------------------------------------

CANOPY_DISTANCE_THRESHOLD = 0.05
DENSE_VEGETATION_THRESHOLD = 0.5
NEIGHBOURHOOD_TARGETS = {90: "90m", 150: "150m", 300: "300m"}


def _prepare_city_lookup(grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    metric_grid = grid_gdf.copy()
    if metric_grid.crs is None or metric_grid.crs.to_epsg() != 3857:
        metric_grid = metric_grid.to_crs(3857)
    metric_grid["centroid"] = metric_grid.geometry.centroid
    metric_grid["x"] = metric_grid.centroid.x.round(3)
    metric_grid["y"] = metric_grid.centroid.y.round(3)
    unique_x = np.sort(metric_grid["x"].unique())
    unique_y = np.sort(metric_grid["y"].unique())[::-1]
    x_to_idx = {val: idx for idx, val in enumerate(unique_x)}
    y_to_idx = {val: idx for idx, val in enumerate(unique_y)}
    metric_grid["col"] = metric_grid["x"].map(x_to_idx).astype(int)
    metric_grid["row"] = metric_grid["y"].map(y_to_idx).astype(int)
    return metric_grid[["global_grid_id", "row", "col"]].set_index("global_grid_id")


def _distance_series(mask_array: np.ndarray, rows: np.ndarray, cols: np.ndarray, cell_size: float) -> np.ndarray:
    distance_pixels = distance_transform_edt(~mask_array)
    distance_m = distance_pixels * cell_size
    return distance_m[rows, cols]


def _rolling_density(array: np.ndarray, size: int) -> np.ndarray:
    presence = np.isfinite(array).astype(float)
    smoothed_presence = uniform_filter(presence, size=size, mode="constant", cval=0.0)
    smoothed_values = uniform_filter(np.nan_to_num(array), size=size, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(smoothed_presence > 0, smoothed_values / smoothed_presence, np.nan)


def _resolve_kernel_sizes(grid_size: int) -> Dict[int, str]:
    kernels: Dict[int, str] = {}
    for target_m, label in NEIGHBOURHOOD_TARGETS.items():
        size = max(1, int(round(target_m / grid_size)))
        if size % 2 == 0:
            size += 1
        kernels[size] = label
    return kernels


__all__ = [
    "configure_pipeline",
    "load_city_grid",
    "check_existing_files",
    "get_cities_to_process",
    "export_city_features_batch",
    "process_city",
    "main_processing_with_cache",
    "add_neighbourhood_context_features",
    "export_combined_dataset",
    "create_city_visualisation",
]
