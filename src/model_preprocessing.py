"""Model-ready feature assembly for Urban Heat Island (UHI) modelling.

This module consolidates the processed data products (buildings, streetscapes,
Urbanity, Google Earth Engine, VoxCity) into a single modelling table. It also
handles feature renaming, missing-value indicators, within-city scaling, and the
creation of a small set of physics-inspired derived features. The goal is to
produce a consistent dataframe that can be consumed directly by modelling code
(e.g. XGBoost baselines).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from src.spatial_cv_utils import load_grid_centroids

def _tokenise_feature_block(block: str) -> Tuple[str, ...]:
    return tuple(
        line.strip()
        for line in block.strip().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


LOG1P_FEATURES: Tuple[str, ...] = _tokenise_feature_block(
    """
    bld_frontal_area_index
    bld_wall_to_plan_ratio
    gee_distance_to_water
    gee_tree_canopy_distance
    """
)

WINSOR_FEATURES: Dict[str, float] = {
    "bld_height_relative": 0.01,
    "vox_solar": 0.01,
    "urb_streetview_road_view": 0.01,
}

# Toggle whether per-feature missing-value indicator columns are added to the
# modelling frame. Setting to False keeps the dataset lean (no *_missing columns).
ADD_MISSING_INDICATORS = False  # adjust to True if downstream models rely on indicators

# Optional list of scaled feature names to retain (e.g. gain-ranked subset).
# Set to None if you want to keep every available feature.
# Updated 2025-01-10: Set to None to let correlation filter do all feature selection
SELECTED_FEATURES: Optional[Tuple[str, ...]] = None

NEIGHBOR_FEATURE_SPECS: Tuple[Tuple[str, float, str], ...] = (
    ("gee_impervious_fraction", 150.0, "mean"),
    ("gee_impervious_fraction", 150.0, "std"),
    ("gee_impervious_fraction", 300.0, "mean"),
    ("gee_tree_canopy_cover", 150.0, "mean"),
    ("gee_tree_canopy_cover", 300.0, "mean"),
    ("gee_vegetation_fraction", 150.0, "mean"),
    ("gee_vegetation_fraction", 300.0, "mean"),
    ("derived_ventilation_proxy", 150.0, "mean"),
)


@dataclass(frozen=True)
class DatasetSpec:
    """Configuration describing one processed Parquet dataset."""

    relative_path: Path
    prefix: str
    drop_geometry: bool = True


@dataclass
class ModelDataset:
    """Container for the assembled modelling dataset and metadata."""

    frame: pd.DataFrame
    feature_columns: List[str]
    indicator_columns: List[str]
    target_column: str
    raw_target_column: str
    uhi_target_column: str
    city_column: str
    feature_scaling: pd.DataFrame
    feature_summary: pd.DataFrame
    target_stats: pd.DataFrame
    uhi_target_stats: pd.DataFrame


def _get_dataset_specs(grid_resolution_m: int = 30) -> Dict[str, DatasetSpec]:
    """Generate dataset specifications for the requested grid resolution.

    Parameters
    ----------
    grid_resolution_m : int
        Grid resolution in metres (default 30).

    Returns
    -------
    Dict[str, DatasetSpec]
        Dictionary mapping dataset names to their specifications.
    """
    grid_suffix = f"_{grid_resolution_m}m" if grid_resolution_m != 30 else "_30m"

    return {
        "buildings": DatasetSpec(
            Path(f"data/1-processed/Buildings/All_cities_buildings_grid{grid_suffix}.parquet"),
            prefix="bld",
        ),
        "streetscapes": DatasetSpec(
            Path(f"data/1-processed/GlobalStreetscapes/GlobalStreetscapes_grid{grid_suffix}.parquet"),
            prefix="ss",
        ),
        "urbanity": DatasetSpec(
            Path(f"data/1-processed/Urbanity/Urbanity_grid{grid_suffix}.parquet"),
            prefix="urb",
        ),
        "gee": DatasetSpec(
            Path(f"data/1-processed/GoogleEarthEngine/All_cities_GEE_features{grid_suffix}.parquet"),
            prefix="gee",
            drop_geometry=False,
        ),
        "voxcity": DatasetSpec(
            Path(f"data/1-processed/VoxCity/VoxCity_grid{grid_suffix}.parquet"),
            prefix="vox",
        ),
    }


# Default dataset specs for 30m grid (backward compatibility)
DATASET_SPECS: Dict[str, DatasetSpec] = _get_dataset_specs(30)


FEATURE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "target": (
        "gee_lst_mean",
        "gee_lst_day_mean",
        "gee_lst_night_mean",
        "gee_lst_day_night_delta",
        "gee_uhi_intensity",
        "gee_uhi_day_intensity",
    ),
    "green_context": (
        "gee_tree_canopy_cover",
        "gee_tree_canopy_distance",
        "gee_tree_canopy_proximity",
        "gee_tree_canopy_fraction_90m",
        "gee_tree_canopy_fraction_150m",
        "gee_tree_canopy_fraction_300m",
        "gee_vegetation_fraction",
        "gee_vegetation_fraction_90m",
        "gee_vegetation_fraction_150m",
        "gee_vegetation_fraction_300m",
        "gee_dense_vegetation_distance",
        "gee_dense_vegetation_proximity",
    ),
    "hardscape_context": (
        "gee_impervious_fraction",
        "gee_impervious_fraction_90m",
        "gee_impervious_fraction_150m",
        "gee_impervious_fraction_300m",
    ),
    "water_context": (
        "gee_water_fraction",
        "gee_distance_to_water",
        "gee_distance_to_coast",
    ),
}

REMOVE_HIGHLY_CORRELATED: bool = True
HIGH_CORRELATION_THRESHOLD: float = 0.92


def load_processed_sources(
    project_root: Path,
    dataset_specs: Optional[Dict[str, DatasetSpec]] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, str]]]:
    """Load processed feature tables and provenance by prefix."""

    specs = dataset_specs or DATASET_SPECS
    tables: Dict[str, pd.DataFrame] = {}
    provenance: Dict[str, Dict[str, str]] = {}

    for name, spec in specs.items():
        table, sources = _load_dataset(project_root, spec)
        tables[name] = table
        provenance[name] = sources

    return tables, provenance


# Columns that should never be used as modelling features.
EXCLUDE_FEATURE_NAMES: Tuple[str, ...] = (
    "city",
    "global_grid_id",
    "geometry",
)

# Prefixes that indicate direct leakage from the target (temperature-derived
# metrics). These get excluded from the feature matrix.
LEAKAGE_PREFIXES: Tuple[str, ...] = (
    "gee_lst",
    "gee_uhi",
)


def run_preprocessing_pipeline(
    project_root: Path, *, return_artifacts: bool = False, grid_resolution_m: int = 30
) -> Tuple[ModelDataset, Dict[str, object]]:
    """Execute the full feature-engineering pipeline.

    Parameters
    ----------
    project_root:
        Repository root containing the ``data/`` directory.
    return_artifacts:
        When True, return intermediate tables and metadata for notebook use.
    grid_resolution_m:
        Grid resolution in metres (default 30).

    Returns
    -------
    Tuple[ModelDataset, Dict[str, object]]
        The assembled modelling dataset plus optional diagnostic artefacts.
    """

    artifacts: Dict[str, object] = {}
    tables: List[pd.DataFrame] = []
    table_lookup: Dict[str, pd.DataFrame] = {}
    column_sources: Dict[str, str] = {}

    dataset_specs = _get_dataset_specs(grid_resolution_m)
    for name, spec in dataset_specs.items():
        table, sources = _load_dataset(project_root, spec)
        tables.append(table)
        table_lookup[name] = table
        column_sources.update(sources)

    merged = _merge_tables(tables)
    merged = _coerce_types(merged)

    if "global_grid_id" not in merged.columns or "city" not in merged.columns:
        raise KeyError("Merged dataframe lacks 'global_grid_id' or 'city'.")

    merged = merged.sort_values(["city", "global_grid_id"]).reset_index(drop=True)
    merged["global_grid_id"] = merged["global_grid_id"].astype(str)

    # Attach centroid coordinates so neighbour-context features can be generated.
    coords = load_grid_centroids()[["global_grid_id", "x", "y"]].drop_duplicates("global_grid_id")
    merged = merged.merge(coords, on="global_grid_id", how="left")

    artifacts["tables"] = table_lookup
    artifacts["merged_initial"] = merged.copy()

    target_raw_col = "target_lst_mean_celsius"
    target_z_col = "target_lst_mean_zscore"

    target_source_col = "gee_lst_mean"
    if target_source_col not in merged.columns:
        raise KeyError("Expected column 'gee_lst_mean' not present after merging.")

    merged[target_raw_col] = merged[target_source_col]
    merged[target_z_col], target_stats = _compute_citywise_zscores(
        merged, target_source_col
    )

    city_medians = merged.groupby("city")[target_source_col].transform("median")
    merged["target_uhi_raw"] = merged[target_source_col] - city_medians
    uhi_target_stats = (
        merged.groupby("city")["target_uhi_raw"].agg(["mean", "std", "median"])
        .rename(columns={"mean": "city_mean", "std": "city_std", "median": "city_median"})
        .reset_index()
    )
    uhi_target_stats["city_std"] = uhi_target_stats["city_std"].replace(0, 1.0).fillna(1.0)

    artifacts["target_stats"] = target_stats
    artifacts["uhi_target_stats"] = uhi_target_stats

    merged, column_sources = add_derived_features(merged, column_sources)

    merged, column_sources = _add_neighbor_context_features(
        merged,
        column_sources,
        specs=NEIGHBOR_FEATURE_SPECS,
        city_col="city",
    )
    artifacts["merged_with_derived"] = merged.copy()

    merged = _apply_value_transforms(merged)
    artifacts["merged_transformed"] = merged.copy()

    candidate_features = _select_feature_columns(merged.columns)
    artifacts["candidate_features"] = candidate_features

    missing_counts = merged[candidate_features].isna().sum()
    indicator_columns: List[str] = []
    indicator_shares: Dict[str, float] = {}
    if ADD_MISSING_INDICATORS:
        for col, missing in missing_counts.items():
            if missing > 0:
                indicator_col = f"{col}_missing"
                indicator_values = merged[col].isna().astype(np.int8)
                merged[indicator_col] = indicator_values
                indicator_columns.append(indicator_col)
                indicator_shares[indicator_col] = float(indicator_values.mean())
                column_sources[indicator_col] = (
                    f"missing_indicator:{column_sources.get(col, 'unknown')}"
                )

    skip_imputation = [
        col
        for col in candidate_features
        if col.startswith("urb_streetview_")
        or col in {
            "bld_area_weighted_building_year",
            "bld_age_pre1950_share",
            "bld_age_1950_1990_share",
            "bld_age_post1990_share",
        }
    ]

    merged[candidate_features] = _impute_by_city(
        merged,
        candidate_features,
        city_col="city",
        skip_columns=skip_imputation,
    )

    non_skip_columns = [col for col in candidate_features if col not in skip_imputation]
    merged[non_skip_columns] = merged[non_skip_columns].fillna(0.0)

    scaling_columns = [col for col in candidate_features if col not in indicator_columns]
    scaled_feature_values, scaling_stats = _scale_within_city(
        merged, scaling_columns, city_col="city"
    )
    scaled_feature_names = [f"{col}_scaled" for col in scaling_columns]
    scaled_df = pd.DataFrame(
        scaled_feature_values,
        columns=scaled_feature_names,
        index=merged.index,
    )
    merged = pd.concat([merged, scaled_df], axis=1)

    if SELECTED_FEATURES is not None:
        selected_set = set(SELECTED_FEATURES)
        feature_pairs = [
            (base, scaled)
            for base, scaled in zip(scaling_columns, scaled_feature_names)
            if scaled in selected_set
        ]
        if not feature_pairs:
            raise ValueError("SELECTED_FEATURES produced an empty feature set")
        scaling_columns, scaled_feature_names = zip(*feature_pairs)
        scaling_columns = list(scaling_columns)
        scaled_feature_names = list(scaled_feature_names)

    scaled_to_base = {scaled: base for base, scaled in zip(scaling_columns, scaled_feature_names)}

    high_corr_pairs = pd.DataFrame(columns=["feature_a", "feature_b", "abs_corr"])
    if REMOVE_HIGHLY_CORRELATED and scaled_feature_names:
        kept_scaled, high_corr_pairs = _drop_highly_correlated_features(
            merged, scaled_feature_names, HIGH_CORRELATION_THRESHOLD
        )
        dropped_scaled = [col for col in scaled_feature_names if col not in kept_scaled]
        if dropped_scaled:
            print(
                f"Removed {len(dropped_scaled)} highly correlated features "
                f"(abs(r) > {HIGH_CORRELATION_THRESHOLD:.2f})."
            )
            scaled_feature_names = kept_scaled
            scaling_columns = [scaled_to_base[col] for col in scaled_feature_names]
            merged = merged.drop(columns=dropped_scaled)
            scaling_stats = scaling_stats[scaling_stats["feature"].isin(scaling_columns)].reset_index(drop=True)
            missing_counts = missing_counts.reindex(scaling_columns).fillna(0)
            for col in dropped_scaled:
                column_sources.pop(col, None)

    column_sources.update(
        {
            name: column_sources.get(col, "unknown")
            for name, col in zip(scaled_feature_names, scaling_columns)
        }
    )

    feature_summary = _build_feature_summary(
        scaling_columns,
        scaled_feature_names,
        indicator_columns,
        missing_counts,
        column_sources,
        row_count=len(merged),
        indicator_shares=indicator_shares,
    )

    modelling_columns = (
        [
            "global_grid_id",
            "city",
            target_raw_col,
            target_z_col,
            "target_uhi_raw",
        ]
        + scaled_feature_names
        + indicator_columns
    )
    frame = merged[modelling_columns].copy()

    feature_columns = list(scaled_feature_names) + indicator_columns

    dataset = ModelDataset(
        frame=frame,
        feature_columns=feature_columns,
        indicator_columns=indicator_columns,
        target_column=target_z_col,
        raw_target_column=target_raw_col,
        uhi_target_column="target_uhi_raw",
        city_column="city",
        feature_scaling=scaling_stats,
        feature_summary=feature_summary,
        target_stats=target_stats,
        uhi_target_stats=uhi_target_stats,
    )

    if return_artifacts:
        artifacts.update(
            {
                "column_sources": column_sources,
                "missing_counts": missing_counts,
                "indicator_columns": indicator_columns,
                "high_corr_pairs": high_corr_pairs,
                "scaling_columns": scaling_columns,
                "scaled_feature_names": scaled_feature_names,
                "scaling_stats": scaling_stats,
                "feature_summary": feature_summary,
                "final_frame": frame,
            }
        )
        return dataset, artifacts

    return dataset, {}


def build_model_dataset(project_root: Path, grid_resolution_m: int = 30) -> ModelDataset:
    """Create a model-ready dataset from the processed feature tables.

    Parameters
    ----------
    project_root : Path
        Repository root containing the ``data/`` directory.
    grid_resolution_m : int
        Grid resolution in metres (default 30).

    Returns
    -------
    ModelDataset
        The assembled modelling dataset.
    """

    dataset, _ = run_preprocessing_pipeline(
        project_root, return_artifacts=False, grid_resolution_m=grid_resolution_m
    )
    return dataset


def _load_dataset(project_root: Path, spec: DatasetSpec) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load a single dataset and apply prefix-based renaming."""

    path = project_root / spec.relative_path
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    table = pd.read_parquet(path)

    if spec.drop_geometry and "geometry" in table.columns:
        table = table.drop(columns=["geometry"])

    rename_map: Dict[str, str] = {}
    column_sources: Dict[str, str] = {}

    for col in table.columns:
        if col in EXCLUDE_FEATURE_NAMES:
            continue
        new_name = f"{spec.prefix}_{col.lower()}"
        rename_map[col] = new_name
        column_sources[new_name] = spec.prefix

    table = table.rename(columns=rename_map)

    return table, column_sources


def _merge_tables(tables: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Merge all tables on `global_grid_id` and `city`."""

    iterator = iter(tables)
    merged = next(iterator)
    for table in iterator:
        merged = merged.merge(
            table,
            on=["global_grid_id", "city"],
            how="outer",
            validate="one_to_one",
        )
    return merged


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce obvious dtypes (booleans to ints, ensure floats for metrics)."""

    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "boolean" or df[col].dtype == bool:
            df[col] = df[col].astype(np.int8)

        if df[col].dtype == "float16":
            df[col] = df[col].astype(np.float32)

    return df


def _compute_citywise_zscores(
    df: pd.DataFrame, column: str, city_col: str = "city"
) -> Tuple[pd.Series, pd.DataFrame]:
    """Compute z-scores of a column within each city and return stats."""

    group_stats = (
        df.groupby(city_col)[column]
        .agg(["mean", "std"])
        .rename(columns={"mean": "city_mean", "std": "city_std"})
    )

    # Avoid division by zero by replacing extremely small std with 1.
    group_stats["city_std"] = group_stats["city_std"].replace(0, 1.0).fillna(1.0)

    def _zscore(row: pd.Series) -> float:
        stats = group_stats.loc[row[city_col]]
        return float((row[column] - stats["city_mean"]) / stats["city_std"])

    zscores = df.apply(_zscore, axis=1)

    stats = group_stats.reset_index().rename(columns={city_col: "city"})

    return zscores, stats


def add_derived_features(
    df: pd.DataFrame, column_sources: Dict[str, str]
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Create lightweight physics-inspired features from existing columns."""

    df = df.copy()

    def _add(name: str, values: pd.Series) -> None:
        df[name] = values
        column_sources[name] = "derived"

    if {"bld_total_footprint_area", "bld_area_weighted_height"}.issubset(df.columns):
        volume = df["bld_total_footprint_area"] * df["bld_area_weighted_height"]
        _add("derived_building_volume_density", volume)

    if {"bld_area_weighted_height", "bld_building_coverage_ratio"}.issubset(df.columns):
        ratio = df["bld_area_weighted_height"] / (df["bld_building_coverage_ratio"] + 1e-3)
        _add("derived_height_to_coverage_ratio", ratio)

    if {"gee_vegetation_fraction", "vox_tree_canopy_ratio"}.issubset(df.columns):
        combined_green = (
            df["gee_vegetation_fraction"].fillna(0) * 0.6
            + df["vox_tree_canopy_ratio"].fillna(0) * 0.4
        )
        _add("derived_green_cooling_potential", combined_green)

    if {"gee_impervious_fraction", "gee_albedo"}.issubset(df.columns):
        impervious_heat = df["gee_impervious_fraction"] * (1 - df["gee_albedo"].fillna(0.3))
        _add("derived_impervious_heat_potential", impervious_heat)

    if {"vox_svf", "bld_building_coverage_ratio"}.issubset(df.columns):
        svf_density = df["vox_svf"].fillna(0.5) * (1 - df["bld_building_coverage_ratio"].fillna(0))
        _add("derived_ventilation_proxy", svf_density)

    if {"urb_street_length", "urb_edge_count"}.issubset(df.columns):
        edge_norm = df["urb_edge_count"].replace(0, np.nan)
        street_closeness = df["urb_street_length"] / edge_norm
        street_closeness = street_closeness.fillna(0)
        _add("derived_street_centrality_proxy", street_closeness)

    # Surface balance: hardscape minus shading fraction (e.g., Oke 1982 urban canyon energy balance).
    if {"gee_impervious_fraction_150m", "gee_tree_canopy_fraction_150m"}.issubset(df.columns):
        canopy = df["gee_tree_canopy_fraction_150m"].fillna(0.0)
        impervious = df["gee_impervious_fraction_150m"].fillna(0.0)
        _add("derived_impervious_canopy_balance", impervious - canopy)

    if {"gee_urban_radiative_index", "gee_era5_vpd_mean"}.issubset(df.columns):
        radiative = df["gee_urban_radiative_index"].fillna(0.0)
        vpd = df["gee_era5_vpd_mean"].fillna(0.0)
        _add("derived_radiative_vpd_interaction", radiative * vpd)

    # Non-linear physics features for SHAP (added based on XGBoost analysis showing
    # impervious_300m as #1 feature - these capture threshold/saturation effects)

    # Urban Heat Trap Index: impervious/albedo ratio (threshold effects)
    if "gee_impervious_fraction_300m" in df and "gee_albedo" in df:
        impervious_300m = df["gee_impervious_fraction_300m"].fillna(0.0)
        albedo = df["gee_albedo"].fillna(0.3) + 0.1  # +0.1 prevents division by zero
        _add("derived_urban_heat_trap_index", impervious_300m / albedo)

    # Vegetation Cooling Saturation: log-ratio capturing diminishing returns
    if "gee_vegetation_fraction_300m" in df and "gee_impervious_fraction_300m" in df:
        veg_300m = df["gee_vegetation_fraction_300m"].fillna(0.0)
        imp_300m = df["gee_impervious_fraction_300m"].fillna(0.0)
        _add("derived_vegetation_cooling_saturation",
             np.log1p(veg_300m * 10) / (np.log1p(imp_300m + 1) + 0.01))

    # Canyon Sky Openness: inverse of aspect ratio (sky view proxy)
    if "vox_canyon_aspect_ratio" in df:
        canyon_ar = df["vox_canyon_aspect_ratio"].fillna(0.0) + 0.5
        _add("derived_canyon_sky_openness", 1.0 / canyon_ar)

    # Impervious Spatial Contrast: coefficient of variation (edge effects)
    if "gee_impervious_fraction_neighbor_std_150m" in df and "gee_impervious_fraction_neighbor_mean_300m" in df:
        imp_std = df["gee_impervious_fraction_neighbor_std_150m"].fillna(0.0)
        imp_mean = df["gee_impervious_fraction_neighbor_mean_300m"].fillna(0.0) + 0.1
        _add("derived_impervious_spatial_contrast", imp_std / imp_mean)

    return df, column_sources


def _add_neighbor_context_features(
    df: pd.DataFrame,
    column_sources: Dict[str, str],
    *,
    specs: Iterable[Tuple[str, float, str]],
    city_col: str,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Add neighbour statistics (mean/std) for selected columns."""

    if not specs:
        return df, column_sources

    if not {"x", "y"}.issubset(df.columns):
        return df, column_sources

    df = df.copy()
    updated_sources = dict(column_sources)

    for base_col, radius, statistic in specs:
        if base_col not in df.columns:
            continue
        radius_int = int(round(radius))
        stat_key = statistic.lower()
        if stat_key not in {"mean", "std"}:
            continue
        neighbor_col = f"{base_col}_neighbor_{stat_key}_{radius_int}m"
        df[neighbor_col] = np.nan
        updated_sources[neighbor_col] = f"neighbor:{column_sources.get(base_col, 'unknown')}"

        for _, indices in df.groupby(city_col).indices.items():
            idx_arr = np.asarray(indices)
            coords = df.loc[idx_arr, ["x", "y"]].to_numpy(dtype=float)
            if np.isnan(coords).any():
                continue
            tree = BallTree(coords, leaf_size=32)
            neighbors = tree.query_radius(coords, r=radius)
            values = df.loc[idx_arr, base_col].to_numpy(dtype=float)
            neighbor_stats = np.full(len(idx_arr), np.nan, dtype=float)
            for i, neigh in enumerate(neighbors):
                neigh = [int(n) for n in neigh if n != i]
                if not neigh:
                    continue
                sample = values[neigh]
                if np.isnan(sample).all():
                    continue
                if stat_key == "mean":
                    neighbor_stats[i] = float(np.nanmean(sample))
                else:
                    neighbor_stats[i] = float(np.nanstd(sample))
            df.loc[idx_arr, neighbor_col] = neighbor_stats

    return df, updated_sources


def _apply_value_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce skew/outliers via winsorisation and log transforms."""

    df = df.copy()

    for col, quant in WINSOR_FEATURES.items():
        if col in df.columns:
            lower = df[col].quantile(quant)
            upper = df[col].quantile(1 - quant)
            df[col] = df[col].clip(lower=lower, upper=upper)

    for col in LOG1P_FEATURES:
        if col in df.columns:
            clipped = df[col].clip(lower=0)
            df[col] = np.log1p(clipped)

    return df


def _select_feature_columns(columns: Iterable[str]) -> List[str]:
    """Select modelling feature columns, excluding obvious leakages and metadata."""

    leakage_feature_names = set()
    for group_name in ("target",):
        leakage_feature_names.update(FEATURE_GROUPS.get(group_name, ()))

    selected: List[str] = []
    for col in columns:
        if col in EXCLUDE_FEATURE_NAMES:
            continue
        if any(col.startswith(prefix) for prefix in LEAKAGE_PREFIXES):
            continue
        if col.startswith("target_"):
            continue
        if col in leakage_feature_names:
            continue
        # Exclude grid_size_m (constant metadata field, always 30m)
        if col.endswith("_grid_size_m"):
            continue
        selected.append(col)
    return selected



def _drop_highly_correlated_features(
    df: pd.DataFrame, columns: Sequence[str], threshold: float
) -> Tuple[List[str], pd.DataFrame]:
    """Identify and remove highly correlated features."""

    if not columns:
        empty = pd.DataFrame(columns=["feature_a", "feature_b", "abs_corr"])
        return list(columns), empty

    corr = df[columns].corr().abs()
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    upper = corr.where(mask)

    to_drop: set[str] = set()
    records: List[Dict[str, float]] = []

    for col in upper.columns:
        high = upper[col][upper[col] > threshold].dropna()
        for other, value in high.items():
            records.append({"feature_a": other, "feature_b": col, "abs_corr": float(value)})
            to_drop.add(col)

    kept = [col for col in columns if col not in to_drop]

    if records:
        pairs = (pd.DataFrame(records)
                    .sort_values("abs_corr", ascending=False)
                    .reset_index(drop=True))
    else:
        pairs = pd.DataFrame(columns=["feature_a", "feature_b", "abs_corr"])

    return kept, pairs


def _impute_by_city(
    df: pd.DataFrame,
    columns: List[str],
    city_col: str,
    skip_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Impute numeric columns using city medians, unless column is skipped."""

    result = df[columns].copy()
    skip_set = set(skip_columns or [])

    city_groups = df.groupby(city_col)
    columns_to_impute = [col for col in columns if col not in skip_set]

    if columns_to_impute:
        city_medians = city_groups[columns_to_impute].median()
        for city, idx in city_groups.indices.items():
            result_subset = result.loc[idx, columns_to_impute]
            result.loc[idx, columns_to_impute] = result_subset.fillna(
                city_medians.loc[city]
            )

        global_medians = result[columns_to_impute].median()
        result[columns_to_impute] = result[columns_to_impute].fillna(global_medians)

    return result


def _scale_within_city(
    df: pd.DataFrame, columns: List[str], city_col: str
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Standardise features within each city. Returns array and stats."""

    scaled = np.empty((df.shape[0], len(columns)), dtype=np.float32)
    stats_records: List[Dict[str, object]] = []

    for city, idx in df.groupby(city_col).indices.items():
        subset = df.loc[idx, columns].astype(float)
        means = subset.mean(axis=0)
        stds = subset.std(axis=0, ddof=0)
        stds = stds.replace(0, 1.0)

        scaled[idx, :] = (subset - means) / stds

        for col, mean, std in zip(columns, means, stds):
            stats_records.append(
                {
                    "city": city,
                    "feature": col,
                    "mean": float(mean),
                    "std": float(std),
                }
            )

    scaling_stats = pd.DataFrame(stats_records)
    return scaled, scaling_stats


def _build_feature_summary(
    base_features: List[str],
    scaled_features: List[str],
    indicator_features: List[str],
    missing_counts: pd.Series,
    column_sources: Dict[str, str],
    *,
    row_count: int,
    indicator_shares: Dict[str, float],
) -> pd.DataFrame:
    """Create summary metadata for the feature set."""

    records: List[Dict[str, object]] = []

    denominator = max(row_count, 1)

    for base, scaled in zip(base_features, scaled_features):
        missing_fraction = float(missing_counts.get(base, 0) / denominator)
        records.append(
            {
                "feature": base,
                "scaled_feature": scaled,
                "source": column_sources.get(base, "unknown"),
                "missing_fraction": missing_fraction,
                "type": "continuous",
            }
        )

    for indicator in indicator_features:
        records.append(
            {
                "feature": indicator,
                "scaled_feature": indicator,
                "source": column_sources.get(indicator, "missing_indicator"),
                "missing_fraction": indicator_shares.get(indicator, 0.0),
                "type": "indicator",
            }
        )

    summary = pd.DataFrame(records)
    return summary


# Public aliases for notebooks / external callers
load_dataset = _load_dataset
merge_tables = _merge_tables
coerce_types = _coerce_types
compute_citywise_zscores = _compute_citywise_zscores
apply_value_transforms = _apply_value_transforms
select_feature_columns = _select_feature_columns
impute_by_city = _impute_by_city
scale_within_city = _scale_within_city
add_neighbor_context_features = _add_neighbor_context_features
build_feature_summary = _build_feature_summary
