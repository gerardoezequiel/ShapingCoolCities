"""Shared spatial cross-validation utilities for UHI modelling notebooks."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from shapely import wkb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID_PATH = PROJECT_ROOT / "data/utils/grids/all_cities_30m_grid.parquet"

DENSITY_FEATURE = "bld_total_footprint_area_scaled"

# Fallbacks for notebooks that still request deprecated density columns.
_DENSITY_FALLBACKS: Dict[str, str] = {
    "bld_building_coverage_ratio_scaled": DENSITY_FEATURE,
}

# Default parameters for contiguous spatial blocking.
DEFAULT_BLOCK_SIZE = 600.0
MAX_BLOCK_SIZE = 2400.0
BLOCK_GROWTH_FACTOR = 2.0


def _quantise_coordinates(
    coords: np.ndarray,
    block_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantise x/y coordinates into integer grid bins for given block size."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    scaled = coords / block_size
    # Use floor to keep bins stable across positive/negative coordinates.
    bins = np.floor(scaled).astype(np.int64)
    return bins[:, 0], bins[:, 1]


def _compute_spatial_blocks(
    frame: pd.DataFrame,
    *,
    city_col: str,
    x_col: str,
    y_col: str,
    base_block_size: float,
    min_block_size: int,
    max_block_size: float,
    growth_factor: float,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Assign contiguous spatial blocks per city using coarse grid binning."""

    block_ids = np.full(len(frame), fill_value=-1, dtype=int)
    block_labels: Dict[Tuple[str, int, int], int] = {}
    block_metadata: List[Dict[str, object]] = []
    next_block_id = 0

    grouped = frame.groupby(city_col, sort=False)

    for city, city_df in grouped:
        indices = city_df.index.to_numpy()
        coords = city_df[[x_col, y_col]].to_numpy(dtype=float)

        if len(coords) == 0:
            continue

        block_size = base_block_size

        while True:
            bx, by = _quantise_coordinates(coords, block_size)
            block_keys = pd.Series(zip(bx, by))
            counts = block_keys.value_counts()

            if counts.empty:
                break

            if counts.min() >= min_block_size or block_size >= max_block_size or len(counts) == 1:
                for (cell_x, cell_y), count in counts.items():
                    label_key = (city, int(cell_x), int(cell_y))
                    if label_key not in block_labels:
                        block_labels[label_key] = next_block_id
                        next_block_id += 1
                    block_metadata.append(
                        {
                            "city": city,
                            "block_label": block_labels[label_key],
                            "block_size": block_size,
                            "cell_x": int(cell_x),
                            "cell_y": int(cell_y),
                            "tile_count": int(count),
                        }
                    )
                block_ids[indices] = [
                    block_labels[(city, int(cx), int(cy))]
                    for cx, cy in zip(bx, by)
                ]
                break

            block_size = min(block_size * growth_factor, max_block_size)

    metadata_df = pd.DataFrame(block_metadata)
    return block_ids, metadata_df


def load_grid_centroids() -> pd.DataFrame:
    """Load grid centroids used for stratified spatial folds."""
    grid_df = pd.read_parquet(GRID_PATH).copy()
    grid_df["geometry"] = grid_df["geometry"].apply(wkb.loads)
    grid_df["centroid"] = grid_df["geometry"].apply(lambda geom: geom.centroid)
    grid_df["x"] = grid_df["centroid"].apply(lambda pt: pt.x)
    grid_df["y"] = grid_df["centroid"].apply(lambda pt: pt.y)
    coords = grid_df[["global_grid_id", "x", "y"]].copy()
    # Provide duplicate columns for notebooks that expect a suffixed merge result.
    coords["x_coord"] = coords["x"]
    coords["y_coord"] = coords["y"]
    return coords


def build_stratified_clusters(
    frame: pd.DataFrame,
    *,
    target_col: str,
    density_feature: str,
    n_folds: int,
    min_cluster_size: int = 400,
    max_clusters_per_stratum: int = 8,
    block_size: float = DEFAULT_BLOCK_SIZE,
    block_growth_factor: float = BLOCK_GROWTH_FACTOR,
    max_block_size: float = MAX_BLOCK_SIZE,
    return_block_metadata: bool = False,
) -> np.ndarray | Tuple[np.ndarray, pd.DataFrame]:
    """Cluster tiles by city/target/density with contiguous spatial blocks.

    Parameters
    ----------
    frame:
        Modelling dataframe containing at least ``city``, ``x``, ``y`` and the
        specified target/density columns.
    target_col:
        Column containing the regression target.
    density_feature:
        Name of the built-density proxy used for stratification. Fallbacks are
        applied when this column has been removed upstream.
    n_folds:
        Number of spatial folds to generate.
    min_cluster_size:
        Minimum desired number of tiles per spatial block. Blocks smaller than
        this threshold trigger block-size growth during quantisation.
    max_clusters_per_stratum:
        Kept for backwards compatibility (unused in the contiguous implementation).
    block_size:
        Initial grid size (same units as ``x``/``y``) used to derive contiguous
        spatial blocks.
    block_growth_factor:
        Factor applied when blocks fall below ``min_cluster_size``. The grid
        expands until blocks meet the threshold or ``max_block_size`` is reached.
    max_block_size:
        Upper bound on the grid size during expansion.
    return_block_metadata:
        When True, returns a tuple of (fold_labels, block_metadata_df).
    """

    resolved_density_feature = density_feature
    if resolved_density_feature not in frame.columns:
        fallback = _DENSITY_FALLBACKS.get(resolved_density_feature)
        if fallback and fallback in frame.columns:
            warnings.warn(
                "Density feature '%s' missing; using fallback '%s' instead." % (
                    resolved_density_feature,
                    fallback,
                ),
                UserWarning,
                stacklevel=2,
            )
            resolved_density_feature = fallback
        else:
            available = [col for col in _DENSITY_FALLBACKS.values() if col in frame.columns]
            hint = (
                f" Try one of {available}."
                if available
                else " No fallback density features available."
            )
            raise KeyError(
                f"Density feature '{density_feature}' not found in frame columns." + hint
            )

    data = frame[["city", target_col, resolved_density_feature, "x", "y"]].copy()

    try:
        data["uhi_bin"] = pd.qcut(data[target_col], q=5, labels=False, duplicates="drop")
    except ValueError:
        data["uhi_bin"] = 0

    try:
        data["density_bin"] = pd.qcut(
            data[resolved_density_feature], q=3, labels=False, duplicates="drop"
        )
    except ValueError:
        data["density_bin"] = 0

    block_labels, block_metadata = _compute_spatial_blocks(
        data,
        city_col="city",
        x_col="x",
        y_col="y",
        base_block_size=block_size,
        min_block_size=max(1, min_cluster_size),
        max_block_size=max(block_size, max_block_size),
        growth_factor=max(1.0, block_growth_factor),
    )

    if (block_labels == -1).any():
        raise RuntimeError("Spatial block assignment failed for some rows.")

    data["block_label"] = block_labels

    cluster_keys = (
        data["city"].astype(str)
        + "|u"
        + data["uhi_bin"].fillna(-1).astype(int).astype(str)
        + "|d"
        + data["density_bin"].fillna(-1).astype(int).astype(str)
        + "|b"
        + data["block_label"].astype(int).astype(str)
    )

    cluster_labels = np.full(len(data), fill_value=-1, dtype=int)
    cluster_info: List[Tuple[int, str, int]] = []
    next_cluster_id = 0

    for key, idx in cluster_keys.groupby(cluster_keys).groups.items():
        indices = np.array(list(idx))
        if len(indices) == 0:
            continue
        cluster_labels[indices] = next_cluster_id
        cluster_info.append((next_cluster_id, key, len(indices)))
        next_cluster_id += 1

    if (cluster_labels == -1).any():
        unknown_idx = np.where(cluster_labels == -1)[0]
        for ridx in unknown_idx:
            cluster_labels[ridx] = next_cluster_id
            cluster_info.append((next_cluster_id, f"orphan_{next_cluster_id}", 1))
            next_cluster_id += 1

    cluster_info.sort(key=lambda item: item[2], reverse=True)

    fold_sizes = np.zeros(n_folds, dtype=int)
    cluster_to_fold: Dict[int, int] = {}

    for cluster_id, _key, size in cluster_info:
        fold = int(np.argmin(fold_sizes))
        cluster_to_fold[cluster_id] = fold
        fold_sizes[fold] += int(size)

    fold_labels = np.array([cluster_to_fold[int(c)] for c in cluster_labels])

    if return_block_metadata:
        return fold_labels, block_metadata

    return fold_labels
