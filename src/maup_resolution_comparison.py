"""Multi-resolution stratified CV evaluation to quantify MAUP effects."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from shapely import wkb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.model_preprocessing import build_model_dataset
from src.spatial_cv_utils import build_stratified_clusters, DENSITY_FEATURE

RESULTS_DIR = PROJECT_ROOT / "results/stratified_spatial_cv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_RESOLUTIONS: Tuple[int, ...] = (30, 60, 90)

XGB_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=799,
    max_depth=6,
    learning_rate=0.03298058136368126,
    subsample=0.7798801010280795,
    colsample_bytree=0.7438641518314566,
    min_child_weight=4.990150425980441,
    gamma=0.10190525873529291,
    reg_lambda=0.777874280293406,
    reg_alpha=0.012834166331645547,
    random_state=42,
    n_jobs=-1,
)


def _grid_suffix(resolution_m: int) -> str:
    return "" if resolution_m == 30 else f"_{resolution_m}m"


def load_centroids(grid_resolution_m: int) -> pd.DataFrame:
    """Load centroid coordinates for the requested grid size."""

    grid_path = (
        PROJECT_ROOT
        / "data/utils/grids"
        / f"all_cities{_grid_suffix(grid_resolution_m)}_grid.parquet"
    )
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid parquet not found: {grid_path}")

    grid_df = pd.read_parquet(grid_path).copy()

    if {"x", "y"}.issubset(grid_df.columns):
        centroids = grid_df[["global_grid_id", "x", "y"]].copy()
    elif "geometry" in grid_df.columns:
        grid_df["geometry"] = grid_df["geometry"].apply(wkb.loads)
        grid_df["centroid"] = grid_df["geometry"].apply(lambda geom: geom.centroid)
        centroids = pd.DataFrame(
            {
                "global_grid_id": grid_df["global_grid_id"].astype(str),
                "x": grid_df["centroid"].apply(lambda pt: pt.x),
                "y": grid_df["centroid"].apply(lambda pt: pt.y),
            }
        )
    else:
        raise ValueError(
            "Grid parquet must contain geometry or x/y columns for centroid derivation"
        )

    centroids["global_grid_id"] = centroids["global_grid_id"].astype(str)
    return centroids


def _compute_city_stats(frame: pd.DataFrame, target_col: str) -> Dict[str, Dict[str, float]]:
    stats = (
        frame.groupby("city")[target_col]
        .agg(["mean", "std"])
        .rename(columns={"mean": "city_mean", "std": "city_std"})
    )
    stats["city_std"] = stats["city_std"].replace(0, 1.0).fillna(1.0)
    return stats.to_dict("index")


def evaluate_resolution(
    grid_resolution_m: int,
    *,
    n_folds: int = 5,
    force: bool = False,
    save: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run stratified spatial CV for a specific grid size."""

    metrics_path = (
        RESULTS_DIR / f"stratified_spatial_cv_metrics_{grid_resolution_m}m.csv"
    )
    if metrics_path.exists() and not force:
        metrics_df = pd.read_csv(metrics_path)
    else:
        dataset = build_model_dataset(
            PROJECT_ROOT, grid_resolution_m=grid_resolution_m
        )
        frame = dataset.frame.copy()
        coords = load_centroids(grid_resolution_m)
        frame = frame.merge(coords, on="global_grid_id", how="left")

        if frame[["x", "y"]].isna().any().any():
            raise ValueError(
                f"Missing centroid coordinates after merge for {grid_resolution_m} m grid"
            )

        target_col = dataset.uhi_target_column
        if target_col not in frame.columns:
            raise KeyError(f"Target column '{target_col}' not found in modelling frame")

        if DENSITY_FEATURE not in frame.columns:
            raise KeyError(
                f"Required density feature '{DENSITY_FEATURE}' missing for {grid_resolution_m} m grid"
            )

        city_stats_lookup = _compute_city_stats(frame, target_col)
        fold_assignments = build_stratified_clusters(
            frame,
            target_col=target_col,
            density_feature=DENSITY_FEATURE,
            n_folds=n_folds,
        )

        model = xgb.XGBRegressor(**XGB_PARAMS)
        X = frame[dataset.feature_columns]
        y = frame[target_col]

        gkf = GroupKFold(n_splits=n_folds)

        rows = []
        for fold_idx, (train_idx, test_idx) in enumerate(
            gkf.split(X, y, groups=fold_assignments),
            start=1,
        ):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            preds_raw = model.predict(X_test)

            test_cities = frame.iloc[test_idx]["city"].values
            stds = np.array([city_stats_lookup[city]["city_std"] for city in test_cities])
            stds[stds == 0] = 1.0

            y_test_z = y_test.to_numpy() / stds
            preds_z = preds_raw / stds

            row = {
                "fold": fold_idx,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "r2_z": r2_score(y_test_z, preds_z),
                "rmse_z": float(np.sqrt(mean_squared_error(y_test_z, preds_z))),
                "mae_z": mean_absolute_error(y_test_z, preds_z),
                "r2_raw": r2_score(y_test, preds_raw),
                "rmse_raw": float(np.sqrt(mean_squared_error(y_test, preds_raw))),
                "mae_raw": mean_absolute_error(y_test, preds_raw),
            }
            rows.append(row)

        metrics_df = pd.DataFrame(rows)
        if save:
            metrics_df.to_csv(metrics_path, index=False)

    summary = {
        "grid_m": grid_resolution_m,
        "r2_z_mean": metrics_df["r2_z"].mean(),
        "r2_z_std": metrics_df["r2_z"].std(ddof=1),
        "r2_raw_mean": metrics_df["r2_raw"].mean(),
        "r2_raw_std": metrics_df["r2_raw"].std(ddof=1),
        "rmse_raw_mean": metrics_df["rmse_raw"].mean(),
        "mae_raw_mean": metrics_df["mae_raw"].mean(),
    }
    return metrics_df, summary


def collect_maup_metrics(
    resolutions: Iterable[int] = DEFAULT_RESOLUTIONS,
    *,
    n_folds: int = 5,
    force: bool = False,
) -> pd.DataFrame:
    """Ensure MAUP metrics exist for the requested resolutions and return summary."""

    summaries = []
    for res in resolutions:
        _, summary = evaluate_resolution(res, n_folds=n_folds, force=force)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries).sort_values("grid_m").reset_index(drop=True)
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MAUP impacts across grid resolutions"
    )
    parser.add_argument(
        "--resolutions",
        metavar="N",
        type=int,
        nargs="+",
        default=DEFAULT_RESOLUTIONS,
        help="Grid resolutions (metres) to evaluate",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of spatial CV folds",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute metrics even if cached CSVs exist",
    )

    args = parser.parse_args()

    summary_df = collect_maup_metrics(
        resolutions=args.resolutions,
        n_folds=args.n_folds,
        force=args.force,
    )

    print("MAUP evaluation summary (higher R²_z is better, lower MAE/RMSE better):")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
