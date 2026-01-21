"""Risk and vulnerability mapping utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.model_preprocessing import build_model_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "risk_vulnerability"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HOTSPOT_PRED_PATH = PROJECT_ROOT / "results/policy_hotspot_classifier/hotspot_predictions.csv"

HIGH_THRESHOLD = 0.75
EXTREME_THRESHOLD = 1.25
VULN_MED_THRESHOLD = 0.0
VULN_HIGH_THRESHOLD = 0.8


def load_predictions() -> pd.DataFrame:
    # Load hierarchical blend predictions (already computed in notebook 07)
    blend_pred_path = PROJECT_ROOT / "results/density_stratified_uhi_raw/hierarchical_blend_target_uhi_raw_predictions.csv"
    preds = pd.read_csv(blend_pred_path)[["global_grid_id", "city", "pred_global", "pred_blend_cv"]]

    # Rename pred_blend_cv to pred_uhi_blend for clarity
    preds = preds.rename(columns={"pred_blend_cv": "pred_uhi_blend"})

    return preds


def zscore_within_city(series: pd.Series, city: pd.Series) -> pd.Series:
    def _z(s):
        std = s.std(ddof=0)
        if std == 0:
            std = 1.0
        return (s - s.mean()) / std

    return series.groupby(city).transform(_z)


def assign_heat_tier(pred_uhi: pd.Series, city: pd.Series, stats: pd.DataFrame) -> pd.Series:
    stats_idx = stats.set_index("city")
    stds = city.map(stats_idx["city_std"]).fillna(pred_uhi.std())
    heat = np.where(
        pred_uhi > stds * EXTREME_THRESHOLD,
        "extreme",
        np.where(pred_uhi > stds * HIGH_THRESHOLD, "high", "moderate"),
    )
    return pd.Series(heat, index=pred_uhi.index)


def assign_vulnerability(children_z: pd.Series, elderly_z: pd.Series) -> pd.Series:
    combined = np.maximum(children_z, elderly_z)
    vuln = np.where(
        combined > VULN_HIGH_THRESHOLD,
        "high",
        np.where(combined > VULN_MED_THRESHOLD, "medium", "low"),
    )
    return pd.Series(vuln, index=children_z.index)


def risk_category(heat: pd.Series, vuln: pd.Series) -> pd.Series:
    mapping = {
        ("extreme", "high"): "severe",
        ("extreme", "medium"): "very_high",
        ("extreme", "low"): "high",
        ("high", "high"): "very_high",
        ("high", "medium"): "high",
        ("high", "low"): "moderate",
        ("moderate", "high"): "moderate",
        ("moderate", "medium"): "elevated",
        ("moderate", "low"): "baseline",
    }
    return pd.Series([mapping.get((h, v), "baseline") for h, v in zip(heat, vuln)], index=heat.index)


def run_risk_map(
    project_root: Path = PROJECT_ROOT,
    *,
    dataset=None,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Generate risk map and summary tables."""

    if dataset is None:
        dataset = build_model_dataset(project_root)

    frame = dataset.frame.copy()

    preds = load_predictions()
    frame = frame.merge(preds, on=["global_grid_id", "city"], how="left")
    # pred_uhi_blend is already in the preds dataframe from load_predictions()

    urbanity_path = project_root / "data" / "1-processed" / "Urbanity" / "Urbanity_grid_30m.parquet"
    if urbanity_path.exists():
        demo_cols = [
            "global_grid_id",
            "population_children_density",
            "population_elderly_density",
        ]
        urbanity_demo = pd.read_parquet(urbanity_path, columns=demo_cols)
        urbanity_demo = urbanity_demo.rename(
            columns={
                "population_children_density": "urb_population_children_density",
                "population_elderly_density": "urb_population_elderly_density",
            }
        )
        frame = frame.merge(urbanity_demo, on="global_grid_id", how="left")

    def _city_z_or_scaled(col_base: str) -> pd.Series:
        if col_base in frame.columns:
            return zscore_within_city(frame[col_base], frame["city"])
        scaled_name = f"{col_base}_scaled"
        if scaled_name in frame.columns:
            return frame[scaled_name]
        raise KeyError(
            f"Required column '{col_base}' or its scaled counterpart is missing from the model dataset"
        )

    frame["children_z"] = _city_z_or_scaled("urb_population_children_density")
    frame["elderly_z"] = _city_z_or_scaled("urb_population_elderly_density")

    frame["heat_tier"] = assign_heat_tier(frame["pred_uhi_blend"], frame["city"], dataset.target_stats)
    frame["vulnerability_tier"] = assign_vulnerability(frame["children_z"], frame["elderly_z"])
    frame["risk_category"] = risk_category(frame["heat_tier"], frame["vulnerability_tier"])

    if HOTSPOT_PRED_PATH.exists():
        hotspot = pd.read_csv(HOTSPOT_PRED_PATH)
        frame = frame.merge(
            hotspot[["global_grid_id", "hotspot_probability_calibrated", "hotspot_prediction"]],
            on="global_grid_id",
            how="left",
        )

    cols = [
        "global_grid_id",
        "city",
        "pred_uhi_blend",
        "heat_tier",
        "vulnerability_tier",
        "risk_category",
        "children_z",
        "elderly_z",
    ]
    if "hotspot_probability_calibrated" in frame:
        cols += ["hotspot_probability_calibrated", "hotspot_prediction"]

    risk_map = frame[cols].copy()
    summary = (
        risk_map.assign(count=1)
        .groupby(["city", "risk_category"])
        .agg({"count": "sum"})
        .reset_index()
        .pivot(index="city", columns="risk_category", values="count")
        .fillna(0)
    )

    if verbose:
        print("Risk tiers computed for", len(risk_map), "grid cells")

    return {"risk_map": risk_map, "summary": summary}
