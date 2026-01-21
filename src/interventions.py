"""NB09 interventions and optimisation API.

This module centralises the logic needed by Notebook 09 to work with
cooling intervention optimisation:

* Running the 400‑combination parameter grid search (optional and heavy).
* Loading the grid-search artefacts written to ``results/scenarios``.
* Exposing optimised intervention parameter sets as typed objects.

The intent is that the notebook only needs to import from this module,
keeping complex logic in versioned Python rather than in ad‑hoc cells.

Typical usage from ``notebooks/09_Cooling_Interventions.ipynb``::

    from pathlib import Path
    from interventions import (
        run_comprehensive_grid_search,
        load_grid_search_results,
        load_intervention_scenarios,
        scenarios_to_frame,
        InterventionScenario,
        GridSearchResult,
    )

    PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()

    # Normal use: just load optimised scenarios
    scenarios = load_intervention_scenarios(PROJECT_ROOT)
    display(scenarios_to_frame(scenarios))

    # Only if needed: re-run the full grid search (5–10 minutes)
    full, best = run_comprehensive_grid_search(PROJECT_ROOT)
    display(best)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from src.model_preprocessing import build_model_dataset


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GridSearchResult:
    """Bundle full grid-search output and objective-specific optima."""

    full_results: pd.DataFrame
    best_by_objective: pd.DataFrame


@dataclass
class InterventionScenario:
    """Optimised intervention configuration for NB09.

    Parameters are expressed as additive deltas relative to the raw
    features (e.g. ``gee_impervious_fraction_300m``) and accompanied by
    the key summary metrics used in NB09.
    """

    name: str
    imp_delta: float
    alb_delta: float
    veg_delta: float
    tree_delta: float
    mean_cooling: float
    max_cooling: float
    pct_cells_ge_1C: float
    pct_cells_ge_0_5C: float
    n_extremes: int


# ---------------------------------------------------------------------------
# Core grid-search implementation (ported from /tmp/comprehensive_grid_search_with_albedo.py)
# ---------------------------------------------------------------------------


def run_comprehensive_grid_search(
    project_root: Path | str,
    grid_params: Optional[Dict[str, Iterable[float]]] = None,
    save_results: bool = True,
    priority_pct: float = 10.0,
    extreme_cold_threshold: float = -5.0,
    extreme_warm_threshold: float = 0.5,
    priority_weights: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the 400‑combination NB09 parameter grid search.

    This is a direct, cleaned-up port of the optimisation logic used in
    ``/tmp/comprehensive_grid_search_with_albedo.py``. It:

    * Loads the NB07 models and blend weights.
    * Builds the model-ready dataset via :func:`build_model_dataset`.
    * Joins risk/vulnerability and raw GEE fractions.
    * Computes SHAP-inspired priority scores.
    * Tests all combinations of (impervious, albedo, vegetation, trees).
    * Computes cooling metrics and counts of extreme predictions.
    * Derives objective-specific "best" rows.

    Parameters
    ----------
    project_root:
        Path to the ``ShapingCoolCities`` project root.
    grid_params:
        Optional override for the parameter grid. When ``None``, uses the
        original search space from the tmp script.
    save_results:
        When True, writes CSVs under ``results/scenarios``.
    priority_pct:
        Top percentage of cells to consider as priority zones (default 10%).
    extreme_cold_threshold:
        Cooling predictions below this are flagged as extreme (default -5.0).
    extreme_warm_threshold:
        Warming predictions above this are flagged as extreme (default 0.5).
    priority_weights:
        Dict with keys 'risk', 'vuln', 'potential' for priority scoring.
        Default: {'risk': 0.40, 'vuln': 0.35, 'potential': 0.25}

    Returns
    -------
    full_results:
        DataFrame with one row per parameter combination.
    best_by_objective:
        DataFrame indexed by objective name (``max_mean_cooling``,
        ``max_peak_cooling``, ``max_coverage_1deg``, ``safe_conservative``,
        ``cost_effective``) with the corresponding best rows.
    """

    project_root = Path(project_root)

    # Set default priority weights
    if priority_weights is None:
        priority_weights = {'risk': 0.40, 'vuln': 0.35, 'potential': 0.25}

    print("=" * 80)
    print("COMPREHENSIVE GRID SEARCH: Including Albedo (SHAP Rank 6)")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")

    # ------------------------------------------------------------------
    # 1. Load models and data
    # ------------------------------------------------------------------
    print("\n[1/7] Loading models and data...")

    model_dir = project_root / "results" / "modeling"
    risk_map_path = project_root / "results" / "risk_vulnerability" / "risk_map.parquet"

    global_model = xgb.XGBRegressor()
    global_model.load_model(str(model_dir / "xgb_global_model.json"))

    cfb_model = xgb.XGBRegressor()
    cfb_model.load_model(str(model_dir / "xgb_cfb_model.json"))

    csa_model = xgb.XGBRegressor()
    csa_model.load_model(str(model_dir / "xgb_csa_model.json"))

    blend_weights = pd.read_csv(model_dir / "climate_blend_weights.csv")

    dataset = build_model_dataset(project_root)
    features_df = dataset.frame.copy()
    feature_cols = dataset.feature_columns

    risk_df = pd.read_parquet(risk_map_path)
    baseline_df = features_df.merge(
        risk_df[[
            "global_grid_id",
            "city",
            "pred_uhi_blend",
            "risk_category",
            "vulnerability_tier",
        ]],
        on=["global_grid_id", "city"],
        how="left",
    )

    # Load GEE features (raw fractions + albedo)
    gee_features = pd.read_parquet(
        project_root
        / "data"
        / "1-processed"
        / "GoogleEarthEngine"
        / "All_cities_GEE_features_30m.parquet"
    )

    gee_subset = gee_features[
        [
            "global_grid_id",
            "city",
            "tree_canopy_fraction_300m",
            "vegetation_fraction_300m",
            "impervious_fraction_300m",
            "albedo",
        ]
    ].copy()

    gee_subset = gee_subset.rename(
        columns={
            "tree_canopy_fraction_300m": "gee_tree_canopy_fraction_300m",
            "vegetation_fraction_300m": "gee_vegetation_fraction_300m",
            "impervious_fraction_300m": "gee_impervious_fraction_300m",
            "albedo": "gee_albedo",
        }
    )

    baseline_df = baseline_df.merge(gee_subset, on=["global_grid_id", "city"], how="left")

    print(f"  ✓ Loaded {len(baseline_df):,} cells")

    # Blend prediction helper (uses per‑city climate blend weights)
    def predict_blend(df: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(df))
        global_pred = global_model.predict(df[feature_cols])
        cfb_pred = cfb_model.predict(df[feature_cols])
        csa_pred = csa_model.predict(df[feature_cols])

        for city in df["city"].unique():
            city_mask = df["city"] == city
            city_weights = blend_weights[blend_weights["city"] == city].iloc[0]
            climate_zone = city_weights["climate_zone"]
            specialist_pred = cfb_pred if climate_zone == "Cfb" else csa_pred
            intercept = city_weights["intercept"]
            global_weight = city_weights["global_weight"]
            specialist_weight = city_weights["specialist_weight"]
            blend_pred = (
                intercept
                + (global_weight * global_pred)
                + (specialist_weight * specialist_pred)
            )
            preds[city_mask] = blend_pred[city_mask]

        return preds

    # ------------------------------------------------------------------
    # 2. Scaling stats
    # ------------------------------------------------------------------
    print("\n[2/7] Loading scaling statistics...")
    scaling_stats = pd.read_csv(
        project_root / "data" / "2-model-ready" / "model_ready_feature_scaling.csv"
    )
    print(f"  ✓ {scaling_stats['feature'].nunique()} features")

    def apply_delta_to_feature(
        df: pd.DataFrame,
        raw_feature: str,
        delta: float,
        scaling_stats_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Unscale → apply delta → rescale for a single raw feature."""

        scaled_feature = f"{raw_feature}_scaled"
        if scaled_feature not in df.columns:
            return df

        df_modified = df.copy()

        for city in df["city"].unique():
            city_mask = df["city"] == city
            city_stats = scaling_stats_df[
                (scaling_stats_df["city"] == city)
                & (scaling_stats_df["feature"] == raw_feature)
            ]
            if len(city_stats) == 0:
                continue

            mean = city_stats.iloc[0]["mean"]
            std = city_stats.iloc[0]["std"]

            unscaled = df_modified.loc[city_mask, scaled_feature] * std + mean
            modified = unscaled + delta

            # Clip fractions and albedo into valid ranges
            if "fraction" in raw_feature or "albedo" in raw_feature:
                modified = modified.clip(0, 1)

            rescaled = (modified - mean) / std
            df_modified.loc[city_mask, scaled_feature] = rescaled

        return df_modified

    # ------------------------------------------------------------------
    # 3. Priority scores (risk × vulnerability × cooling potential)
    # ------------------------------------------------------------------
    print("\n[3/7] Computing priority scores...")
    print(f"  Weights: risk={priority_weights['risk']:.0%}, vuln={priority_weights['vuln']:.0%}, potential={priority_weights['potential']:.0%}")

    risk_map = {
        "baseline": 0.0,
        "moderate": 0.3,
        "elevated": 0.5,
        "high": 0.7,
        "very_high": 0.85,
        "severe": 1.0,
    }
    vuln_map = {"low": 0.2, "medium": 0.6, "high": 1.0}

    baseline_df["risk_score"] = baseline_df["risk_category"].map(risk_map).fillna(0.2)
    baseline_df["vuln_score"] = baseline_df["vulnerability_tier"].map(vuln_map).fillna(0.2)
    baseline_df["cooling_potential"] = (
        baseline_df["gee_impervious_fraction_300m"] * 0.50
        + (1 - baseline_df["gee_tree_canopy_fraction_300m"]) * 0.25
        + (1 - baseline_df["gee_albedo"]) * 0.25
    )

    baseline_df["priority_score"] = (
        baseline_df["risk_score"] * priority_weights['risk']
        + baseline_df["vuln_score"] * priority_weights['vuln']
        + baseline_df["cooling_potential"] * priority_weights['potential']
    )

    threshold = np.percentile(baseline_df["priority_score"].dropna(), 100 - priority_pct)
    priority_mask = baseline_df["priority_score"] >= threshold
    print(f"  ✓ Priority cells: {priority_mask.sum():,} ({priority_pct:.0f}%)")

    # ------------------------------------------------------------------
    # 4. Grid search space
    # ------------------------------------------------------------------
    print("\n[4/7] Defining grid search space...")

    default_grid: Dict[str, Iterable[float]] = {
        "impervious": [-0.30, -0.35, -0.40, -0.45, -0.50],
        "albedo": [0.00, 0.05, 0.10, 0.15, 0.20],
        "vegetation": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        "trees": [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    }

    grid = grid_params or default_grid

    combinations = list(
        product(
            grid["impervious"],
            grid["albedo"],
            grid["vegetation"],
            grid["trees"],
        )
    )

    print("  Grid space:")
    print(f"    Impervious: {list(grid['impervious'])}")
    print(f"    Albedo:     {list(grid['albedo'])}")
    print(f"    Vegetation: {list(grid['vegetation'])}")
    print(f"    Trees:      {list(grid['trees'])}")
    print(f"  Total combinations: {len(combinations)}")

    # ------------------------------------------------------------------
    # 5. Run grid search
    # ------------------------------------------------------------------
    print("\n[5/7] Running grid search...")
    print(f"  Applying interventions ONLY to priority zones ({priority_mask.sum():,} cells)")
    print("  Estimated time: 5–10 minutes for full 400‑combo search")

    results = []

    for idx, (imp_delta, alb_delta, veg_delta, tree_delta) in enumerate(combinations):
        if (idx + 1) % 50 == 0:
            pct = (idx + 1) / len(combinations) * 100
            print(f"  Progress: {idx + 1}/{len(combinations)} ({pct:.1f}%)")

        # Create temporary scenario object for apply_scenario_deltas
        temp_scenario = InterventionScenario(
            name="temp",
            imp_delta=imp_delta,
            alb_delta=alb_delta,
            veg_delta=veg_delta,
            tree_delta=tree_delta,
            mean_cooling=0.0,
            max_cooling=0.0,
            pct_cells_ge_1C=0.0,
            pct_cells_ge_0_5C=0.0,
            n_extremes=0
        )

        # Apply deltas ONLY to priority zones using target_mask
        df_scenario = apply_scenario_deltas(
            baseline_df.copy(),
            temp_scenario,
            scaling_stats,
            feature_cols,
            target_mask=priority_mask.values
        )

        scenario_predictions = predict_blend(df_scenario)
        baseline_predictions = baseline_df["pred_uhi_blend"].values
        cooling_delta = scenario_predictions - baseline_predictions

        extreme_mask = (cooling_delta < extreme_cold_threshold) | (cooling_delta > extreme_warm_threshold)
        n_extremes = int(extreme_mask.sum())

        # Compute metrics on VALID priority cells only (exclude extremes)
        valid_priority_mask = priority_mask & ~extreme_mask
        priority_cooling = cooling_delta[valid_priority_mask]

        mean_cool = float(priority_cooling.mean())
        median_cool = float(np.median(priority_cooling))
        max_cool = float(priority_cooling.min())
        pct_1deg = float((priority_cooling <= -1.0).sum() / len(priority_cooling) * 100)
        pct_05deg = float((priority_cooling <= -0.5).sum() / len(priority_cooling) * 100)

        results.append(
            {
                "imp_delta": imp_delta,
                "alb_delta": alb_delta,
                "veg_delta": veg_delta,
                "tree_delta": tree_delta,
                "mean_cooling": mean_cool,
                "median_cooling": median_cool,
                "max_cooling": max_cool,
                "pct_1deg": pct_1deg,
                "pct_05deg": pct_05deg,
                "n_extremes": n_extremes,
            }
        )

    full_results = pd.DataFrame(results)
    print(f"\n  ✓ Grid search complete: {len(full_results)} combinations tested")

    # ------------------------------------------------------------------
    # 6. Analyse results and derive objective‑specific optima
    # ------------------------------------------------------------------
    print("\n[6/7] Analysing results...")

    valid_results = full_results[full_results["n_extremes"] == 0].copy()
    print(f"  Valid (no extremes): {len(valid_results)}/{len(full_results)}")

    if len(valid_results) == 0:
        print("\n  ⚠️ All combinations produced extremes – using top 50 by fewest extremes.")
        valid_results = full_results.nsmallest(50, "n_extremes").copy()

    # 1. Maximum mean cooling
    best_mean = valid_results.nsmallest(1, "mean_cooling").iloc[0]

    # 2. Maximum peak cooling
    best_peak = valid_results.nsmallest(1, "max_cooling").iloc[0]

    # 3. Maximum coverage ≥1.0°C
    best_coverage = valid_results.nlargest(1, "pct_1deg").iloc[0]

    # 4. Safe conservative (impervious ≥ -0.35, fewer extremes)
    safe = valid_results[valid_results["imp_delta"] >= -0.35]
    if len(safe) > 0:
        best_safe = safe.nsmallest(1, "mean_cooling").iloc[0]
    else:
        best_safe = best_mean

    # 5. Cost‑effective (best cooling per unit total delta)
    valid_results = valid_results.copy()
    valid_results["total_delta"] = (
        valid_results["imp_delta"].abs()
        + valid_results["alb_delta"].abs()
        + valid_results["veg_delta"].abs()
        + valid_results["tree_delta"].abs()
    )
    valid_results["cooling_per_delta"] = (
        valid_results["mean_cooling"].abs() / valid_results["total_delta"]
    )
    best_cost = valid_results.nlargest(1, "cooling_per_delta").iloc[0]

    best_by_objective = pd.DataFrame(
        {
            "max_mean_cooling": best_mean,
            "max_peak_cooling": best_peak,
            "max_coverage_1deg": best_coverage,
            "safe_conservative": best_safe,
            "cost_effective": best_cost,
        }
    ).T

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    if save_results:
        output_dir = project_root / "results" / "scenarios"
        output_dir.mkdir(parents=True, exist_ok=True)

        full_results.to_csv(output_dir / "grid_search_full_results.csv", index=False)
        best_by_objective.to_csv(
            output_dir / "optimal_scenarios_by_objective.csv", index=True
        )

        print("\n[7/7] Saving results...")
        print(
            f"  ✓ Full results: grid_search_full_results.csv "
            f"({len(full_results)} combinations)"
        )
        print("  ✓ Best by objective: optimal_scenarios_by_objective.csv")

    print("\n" + "=" * 80)
    print("KEY FINDINGS (see optimal_scenarios_by_objective.csv for details)")
    print("=" * 80)

    return full_results, best_by_objective


# ---------------------------------------------------------------------------
# Lightweight loaders for notebook use
# ---------------------------------------------------------------------------


def load_grid_search_results(
    project_root: Path | str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load grid-search artefacts from ``results/scenarios``.

    Parameters
    ----------
    project_root:
        Path to the project root.

    Returns
    -------
    full_results:
        DataFrame of all parameter combinations.
    best_by_objective:
        DataFrame indexed by objective name.
    """

    project_root = Path(project_root)
    output_dir = project_root / "results" / "scenarios"
    full_results = pd.read_csv(output_dir / "grid_search_full_results.csv")
    best_by_objective = pd.read_csv(output_dir / "optimal_scenarios_by_objective.csv")

    # Normalise index from the first column if needed
    if "objective" in best_by_objective.columns:
        best_by_objective = best_by_objective.set_index("objective")
    elif "Unnamed: 0" in best_by_objective.columns:
        best_by_objective = best_by_objective.rename(columns={"Unnamed: 0": "objective"}).set_index(
            "objective"
        )

    return full_results, best_by_objective


def load_intervention_scenarios(
    project_root: Path | str,
) -> Dict[str, InterventionScenario]:
    """Load optimised NB09 intervention scenarios from grid-search outputs."""

    _, best_by_objective = load_grid_search_results(project_root)

    scenarios: Dict[str, InterventionScenario] = {}
    for name, row in best_by_objective.iterrows():
        scenarios[name] = InterventionScenario(
            name=name,
            imp_delta=float(row["imp_delta"]),
            alb_delta=float(row.get("alb_delta", 0.0)),
            veg_delta=float(row["veg_delta"]),
            tree_delta=float(row["tree_delta"]),
            mean_cooling=float(row["mean_cooling"]),
            max_cooling=float(row["max_cooling"]),
            pct_cells_ge_1C=float(row["pct_1deg"]),
            pct_cells_ge_0_5C=float(row["pct_05deg"]),
            n_extremes=int(row["n_extremes"]),
        )

    return scenarios


def scenarios_to_frame(scenarios: Dict[str, InterventionScenario]) -> pd.DataFrame:
    """Convert a scenario dictionary to a tidy DataFrame for display."""

    records = []
    for key, s in scenarios.items():
        records.append(
            {
                "objective": key,
                "imp_delta": s.imp_delta,
                "alb_delta": s.alb_delta,
                "veg_delta": s.veg_delta,
                "tree_delta": s.tree_delta,
                "mean_cooling": s.mean_cooling,
                "max_cooling": s.max_cooling,
                "pct_cells_ge_1C": s.pct_cells_ge_1C,
                "pct_cells_ge_0_5C": s.pct_cells_ge_0_5C,
                "n_extremes": s.n_extremes,
            }
        )

    return pd.DataFrame.from_records(records).set_index("objective")


# ---------------------------------------------------------------------------
# Additional helper functions for Notebook 09
# ---------------------------------------------------------------------------


def apply_scenario_deltas(
    baseline_df: pd.DataFrame,
    scenario: InterventionScenario,
    scaling_stats: pd.DataFrame,
    feature_cols: list,
    target_mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Apply intervention scenario deltas to scaled features.

    Process:
    1. Unscale features using city-specific mean/std
    2. Apply delta (additive change)
    3. Clip to valid range (0-1 for fractions)
    4. Rescale using same city-specific statistics

    Parameters
    ----------
    baseline_df : pd.DataFrame
        Model-ready dataset with scaled features
    scenario : InterventionScenario
        Scenario with parameter deltas
    scaling_stats : pd.DataFrame
        Feature scaling statistics (columns: feature, city, mean, std)
    feature_cols : list
        List of feature column names for model prediction
    target_mask : Optional[np.ndarray]
        Boolean array (same length as ``baseline_df``) indicating which
        rows should receive the intervention. When ``None`` (default),
        deltas are applied to *all* rows.

    Returns
    -------
    pd.DataFrame
        Modified dataframe with scenario deltas applied
    """
    df_modified = baseline_df.copy()

    # Map scenario deltas to feature names
    delta_mapping = {
        "gee_impervious_fraction_300m": scenario.imp_delta,
        "gee_albedo": scenario.alb_delta,
        "gee_vegetation_fraction_300m": scenario.veg_delta,
        "gee_tree_canopy_fraction_300m": scenario.tree_delta,
    }

    if target_mask is None:
        target_mask = np.ones(len(df_modified), dtype=bool)

    for raw_feature, delta in delta_mapping.items():
        if delta == 0:
            continue

        scaled_feature = f"{raw_feature}_scaled"
        if scaled_feature not in df_modified.columns:
            continue

        for city in df_modified["city"].unique():
            city_mask = (df_modified["city"] == city) & target_mask
            city_stats = scaling_stats[
                (scaling_stats["city"] == city) & (scaling_stats["feature"] == raw_feature)
            ]

            if len(city_stats) == 0:
                continue

            mean = city_stats.iloc[0]["mean"]
            std = city_stats.iloc[0]["std"]

            # Unscale → apply delta → clip → rescale
            unscaled = df_modified.loc[city_mask, scaled_feature] * std + mean
            modified = unscaled + delta

            # Clip fractions and albedo to valid ranges
            if "fraction" in raw_feature or "albedo" in raw_feature:
                modified = modified.clip(0, 1)

            rescaled = (modified - mean) / std
            df_modified.loc[city_mask, scaled_feature] = rescaled

    return df_modified


def filter_extreme_predictions(
    cooling_delta: pd.Series,
    threshold_cold: float = -5.0,
    threshold_warm: float = 0.5,
) -> pd.Series:
    """Filter cells with extreme cooling predictions.

    Grid search validation showed ALL scenarios produce ~6,500-7,000
    extreme predictions (16-17% of cells). These represent model
    extrapolation beyond training bounds and should be excluded.

    Parameters
    ----------
    cooling_delta : pd.Series
        Predicted cooling effect (scenario - baseline)
    threshold_cold : float
        Minimum valid cooling (more negative = more extreme)
    threshold_warm : float
        Maximum valid warming (reject warming predictions)

    Returns
    -------
    pd.Series (bool)
        Boolean mask where True = valid prediction
    """
    extreme_mask = (cooling_delta < threshold_cold) | (cooling_delta > threshold_warm)
    return ~extreme_mask


def predict_blend(
    df: pd.DataFrame,
    feature_cols: list,
    global_model: xgb.XGBRegressor,
    cfb_model: xgb.XGBRegressor,
    csa_model: xgb.XGBRegressor,
    blend_weights: pd.DataFrame,
) -> np.ndarray:
    """Predict using hierarchical blend (global + climate specialists).

    For each city:
    1. Predict with global model
    2. Predict with climate-specific model (Cfb or Csa)
    3. Blend using city-specific weights and intercept

    Parameters
    ----------
    df : pd.DataFrame
        Features dataframe with 'city' column
    feature_cols : list
        Feature columns for prediction
    global_model : xgb.XGBRegressor
        Global XGBoost model
    cfb_model : xgb.XGBRegressor
        Temperate oceanic (Cfb) specialist model
    csa_model : xgb.XGBRegressor
        Mediterranean (Csa) specialist model
    blend_weights : pd.DataFrame
        City-specific blend coefficients

    Returns
    -------
    np.ndarray
        Blended predictions
    """
    preds = np.zeros(len(df))
    global_pred = global_model.predict(df[feature_cols])
    cfb_pred = cfb_model.predict(df[feature_cols])
    csa_pred = csa_model.predict(df[feature_cols])

    for city in df["city"].unique():
        city_mask = df["city"] == city
        city_weights = blend_weights[blend_weights["city"] == city].iloc[0]
        climate_zone = city_weights["climate_zone"]
        specialist_pred = cfb_pred if climate_zone == "Cfb" else csa_pred
        intercept = city_weights["intercept"]
        global_weight = city_weights["global_weight"]
        specialist_weight = city_weights["specialist_weight"]
        blend_pred = (
            intercept
            + (global_weight * global_pred)
            + (specialist_weight * specialist_pred)
        )
        preds[city_mask] = blend_pred[city_mask]

    return preds


__all__ = [
    "GridSearchResult",
    "InterventionScenario",
    "run_comprehensive_grid_search",
    "load_grid_search_results",
    "load_intervention_scenarios",
    "scenarios_to_frame",
    "apply_scenario_deltas",
    "filter_extreme_predictions",
    "predict_blend",
]
