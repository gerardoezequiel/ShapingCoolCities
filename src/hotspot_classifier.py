"""Helpers for training the calibrated hotspot classifier used in notebooks."""
from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import median_abs_deviation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from src.model_preprocessing import build_model_dataset
from src.spatial_cv_utils import (
    DENSITY_FEATURE,
    build_stratified_clusters,
    load_grid_centroids,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "policy_hotspot_classifier"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BEST_PARAM_PATH = RESULTS_DIR / "best_params.json"

BASE_MODEL_PARAMS: Dict[str, float] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": -1,
    "random_state": 42,
    "tree_method": "hist",
    "verbosity": 0,
}


def derive_hotspot_labels(
    frame: pd.DataFrame,
    *,
    city_col: str,
    uhi_col: str,
    city_stats: pd.DataFrame = None,
) -> pd.Series:
    """Binary label: hotspot if UHI exceeds city-level MAD threshold (robust to outliers).

    Uses Median Absolute Deviation (MAD) scaled to match standard deviation units.
    MAD = median(|x - median|) × 1.4826, providing robust threshold unaffected by outliers.

    This approach addresses outlier contamination (e.g., Barcelona's 4.2% extreme cold cells
    inflating σ from ~1.5°C to 3.13°C), ensuring realistic hotspot prevalence across cities.
    """
    labels = np.zeros(len(frame), dtype=np.int8)

    for city in frame[city_col].unique():
        city_mask = frame[city_col] == city
        city_uhi = frame.loc[city_mask, uhi_col].values

        # Calculate robust threshold using MAD (scaled to match σ for normal distributions)
        mad = median_abs_deviation(city_uhi, scale='normal')  # scale='normal' applies 1.4826 factor

        # Hotspot: UHI > +1 MAD (analogous to +1σ but robust)
        labels[city_mask] = (city_uhi > mad).astype(np.int8)

    return pd.Series(labels, index=frame.index, dtype=np.int8)


INT_PARAMS = {"max_depth", "n_estimators"}


def _load_tuned_params() -> tuple[Dict[str, float], Optional[float]]:
    if not BEST_PARAM_PATH.exists():
        return {}, None
    try:
        payload = json.loads(BEST_PARAM_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, None
    if not isinstance(payload, dict):
        return {}, None
    params = payload.get("params", {})
    threshold = payload.get("threshold")
    clean_params: Dict[str, float] = {}
    if isinstance(params, dict):
        for key, value in params.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            clean_params[key] = int(numeric) if key in INT_PARAMS else numeric
    tuned_threshold = float(threshold) if threshold is not None else None
    return clean_params, tuned_threshold


def run_hotspot_classifier(
    project_root: Path = PROJECT_ROOT,
    *,
    dataset=None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Train the hotspot classifier and return evaluation artefacts."""

    if dataset is None:
        dataset = build_model_dataset(project_root)

    frame = dataset.frame.copy()
    coords = load_grid_centroids()
    frame = frame.merge(coords, on="global_grid_id", how="left")

    frame["hotspot"] = derive_hotspot_labels(
        frame,
        city_col=dataset.city_column,
        uhi_col=dataset.uhi_target_column,
        city_stats=dataset.target_stats,
    )

    X = frame[dataset.feature_columns]
    y = frame["hotspot"]

    groups = build_stratified_clusters(
        frame,
        target_col=dataset.uhi_target_column,
        density_feature=DENSITY_FEATURE,
        n_folds=5,
    )

    pos_weight = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)

    model_params: Dict[str, float] = {**BASE_MODEL_PARAMS}
    model_params.update(
        max_depth=4,
        learning_rate=0.05,
        n_estimators=400,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=3.0,
        gamma=0.1,
        scale_pos_weight=pos_weight * 1.5,
    )

    tuned_params, tuned_threshold = _load_tuned_params()
    model_params.update(tuned_params)

    clf = xgb.XGBClassifier(**model_params)
    gkf = GroupKFold(n_splits=5)

    probabilities = np.zeros(len(frame), dtype=float)
    fold_reports: List[Dict[str, float]] = []
    per_city_records: List[Dict[str, float]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_prob = clf.predict_proba(X.iloc[test_idx])[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        y_true = y.iloc[test_idx]

        probabilities[test_idx] = y_prob

        fold_reports.append(
            {
                "fold": fold_idx,
                "auc": roc_auc_score(y_true, y_prob),
                "f1": f1_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
            }
        )

        fold_df = frame.iloc[test_idx][[dataset.city_column]].copy()
        fold_df["y_true"] = y_true.to_numpy()
        fold_df["y_pred"] = y_pred
        fold_df["y_prob"] = y_prob

        for city, city_df in fold_df.groupby(dataset.city_column):
            if city_df["y_true"].nunique() < 2:
                auc = np.nan
            else:
                auc = roc_auc_score(city_df["y_true"], city_df["y_prob"])
            per_city_records.append(
                {
                    "fold": fold_idx,
                    "city": city,
                    "auc": auc,
                    "f1": f1_score(city_df["y_true"], city_df["y_pred"], zero_division=0),
                    "precision": precision_score(city_df["y_true"], city_df["y_pred"], zero_division=0),
                    "recall": recall_score(city_df["y_true"], city_df["y_pred"], zero_division=0),
                }
            )

    truth_array = y.to_numpy()

    rng = np.random.default_rng(42)
    calibration_mask = np.zeros(len(frame), dtype=bool)
    calibration_share = 0.15
    random_values = rng.random(len(frame))

    # Create 15% holdout per city for calibration
    for city, idx in frame.groupby(dataset.city_column).indices.items():
        city_idx = np.asarray(idx)
        if city_idx.size == 0:
            continue
        n_cal = max(1, int(np.ceil(calibration_share * city_idx.size)))
        order = np.argsort(random_values[city_idx])
        calibration_mask[city_idx[order[:n_cal]]] = True

    # Safety checks for global calibration set
    positives = np.where(truth_array == 1)[0]
    negatives = np.where(truth_array == 0)[0]
    if calibration_mask.sum() == 0 or positives.size == 0 or negatives.size == 0:
        calibration_mask[:] = False
    if calibration_mask.sum() == 0:
        calibration_mask[rng.choice(len(frame), size=max(1, int(0.15 * len(frame))), replace=False)] = True
    if truth_array[calibration_mask].sum() == 0 and positives.size:
        calibration_mask[positives[rng.integers(positives.size)]] = True
    if truth_array[calibration_mask].sum() == calibration_mask.sum() and negatives.size:
        calibration_mask[negatives[rng.integers(negatives.size)]] = True

    # Initialize calibrated probabilities array
    y_prob_cal = probabilities.copy()

    # Fit city-specific Platt calibrators
    MIN_CALIBRATION_POSITIVES = 20
    for city, idx in frame.groupby(dataset.city_column).indices.items():
        city_idx = np.asarray(idx)
        city_cal_mask = calibration_mask[city_idx]
        city_cal_prob = probabilities[city_idx][city_cal_mask]
        city_cal_y = truth_array[city_idx][city_cal_mask]

        n_pos_cal = (city_cal_y == 1).sum()

        if n_pos_cal < MIN_CALIBRATION_POSITIVES:
            # Insufficient positive samples - use raw probabilities
            if verbose:
                print(f"  {city}: Skipping calibration (only {n_pos_cal} positive samples in holdout, need {MIN_CALIBRATION_POSITIVES})")
            # y_prob_cal already initialized with raw probabilities
            continue

        # Fit city-specific Platt calibrator
        city_calibrator = LogisticRegression(max_iter=1000, random_state=42)
        city_calibrator.fit(city_cal_prob.reshape(-1, 1), city_cal_y)

        # Apply calibration to all samples from this city
        city_prob_all = probabilities[city_idx]
        y_prob_cal[city_idx] = city_calibrator.predict_proba(city_prob_all.reshape(-1, 1))[:, 1]

        if verbose:
            print(f"  {city}: Calibrated using {len(city_cal_prob)} samples ({n_pos_cal} positive, {len(city_cal_prob) - n_pos_cal} negative)")

    base_thresholds = np.arange(0.05, 0.351, 0.025)
    extended_thresholds = np.array([0.375, 0.4, 0.45, 0.5])
    threshold_list = np.concatenate([base_thresholds, extended_thresholds])
    if tuned_threshold is not None:
        threshold_list = np.append(threshold_list, tuned_threshold)
    thresholds = np.unique(np.round(threshold_list, 4))

    threshold_records = []
    for thr in thresholds:
        y_pred_thr = (y_prob_cal >= thr).astype(int)
        threshold_records.append(
            {
                "threshold": float(thr),
                "precision": float(precision_score(truth_array, y_pred_thr, zero_division=0)),
                "recall": float(recall_score(truth_array, y_pred_thr, zero_division=0)),
                "f1": float(f1_score(truth_array, y_pred_thr, zero_division=0)),
            }
        )

    candidates = [r for r in threshold_records if r["recall"] >= 0.6 and r["precision"] >= 0.25]
    if candidates:
        best_row = max(candidates, key=lambda r: (r["f1"], r["precision"]))
    else:
        best_row = max(threshold_records, key=lambda r: r["f1"])
    selected_threshold = best_row["threshold"]

    y_pred_selected = (y_prob_cal >= selected_threshold).astype(int)

    overall_report = classification_report(
        truth_array, y_pred_selected, output_dict=True, zero_division=0
    )
    overall_auc = roc_auc_score(truth_array, y_prob_cal)
    cm = confusion_matrix(truth_array, y_pred_selected)

    if verbose:
        print("Fold metrics:\n", pd.DataFrame(fold_reports))
        print("\nSelected threshold:", selected_threshold)
        print("\nOverall AUC:", overall_auc)
        print(
            "\nGlobal classification report (threshold {:.2f}):".format(selected_threshold)
        )
        print(classification_report(truth_array, y_pred_selected, zero_division=0))
        print("Confusion matrix:\n", cm)

    predictions = frame[["global_grid_id", dataset.city_column]].copy()
    predictions["hotspot_probability_raw"] = probabilities
    predictions["hotspot_probability_calibrated"] = y_prob_cal
    predictions["hotspot_prediction"] = y_pred_selected

    return {
        "fold_metrics": pd.DataFrame(fold_reports),
        "per_city_metrics": pd.DataFrame(per_city_records),
        "threshold_metrics": pd.DataFrame(threshold_records),
        "selected_threshold": float(selected_threshold),
        "overall_auc": float(overall_auc),
        "classification_report": pd.DataFrame(overall_report).transpose(),
        "confusion_matrix": cm,
        "predictions": predictions,
        "calibrator": None,  # City-specific calibrators used, not returning single global calibrator
        "calibration_mask": calibration_mask,
        "model_params": model_params,
    }
