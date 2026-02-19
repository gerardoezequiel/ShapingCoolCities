#!/usr/bin/env python3
"""Extract response curve data for the 50% regime boundary visualization.

Reads grid_search_full_results.csv and outputs JSON data showing
depaving % vs. cooling °C for Chart.js rendering.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parents[2] / "results" / "scenarios" / "grid_search_full_results.csv"
OUT_PATH = Path(__file__).resolve().parents[1] / "assets" / "response_curve_data.json"


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} scenarios from {CSV_PATH}")

    # Group by depaving level (imp_delta), compute mean cooling across all combos
    depaving_groups = defaultdict(list)
    for row in rows:
        imp_delta = abs(float(row["imp_delta"]))  # Convert negative to positive percentage
        cooling = abs(float(row["mean_cooling"]))
        depaving_groups[imp_delta].append(cooling)

    # Also extract the "cost-effective" line: veg_delta=0.1, tree_delta=0.2
    cost_effective = defaultdict(list)
    for row in rows:
        veg = round(float(row["veg_delta"]), 2)
        tree = round(float(row["tree_delta"]), 2)
        if abs(veg - 0.1) < 0.01 and abs(tree - 0.2) < 0.01:
            imp_delta = abs(float(row["imp_delta"]))
            cooling = abs(float(row["mean_cooling"]))
            cost_effective[imp_delta].append(cooling)

    # Build output data
    all_depaving_levels = sorted(depaving_groups.keys())
    print(f"Depaving levels: {all_depaving_levels}")

    mean_line = []
    min_line = []
    max_line = []
    cost_line = []

    for level in all_depaving_levels:
        values = depaving_groups[level]
        pct = round(level * 100)
        mean_line.append({"x": pct, "y": round(sum(values) / len(values), 3)})
        min_line.append({"x": pct, "y": round(min(values), 3)})
        max_line.append({"x": pct, "y": round(max(values), 3)})

        if level in cost_effective:
            ce_vals = cost_effective[level]
            cost_line.append({"x": pct, "y": round(sum(ce_vals) / len(ce_vals), 3)})

    result = {
        "mean": mean_line,
        "min": min_line,
        "max": max_line,
        "costEffective": cost_line,
        "regimeBoundary": 50
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nExported to {OUT_PATH}")
    for key in ["mean", "min", "max", "costEffective"]:
        print(f"  {key}: {len(result[key])} points")


if __name__ == "__main__":
    main()
