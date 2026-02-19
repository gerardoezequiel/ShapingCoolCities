#!/usr/bin/env python3
"""Export all 6 cities' VoxCity voxel data to lightweight JSON for Three.js viewer.

Reads voxcity.npy (class labels) and solar_peak.npy (irradiance W/m²),
filters to visible surface voxels, and outputs compact JSON for each city.
Also generates a combined metadata file.
"""

import json
import sys
from pathlib import Path

import numpy as np

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
CACHE_BASE = BASE_DIR / "data" / "1-processed" / "VoxCity" / "cache"
ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"

CITIES = ["amsterdam", "athens", "barcelona", "berlin", "madrid", "paris"]

# VoxCity class mapping
VISIBLE_CLASSES = [-3, -2, -1, 1, 2, 6, 9, 11, 12, 13]
CLASS_TO_INT = {
    -3: 0, -2: 0, -1: 0,  # ground variants
    1: 1,   # wall
    2: 2,   # tree
    6: 3,   # sidewalk/road
    9: 4,   # building
    11: 4, 12: 4, 13: 4,  # built structures
}

# Lower budget per city since we're rendering 6 at once
MAX_VOXELS_PER_CITY = 40000


def load_data(city):
    cache_dir = CACHE_BASE / city
    voxcity_path = cache_dir / "voxcity.npy"
    solar_path = cache_dir / "solar_peak.npy"

    if not voxcity_path.exists():
        print(f"  WARNING: {voxcity_path} not found, skipping", file=sys.stderr)
        return None, None

    voxcity = np.load(voxcity_path)
    print(f"  Shape: {voxcity.shape}, classes: {np.unique(voxcity)}")

    solar = None
    if solar_path.exists():
        solar = np.load(solar_path).astype(np.float64)
        valid = ~np.isnan(solar)
        if valid.sum() > 0:
            solar[np.isnan(solar)] = np.nanmedian(solar)
            print(f"  Solar: {solar[valid].min():.0f}-{solar[valid].max():.0f} W/m²")
        else:
            solar = None

    return voxcity, solar


def extract_surface_voxels(voxcity):
    mask = np.isin(voxcity, VISIBLE_CLASSES)
    surface_mask = np.zeros_like(mask)

    for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        shifted = np.roll(voxcity, (dx, dy, dz), axis=(0, 1, 2))
        surface_mask |= (mask & (shifted == 0))

    always_visible = np.isin(voxcity, [-3, -2, -1, 2, 6])
    surface_mask |= (mask & always_visible)

    coords = np.argwhere(surface_mask)
    classes = voxcity[surface_mask]
    print(f"  Surface voxels: {len(coords):,}")
    return coords, classes


def downsample(coords, classes, max_voxels):
    if len(coords) <= max_voxels:
        return coords, classes
    step = 2
    while True:
        mask = ((coords[:, 0] % step == 0) & (coords[:, 1] % step == 0))
        if mask.sum() <= max_voxels:
            break
        step += 1
    print(f"  Downsampled: {len(coords):,} → {mask.sum():,} (step={step})")
    return coords[mask], classes[mask]


def build_json(coords, classes, solar, shape):
    int_classes = np.array([CLASS_TO_INT.get(int(c), 0) for c in classes])

    solar_values = None
    if solar is not None:
        s_min, s_max = float(np.nanmin(solar)), float(np.nanmax(solar))
        s_range = max(s_max - s_min, 1.0)
        solar_values = []
        for i, (x, y, z) in enumerate(coords):
            if x < solar.shape[0] and y < solar.shape[1]:
                val = solar[x, y]
                normalized = int(255 * (val - s_min) / s_range) if not np.isnan(val) else 128
                solar_values.append(max(0, min(255, normalized)))
            else:
                solar_values.append(128)

    voxels = []
    for i in range(len(coords)):
        entry = [int(coords[i][0]), int(coords[i][1]), int(coords[i][2]), int(int_classes[i])]
        if solar_values is not None:
            entry.append(solar_values[i])
        voxels.append(entry)

    return {
        "dims": [int(shape[0]), int(shape[1]), int(shape[2])],
        "classes": {"0":"ground","1":"wall","2":"tree","3":"road","4":"building"},
        "hasSolar": solar_values is not None,
        "count": len(voxels),
        "voxels": voxels
    }


def main():
    print("=== VoxCity Multi-City Export ===\n")
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {}

    for city in CITIES:
        print(f"\n[{city.upper()}]")
        voxcity, solar = load_data(city)
        if voxcity is None:
            continue

        coords, classes = extract_surface_voxels(voxcity)
        coords, classes = downsample(coords, classes, MAX_VOXELS_PER_CITY)
        data = build_json(coords, classes, solar, voxcity.shape)

        out_path = ASSETS_DIR / f"{city}_voxels.json"
        with open(out_path, "w") as f:
            json.dump(data, f, separators=(",", ":"))

        size_kb = out_path.stat().st_size / 1024
        print(f"  → {out_path.name}: {data['count']:,} voxels, {size_kb:.0f} KB")

        metadata[city] = {
            "file": f"{city}_voxels.json",
            "count": data["count"],
            "dims": data["dims"],
            "hasSolar": data["hasSolar"]
        }

    # Write metadata
    meta_path = ASSETS_DIR / "voxels_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata: {meta_path}")
    print("Done!")


if __name__ == "__main__":
    main()
