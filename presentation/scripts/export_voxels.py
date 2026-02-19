#!/usr/bin/env python3
"""Export Barcelona VoxCity voxel data to lightweight JSON for Three.js viewer.

Reads voxcity.npy (class labels) and solar_peak.npy (irradiance W/m²),
filters to visible surface voxels (buildings, trees, ground), and outputs
a compact JSON file for the browser-based 3D viewer.

Actual classes found in Barcelona data: [-3, -2, -1, 0, 1, 2, 6, 9, 11, 12, 13]
"""

import json
import sys
from pathlib import Path

import numpy as np

# Paths
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "1-processed" / "VoxCity" / "cache" / "barcelona"
OUT_PATH = Path(__file__).resolve().parents[1] / "assets" / "barcelona_voxels.json"

# VoxCity class mapping (discovered from actual data)
# Negative values: -3=outside/boundary, -2=underground, -1=ground
# Positive: 0=air, 1=wall, 2=tree, 6=sidewalk/road, 9=building, 11-13=other built
VISIBLE_CLASSES = [-3, -2, -1, 1, 2, 6, 9, 11, 12, 13]

# Compact integer IDs for JSON output
CLASS_TO_INT = {
    -3: 0,  # boundary/ground
    -2: 0,  # underground → treat as ground
    -1: 0,  # ground
    1: 1,   # wall
    2: 2,   # tree
    6: 3,   # sidewalk/road
    9: 4,   # building
    11: 4,  # built structure → building
    12: 4,  # built structure → building
    13: 4,  # built structure → building
}

MAX_VOXELS = 80000  # Target maximum for browser performance


def load_data():
    """Load voxel array and solar data."""
    voxcity_path = CACHE_DIR / "voxcity.npy"
    solar_path = CACHE_DIR / "solar_peak.npy"

    if not voxcity_path.exists():
        print(f"ERROR: {voxcity_path} not found", file=sys.stderr)
        sys.exit(1)

    voxcity = np.load(voxcity_path)
    print(f"Voxcity shape: {voxcity.shape}, dtype: {voxcity.dtype}")
    print(f"Unique classes: {np.unique(voxcity)}")

    solar = None
    if solar_path.exists():
        solar = np.load(solar_path).astype(np.float64)
        nan_count = np.isnan(solar).sum()
        valid_count = (~np.isnan(solar)).sum()
        print(f"Solar shape: {solar.shape}, dtype: {solar.dtype}")
        print(f"Solar NaN count: {nan_count}, valid: {valid_count}")
        if valid_count > 0:
            valid_mask = ~np.isnan(solar)
            print(f"Solar range (valid): {solar[valid_mask].min():.1f} - {solar[valid_mask].max():.1f} W/m²")
            # Replace NaN with median for rendering
            solar[np.isnan(solar)] = np.nanmedian(solar)
        else:
            print("WARNING: All solar values are NaN, skipping solar layer")
            solar = None

    return voxcity, solar


def extract_surface_voxels(voxcity):
    """Extract only visible surface voxels (not interior or air)."""
    nx, ny, nz = voxcity.shape

    # Find all non-air voxels that belong to visible classes
    mask = np.isin(voxcity, VISIBLE_CLASSES)

    # For buildings: only keep surface voxels (those adjacent to air/outside)
    surface_mask = np.zeros_like(mask)

    for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        shifted = np.roll(voxcity, (dx, dy, dz), axis=(0, 1, 2))
        neighbor_is_air = (shifted == 0)
        surface_mask |= (mask & neighbor_is_air)

    # Also include ground-level and tree voxels (always visible)
    always_visible = np.isin(voxcity, [-3, -2, -1, 2, 6])
    surface_mask |= (mask & always_visible)

    coords = np.argwhere(surface_mask)
    classes = voxcity[surface_mask]

    print(f"Total visible mask: {mask.sum():,}")
    print(f"Surface voxels: {len(coords):,}")

    return coords, classes


def downsample_if_needed(coords, classes, max_voxels=MAX_VOXELS):
    """Downsample voxels if exceeding browser performance budget."""
    if len(coords) <= max_voxels:
        return coords, classes

    # Spatial downsampling: skip every Nth voxel in x/y
    step = 2
    while True:
        mask = ((coords[:, 0] % step == 0) & (coords[:, 1] % step == 0))
        if mask.sum() <= max_voxels:
            break
        step += 1

    filtered_coords = coords[mask]
    filtered_classes = classes[mask]
    print(f"Downsampled from {len(coords):,} to {len(filtered_coords):,} (step={step})")
    return filtered_coords, filtered_classes


def build_json(coords, classes, solar, voxcity_shape):
    """Build compact JSON structure for Three.js."""
    nx, ny, nz = voxcity_shape

    # Convert classes to compact integers
    int_classes = np.array([CLASS_TO_INT.get(int(c), 0) for c in classes])

    # Build solar values
    solar_values = None
    if solar is not None:
        solar_min = float(np.nanmin(solar))
        solar_max = float(np.nanmax(solar))
        solar_range = solar_max - solar_min if solar_max > solar_min else 1.0
        print(f"Solar normalization: min={solar_min:.1f}, max={solar_max:.1f}, range={solar_range:.1f}")

        solar_values = []
        for i, (x, y, z) in enumerate(coords):
            if x < solar.shape[0] and y < solar.shape[1]:
                val = solar[x, y]
                if np.isnan(val):
                    solar_values.append(128)
                else:
                    normalized = int(255 * (val - solar_min) / solar_range)
                    solar_values.append(max(0, min(255, normalized)))
            else:
                solar_values.append(128)

    # Pack data compactly: [x, y, z, class] or [x, y, z, class, solar]
    voxels = []
    for i in range(len(coords)):
        entry = [int(coords[i][0]), int(coords[i][1]), int(coords[i][2]), int(int_classes[i])]
        if solar_values is not None:
            entry.append(solar_values[i])
        voxels.append(entry)

    result = {
        "dims": [int(nx), int(ny), int(nz)],
        "classes": {
            "0": "ground",
            "1": "wall",
            "2": "tree",
            "3": "road",
            "4": "building"
        },
        "hasSolar": solar_values is not None,
        "count": len(voxels),
        "voxels": voxels
    }

    return result


def main():
    print("=== VoxCity Export for Three.js ===")
    print(f"Input: {CACHE_DIR}")
    print(f"Output: {OUT_PATH}")
    print()

    voxcity, solar = load_data()
    coords, classes = extract_surface_voxels(voxcity)
    coords, classes = downsample_if_needed(coords, classes)

    data = build_json(coords, classes, solar, voxcity.shape)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nExported {data['count']:,} voxels to {OUT_PATH}")
    print(f"File size: {size_mb:.1f} MB")

    if size_mb > 5:
        print("WARNING: File exceeds 5MB. Consider further downsampling.")


if __name__ == "__main__":
    main()
