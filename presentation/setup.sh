#!/bin/bash
# ============================================================
# Shaping Cool Cities — Presentation Setup
# ============================================================
# Run this from the ShapingCoolCities repo root:
#   chmod +x presentation/setup.sh && ./presentation/setup.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ASSETS_DIR="$SCRIPT_DIR/assets"
RESULTS_DIR="$(dirname "$SCRIPT_DIR")/results"
FIGURES_DIR="$RESULTS_DIR/figures"

echo "🔥 Setting up Shaping Cool Cities presentation..."
echo ""

# Create assets directory
mkdir -p "$ASSETS_DIR"

# Copy figures
echo "📋 Copying figures to presentation/assets/..."

# Methodology overview
cp "$RESULTS_DIR/Shapping_cool_cities_method.png" "$ASSETS_DIR/" 2>/dev/null && echo "  ✓ Methodology overview" || echo "  ✗ Methodology overview (not found)"

# Study grids
cp "$FIGURES_DIR/study_grids_30m.png" "$ASSETS_DIR/" 2>/dev/null && echo "  ✓ Study grids" || echo "  ✗ Study grids (not found)"

# LST global
cp "$FIGURES_DIR/gee_LST_mean_global.png" "$ASSETS_DIR/" 2>/dev/null && echo "  ✓ LST global map" || echo "  ✗ LST global map (not found)"

# VoxCity SVF
cp "$FIGURES_DIR/voxcity_svf_panel.png" "$ASSETS_DIR/" 2>/dev/null && echo "  ✓ VoxCity SVF" || echo "  ✗ VoxCity SVF (not found)"

# SHAP plot
cp "$FIGURES_DIR/modeling_shap/global/shap_target_uhi_raw_global.png" "$ASSETS_DIR/" 2>/dev/null && echo "  ✓ SHAP global" || echo "  ✗ SHAP global (not found)"

# Priority zones
cp "$FIGURES_DIR/priority_zones_spatial.png" "$ASSETS_DIR/" 2>/dev/null && echo "  ✓ Priority zones" || echo "  ✗ Priority zones (not found)"

# Vulnerability tiers
cp "$FIGURES_DIR/vulnerability_tiers_panels.png" "$ASSETS_DIR/" 2>/dev/null && echo "  ✓ Vulnerability tiers" || echo "  ✗ Vulnerability tiers (not found)"

# Temperature reduction
cp "$FIGURES_DIR/temperature_reduction_map.png" "$ASSETS_DIR/" 2>/dev/null && echo "  ✓ Temperature reduction" || echo "  ✗ Temperature reduction (not found)"

echo ""
echo "✅ Setup complete!"
echo ""
echo "📂 Files are in: $ASSETS_DIR/"
echo ""
echo "To preview locally:"
echo "  cd $SCRIPT_DIR"
echo "  python -m http.server 8000"
echo "  # Then open http://localhost:8000"
echo ""
echo "To deploy to GitHub Pages:"
echo "  1. Push the presentation/ folder to your repo"
echo "  2. Go to repo Settings → Pages"
echo "  3. Set source to 'Deploy from a branch'"
echo "  4. Select the branch and /presentation folder"
echo "  5. Your URL: https://gerardoezequiel.github.io/ShapingCoolCities/presentation/"
echo ""
echo "💡 Presenter tips:"
echo "  - Press 'S' to open speaker notes view"
echo "  - Press 'F' for fullscreen"
echo "  - Press 'O' for slide overview"
echo "  - Arrow keys or Space to navigate"
echo ""
