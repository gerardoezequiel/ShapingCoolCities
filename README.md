# Shaping Cool Cities

**Multi-Source Data Fusion for Modelling Urban Heat Mitigation Across Six European Cities**

The summer of 2022 killed over 61,000 Europeans from heat-related causes—the deadliest heatwave on record. Yet cities remain unprepared, not because data is scarce, but because translating observations into actionable cooling strategies remains unsolved.

This repository contains a complete analytical framework that bridges that gap: from satellite imagery and 3D urban models to interpretable predictions and intervention priorities.

---

## Why This Matters

European cities have never had more data about urban heat. Satellites capture surface temperatures at metre-scale resolution. Street-level imagery reveals pavement, vegetation, and shadow. 3D city models show building heights and street configurations. Yet planners facing decisions about where to plant trees, which streets to depave, or how to prioritise cooling budgets find little guidance in this abundance.

**Three findings from this research challenge prevailing assumptions:**

| Finding | Implication |
|---------|-------------|
| Thermal effects operate at **300-metre scales** | Cooling one plot achieves nothing if surroundings remain sealed. Most programmes target individual properties—the wrong scale. |
| **Surface permeability matters 3× more than vegetation** | The policy obsession with tree planting misallocates resources. De-sealing delivers greater cooling returns. |
| A **50% de-sealing threshold** marks a regime boundary | Below this, evaporative cooling becomes viable. Incremental improvements that fail to cross this threshold waste money. |

---

## What This Framework Does

The pipeline processes 6 regular-size study areas across six European cities (Amsterdam, Athens, Barcelona, Berlin, Madrid, Paris) through two phases:

### Phase 1: Multi-Source Feature Engineering
- **Satellite imagery** — Extract land surface temperature, NDVI, spectral indices, and impervious fraction from Landsat 8/9 via Google Earth Engine
- **3D morphology** — Generate sky view factor, solar irradiance, and canyon geometry from voxelised building models (VoxCity)
- **Street networks** — Compute centrality, connectivity, and demographic density from OpenStreetMap (Urbanity)
- **Computer vision** — Derive greenery, enclosure, and surface semantics from street-level imagery (GlobalStreetscapes)

These heterogeneous sources are spatially integrated at 30m resolution with climate-informed neighbourhood aggregations (150m, 300m buffers).

### Phase 2: Explainable Prediction & Intervention
1. **Predict heat risk** — Hierarchical XGBoost achieves R² = 0.841, RMSE = 0.85°C
2. **Explain why** — SHAP analysis reveals which features drive heating and can be modified
3. **Prioritise where** — Combine physical cooling potential with demographic vulnerability
4. **Quantify interventions** — Scenario analysis shows achievable cooling: **1.26°C mean reduction** in priority zones

![Methodology overview](results/Shapping_cool_cities_method.png)

---

## Key Results

### Feature Importance Hierarchy

| Category | Importance | Key Insight |
|----------|------------|-------------|
| Surface/spectral | **21%** | NDBI + impervious fraction dominate |
| Water proximity | **13%** | Cooling extends ~500m from water bodies |
| Vegetation | **7%** | Trees help, but less than assumed |

### Climate-Specific Patterns

- **Continental cities** (Paris, Berlin): Centre-periphery gradients; de-sealing urban cores most effective
- **Mediterranean cities** (Athens, Barcelona): Topography dominates; elevation-aware planning needed
- **Maritime cities** (Amsterdam): Water omnipresent; building density and materials matter most

---

## Repository Structure

```
├── src/                    # Python modules
│   ├── gee_pipeline.py     # Google Earth Engine feature extraction
│   ├── model_preprocessing.py  # Feature engineering & scaling
│   ├── hotspot_classifier.py   # Binary hotspot classification
│   ├── risk_mapping.py     # Vulnerability assessment
│   └── interventions.py    # Cooling scenario optimisation
│
├── notebooks/              # Analysis pipeline (run in order)
│   ├── 00_Study_areas.ipynb      # Define cities, generate grids
│   ├── 01_Buildings.ipynb        # EUBUCCO building features
│   ├── 02_GlobalStreetScapes.ipynb  # Street-level imagery
│   ├── 03_Urbanity.ipynb         # Demographics & network metrics
│   ├── 04_GoogleEarthEngine.ipynb   # LST, NDVI, spectral indices
│   ├── 05_VoxCity.ipynb          # 3D morphology (SVF, canyon ratio)
│   ├── 06_Feature_engineering.ipynb # Merge & derive features
│   ├── 07_Modelling_XGBoost.ipynb   # Train models, SHAP analysis
│   ├── 08_Risk_Hotspot.ipynb     # Hotspot & vulnerability mapping
│   └── 09_Interventions.ipynb    # Cooling scenarios
│
└── data/
    ├── 0-raw/              # External data sources
    ├── 1-processed/        # Processed features per source
    └── 2-model-ready/      # Final model input
```

---

## Installation

```bash
git clone https://github.com/gerardoezequiel/ShapingCoolCities.git
cd ShapingCoolCities
```

### Option A: Conda (recommended)
```bash
conda env create -f environment.yml
conda activate shapingcoolcities
```

### Option B: pip
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure Google Earth Engine
```bash
earthengine authenticate
```

Create a `.env` file with your GEE project ID:
```
GEE_PROJECT=your-project-id
```

---

## Quick Start

| Step | Notebook | Output |
|------|----------|--------|
| 1 | `00_Study_areas` | City grids at 30m resolution |
| 2 | `01-05` | Feature extraction from each data source |
| 3 | `06_Feature_engineering` | Merged, scaled model-ready dataset |
| 4 | `07_Modelling_XGBoost` | Trained models + SHAP importance |
| 5 | `08_Risk_Hotspot` | Priority zones weighted by vulnerability |
| 6 | `09_Interventions` | Cooling scenarios + recommendations |

---

## Data Sources

| Source | Description | Reference |
|--------|-------------|-----------|
| [EUBUCCO](https://eubucco.com/) | European building footprints & heights (202M buildings) | [Milojevic-Dupont et al. (2023)](https://doi.org/10.1038/s41597-023-02040-2) |
| [Urbanity](https://github.com/winstonyym/urbanity) | Street network metrics, demographics, centrality | [Yap et al. (2023)](https://doi.org/10.1038/s42949-023-00125-w) |
| [GlobalStreetscapes](https://github.com/ualsg/global-streetscapes) | Street-level imagery with 300+ semantic attributes | [Hou et al. (2024)](https://doi.org/10.1016/j.isprsjprs.2024.06.023) |
| [VoxCity](https://github.com/kunifujiwara/VoxCity) | 3D voxel morphology (SVF, solar irradiance) | [Fujiwara et al. (2025)](https://arxiv.org/abs/2504.13934) |
| Landsat 8/9 | Surface temperature & spectral indices | [Google Earth Engine](https://earthengine.google.com/) |
| OpenStreetMap | Street network & POIs | [OSM](https://www.openstreetmap.org/) |

---

## Citation

If you use this code or methodology, please cite:

```bibtex
@mastersthesis{martincarreno2026,
  author  = {Martín Carreño, Gerardo Ezequiel},
  title   = {Shaping Cool Cities:
             Multi-Source Data Fusion for Modelling Urban Heat Mitigation Across Six European Cities},
  school  = {University College London},
  year    = {2026},
  type    = {MSc Thesis}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

This research was conducted as part of the MSc Urban Spatial Science programme at the Centre for Advanced Spatial Analysis (CASA), University College London.
