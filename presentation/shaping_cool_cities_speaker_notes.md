# SHAPING COOL CITIES — Speaker Notes (15-Minute Version)
## Urban AI Level 1 Capstone
### Jury: Or Aleksandrowicz (Technion) + Federico Fernandez (Urbanly)

---

## How to Use This Document

**SPEAK** = Exactly what you say out loud. Rehearse this verbatim.
**NOTES** = Depth you hold in reserve. Only use if a question touches the topic.
**⏱️** = Target time per slide. Running total tracks your 15-minute budget.
**⚠️ CUT** = Skip this slide if behind schedule. The presentation works without it.
**[PAUSE]** = Deliberate silence, minimum 2 seconds. Do not rush through these.
**[CLICK]** = Advance animation or slide.

**Total spoken time budget: ~14 minutes** (leaves 1 minute for natural pauses/transitions)

---
---

## SLIDE 1 — TITLE
**⏱️ 0:30 | Running: 0:30**

### SPEAK

Thank you. I'm Gerardo, I completed my MSc in Urban Spatial Science at UCL's Centre for Advanced Spatial Analysis. Today's question: how do we move from observing urban heat to prescribing cooling interventions — city by city, climate by climate?

### NOTES

- Dissertation title: "Shaping Cool Cities: A Multi-Source Machine Learning Framework for Urban Heat Prediction and Cooling Intervention Optimisation Across European Climates"
- Supervised by Dr Adam Dennett (CASA, UCL). Second marker: Dr Bonnie Buyuklieva.
- If asked about Urban AI connection: the diagnosis-to-prescription pipeline is what Urban AI frames as "from intelligence to action." The framework operationalises that framing for heat.

---

## SLIDE 2 — THE HOOK (61,000)
**⏱️ 0:45 | Running: 1:15**

### SPEAK

[PAUSE — let the number land]

Sixty-one thousand Europeans died from heat in the summer of 2022. The deadliest natural hazard event in European history. Not a flood, not a storm — heat.

And cities had more climate data available than ever before. Yet when planners asked "which streets should we depave?" — there was remarkably little useful guidance.

### NOTES

- Source: Ballester et al. 2023, Nature Medicine. 61,672 deaths across 35 countries, June-September 2022.
- This is surface UHI (SUHI), not canopy-layer UHI (CUHI). Landsat measures radiative skin temperature at surface level. CUHI is air temperature at 1.5-2m measured by weather stations — too sparse for city-scale spatial analysis.
- If asked "why not air temperature?": SUHI enables spatially comprehensive analysis at 30m. CUHI-SUHI correlation is imperfect, but neighbourhood-level relative rankings are preserved. Framework maps anomalies (relative hot/cool), so ordering holds even if absolute magnitude differs from air temperature.
- Previous worst: 2003 European heatwave, ~70,000 deaths but over a longer period and less precisely attributed.

---

## SLIDE 3 — THE DATA PARADOX
**⏱️ 0:30 | Running: 1:45**

### SPEAK

The challenge is no longer data availability. It's integration — combining complementary sources into guidance planners can act on. That's what this research does. A diagnosis-to-prescription pipeline across six European cities, three climate zones. Only open-source tools. No proprietary data, no black boxes.

### NOTES

- "Open-source" means: GEE (free academic access), EUBUCCO (CC-BY 4.0), GlobalStreetscapes (CC-BY 4.0), Urbanity (MIT licence), VoxCity (open access). Full pipeline on GitHub under MIT licence.
- "No black boxes" connects to SHAP explainability — every prediction decomposes into individual feature contributions. This matters for Or (needs to know *why* a block is hot to write a design brief) and Federico (product requires transparency for user trust and calibration).
- "Diagnosis-to-prescription" is deliberate framing: Stage 1 diagnoses (what's hot, why), Stage 2 detects (where heat meets vulnerability), Stage 3 prescribes (what interventions, how much cooling).

---

## SLIDE 4 — ROADMAP
**⏱️ 0:10 | Running: 1:55**
**⚠️ CUT if behind. Gesture and advance.**

### SPEAK

Quick roadmap. Problem, data, model, findings, implications.

### NOTES

- This slide exists as a visual anchor. Don't narrate it. Gesture at the structure and move on.
- If jury asks you to elaborate on any section, you know where everything lives.

---

## SLIDE 5 — THREE GAPS
**⏱️ 1:00 | Running: 2:55**

### SPEAK

Three gaps prevent cities from acting effectively.

First — integration. Most studies examine morphology, vegetation, or connectivity in isolation. But heat emerges from their interactions. How building arrangements channel winds determines whether a tree's cooling reaches pedestrians.

Second — transferability. What works in Berlin doesn't work in Barcelona. A tree-planting programme calibrated in Amsterdam produces a third of expected cooling in Athens. That's wasted public money.

Third — the action gap. ML models achieve impressive accuracy but function as black boxes. Planners don't need to know a neighbourhood will be hot — they need to know why, and what they can change.

[CLICK] Two research questions. What drives urban heat across contexts? And where should cities intervene to protect the most vulnerable?

### NOTES

- "A third of expected cooling": Schwaab et al. 2021, analysing 293 European cities. Identical tree species cool 8-12°C in Central Europe (Cfb climate) but only 0-4°C in Mediterranean (Csa). Mechanism: evapotranspiration constrained by vapour pressure deficit — dry air limits transpiration regardless of soil moisture.
- Integration gap references: Li et al. 2020 (morphology studies ignore vegetation obstruction), Li et al. 2024 (vegetation analyses assume uniform building contexts), Chenary et al. 2023 (network models omit surface roughness).
- Action gap: Camps-Valls et al. 2025 tripartite framework — quantification, understanding, communication. Most ML-UHI studies stop at quantification. This research addresses all three through SHAP decomposition (understanding) and scenario modelling (communication).
- If asked "why not physics-based models like ENVI-met?": computationally prohibitive across 40,344 cells and six cities. ENVI-met operates at street-section scale (metres, hours of compute per block). This framework screens at neighbourhood scale (30m, minutes of compute for all six cities). Complementary, not competing — ENVI-met should follow this framework for detailed design.

---

## SLIDE 6 — SIX CITIES
**⏱️ 0:45 | Running: 3:40**

### SPEAK

Six cities across a climate gradient. Amsterdam and Paris oceanic. Athens and Barcelona Mediterranean. Berlin and Madrid deliberately transitional — they test whether the framework handles climate ambiguity. Each gridded at 30-metre resolution, 40,344 cells total, roughly 6 square kilometres per city.

### NOTES

- Köppen classes: Amsterdam/Paris = Cfb (oceanic). Athens/Barcelona = Csa (Mediterranean). Berlin = Cfb/Dfb boundary. Madrid = Csa/BSk (semi-arid transition).
- 30m resolution matches Landsat native thermal band (Band 10, 100m resampled to 30m by USGS). MAUP sensitivity tested: 60m loses 17% variance (falls between building-scale radiative processes and neighbourhood-scale advective transport — misaligns with both). 90m recovers non-monotonically (aligns with broader advective scale).
- Equal-area approach: UHI intensity correlates with total urban extent (Oke 2017). Unequal study areas would confound morphological effects with coverage differences. Each ~6 km² captures the dense urban core.
- Grid alignment: Web Mercator (EPSG:3857), zoom level 14 tiles aligned with GlobalStreetscapes sampling framework.
- If Federico asks "why these six and not others?": maximise climatic variation within constraints of all five data sources. EUBUCCO limits to Europe. Adding non-European cities requires substituting EUBUCCO with OSM buildings (less accurate heights) or Google Open Buildings.
- Multi-seed stability: σ(R²) = 0.064 across 5 random seeds.

---

## SLIDE 7 — FIVE DATA SOURCES (FRAMEWORK)
**⏱️ 0:45 | Running: 4:25**

### SPEAK

The novelty isn't any single dataset — it's the integration. Five complementary sources fused into 118 features. Google Earth Engine gives us surface temperature. EUBUCCO provides 3D building morphology. Urbanity captures street network topology. GlobalStreetscapes adds the pedestrian perspective — vegetation at eye level, which satellites miss in urban canyons. And VoxCity ray-traces solar geometry.

All five are open. Any European city can replicate this tomorrow.

### NOTES

- Coverage by source: GEE 95-99%, VoxCity 92-99%, EUBUCCO 34-92% (Madrid lowest at 65.6% missing), GlobalStreetscapes 29-71% (Athens lowest at 71.2% missing), Urbanity 7-12% (but this is network node architecture — every intersection captured, not 7% of area).
- Missingness: city-specific median imputation with binary indicator flags. Model learns to treat imputed vs. observed values differently.
- Feature engineering pipeline: 165 base features → spatial lags at 150m (pedestrian scale) and 300m (neighbourhood/advective scale) → 189 candidates → Pearson correlation filter |r| > 0.92 → 118 final features. Six physics-informed derived features: urban heat trap index, vegetation cooling saturation (log-scaled), canyon sky openness, ventilation proxy (SVF × (1 − building coverage)), impervious-canopy balance, height-to-coverage ratio.
- Temporal mismatch caveat: GEE = summer 2024 composites. GlobalStreetscapes = undated imagery (various years). EUBUCCO = static footprints. VoxCity = static DSM/DTM. Pipeline assumes morphological features change slowly relative to seasonal thermal dynamics — true for buildings, less so for vegetation (new park vs. 15-year-old tree canopy).
- If Or asks about GVI sources: VoxCity GVI derived from voxel-based canopy volume. GlobalStreetscapes GVI derived from street-level panoramic segmentation. They capture vegetation from different perspectives (overhead vs. eye-level). Divergence itself is a finding — validates multi-source approach and supports intuition that pedestrian-perspective greenery matters differently than overhead canopy for experienced comfort.

---

## SLIDES 8-11 — INDIVIDUAL DATA SOURCES
**⚠️ CUT ALL FOUR. Skip directly to Slide 12 (VoxCity).**

**If you keep them, spend maximum 15 seconds each = 1 minute total. Here's what to say:**

### SLIDE 8 (GEE) — SPEAK (15 sec)
Landsat at 30m — our dependent variable. Summer composites, cloud-filtered, intra-city anomalies.

### SLIDE 9 (Urbanity) — SPEAK (15 sec)
Street network topology from OpenStreetMap. Connectivity determines how cooling propagates — Amsterdam's canal grid vs Athens's organic fabric.

### SLIDE 10 (GlobalStreetscapes) — SPEAK (15 sec)
10 million street-level images. From these we derive Green View Index — vegetation at eye level. Coverage varies 29-71% by city.

### SLIDE 11 (EUBUCCO) — SPEAK (15 sec)
3D building morphology. Canyon geometry, sky view factor, surface-to-volume ratios. Notice the dramatically different morphologies.

### NOTES (shared for slides 8-11)

- GEE processing: Landsat 8/9 Collection 2 Level 2. Summer months (June-August). Cloud masking via QA_PIXEL band. Compositing: per-pixel median. Anomaly = LST_cell − LST_city_mean. This isolates morphological heating from background climate.
- NDVI calculated from same Landsat scenes. Inverse LST-NDVI pattern visible but non-linear — log relationship, not linear. The model captures this.
- Urbanity metrics: betweenness centrality (ventilation corridors), meshedness (grid regularity), connectivity (intersection density). Amsterdam canal grid: high meshedness, regular spacing → cooling propagates ~3× further than Athens's organic fabric (low meshedness, irregular block sizes).
- GlobalStreetscapes: Mapillary + KartaView sources. Berlin 277K images, Paris only 14K. Coverage is systematic not random — commercial/tourist areas overrepresented, peripheral/low-income areas underrepresented. This is a known bias addressed in scope conditions.
- EUBUCCO: 6,500-8,500 buildings per study area. Heights, footprints, aspect ratios. Paris's uniform Haussmann fabric (18-25m uniform height) vs. Athens's irregular organic growth (4-30m mixed heights).

---

## SLIDE 12 — VOXCITY 3D
**⏱️ 1:00 | Running: 5:25** (or 6:25 if you kept slides 8-11)

### SPEAK

VoxCity converts open data into volumetric 3D models — buildings, canopy, terrain.

[CLICK] Six cities voxelised. Notice the different morphologies.

[CLICK] Solar irradiance on ground surfaces. Barcelona's grid distributes heat evenly. Athens's basin traps it.

[CLICK] Sky View Factor — how much sky each ground cell sees.

[CLICK] Green View Index — vegetation as pedestrians experience it.

This 30m grid tells the designer which blocks to focus on. When we decompose each cell with SHAP, it tells them whether to de-seal, shade, or plant — a brief for street-section design. Though configuring the actual street section requires sub-30m microclimate tools. These are complementary scales.

### NOTES

- VoxCity technical: 5m voxel resolution, aggregated to 30m grid cells. EPW weather files for solar simulation (15 June, 14:00 local time, TMYx 2007-2021 climatology). Ray-tracing accounts for building shadows and tree canopy transmittance.
- SVF = ratio of visible sky hemisphere to total hemisphere at ground level. Low SVF = deep canyons restricting longwave radiation loss at night → nocturnal heat retention. Dirksen et al. 2019 found non-linear SVF-temperature relationships.
- Solar irradiance was generated but used primarily for visualisation, not as a direct model feature. At 5m resolution aggregated to 30m, the shadow signal smooths. SVF served as the proxy that entered the model. Including direct solar irradiance as a feature is a legitimate gap and future improvement.
- **The "complementary scales" sentence is the bridge both jury members will probe.** This framework screens at neighbourhood scale (which blocks, 30m). Or's shade mapping methodology operates at street-section scale (how to configure, 5-15m). Federico's CityCompass operates at scenario comparison scale (what parameters to vary, policy interface). Three layers of a complete planning workflow.
- If Or asks "how does the voxel grid relate to what I'd draw as an architect?": the grid tells you Block X is hot because imperviousness is 94% and SVF is 0.3. That's a design brief: "this block needs de-sealing and shade intervention." How you configure the street section — tree species, paving material, canopy height, setback — that's architectural judgment operating below 30m resolution. The framework identifies *what to change*; the architect determines *how to change it*.
- Fallback if 3D visualisation doesn't load: static PNG images of voxelised cities + SVF panels stored as backup.

---

## SLIDE 13 — THREE-STAGE PIPELINE
**⏱️ 0:30 | Running: 5:55**

### SPEAK

Three stages. Regression: predict temperature anomalies for each cell, city mean subtracted. Classification: calibrated hotspot probabilities — and here the feature ranking shifts, with demographic vulnerability rising to the top. Scenario optimisation: 648 intervention combinations, each bootstrapped 500 times.

The model shows its work at every stage. The planner keeps the final call.

### NOTES

- Stage 1 target: LST_anomaly = LST_observed − LST_city_mean. Positive = relative hotspot, negative = cool island. City-demeaning enables cross-city comparison of morphological effects without confounding by background climate.
- Stage 2: binary hotspot = any cell >1 MAD above city median (MAD scaled by 1.4826 for normal equivalence ≈ +1σ). 1.5× minority class weighting because hotspots are only 15.8% of cells. Platt scaling converts XGBoost raw scores to calibrated probabilities. Threshold = 0.30 to ensure recall ≥ 60% — better to flag a non-hotspot than miss a real one.
- Stage 3: 3 depaving levels (30/40/50%) × 9 vegetation (10-50% in 5% steps) × 8 tree canopy (15-50% in 5% steps) × 3 albedo (0/10/20%) = 648 combinations. Bootstrap n=500. Extreme prediction filter: >5°C change removed (~21% of predictions, mostly peripheral/industrial zones; core urban ~8%).
- Why XGBoost not neural networks: (1) native SHAP TreeExplainer support for exact decomposition, (2) captures non-linear thresholds linear models miss, (3) handles mixed feature types without scaling. Not because it's highest-accuracy — interpretability is non-negotiable for planning applications.
- Hyperparameters: 200 Bayesian trials (Optuna), minimising 5-fold CV RMSE. Final: 820 estimators, max_depth=6, learning_rate=0.03, subsample=0.78, colsample_bytree=0.74.
- If Federico asks "why grid search not Bayesian optimisation for scenarios?": grid search maps the full response surface — reveals non-linear shape (the 30→40→50% regime boundary). Optimisation finds optima faster but doesn't reveal the landscape. For a planning tool, understanding the shape matters more than finding the peak. Multi-objective Pareto optimisation (cooling × cost × equity) would be the natural product extension.

---

## SLIDE 14 — MODEL ARCHITECTURE
**⏱️ 0:45 | Running: 6:40**

### SPEAK

Three models — global, Mediterranean specialist, oceanic specialist — blended with city-specific weights. The global model gets a negative weight: it's subtracted. The specialists dominate.

That blend cuts unexplained variance by 22% over a global-only model. The improvement comes entirely from respecting geographic context.

[PAUSE]

The architecture is the finding.

### NOTES

- Global-only R² = 0.795. Blended R² = 0.840. Improvement = (0.840−0.795)/(1−0.795) = 22% reduction in unexplained variance.
- Blend equation: pred_blended = β₀ + β₁·pred_global + β₂·pred_specialist. Coefficients optimised per city via least-squares on held-out validation fold.
- Negative global weight means: the global model captures universal patterns (sealed surfaces heat everywhere) but also introduces systematic errors from averaging across climates. Subtracting it corrects the specialists' residuals.
- SHAP built into every prediction: for each of 40,344 cells, SHAP decomposes the predicted anomaly into 118 feature contributions. "This cell is +2.1°C because imperviousness contributes +1.4, water distance +0.5, vegetation −0.3, building height +0.5." That's explainability — it converts a black-box prediction into a design brief.
- 118 features from: 165 base → spatial lags at 150m and 300m → 189 → correlation filter |r|>0.92 → 118. Six physics-informed derived features.
- MAUP sensitivity: 60m loses 17% variance. 90m recovers non-monotonically. 30m provides best balance.
- If asked to define terms quickly: R² = proportion of temperature variation explained (1.0 = perfect). SHAP = feature contribution receipt for each prediction. AUC = hotspot discrimination accuracy (1.0 = perfect, 0.5 = coin flip; ours = 0.911).

---

## SLIDE 15 — SPATIAL VALIDATION
**⚠️ CUT. Move directly to Slide 16. This is Q&A material.**

### SPEAK (only if kept — 20 seconds max)
Spatial validation: 5-fold cross-validation with 600m block grouping — the model never sees neighbours of test cells. Moran's I confirms minimal residual spatial autocorrelation. The model learns morphological relationships, not spatial patterns.

### NOTES

- 5-fold GroupKFold. Stratified by UHI quintile × built density tertile → 600m contiguous blocks.
- 600m block size exceeds 300m advective transport scale (Oke 2017) — ensures spatial independence between train and test folds.
- Moran's I on residuals: confirmed minimal remaining autocorrelation. The city-demeaning (anomalies not absolute temperatures) removes the large-scale spatial trend. The 300m buffer aggregation is physically grounded in advective transport.
- Variance weighting by city: w = σ²_city / mean(σ²). Cities with wider temperature distributions contribute proportionally more to loss function.
- Multi-seed validation: 5 random seeds, σ(R²) = 0.064. Results stable.
- MAUP: tested 30m, 60m, 90m. 30m best performance. 60m falls between two physical scales. 90m partial recovery aligns with broader advective scale.
- If Federico asks "how do you prevent spatial data leakage?": three defences. (1) 600m block grouping exceeds advective scale. (2) Anomaly target removes large-scale trends. (3) 300m buffer features physically grounded, not arbitrary radii. Moran's I on residuals is the empirical check.

---

## SLIDE 16 — PERFORMANCE RESULTS
**⏱️ 0:45 | Running: 7:25**

### SPEAK

R² of 0.93 in Paris and Barcelona, down to 0.73 in Athens. The spatial patterns match reality — centre-periphery gradient in Paris, coastal cooling in Barcelona, basin trapping in Athens.

Honest exception: blending actually hurts Athens. The topographic basin overrides the climate-zone correction. And Madrid's drought stress introduces dynamics beyond morphological control. These are scope conditions, not failures.

But model performance isn't the finding. The finding is what the model reveals about what drives heat.

### NOTES

- Full city-level R²: Paris 0.932, Barcelona 0.926, Amsterdam 0.880, Athens 0.730, Berlin 0.727, Madrid 0.718.
- RMSE: overall 0.85°C. City range: Paris 0.68°C to Madrid 1.12°C.
- Athens blending: R² drops 0.751→0.730 (−2.8%). Topographic basin creates katabatic flows (cold air drainage from surrounding mountains) and thermal inversions that override climate-zone correction. Scope condition: basin cities need terrain-stratified sub-models.
- Madrid: semi-arid drought stress creates soil moisture dynamics that morphological features can't capture. Vegetation effectiveness depends on irrigation — which isn't in the model. R² = 0.718 is still usable for screening but lower confidence.
- Target variable distributions: Barcelona σ=3.13°C, Madrid σ=2.89°C (widest — easiest to predict). Amsterdam σ≈1.2°C, Athens σ≈1.2°C (narrowest — hardest to predict).
- If Or asks "does R²=0.73 in Athens mean the model is wrong there?": it means 27% of variance is unexplained — likely terrain effects, sea breeze channelling, and thermal inversions the morphological features don't capture. The neighbourhood-level relative rankings are still informative for screening. Athens would benefit from a terrain-enhanced model variant.

---

## SLIDE 17 — SHAP HIERARCHY (KEY SLIDE)
**⏱️ 1:45 | Running: 9:10**

### SPEAK

[SLOW DOWN — this is the central finding]

This is the SHAP analysis. Each dot is one of 40,344 grid cells. What it reveals is a clear hierarchy that challenges planning assumptions.

Surface characteristics — impervious fraction, built-up index — account for 21% of total feature importance. Water features: 13%. And vegetation — tree canopy, green view index — just 7%.

[PAUSE — let the 3:1 ratio land]

Surface permeability matters roughly three times more than vegetation for cooling. The physics: natural surfaces partition 30-50% of solar radiation into latent heat through evapotranspiration. Seal that surface and the energy goes directly into warming the atmosphere.

This doesn't mean trees don't matter — they provide shade, biodiversity, air quality. But if the primary goal is temperature reduction, de-sealing is the stronger lever. And that's not how most cities invest.

### NOTES

- SHAP methodology: TreeExplainer provides exact Shapley values for tree ensembles. Mean absolute SHAP value (|SHAP|) measures average contribution magnitude across all cells. Category aggregation (surface, water, vegetation, morphology, network, demographic) is more robust than individual feature rankings because correlated features redistribute importance within categories.
- Top individual features: NDBI (normalised difference built-up index) |SHAP|=0.321, impervious_300m=0.289, water_distance=0.245, NDVI=0.198, building_density=0.176.
- Physics of de-sealing: impervious surfaces have Bowen ratio >5 (nearly all energy into sensible heat). Permeable vegetated surfaces: Bowen ratio 0.5-1.5 (30-50% into latent heat). Transitioning from 95% to 50% imperviousness shifts the energy balance fundamentally — this is the physical basis for the regime boundary.
- 300m optimal radius: buffer optimisation showed impervious effects peak at 300m (SHAP=0.321 at 300m vs. 0.085 at 90m, 0.067 at 150m). Aligns with Oke 2017 advective transport scale under 2-3 m/s winds.
- Schwaab et al. 2021: 293 European cities. Trees cool 8-12°C in Cfb but 0-4°C in Csa. Mechanism: evapotranspiration efficiency depends on vapour pressure deficit. Dry Mediterranean air limits transpiration regardless of soil moisture.
- If Or asks "how does permeability interact with canyon geometry?": SHAP dependence plots show interaction. High imperviousness hurts most when SVF is also high (open plazas — full solar exposure, no shade, no evaporative pathway). In deep canyons (low SVF), shadow already reduces surface heating, so permeability matters less. **Design rule: open plazas → de-seal first. Narrow streets → shade first.**
- If Or asks "does the framework distinguish types of permeability?": No. Permeable paving, rain gardens, pocket parks, bioswales all register as reduced imperviousness. The model captures the thermal effect of resulting surface composition, not the design typology. A rain garden introduces vegetation and water retention; permeable paving maintains hardscape function. Translating "reduce imperviousness by 50%" into specific street-section design is the architect's judgment — the handoff point between this framework and Or's expertise.
- SHAP caveat: assumes feature independence in attribution, but urban variables correlate (NDBI and impervious fraction r=0.76). Category-level hierarchy (surface > water > vegetation) is robust. Individual feature redistribution within categories has uncertainty. Correlated features share importance — true individual effect may be larger for any one feature.

---

## SLIDE 18 — GEOGRAPHIC SHIFT
**⏱️ 0:45 | Running: 9:55**

### SPEAK

Same model, same features, but the hierarchy fundamentally shifts by climate. Mediterranean cities: albedo and built-up index dominate. Temperate cities: impervious fraction.

Athens's basin traps circulation. Amsterdam's canals flatten water proximity because 90% of cells are already near water. Each city needs a different strategy. One-size-fits-all EU guidance ignores this.

### NOTES

- Cross-city SHAP variation: vegetation importance varies 3× between Mediterranean (low — moisture-constrained) and temperate (higher — evapotranspiration effective). This is not an intercept shift — it's a fundamentally different feature interaction structure. That's why hierarchical blending (separate specialist models) outperforms a single global model with climate dummies.
- Continental breakdown: Oceanic (Amsterdam/Paris) — impervious fraction dominant, water effective, vegetation moderate. Mediterranean (Athens/Barcelona) — albedo dominant, built-up index high, vegetation constrained by moisture. Transitional: Berlin follows oceanic pattern. Madrid shows unique drought-stress amplification.
- Amsterdam canal effect: 90% of cells within 200m of water. Water distance SHAP nearly flat — no predictive power because there's no variation. Additional water infrastructure has diminishing returns. Primary lever shifts to building materials and surface permeability.
- Athens elevation: topographic position (elevation within basin) has SHAP=0.12, exceeding several morphological features. Basin floor traps heat; hillsides receive katabatic drainage. Terrain is a first-order control that other cities don't face.
- If asked "what does this mean for EU adaptation policy?": current frameworks (European Climate Pact, Covenant of Mayors) provide generic guidance — "increase urban greening," "reduce heat island effect." The geographic shift evidence argues for climate-zone-specific toolkits: different intervention priorities, different budget allocations, different performance targets. A tree-first strategy makes sense in Berlin; a de-sealing-first strategy makes sense in Barcelona.

---

## SLIDE 19 — HOTSPOT CLASSIFICATION
**⚠️ CUT. Merge key content into Slide 21 delivery. Skip to Slide 20 or 21.**

### SPEAK (only if kept — 20 seconds)
We classify hotspots using calibrated probabilities. AUC = 0.911. Risk concentrates differently — Athens nearly a third of cells, Paris under 8%.

### NOTES

- Hotspot definition: >1 MAD above city median, scaled by 1.4826 for Gaussian equivalence (~+1σ). More robust to outliers than standard deviation.
- XGBoost classification with 1.5× minority class weight (hotspots = 15.8% of cells).
- Platt scaling: logistic regression on held-out predictions to calibrate probabilities. When model outputs 0.7, empirical frequency of hotspots is genuinely ~70%.
- Threshold = 0.30 (sensitivity-tuned). Ensures recall ≥ 60%. Precision = 56.3%, Recall = 68.0%, F1 = 0.616. Trade-off: flag some non-hotspots rather than miss real ones.
- AUC = 0.911 overall. Fold-level: 0.949, 0.924, 0.932, 0.894, 0.858. Madrid Fold 5 = 0.511 — near random. Scope condition: semi-arid instability in Madrid's temperature distribution makes classification unreliable in some spatial folds.
- Hotspot prevalence: Athens 30.8%, Amsterdam 20.5%, Berlin 13.3%, Madrid 11.6%, Barcelona 11.6%, Paris 7.2%.

---

## SLIDE 20 — SPATIAL RISK TIERS
**⚠️ CUT. This is a visual transition — advance through it in 5 seconds.**

### SPEAK (5 seconds if shown)
Risk tiers across all six cities. Severe heat concentrates in dense cores.

### NOTES

- Five tiers: low, moderate, high, very high, severe. Based on predicted probability quintiles within each city.
- Visual patterns: Athens shows contiguous severe-tier zones across basin floor. Amsterdam's risk is more distributed (canal network fragments heat islands). Paris shows clear centre-periphery gradient.
- These tiers feed into priority zone scoring (Slide 21-22). Priority = f(heat risk 40%, vulnerability 35%, cooling potential 25%).

---

## SLIDE 21 — VULNERABILITY PIVOT
**⏱️ 1:00 | Running: 10:55**

### SPEAK

When we shift from predicting temperature to classifying hotspots, something important happens. The feature ranking inverts. Child and elderly density displace morphological features as the top predictors. Areas with high vulnerability have 2.3 times higher hotspot probability.

The model captures a spatial co-occurrence: vulnerable populations concentrate in older, denser fabric with less green space. The heat-trapping morphologies.

Priority zones weight vulnerability at 35%, heat risk at 40%, cooling potential at 25%. Across all cities: 15,734 cells flagged. Athens: 78%. Madrid: just 21%.

### NOTES

- Feature ranking inversion: child density |SHAP| = 0.182, total population density = 0.156. Impervious surface drops to third (0.089) in classification model. This inversion means: for predicting *where* heat is, surface properties dominate. For predicting *where heat is dangerous*, demographics dominate.
- Physiology: children have higher surface-area-to-mass ratio (absorb heat faster), lower sweat rates, less behavioural adaptation (don't self-regulate exposure). Elderly: chronic conditions, medication effects (diuretics, beta-blockers impair thermoregulation), reduced mobility limits escape. Source: Kovats and Hajat 2008.
- Vulnerability tier construction: z-scored within each city (so local patterns emerge, not cross-city income differences). Low = z ≤ 0, Medium = 0 < z ≤ 0.8, High = z > 0.8. Combined index = max(children_z, elderly_z). Flags cells where either group is disproportionately present.
- Priority zone composition: total 15,734 cells (39.0%). Athens 5,248 (78.2%), Paris 2,875 (43.7%), Amsterdam 2,422 (35.6%), Berlin 2,100 (30.2%), Barcelona 1,684 (25.3%), Madrid 1,405 (20.9%).
- Priority zone characteristics vs citywide: mean UHI anomaly +0.73°C vs −0.25°C, 93.6% impervious vs 77.2%, 3.0% tree canopy vs 7.2%. These are the areas with most heating and most room for intervention.
- If asked "why those specific weights (40/35/25)?": analytical judgment, not empirical optimum. Heat risk largest because it's the direct target variable. Vulnerability substantial because exposure × adaptive capacity determines mortality (Harlan et al. 2006). Cooling potential as feasibility modifier — no point prioritising a zone with no intervention headroom. Sensitivity testing: rankings stable across ±10% weight variation. Different municipalities could weight differently — that's a democratic decision.
- Madrid 20.9%: fewest priority zones because existing tree canopy (21.7%) already buffers many areas. Semi-arid climate means the *hottest* areas aren't necessarily the most morphologically improvable.

---

## SLIDE 22 — 648 SCENARIOS / REGIME BOUNDARY
**⏱️ 1:15 | Running: 12:10**

### SPEAK

648 intervention combinations across four levers. Every optimal strategy converges on 50% de-sealing.

The response is non-linear. 30% depaving delivers 49% of maximum cooling. 40% delivers 71%. But 50% delivers 100%. That steep jump marks a regime boundary — below 50% sealed, enough permeable area enables evapotranspiration.

Maximum impact: all levers at max, 1.27 degrees cooling. Cost-effective strategy — 50% depaving with modest vegetation and 20% trees — achieves 1.20 degrees. Those confidence intervals overlap — statistically indistinguishable.

[PAUSE]

That's 95% of the benefit with substantially fewer resources.

### NOTES

- 648 = 3 depaving (30/40/50%) × 9 vegetation (10-50% in 5% steps) × 8 tree canopy (15-50% in 5% steps) × 3 albedo (0/10/20%).
- Maximum impact: 1.27°C (95% CI: 1.23-1.31°C). Cost-effective: 1.20°C (95% CI: 1.16-1.24°C). CIs overlap → not statistically different at α=0.05.
- "Cost-effective" is a proxy: de-sealing during routine maintenance cycles is cheaper than establishing mature tree canopy (15-20 years to maturity, irrigation requirements, ongoing maintenance). No actual €/m² cost function in the framework — that's a limitation and future work.
- Regime boundary physics: below 50% sealed, sufficient permeable surface enables 30-50% of absorbed solar energy to redirect from sensible to latent heat via evapotranspiration. Above 50%, evaporative pathways are too fragmented for continuous cooling. Aligns with "sponge city" threshold concepts — same intervention addresses both flood and heat resilience.
- Bootstrap uncertainty: n=500 iterations. Scenario uncertainty = ±1.0-1.1% across strategies. Extreme filter: predictions >5°C change removed (~21% of all predictions, ~8% in core urban areas — rest are peripheral/industrial where extrapolation is most severe).
- If Federico asks about counterfactual validity: legitimate concern. Model learns correlations from current configurations, not causal consequences of changing configurations. Setting imperviousness from 95% to 50% is extrapolation — the model has limited training examples of 50%-impervious cells in dense urban cores. The extreme filter is a crude safeguard. True validation requires field experiments: pilot de-sealing projects with before-after temperature monitoring. Predictions are hypotheses, not guarantees. This aligns with Federico's philosophy of "calibrated AI, not oracle."
- If Federico asks "why not multi-objective optimisation?": grid search reveals full response landscape. Bayesian or genetic algorithms find optima faster but don't show the non-linear shape. For planning, the shape matters — the 30→40→50% jump is the finding, not just the optimum. Future product: Pareto front across cooling × cost × equity × feasibility.
- If Or asks "is 50% realistic in heritage European cores?": probably not everywhere. Underground infrastructure, heritage protections, property rights constrain. But the non-linear response means even 30% delivers 49% of benefit — meaningful even where 50% is impossible. And de-sealing can phase into routine infrastructure maintenance (street resurfacing cycles, utility upgrades).
- If Or asks "does the 50% threshold shift with canyon geometry?": threshold emerges from global model pooling all morphologies. SHAP dependence plots suggest variation — Mediterranean cities likely need lower thresholds (higher solar loads demand more evaporative compensation). Threshold is central tendency, not universal constant. Local calibration would refine for specific canyon geometries.

---

## SLIDE 23 — CITY-SPECIFIC COOLING
**⏱️ 0:45 | Running: 12:55**

### SPEAK

Same interventions, six different outcomes. Athens gains most — 1.45 degrees — because it has the most headroom: 95% impervious. Amsterdam gains least — 0.92 degrees — because 90% of cells are already near water. The canal network caps additional cooling.

Each city needs a tailored strategy. De-sealing is the anchor everywhere, but the supporting interventions differ.

### NOTES

- Full city cooling: Athens 1.45°C (95.2% impervious, 78.2% priority), Barcelona 1.31°C (94.7%), Paris 1.22°C (94.1%), Berlin 1.08°C (93.5%), Amsterdam 0.92°C (92.8%), Madrid 0.89°C (91.3%, only 20.9% priority).
- Athens: Attica Basin traps heat (scope condition for model) but also means interventions have maximum impact — nowhere for additional heat to go, so removing sources yields big returns. Sea breeze from Saronic Gulf creates corridors that de-sealing can amplify.
- Barcelona: coastal-hill interface creates sea-breeze corridors. De-sealing in these corridors amplifies natural ventilation cooling. Eixample block interiors are prime de-sealing targets — current courtyards often paved.
- Paris: strong centre-periphery gradient. Core de-sealing extends Seine corridor cooling into dense Right Bank neighbourhoods. Haussmann's uniform canyon geometry means predictable intervention responses.
- Berlin: polycentric structure — no single heat core, multiple distributed centres. Distributed intervention strategy needed. Impervious fraction is dominant driver across all sub-centres.
- Amsterdam: water proximity saturated as a predictor. 90% within 200m. Additional water infrastructure has diminishing returns. Primary lever: building material albedo and surface permeability in non-canal areas.
- Madrid: fewest priority zones (20.9%). Existing tree canopy (21.7%) already moderates many areas. Semi-arid climate means vegetation interventions need drought-tolerant species — water-intensive greening counterproductive (Schwaab findings). Albedo modification (cool roofs/pavements) may be more effective than vegetation in this climate.

---

## SLIDE 24 — DE-SEALING CO-BENEFITS
**⚠️ CUT. Deliver the key point in one sentence within Slide 22.**

**If you say it: "De-sealing doesn't mean stop planting trees. It means rebalancing — invest in the surface beneath, not just the canopy above."**

### NOTES

- Co-benefits of de-sealing: restores evaporative cooling (heat), improves stormwater infiltration (flooding), enables soil biota (biodiversity), recharges aquifers (water security). Single intervention, four resilience domains. "Sponge city" framework alignment.
- Cost-effective strategy still includes 20% tree canopy. Trees provide irreplaceable shade, biodiversity habitat, air quality improvement (PM2.5, O3 uptake), aesthetic/psychological wellbeing. De-sealing cannot replace these functions. The argument is priority ordering for budget allocation, not elimination of tree planting.
- Amsterdam diminishing returns: 90% of cells within 200m of water, yet city still has hotspots. Once water proximity saturates as a cooling source, intervention lever shifts to surface materials and permeability. This is a practical insight for canal cities — "more water" is not the answer when you're already a canal city.
- Eyni et al. 2025: independently confirmed diversified depaving outperforms tree-only strategies for reducing heat-related health disparities. Corroborates the SHAP hierarchy from a health equity angle.

---

## SLIDE 25 — GREEN GENTRIFICATION
**⏱️ 0:30 | Running: 13:25**
**⚠️ CUT if behind. Compress to one sentence: "Green gentrification is real — technical optimisation without equity safeguards risks improving places while displacing people."**

### SPEAK

The framework identifies where to intervene. It cannot prevent green gentrification — where environmental improvements trigger property value increases that displace the residents the interventions aim to protect. Technical optimisation without anti-displacement policies is incomplete. That's a governance question this analysis raises but cannot answer.

### NOTES

- Anguelovski et al. 2019: documented mechanisms — greening → property value increase → rent escalation → displacement of low-income renters. Barcelona's Superblocks and New York's High Line both documented displacement effects alongside environmental benefits.
- Hoffman et al. 2020: historically redlined US neighbourhoods experience surface temperatures up to 4°C higher than non-redlined districts. Structural racism encoded in urban morphology. European equivalents: social housing peripheries, immigrant-concentration districts.
- Policy instruments: rent stabilisation/control, community land trusts (CLTs), participatory budgeting for intervention prioritisation, inclusionary zoning, anti-displacement ordinances triggered by environmental investment thresholds.
- Framework contribution: vulnerability weighting (35%) ensures priority zones are where vulnerable people live now. But the weighting can't prevent market dynamics after intervention. Policy safeguards must be concurrent with physical interventions, not sequential.
- If asked about Latin American context (Federico): green gentrification dynamics amplified in informal settlements. Favela upgrading programmes in Rio, Medellín's green corridors — both documented gentrification effects. Framework methodology transfers; equity safeguards need local adaptation.

---

## SLIDE 26 — THREE CHALLENGES
**⚠️ CUT. You've already made all three points. If kept, deliver in 30 seconds max.**

### SPEAK (30 seconds if kept)

Three challenges. Heat operates at 300-metre neighbourhood scales — policy targets individual parcels. De-sealing outperforms tree planting 3-to-1 but gets a fraction of greening budgets. And no intervention works the same way in two cities. Geographic contingency is the signal.

### NOTES

- Scale mismatch: Oke 2017 documents advective transport under 2-3 m/s winds propagating cooling 300-500m. Isolated parcel interventions provide localised shade but minimal area-wide cooling. Framework identifies contiguous zones where coordinated intervention crosses neighbourhood-scale effectiveness threshold. "Heat doesn't respect cadastral boundaries. This is not a technical gap — it is an institutional failure."
- Wrong priority: surface permeability 21% SHAP vs vegetation 7%. De-sealing outperforms tree planting for temperature reduction, yet receives a fraction of municipal greening budgets. Most EU adaptation plans lead with tree planting. Eyni et al. 2025 corroborates independently.
- No universal template: Schwaab et al. — 8-12°C cooling in Cfb, 0-4°C in Csa. Same species, same canopy, 3× different thermal effect. EU Covenant of Mayors provides generic adaptation guidance. Geographic contingency demands climate-zone-specific intervention toolkits.
- "Geographic contingency is not noise to be averaged away. It is the signal." — if you use this line, use it once. It's the thesis statement.

---

## SLIDE 27 — SCOPE CONDITIONS
**⏱️ 0:30 | Running: 13:55**

### SPEAK

Two honest limitations. This captures daytime heat at 10:30 AM — most mortality occurs at night. The morphological drivers also trap heat at night, so the screening is useful, but nocturnal data should complement before final decisions. And the 50% threshold awaits field validation. These are predictions, not proven causal effects.

The silver lining: satellite features alone dominate SHAP rankings. A GEE-only pipeline captures most of the signal. Coverage beats sophistication.

### NOTES

- Landsat overpass: 10:30 local solar time. Peak surface temperature typically occurs 13:00-15:00. Peak mortality risk occurs overnight (inability to thermoregulate during sleep). Framework maps 10:30 relative anomalies — neighbourhood rankings are preserved (cells hot at 10:30 are generally hot at 15:00 and retain more heat overnight). But absolute magnitude of surface-to-air relationship varies with wind speed, humidity, and canyon geometry.
- ECOSTRESS: finer temporal sampling (multiple times of day including nighttime) but coarser spatial resolution and inconsistent revisit schedule. Could complement Landsat for nocturnal validation but isn't a drop-in replacement.
- Counterfactual validity: model learns statistical associations between current configurations and temperatures. Modifying features to values underrepresented in training data (e.g., setting 95% impervious to 50% in dense urban core) is extrapolation. The extreme-prediction filter (>5°C) removes obvious failures. But rigorous validation requires pilot projects with before-after monitoring. "The predictions are hypotheses, not guarantees." Federico will respect this framing.
- Coverage bias: GlobalStreetscapes 29-71% by city. Systematic, not random — street-level platforms historically prioritise commercial, tourist, and higher-property-value areas. Predictions least reliable where vulnerability concentrates (peripheral low-income areas). Planning applications should prioritise field validation in underrepresented areas.
- GEE dominance: satellite-derived features (NDBI, impervious fraction, NDVI, albedo) carry ~21% combined SHAP importance with 95-99% spatial coverage. Heavily-imputed sources (GlobalStreetscapes, EUBUCCO in some cities) contribute modestly. Practical implication: a city without VoxCity or EUBUCCO can still run a meaningful screening analysis. GEE alone + XGBoost + SHAP = viable minimum pipeline.
- "Coverage beats sophistication": This is the operational takeaway for Federico. A product built on GEE + open buildings data covers most cities globally. Adding VoxCity/GlobalStreetscapes/EUBUCCO improves predictions but isn't required for a useful first pass.
- If asked about anthropogenic heat (traffic, HVAC, cooking, industry): absent from framework. Likely explains residual errors in commercial/industrial zones. Can add 20-50 W/m² in dense urban cores — comparable in magnitude to moderate de-sealing effects. Including traffic density and building energy use data would improve commercial-area predictions but spatially comprehensive datasets don't exist across six cities.
- If asked about Global South transferability: methodology transfers (multi-source integration, SHAP, vulnerability weighting) but requires local recalibration. EUBUCCO is Europe-only. Substitutes: OSM buildings (less accurate heights), Microsoft Building Footprints (Africa, SE Asia), Google Open Buildings. GEE features work globally. Extension requires local data partnerships and particular attention to informal settlement morphologies (different from European building stock).

---

## SLIDE 28 — CLOSING
**⏱️ 0:45 | Running: ~14:40**

### SPEAK

[SLOW, DELIBERATE — three clicks, three beats]

[CLICK] What works in Amsterdam fails in Athens. That difference is not a problem. It is the finding.

[PAUSE 2 seconds]

[CLICK] Open tools. Open data. Every city can act on this tomorrow. The barrier is not technical.

[PAUSE — CLICK — DO NOT SPEAK. Let the audience read the final line. Hold 3 full seconds of silence.]

Thank you. Happy to discuss.

### NOTES

- The third click reveals: "It is a question of will." (or equivalent closing line). Let it land in silence. Do not narrate it. Do not add caveats. Do not mention Urban AI's portfolio. End clean.
- If jury opens with general "what would you do differently?": (1) Add nocturnal thermal dynamics via ECOSTRESS integration. (2) Higher-resolution sub-block modelling for intervention design — combining 30m screening with street-section microclimate simulation (ENVI-met, RayMan, or Or's shade tools). (3) Longitudinal field validation through pilot monitoring programmes. (4) Cost function integration for true cost-effectiveness (€/°C/m²).
- If asked about connection to Urban AI learning: Or's shade mapping = design-scale complement to neighbourhood screening. Federico's CityCompass scenario comparison = product paradigm for what this framework could become. Extreme Heat module's social autopsy framing (Klinenberg's 1995 Chicago analysis). Nadina Galle's blue-green infrastructure module contextualises vegetation findings within broader ecological urbanism.
- If asked "where do you see yourself applying this?": intersection of spatial data science and urban policy. Framework is a prototype for a planning decision-support service. Open-source repo is step one. Product path: web interface where planner selects city → sees priority zones → adjusts intervention sliders → gets predicted cooling with equity overlays. Federico's journey (UrbanSim research code → Synthicity startup → Autodesk acquisition → Urbanly/CityCompass product) is the template.
- If asked to compare with existing city heat action plans: most emphasise tree planting and reactive cooling centres (libraries, pools). Framework argues for preventive de-sealing at neighbourhood scale. Barcelona's Superblocks partially aligns with de-sealing logic. Paris's Plan Climat emphasises canopy coverage targets. Berlin's Stadtentwicklungsplan Klima incorporates ventilation corridors (connects to network connectivity findings). A systematic policy alignment analysis would strengthen practical contribution — potential follow-up project.

---

## SLIDES 29-30 — REFERENCES + THANK YOU

### SPEAK (Slide 29 — 5 seconds)
Key references are listed here. Full bibliography with 100+ entries is on the GitHub repository.

### SPEAK (Slide 30)
Thank you. Happy to discuss any aspect.

### NOTES
- GitHub repo: MIT licence. Includes full pipeline code, feature engineering scripts, model training, SHAP analysis, scenario generation. README has reproduction instructions.
- If anyone asks for slides: available to share with references and BibTeX file included.

---
---

# TIMING SUMMARY — RECOMMENDED PATH

## Tight 15-Minute Route (cut slides marked ⚠️)

| Slide | Content | Time | Running |
|-------|---------|------|---------|
| 1 | Title | 0:30 | 0:30 |
| 2 | 61,000 hook | 0:45 | 1:15 |
| 3 | Data paradox | 0:30 | 1:45 |
| *4* | *Roadmap — skip* | *0* | *1:45* |
| 5 | Three gaps | 1:00 | 2:45 |
| 6 | Six cities | 0:45 | 3:30 |
| 7 | Five sources framework | 0:45 | 4:15 |
| *8-11* | *Individual sources — skip* | *0* | *4:15* |
| 12 | VoxCity 3D | 1:00 | 5:15 |
| 13 | Pipeline | 0:30 | 5:45 |
| 14 | Model architecture | 0:45 | 6:30 |
| *15* | *Spatial validation — skip* | *0* | *6:30* |
| 16 | Performance | 0:45 | 7:15 |
| 17 | SHAP hierarchy | 1:45 | 9:00 |
| 18 | Geographic shift | 0:45 | 9:45 |
| *19-20* | *Hotspot detail — skip* | *0* | *9:45* |
| 21 | Vulnerability pivot | 1:00 | 10:45 |
| 22 | Regime boundary | 1:15 | 12:00 |
| 23 | City-specific cooling | 0:45 | 12:45 |
| *24* | *Co-benefits — skip* | *0* | *12:45* |
| 25 | Equity (compressed) | 0:30 | 13:15 |
| *26* | *Three challenges — skip* | *0* | *13:15* |
| 27 | Scope conditions | 0:30 | 13:45 |
| 28 | Closing | 0:45 | 14:30 |

**Buffer: 30 seconds for natural pauses and transitions.**

## If You Keep Everything (30 slides, no cuts)

Estimated: ~25-28 minutes. You will not finish. The jury will stop you around slide 18-20, before you reach the intervention findings, the regime boundary, or the closing. Your strongest material (slides 17, 22, 28) will be rushed or lost.

**Cut confidently. Everything cut lives in these NOTES, ready for Q&A.**
