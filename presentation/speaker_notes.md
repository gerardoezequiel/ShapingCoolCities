# Speaker Notes — Shaping Cool Cities (15-Minute Delivery)

**Total spoken words: ~1,500** (~11.5 min at 130 wpm + pauses/clicks/bridges = ~15 min)

**Structure:** Each slide has spoken text first, then `[Q&A DEPTH]` / `[Q&A REFERENCE]` / `[Q&A GLOSSARY]` blocks with detailed backup material for questions. Delivery cues in `[BRACKETS]` are stage directions, not spoken.

---

## Slide 1: TITLE

Thank you for having me. I'm Gerardo, I completed my MSc in Urban Spatial Science at The Bartlett, UCL's Centre for Advanced Spatial Analysis. How do we translate the abundance of urban data into guidance cities can act on? That's the question this research set out to answer. Specifically: how do we move from observing heat to prescribing cooling interventions, city by city?

[Q&A REFERENCE: Full title: "Shaping Cool Cities: A Multi-Source Machine Learning Framework for Urban Heat Prediction and Cooling Intervention Optimisation Across European Climates." Supervised by Dr Adam Dennett (CASA, UCL). Second marker: Dr Bonnie Buyuklieva. Urban AI connection: the diagnosis-to-prescription pipeline is what Urban AI frames as "from intelligence to action" — this framework operationalises that framing for heat.]

---

## Slide 2: THE HOOK - 61,000

[PAUSE — let the number land]

Sixty-one thousand people died from heat across Europe in the summer of 2022. The deadliest natural hazard event in European history. Not a flood, not a storm. Heat. Yet when planners asked "where should we plant trees?" or "which streets should we depave?" there was remarkably little useful guidance.

[Q&A REFERENCE: Ballester et al. 2023 — 61,672 deaths across 35 countries, June-September 2022. Previous worst: 2003 European heatwave (~70,000 deaths but over longer period, less precisely attributed). This is surface UHI (SUHI), not canopy-layer UHI (CUHI). Landsat measures radiative skin temperature at surface level. CUHI is air temperature at 1.5-2m measured by weather stations — too sparse for city-scale spatial analysis. If asked "why not air temperature?": SUHI enables spatially comprehensive analysis at 30m. CUHI-SUHI correlation is imperfect, but neighbourhood-level relative rankings are preserved. Framework maps anomalies (relative hot/cool), so ordering holds even if absolute magnitude differs from air temperature.]

---

## Slide 3: The Data Opportunity

We have more urban data than ever before, and the quality keeps improving. Satellites, 3D models, street imagery. The challenge is no longer availability — it's integration: combining these sources into guidance planners can act on. That's what this research does. Six cities, three climates, open-source tools, diagnosis to prescription.

[BRIDGE] So what's preventing cities from using all this data? Three gaps.

[Q&A REFERENCE: "Open-source" means: GEE (free academic access), EUBUCCO (CC-BY 4.0), GlobalStreetscapes (CC-BY 4.0), Urbanity (MIT licence), VoxCity (open access). Full pipeline on GitHub under MIT licence. "No black boxes" connects to SHAP explainability — every prediction decomposes into individual feature contributions. "Diagnosis-to-prescription" is deliberate framing: Stage 1 diagnoses (what's hot, why), Stage 2 detects (where heat meets vulnerability), Stage 3 prescribes (what interventions, how much cooling).]

---

## Slide 4: Roadmap

Quick roadmap. Problem, data, model, findings, implications. Let's start with what's preventing action.

---

## Slide 5: Three Gaps Preventing Action

Three gaps prevent cities from acting.

First, integration. Studies examine morphology, vegetation, or networks in isolation. Heat emerges from their interactions.

Second, transferability. Schwaab et al. found identical tree species cool 8-12 degrees in Central Europe but only 0-4 in the Mediterranean. A programme designed for one climate context may deliver substantially less in another.

Third, the action gap. Models predict well but explain poorly. Planners need to know why and what to change. [PAUSE]

Two questions: What drives urban heat across contexts? And where should cities intervene?

[Q&A REFERENCE: Integration gap — how building arrangements channel winds determines whether a tree's cooling actually reaches pedestrians. Interventions designed from single-source studies fail when deployed across contexts (Li et al. 2020: morphology ignores vegetation obstruction; Li et al. 2024: vegetation assumes uniform building contexts). Action gap — the barrier between model outputs and actionable design briefs. ML models achieve impressive accuracy but function as black boxes. Camps-Valls et al. 2025 tripartite framework: quantification, understanding, communication — most ML-UHI studies stop at quantification. Schwaab mechanism: evapotranspiration constrained by vapour pressure deficit — dry air limits transpiration regardless of soil moisture. If asked "why not physics-based models like ENVI-met?": computationally prohibitive across 40,344 cells and six cities. ENVI-met operates at street-section scale (metres, hours of compute per block). This framework screens at neighbourhood scale (30m, minutes of compute for all six cities). Complementary, not competing — ENVI-met should follow this framework for detailed design.]

---

## Slide 6: Six Cities, Three Climates

Six cities across a climate gradient. Amsterdam and Paris oceanic. Athens and Barcelona Mediterranean. Berlin and Madrid transitional — they test whether the framework handles climate ambiguity.

[CLICK] Each city gridded at 30-metre resolution, 40,344 cells total. Equal-area approach: roughly 6 square kilometres per urban core.

[Q&A REFERENCE: Berlin sits at the Cfb/Dfb boundary, Madrid at the Csa/BSk semi-arid transition. Equal-area approach lets us compare morphological effects without confounding them with different study extents. Grid alignment: EPSG:3857 zoom level 14 tiles aligned with GlobalStreetscapes sampling framework. MAUP: 60m loses 17% variance (falls between building-scale radiative and neighbourhood-scale advective — misaligns with both). 90m recovers non-monotonically. Multi-seed stability: σ(R²) = 0.064 across 5 random seeds. If asked "why these six?": maximise climatic variation within constraints of all five data sources. EUBUCCO limits to Europe. Adding non-European cities requires substituting EUBUCCO with OSM buildings (less accurate heights) or Google Open Buildings.]

---

## Slide 7: Five Data Sources, One Framework

The novelty is fusion: 118 features from five open-source tools. GEE gives us surface temperature. EUBUCCO gives 3D building morphology. Urbanity captures how streets connect. GlobalStreetscapes adds the pedestrian perspective. All open. Any European city can replicate this tomorrow.

Let me show you these briefly.

[Q&A REFERENCE: No single source captures urban heat. Satellites see surface temperature but not street-level shade. Street imagery captures the pedestrian experience but not morphological context. Building models give geometry but not network connectivity. The novelty is fusing five complementary perspectives into a unified feature space — 118 features, each grounded in urban climate theory. Coverage by source: GEE 95-99%, VoxCity 92-99%, EUBUCCO 34-92% (Madrid lowest at 65.6% missing), GlobalStreetscapes 29-71% (Athens lowest at 71.2% missing), Urbanity 7-12% (network node architecture — every intersection captured). Missingness: city-specific median imputation with binary indicator flags — model learns to treat imputed vs observed differently. Feature pipeline: 165 base → spatial lags at 150m (pedestrian) and 300m (neighbourhood/advective) → 189 candidates → Pearson |r| > 0.92 filter → 118 final. Six physics-informed derived features: urban heat trap index, vegetation cooling saturation (log-scaled), canyon sky openness, ventilation proxy (SVF × (1 − building coverage)), impervious-canopy balance, height-to-coverage ratio. Temporal mismatch caveat: GEE = summer 2024 composites, GlobalStreetscapes = undated imagery (various years), EUBUCCO/VoxCity = static. Assumes morphological features change slowly relative to seasonal thermal dynamics.]

---

## Slide 8: Google Earth Engine

Google Earth Engine is the backbone. Landsat at 30-metre resolution gives us our dependent variable: land surface temperature. Summer composites, cloud-filtered. Notice the inverse pattern: where temperature is high, vegetation is low.

[Q&A REFERENCE: Landsat 8/9 Collection 2 Level 2. Summer months (June-August). Cloud masking via QA_PIXEL band. Compositing: per-pixel median. Anomaly = LST_cell − LST_city_mean — isolates morphological heating from background climate. Summer composites are cloud-filtered, with intra-city anomalies so we're comparing within each city, not between them. The LST-NDVI relationship isn't linear — log relationship, not linear. The model captures this. NDVI calculated from same Landsat scenes.]

---

## Slide 9: Urbanity

Urbanity extracts street network topology from OpenStreetMap. Different network structures mean different ventilation pathways. On the left, the global dataset; on the right, our six cities. Amsterdam's canal grid versus Athens's dense organic fabric.

[Q&A REFERENCE: Urbanity extracts connectivity, betweenness centrality (ventilation corridors), meshedness (grid regularity), intersection density. The network determines how cooling propagates — network connectivity determines whether cooling from parks reaches surrounding streets. Amsterdam canal grid: high meshedness, regular spacing. Athens: low meshedness, irregular block sizes.]

---

## Slide 10: GlobalStreetscapes

GlobalStreetscapes: 10 million street-level images. From these we derive the Green View Index — vegetation visible at eye level, which satellites miss entirely. Coverage varies significantly by city.

[Q&A REFERENCE: 10 million street-level images from Mapillary and KartaView. Berlin has 277K images, Paris only 14K. Coverage is systematic not random — commercial/tourist areas overrepresented, peripheral/low-income areas underrepresented (known bias addressed in scope conditions). NDVI from satellites misses urban canyons entirely. Coverage varies 29-71% by city. VoxCity GVI (voxel-based canopy volume) vs GlobalStreetscapes GVI (street-level panoramic segmentation) capture vegetation from different perspectives (overhead vs eye-level). Divergence validates multi-source approach.]

---

## Slide 11: EUBUCCO

EUBUCCO gives us 3D building morphology across Europe. Heights, footprints, canyon geometry. This is the structural skeleton of the urban heat island.

[Q&A REFERENCE: 6,500 to 8,500 buildings per city in our study areas. Heights, footprints, aspect ratios. Sky view factor, surface-to-volume ratios all derive from this. Paris's uniform Haussmann fabric (18-25m uniform height) vs Athens's irregular organic growth (4-30m mixed heights). Amsterdam's low-rise canal buildings versus Barcelona's Eixample blocks. These morphological signatures drive fundamentally different heat dynamics.]

---

## Slide 12: Six Cities in 3D

VoxCity converts open data into volumetric 3D models. [CLICK] Six cities voxelised. Dramatically different morphologies. [CLICK] Solar irradiance: Barcelona's grid distributes heat evenly, Athens's basin traps it. [CLICK] Sky View Factor. [CLICK] Green View Index.

This 30m framework tells the designer which blocks to focus on. When we decompose each cell with SHAP, it tells them whether to de-seal, shade, or plant.

[Q&A REFERENCE: VoxCity technical: 5m voxel resolution, aggregated to 30m grid cells. EPW weather files for solar simulation (15 June, 14:00 local time, TMYx 2007-2021 climatology). Ray-tracing accounts for building shadows and tree canopy transmittance. SVF = ratio of visible sky hemisphere to total hemisphere at ground level. Low SVF = deep canyons restricting longwave radiation loss at night → nocturnal heat retention. Solar irradiance was generated but used primarily for visualisation, not as a direct model feature — at 5m aggregated to 30m, the shadow signal smooths. SVF served as the proxy that entered the model. Green View Index derived from street-level imagery — vegetation as pedestrians experience it. Complementary scales: this framework screens at neighbourhood scale (which blocks, 30m). Street-section design (how to configure, 5-15m) requires microclimate tools like ENVI-met or RayMan. The 30m grid tells you "Block X is hot because imperviousness is 94% and SVF is 0.3" — that's a design brief: "this block needs de-sealing and shade intervention." How you configure the street section — tree species, paving material, canopy height, setback — is architectural judgment below 30m resolution.]

---

## Slide 13: From Prediction to Prescription

Three stages. Predict temperature anomalies. Classify hotspots with calibrated probabilities. Test 648 intervention scenarios. The model shows its work at every stage, the planner keeps the final call.

[Q&A DEPTH: Stage 1 regression: city mean subtracted, so we isolate what morphology does from background climate. Stage 2 classification: sensitivity-tuned threshold flags cells most likely to overheat. Feature ranking shifts here — demographic vulnerability rises to top. Stage 3: each combination bootstrapped 500 times, physically implausible predictions filtered. Platt scaling converts XGBoost scores to probabilities, threshold = 0.30. 3 depaving × 9 veg × 8 tree × 3 albedo = 648. ±5°C plausibility filter. Why XGBoost not neural networks: (1) native SHAP TreeExplainer support for exact Shapley value decomposition, (2) captures non-linear thresholds linear models miss, (3) handles mixed feature types without scaling. Interpretability is non-negotiable for planning applications. Hyperparameters: 200 Bayesian trials (Optuna), minimising 5-fold CV RMSE. Final: 820 estimators, max_depth=6, learning_rate=0.03, subsample=0.78, colsample_bytree=0.74. Why grid search not Bayesian for scenarios: grid search maps full response surface — reveals non-linear shape (the 30→40→50% regime boundary). Optimisation finds optima faster but doesn't reveal the landscape. For planning, the shape matters more than finding the peak.]

---

## Slide 14: Building the Model

We built three models: global, Mediterranean specialist, oceanic specialist, and blended them. 118 features, SHAP explainability built in.

[GESTURE TO BLEND DIAGRAM] The global model gets subtracted. Climate-zone specialists dominate. The blend cuts unexplained variance by 22%.

[PAUSE]

[BRIDGE] Does the model actually predict heat across contexts?

[Q&A DEPTH: 22% = 4.5 percentage-point R² improvement (0.795→0.840). Denominator is unexplained variance, not total. Blend equation: pred_blended = β₀ + β₁·pred_global + β₂·pred_specialist. Coefficients optimised per city via least-squares on held-out validation fold. Negative global weight means: the global model captures universal patterns but also introduces systematic errors from averaging across climates. Subtracting it corrects the specialists' residuals. Per-cell SHAP example: "This cell is +2.1°C because imperviousness contributes +1.4, water distance +0.5, vegetation −0.3, building height +0.5." 165 base features → spatial lags → 189 → correlation filter → 118. Six physics-informed features. 200 Bayesian trials (Optuna). MAUP: 60m loses 17% variance. See backup B10.]

[Q&A GLOSSARY: R²: How much temperature variation the model explains. 1.0 = perfect, 0 = guessing average. SHAP: Feature contribution receipt — what drove each prediction. AUC: Hotspot discrimination. 1.0 = perfect, 0.5 = coin flip. Ours = 0.911.]

---

## Slide 15: Spatial Cross-Validation

We tested the model rigorously. 5-fold spatial cross-validation, 600m block grouping. No spatial leakage. The model learns morphological relationships, not spatial patterns.

[Q&A DEPTH: Moran's I confirms minimal residual autocorrelation. MAUP tested at 30m, 60m, 90m — 30m provided best balance between resolution and feature stability. Model never sees neighbours of test cells during training. We model anomalies, not absolute temperatures, removing the large-scale spatial trend. 300m buffer aggregation physically grounded in Oke's advective transport scale.]

---

## Slide 16: Can We Predict Heat Across Contexts?

R² of 0.93 in Paris and Barcelona, down to 0.72 in Madrid. The hierarchical blend improves the global model by 22%.

[CLICK] Spatial comparison confirms real patterns. Centre-periphery in Paris, coastal cooling in Barcelona, basin trapping in Athens.

Honest exception: blending hurts Athens. R² drops from 0.751 to 0.730. Topographic basin overrides climate-zone correction. That's a scope condition, not a failure.

But model performance isn't the finding. What the model reveals about what drives heat — that's the finding.

[BRIDGE] The model works. Now the question becomes: what does it see?

[Q&A REFERENCE: Global model alone R² = 0.795. Hierarchical blend pushes to 0.840 — cutting unexplained variance by 22%. That improvement comes entirely from respecting geographic context. The architecture IS the finding. Full RMSE: overall 0.85°C, range Paris 0.68°C to Madrid 1.12°C. Target variable distributions: Barcelona σ=3.13°C, Madrid σ=2.89°C (widest — easiest to predict). Amsterdam σ≈1.2°C, Athens σ≈1.2°C (narrowest — hardest to predict). Athens blending: topographic basin creates katabatic flows (cold air drainage from mountains) and thermal inversions overriding climate-zone correction. Scope condition: basin cities need terrain-stratified sub-models. Madrid: semi-arid drought stress creates soil moisture dynamics morphological features can't capture. Vegetation effectiveness depends on irrigation — not in the model. If asked "does R²=0.73 in Athens mean the model is wrong?": 27% unexplained — likely terrain effects, sea breeze channelling, thermal inversions. Neighbourhood-level relative rankings still informative for screening.]

---

## Slide 17: What Actually Drives Urban Heat?

[SLOW DOWN]

SHAP decomposes each prediction. The hierarchy: surface characteristics 21%. Water features 13%. Vegetation just 7%.

[PAUSE — let the inversion land]

Surface permeability matters three times more than vegetation for temperature reduction. The physical mechanism is well established: permeable ground lets water evaporate, which absorbs heat. Seal the surface, and all that energy warms the air instead. Our SHAP results are consistent with this mechanism.

This doesn't mean trees don't matter. They provide shade, biodiversity, air quality. But for temperature reduction, de-sealing is the stronger lever.

And this hierarchy shifts by climate zone. In the Mediterranean, albedo dominates. In temperate cities, impervious fraction. The same intervention produces fundamentally different outcomes depending on where you apply it.

[Q&A DEPTH: 21%/7% uses top two features per category. Full categories: surface 34%, vegetation 14%, ratio ~2.4:1. Top individual features: NDBI |SHAP|=0.321, impervious_300m=0.289, water_distance=0.245, NDVI=0.198, building_density=0.176. Physics: impervious surfaces have Bowen ratio >5 (nearly all energy into sensible heat). Permeable vegetated surfaces: Bowen ratio 0.5-1.5 (30-50% into latent heat). Transitioning from 95% to 50% imperviousness shifts the energy balance fundamentally. 300m optimal radius: buffer optimisation showed impervious effects peak at 300m (SHAP=0.321 at 300m vs 0.085 at 90m, 0.067 at 150m). Aligns with Oke 2017 advective transport under 2-3 m/s winds. Schwaab et al. 2021, 293 cities — evapotranspiration efficiency depends on vapour pressure deficit. DESIGN RULE: SHAP dependence plots show interaction — high imperviousness hurts most when SVF is also high (open plazas: full solar exposure, no shade, no evaporative pathway). In deep canyons (low SVF), shadow already reduces surface heating, so permeability matters less. Open plazas → de-seal first. Narrow streets → shade first. If asked "does framework distinguish types of permeability?": No. Permeable paving, rain gardens, pocket parks, bioswales all register as reduced imperviousness. Translating "reduce imperviousness by 50%" into specific design is architectural judgment — the handoff point. SHAP caveat: assumes feature independence in attribution, but urban variables correlate (NDBI and impervious fraction r=0.76). Category-level hierarchy (surface > water > vegetation) is robust. Individual feature redistribution within categories has uncertainty.]

---

## Slide 18: Geography Changes Everything

Same model, same features, but the SHAP hierarchy shifts by climate zone. Mediterranean: albedo and built-up index dominate. Temperate: impervious fraction.

Each city needs a different intervention strategy. One-size-fits-all guidance ignores this.

[BRIDGE] We know what drives heat and how it varies. Now: where does heat meet people?

[Q&A REFERENCE: Athens's topographic basin traps circulation. Athens elevation SHAP=0.12 — terrain is a first-order control other cities don't face. Amsterdam's canals flatten water proximity as a predictor because 90% of cells are already near water. Madrid's semi-arid climate amplifies drought stress effects. Cross-city SHAP variation: vegetation importance varies 3× between Mediterranean (low — moisture-constrained) and temperate (higher — evapotranspiration effective). This is not an intercept shift — it's a fundamentally different feature interaction structure. That's why hierarchical blending outperforms a single global model with climate dummies. EU policy implication: current frameworks (European Climate Pact, Covenant of Mayors) provide generic "increase urban greening" guidance. Geographic shift evidence argues for climate-zone-specific toolkits: different intervention priorities, different budget allocations, different performance targets. Tree-first makes sense in Berlin; de-sealing-first makes sense in Barcelona.]

---

## Slide 19: Classifying Hotspots

Now: classifying hotspots. We use a robust statistical threshold that adapts to each city's temperature distribution. The classifier is calibrated: when it says 70% risk, it means 70%. Overall AUC = 0.911. Athens flags nearly a third of its area; Paris, under 8%.

[Q&A DEPTH: Hotspot = any cell exceeding 1 MAD above city median (MAD scaled by 1.4826 for Gaussian equivalence ≈ +1σ) — more robust to outliers than standard deviation. XGBoost classification with 1.5x minority weighting (hotspots only 15.8% of cells). Platt scaling: logistic regression on held-out predictions to calibrate probabilities. Threshold set at 0.30 for ≥60% recall. F1 = 0.616, Precision = 56.3%, Recall = 68.0%. Trade-off: flag some non-hotspots rather than miss real ones. Fold-level AUC: 0.949, 0.924, 0.932, 0.894, 0.858. Madrid Fold 5 = 0.511 — near random, scope condition for semi-arid instability. Hotspot prevalence by city: Athens 30.8%, Amsterdam 20.5%, Berlin 13.3%, Madrid 11.6%, Barcelona 11.6%, Paris 7.2%.]

---

## Slide 20: Risk Tiers Across Cities

Spatial risk tiers across all six cities. Severe risk concentrates in dense urban cores. Athens: nearly a third in the highest tier. These feed directly into priority zone scoring. But the tiers alone don't tell the full story.

---

## Slide 21: Who's Most At Risk?

[SHIFT TONE — this is the equity pivot]

Now something surprising. When we classify hotspots, the feature ranking shifts. Child density displaces morphological features as the top predictor. Areas with high vulnerability: 2.3 times higher hotspot probability.

[PAUSE — 3 seconds. Let this land.]

The model captures where heat meets vulnerability. Dense, older fabric with less green space is where families concentrate. Priority zones weight vulnerability at 35%.

Hotspots aren't just where heat is generated. They're where vulnerable people experience it.

[BRIDGE] We know where heat is, who it affects. Now: what do we do about it?

[Q&A DEPTH: Feature ranking inversion quantified: child density |SHAP| = 0.182, total population density = 0.156, impervious surface drops to third (0.089) in classification model. This means: for predicting WHERE heat is, surface properties dominate. For predicting WHERE HEAT IS DANGEROUS, demographics dominate. Vulnerability tier: z-scored within each city (local patterns, not cross-city income differences). Low = z ≤ 0, Medium = 0 < z ≤ 0.8, High = z > 0.8. Combined index = max(children_z, elderly_z). Priority zone composition: total 15,734 cells (39.0%). Athens 5,248 (78.2%), Paris 2,875 (43.7%), Amsterdam 2,422 (35.6%), Berlin 2,100 (30.2%), Barcelona 1,684 (25.3%), Madrid 1,405 (20.9%). Priority zone characteristics vs citywide: mean UHI anomaly +0.73°C vs −0.25°C, 93.6% impervious vs 77.2%, 3.0% tree canopy vs 7.2%. Thermoregulation: Children have higher surface-area-to-mass ratio, lower sweat rates, less behavioural adaptation. Elderly: chronic conditions, medication effects (diuretics, beta-blockers impair thermoregulation), reduced mobility. Kovats and Hajat 2008. If asked "why those specific weights (40/35/25)?": analytical judgment, not empirical optimum. Sensitivity testing: rankings stable across ±10% weight variation. Different municipalities could weight differently — that's a democratic decision.]

---

## Slide 22: The 50% Regime Boundary

[THIS IS THE POLICY-RELEVANT SLIDE — deliver clearly]

We tested 648 combinations of four levers. Each bootstrapped 500 times.

Every optimal strategy converges on 50% de-sealing. The response is non-linear: the SHAP dependence structure shows a clear inflection around 50% sealed surface, consistent with the regime boundary Oke et al. describe where permeable area enables evapotranspiration.

The cost-effective strategy — 50% depaving, modest vegetation, 20% trees, no albedo treatment — achieves 95% of maximum cooling with substantially fewer resources. Those confidence intervals overlap.

[BRIDGE] That's the universal finding. But each city's baseline is different.

[Q&A DEPTH: Four levers: impervious surface reduction at 30%, 40%, 50%; vegetation increase 10-50% in 5% steps; tree canopy 15-50%; albedo 0%, 10%, 20%. Non-linear response: 30% depaving = 49% of maximum cooling, 40% = 71%, 50% = 100%. Max: −1.27°C (95% CI: 1.23-1.31°C). Cost-effective: −1.20°C (95% CI: 1.16-1.24°C). CIs overlap → not statistically different at α=0.05. "Cost-effective" is a proxy: de-sealing during routine maintenance cycles is cheaper than establishing mature tree canopy (15-20 years to maturity, irrigation). No actual €/m² cost function — that's a limitation and future work. Regime boundary physics: below 50% sealed, sufficient permeable surface enables 30-50% of absorbed solar energy to redirect from sensible to latent heat. Above 50%, evaporative pathways too fragmented for continuous cooling. Aligns with "sponge city" threshold concepts. Extreme filter: predictions >5°C change removed (~21% of all predictions, ~8% in core urban — rest peripheral/industrial where extrapolation is most severe). COUNTERFACTUAL VALIDITY: model learns correlations from current configurations, not causal consequences of changing them. Setting imperviousness from 95% to 50% is extrapolation — limited training examples of 50%-impervious cells in dense urban cores. Predictions are hypotheses, not guarantees. True validation requires pilot de-sealing projects with before-after monitoring. If asked "is 50% realistic in heritage European cores?": probably not everywhere (underground infrastructure, heritage protections). But the non-linear response means even 30% delivers 49% — meaningful even where 50% is impossible. De-sealing can phase into routine infrastructure maintenance. If asked "does threshold shift with canyon geometry?": threshold emerges from global model pooling all morphologies. Mediterranean cities likely need lower thresholds (higher solar loads). Threshold is central tendency, not universal constant. ±5°C plausibility filter.]

---

## Slide 23: What Should Each City Do?

Athens gains the most cooling, 1.45 degrees, because it starts from the worst baseline: 95% impervious, nearly 80% flagged as priority. Amsterdam gains the least, 0.92 degrees, because its canal network already saturates water cooling. The other four fall between these poles, each visible in the table. Cooling scales with baseline imperviousness.

One scope note: this framework screens neighbourhoods. Street-section design — material selection, tree placement, shade geometry — requires sub-30m tools. Complementary scales.

[Q&A REFERENCE: Per-city detail — Athens (1.45°C): 95.2% impervious, 78.2% priority. Attica Basin traps heat but also means interventions have maximum impact — nowhere for additional heat to go, so removing sources yields big returns. Sea breeze from Saronic Gulf creates corridors that de-sealing can amplify. Barcelona (1.31°C): 94.7% impervious. Coastal-hill interface creates sea-breeze corridors. Eixample block interiors are prime de-sealing targets — current courtyards often paved. Paris (1.22°C): strong centre-periphery gradient. Core de-sealing extends Seine corridor cooling into dense Right Bank neighbourhoods. Haussmann's uniform canyon geometry means predictable intervention responses. Berlin (1.08°C): polycentric structure, no single heat core. Distributed intervention strategy needed. Impervious fraction is dominant driver across all sub-centres. Amsterdam (0.92°C): 90% of cells within 200m of water. Canal network caps cooling. Primary lever: building material albedo and surface permeability in non-canal areas. Madrid (0.89°C): fewest priority zones (20.9%). Existing tree canopy (21.7%) already buffers. Semi-arid climate means vegetation interventions need drought-tolerant species — water-intensive greening counterproductive (Schwaab findings). Albedo modification (cool roofs/pavements) may be more effective than vegetation in this climate.]

---

## Slide 24: Blue-Green Synergy & Co-Benefits

De-sealing delivers more than heat reduction. Stormwater infiltration, biodiversity habitat, groundwater recharge. Cities pursuing flood and heat resilience can achieve both.

This is not "stop planting trees." The cost-effective strategy still includes 20% tree canopy increase. It's rebalance budgets.

[Q&A REFERENCE: Co-benefits of de-sealing: restores evaporative cooling (heat), improves stormwater infiltration (flooding), enables soil biota (biodiversity), recharges aquifers (water security). Single intervention, four resilience domains — "sponge city" framework alignment. Amsterdam illustrates diminishing returns of water: 90% of cells within 200m of water, yet still has hotspots. Once water proximity saturates, the intervention lever shifts to building materials and surface permeability. Currently most cities invest heavily in tree planting while ignoring the surface beneath. Eyni et al. 2025: independently confirmed diversified depaving outperforms tree-only strategies for reducing heat-related health disparities — corroborates the SHAP hierarchy from a health equity angle.]

---

## Slide 25: Who Benefits? Who Gets Displaced?

The framework identifies where to intervene. It cannot prevent green gentrification. Cooling investments without anti-displacement policies risk improving places while displacing people. Hoffman et al.: historically redlined neighbourhoods are up to 4 degrees hotter.

Technical optimisation without equity safeguards is incomplete. That's a governance question this analysis raises but cannot answer.

[Q&A REFERENCE: Anguelovski et al. 2019 documented the mechanisms: greening → property value increase → rent escalation → displacement of low-income renters. Barcelona's Superblocks and New York's High Line both documented displacement effects alongside environmental benefits. Hoffman et al. 2020: historically redlined US neighbourhoods experience surface temperatures up to 4°C higher than non-redlined districts. Structural racism encoded in urban morphology. European equivalents: social housing peripheries, immigrant-concentration districts. Policy instruments: rent stabilisation/control, community land trusts (CLTs), participatory budgeting for intervention prioritisation, inclusionary zoning, anti-displacement ordinances triggered by environmental investment thresholds. Framework contribution: vulnerability weighting (35%) ensures priority zones are where vulnerable people live now. But weighting can't prevent market dynamics after intervention. Policy safeguards must be concurrent with physical interventions, not sequential.]

---

## Slide 26: Three Challenges to Prevailing Assumptions

Three challenges. One: heat operates at 300 metres. Policy targets individual plots. Scale mismatch. Two: de-sealing outperforms tree planting three to one, yet gets a fraction of the budget. Wrong priority. Three: the same tree cools 8 degrees in Berlin and zero in Athens. No universal template.

[Q&A REFERENCE: Scale mismatch: Oke 2017 documents advective transport under typical 2-3 m/s winds propagates cooling effects 300-500 metres. Heat doesn't respect cadastral boundaries. Wrong priority: Eyni et al. 2025 confirmed independently — diversified depaving outperforms tree-only strategies for reducing health disparities. No universal template: Schwaab et al. found identical tree species cool 8-12°C Central Europe, 0-4°C Mediterranean. Continental, Mediterranean, and maritime cities need fundamentally different strategies. One-size-fits-all guidance — the default mode of EU adaptation policy — ignores geographic contingency.]

---

## Slide 27: Scope Conditions

Two key scope conditions. Daytime only: Landsat captures 10:30 AM, not nighttime when mortality peaks. And correlations, not causes: the 50% threshold needs field validation.

But: GEE satellite features dominate the SHAP rankings. A satellite-only pipeline captures most of the signal. Coverage beats sophistication.

[PAUSE]

These are real limitations. But waiting for perfect evidence while heatwaves kill thousands is itself a policy choice.

[Q&A REFERENCE: Landsat overpass: 10:30 local solar time. Peak surface temperature typically 13:00-15:00. Peak mortality risk occurs overnight (inability to thermoregulate during sleep). Framework maps 10:30 relative anomalies — neighbourhood rankings preserved (cells hot at 10:30 are generally hot at 15:00 and retain more heat overnight). But absolute magnitude of surface-to-air relationship varies with wind speed, humidity, canyon geometry. ECOSTRESS: finer temporal sampling including nighttime but coarser spatial resolution and inconsistent revisit — could complement, not replace Landsat. 50% threshold: analytically grounded (SHAP dependence structure) but awaits field validation — field-wide limitation, not unique to this study. Counterfactual validity: model learns statistical associations, not causal consequences of changing configurations. Extreme-prediction filter (>5°C) removes obvious failures. Rigorous validation requires pilot projects with before-after monitoring. Data coverage: GlobalStreetscapes 29-71% by city. Systematic not random — street-level platforms prioritise commercial/tourist/higher-property-value areas. Predictions least reliable where vulnerability concentrates. GEE dominance: satellite-derived features (NDBI, impervious fraction, NDVI, albedo) carry ~21% combined SHAP importance with 95-99% spatial coverage. Practical: a city without VoxCity or EUBUCCO can still run meaningful screening. GEE alone + XGBoost + SHAP = viable minimum pipeline. Anthropogenic heat gap: traffic, HVAC, cooking, industry absent from framework. Can add 20-50 W/m² in dense urban cores — comparable in magnitude to moderate de-sealing effects. Spatially comprehensive datasets don't exist across six cities. Global South transferability: methodology transfers but requires local recalibration. EUBUCCO Europe-only. Substitutes: OSM buildings (less accurate heights), Microsoft Building Footprints (Africa, SE Asia), Google Open Buildings. GEE features work globally. Extension requires local data partnerships and attention to informal settlement morphologies.]

---

## Slide 28: CLOSING

[Step back from podium. Hands at sides. Look at the audience, not the screen. Count slowly to three.]

[CLICK] What works in Amsterdam fails in Athens. That difference is not a problem. It is the finding.

[PAUSE 2 seconds — CLICK] Open tools. Open data. Every city can act on this tomorrow. The barrier is not technical.

[PAUSE — CLICK — DO NOT SPEAK. Let the audience read it. Hold 3 full seconds of silence. Then simply: "Thank you."]

[Q&A REFERENCE: If asked "what would you do differently?": (1) Add nocturnal thermal dynamics via ECOSTRESS integration. (2) Higher-resolution sub-block modelling — combining 30m screening with street-section microclimate simulation (ENVI-met, RayMan). (3) Longitudinal field validation through pilot monitoring programmes. (4) Cost function integration for true cost-effectiveness (€/°C/m²). Product path vision: web interface where planner selects city → sees priority zones → adjusts intervention sliders → gets predicted cooling with equity overlays. If asked to compare with existing city heat action plans: most emphasise tree planting and reactive cooling centres. Framework argues for preventive de-sealing at neighbourhood scale. Barcelona's Superblocks partially aligns with de-sealing logic. Paris's Plan Climat emphasises canopy coverage targets. Berlin's Stadtentwicklungsplan Klima incorporates ventilation corridors (connects to network connectivity findings). GitHub repo: MIT licence, full pipeline code, feature engineering scripts, model training, SHAP analysis, scenario generation.]

---

## Slide 29: References

This slide lists the key references cited throughout the presentation. The full bibliography with 100+ entries is available in the GitHub repository. If anyone wants the slides, these references and the full BibTeX file are included.

---

## Slide 30: Any Questions?

Thank you. Happy to discuss any aspect - methodology, city-specific findings, data pipeline, or equity implications.

---
