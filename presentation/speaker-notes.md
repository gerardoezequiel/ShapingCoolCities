# Speaker Notes — Shaping Cool Cities
## UrbanAI Presentation v3 (16 minutes + Q&A)

---

## Slide 1: Title (20 seconds)

Thank you for having me. I'm Gerardo, I recently completed my MSc in Urban Spatial Science at UCL's Centre for Advanced Spatial Analysis. Today I want to share research about a question that I think sits at the heart of what Urban AI is about: how do we translate the abundance of urban data we now have into actionable guidance that cities can actually use? Specifically — how do we move from observing heat to prescribing cooling interventions, city by city?

[NOTE: UrbanAI's Guide 2023 covers water meters, curb digitization, air quality — but heat adaptation is a gap in their case study portfolio. This research fills that gap directly.]

---

## Slide 2: The Hook — 61,000 (40 seconds)

[PAUSE — let the number land]

Sixty-one thousand people died from heat across Europe in the summer of 2022. That makes it the deadliest natural hazard event in European history. Not a flood, not a storm — heat. And what makes this particularly striking is that European cities had more climate data available to them than ever before. Satellites capturing surface temperatures at metre-scale resolution. Three-dimensional building models. Street-level imagery. Yet when planners asked "where should we plant trees?" or "which streets should we depave?" — there was remarkably little useful guidance.

---

## Slide 3: The Paradox — Translation Gap (40 seconds)

This is what I call the translation gap. The problem isn't data scarcity — it's that we haven't been able to translate observation into intervention. We have thermal imagery but no guidance on where to intervene. We have 3D city models but limited understanding of why heat concentrates where it does. We have street-level photographs but no evidence base for what interventions actually work. The gap between what we can observe and what we can act on — that's what this research addresses.

---

## Slide 4: Three Gaps (50 seconds)

More specifically, three analytical gaps prevent cities from acting effectively.

First, the integration gap. Most studies examine morphology, vegetation, or network connectivity in isolation. But heat emerges from their interactions — how building arrangements channel winds determines whether a tree's cooling actually reaches pedestrians.

Second, transferability. What works in Berlin doesn't necessarily work in Barcelona. Climate context shapes everything, yet most studies draw from single cities and assume universal principles.

Third, the action gap. Machine learning models achieve impressive accuracy but function as black boxes. Planners don't need to know that a neighbourhood will be hot — they need to know *why*, and what they can change about it. [TRANSITION] So the research asks two questions...

---

## Slide 5: Research Questions (30 seconds)

First: what actually drives urban heat, and how do those drivers vary across different cities and climates? And second — the applied question: where should cities concentrate cooling interventions to get the most benefit, especially for the people most at risk? [BRIEF PAUSE] Let me show you how we approached this.

---

## Slide 6: Study Domain (40 seconds)

We analysed six European cities spanning three climate zones. Amsterdam, Berlin, and Paris in the oceanic zone. Athens, Barcelona, and Madrid in the Mediterranean. Each city was gridded at 30-metre resolution — roughly the scale of a street block — giving us over 40,000 analysis units across all six cities. The equal-area approach is important: it lets us compare morphological effects between cities without confounding them with different study extents.

---

## Slide 7: Data Integration Overview (30 seconds)

What makes this framework different isn't any single dataset — it's the integration. No single source captures urban heat. Satellites see surface temperature but not street-level shade. Street imagery captures the pedestrian experience but not morphological context. Building models give geometry but not network connectivity.

The novelty is fusing five complementary perspectives into a unified feature space — 118 features, each grounded in urban climate theory. And critically: all five sources are open. Zero proprietary data. Any European city can replicate this. Let me show you each source.

---

## Slide 7a: Google Earth Engine — LST (15 seconds)

[FULL-BLEED FIGURE — let it speak]

The thermal foundation. Landsat surface temperature at 30-metre resolution. Notice within-city variation: 7 to 15 degrees over just a few hundred metres. Amsterdam ranges 29 to 36; Athens hits nearly 50. This is what we're trying to explain and reduce.

---

## Slide 7b: GlobalStreetscapes (15 seconds)

[FULL-BLEED FIGURE — let it speak]

The pedestrian perspective. 10 million street-level images. Green View Index captures what people actually see at eye level — fundamentally different from satellite NDVI. A tree behind a wall shows up in satellite data but provides no shade to pedestrians.

---

## Slide 7c: EUBUCCO — Building Heights (15 seconds)

[FULL-BLEED FIGURE — let it speak]

Three-dimensional building morphology. Not just footprints but heights. Paris's uniform 15-20m Haussmann fabric versus Athens's dense irregular height variation. From these we compute building density, aspect ratios, sky exposure.

---

## Slide 7d: Urbanity — Network Topology (15 seconds)

[FULL-BLEED FIGURE — let it speak]

Street network topology — how connected the grid is determines wind ventilation. Barcelona's regular grid channels sea breezes; Athens's dense network (1,200 nodes, 4,000 edges) creates both ventilation challenges and pedestrian exposure.

---

## Slide 7e: VoxCity — Sky View Factor (15 seconds)

[FULL-BLEED FIGURE — let it speak]

Ray-traced solar geometry at voxel resolution. SVF is the top SHAP driver among 3D features — this is WHY three-dimensional analysis matters. Shadow patterns shift minute-by-minute; standard SVF is static, VoxCity captures the dynamics.

---

## Slide 7f: Six Cities Voxelised (20 seconds)

All six cities voxelised — dramatically different morphologies. [CLICK TOGGLE] Switching to solar irradiance shows how the same morphology creates completely different irradiance patterns on the ground. Five data sources, one integrated framework. Now let me show you what we do with it.

---

## Slide 9: Method (40 seconds)

The analytical pipeline has three stages. Diagnosis: XGBoost gradient boosting predicts temperature anomalies — deviations from city means that isolate local morphological effects from background climate. The model achieves R-squared of 0.84, so it's explaining most of the within-city thermal variation.

Detection: we classify hotspots for policy targeting, achieving AUC of 0.911.

And prescription — and this is where it moves beyond most ML studies — we evaluate 648 intervention scenarios to identify what combinations of de-sealing, vegetation, and tree planting achieve meaningful cooling. Throughout, SHAP decomposition ensures every prediction is interpretable.

Why XGBoost with SHAP rather than deep learning? Because this is designed as a co-pilot for planners, not a black box that replaces their judgement. The AI augments human decision-making: planners see the reasoning, not just the answer. Explainability isn't an add-on — it's the mechanism that transforms prediction into prescription. [TRANSITION] So what did we find?

---

## Slide 10: Model Performance (30 seconds)

Performance varies systematically with city characteristics. Paris and Barcelona achieve the highest accuracy — they have pronounced centre-periphery heat gradients that morphological features capture well. Athens and Madrid are more challenging: Athens because topographic basin effects override built form, Madrid because semi-arid drought stress introduces dynamics beyond morphological control. The hierarchical climate-zone blending improves predictions for every city. [TRANSITION] But the model performance isn't the finding. The finding is what the model reveals about what drives heat.

---

## Slide 10b: Cross-Validation Rigor [NEW] (25 seconds)

[QUICK GLANCE — technical credibility slide]

Before the findings, a note on validation. We use spatially blocked cross-validation with 600-metre grid cells — Moran's I confirms this is essential, not optional. MAUP sensitivity shows resolution matters: 30m captures effects that 60m loses. Multi-seed stability at sigma 0.064 confirms robustness.

And one counterintuitive lesson: comprehensive coverage of simple satellite features outperforms incomplete coverage of sophisticated 3D features. The practical implication: start with what every city already has. [ADVANCE]

---

## Slide 11: The Surprise — SHAP (60 seconds)

[SLOW DOWN — this is the central finding]

This is the SHAP analysis — each dot is one grid cell, coloured by feature value. What it reveals is a clear hierarchy that challenges common planning assumptions.

Surface characteristics — impervious fraction, built-up index — account for 21% of total feature importance. Water features collectively contribute 13%. And vegetation — tree canopy, green view index — just 7%.

[PAUSE]

Surface permeability matters roughly three times more than vegetation for urban cooling. This doesn't mean trees don't matter — they provide shade, biodiversity, air quality. But if your primary goal is temperature reduction, de-sealing is the stronger lever. And that's not how most cities are investing. Most cooling budgets go to tree planting. This evidence suggests they should diversify.

---

## Slide 11b: How the Blend Works [NEW] (20 seconds)

[QUICK GLANCE — mechanism slide]

The hierarchical model does something surprising — it gives the global model a *negative* weight of minus 0.151. It literally subtracts the pooled signal. What remains are climate-zone specialists that capture relationships the global model averages away. Paris improves 6.8%, Amsterdam 6.4%. But Athens actually gets slightly worse — the topographic basin effects are so strong that climate-zone specialisation can't improve on the global model. That's an honest scope condition. [ADVANCE]

---

## Slide 12: Model Performance — Observed vs. Predicted (30 seconds)

Here's the spatial comparison — observed on the left, predicted on the right. The model captures within-city thermal gradients, not just averages. You can see Paris's centre-periphery gradient, Barcelona's coastal cooling, Athens's basin effect. Errors concentrate at urban edges where morphological transitions are sharpest.

---

## Slide 13: Geography Changes Everything (40 seconds)

Now look at the side-by-side SHAP plots — Mediterranean on top, temperate oceanic on the bottom. In the Mediterranean zone, albedo and NDBI dominate. In temperate cities like Paris and Berlin, impervious surface fraction takes over. The feature hierarchies are fundamentally different.

Mediterranean cities show topographic basin effects that overwhelm building morphology. Barcelona's coastal-hill interface channels sea breezes. And Amsterdam's ubiquitous canals flatten water proximity as a predictor, so building density takes over.

This is why one-size-fits-all guidance fails. Geographic contingency isn't noise — it's the signal.

---

## Slide 13b: City Spotlights [NEW] (20 seconds)

[QUICK GLANCE — scan across six panels]

Each city tells a different SHAP story. Amsterdam: water saturation. Athens: topography overwhelms morphology. Barcelona: coastal wind channels. Berlin: classic impervious primacy. Madrid: note the Fold 5 instability. Paris: strongest centre-periphery gradient. Each needs a tailored strategy. [ADVANCE]

---

## Slide 14: The Threshold (50 seconds)

[THIS IS THE POLICY-RELEVANT SLIDE — deliver clearly]

Now, the intervention analysis. We tested 648 combinations of de-sealing, vegetation enhancement, tree planting, and albedo modification.

Every optimal strategy converges on the same number: 50% de-sealing. This isn't arbitrary — it marks a regime boundary. Below 50% sealed surfaces, you have enough permeable area for evapotranspiration to function, redirecting solar energy into latent heat rather than atmospheric warming. The response curve on the right shows that acceleration — cooling steepens as you approach 50%.

And here's the cost-effectiveness finding: the maximum-impact scenario achieves 1.27 degrees of cooling. But a much simpler strategy — 50% depaving with modest tree cover and no albedo treatment — achieves 1.20 degrees. That's 95% of the maximum benefit with 60% fewer resources.

---

## Slide 15: Hotspot Classification [NEW] (30 seconds)

The hotspot classifier achieves AUC 0.911 — this is a strong discrimination metric. We calibrate the threshold at 0.3 for sensitivity. The headline finding: child density becomes the number one hotspot predictor, with 2.3 times higher hotspot probability in high-child-density areas. Hotspots are defined by where vulnerable people experience heat, not just where heat is generated.

One scope condition: Madrid Fold 5 at AUC 0.511 shows where morphological models hit their limits.

---

## Slide 16: Priority Zones (40 seconds)

The composite scoring weights heat risk at 40%, demographic vulnerability at 35%, and cooling potential at 25%. Priority zones aren't uniformly distributed — Athens has 78% of cells flagged as priority, reflecting extreme heat stress, while Madrid has 21%.

[GESTURE TO MAP] You can hover over each city on the interactive map to see city-specific metrics.

---

## Slide 17: Vulnerability Tiers [NEW] (25 seconds)

The vulnerability tier analysis shows that highest-priority zones aren't just the hottest — they're where heat intersects demographic vulnerability. Children have less thermoregulatory capacity than adults, making child density a critical equity metric. This reframes urban heat from a physical phenomenon to a social justice issue.

---

## Slide 18: Cooling Potential (30 seconds)

The predicted cooling varies by city. Athens stands to gain most — 1.45 degrees — because it has the most intervention headroom: extensive sealed surfaces combined with intense summer heat. Amsterdam gains least, not because of climate but because its ubiquitous canal network already provides a cooling baseline that additional surface changes can't easily improve upon.

---

## Slide 19: City-Specific Prescriptions [NEW] (20 seconds)

[QUICK GLANCE — scope conditions table]

Continental cities: de-seal urban cores. Mediterranean: preserve wind corridors and treat albedo. Maritime Amsterdam: building material retrofits. And notice the sponge city synergy — flood and heat resilience share the same de-sealing intervention. Two policy goals, one investment. [ADVANCE]

---

## Slide 20: Equity (40 seconds)

[SHIFT TONE — this is the ethical core]

I want to be honest about what this framework can and cannot do. It identifies where to intervene. But we know from the literature — from Anguelovski's work and others — that greening interventions can trigger displacement. Trees raise property values. Landlords raise rents. The populations we're trying to protect get priced out.

The framework includes vulnerability weighting, but technical optimisation without equity safeguards is incomplete. Rent stabilisation, community land trusts, participatory governance — these are policy instruments that must accompany physical interventions.

---

## Slide 21: Three Takeaways (45 seconds)

[COUNT ON FINGERS — clear enumeration]

Three challenges to prevailing assumptions.

One: scale mismatch. Heat operates at 300-metre neighbourhood scales. Most policy targets individual properties.

Two: wrong priority. Surface permeability matters three times more than vegetation. De-sealing outperforms tree planting for temperature reduction, yet receives a fraction of the investment.

Three: no universal template. Continental, Mediterranean, and maritime cities need fundamentally different strategies.

---

## Slide 22: Limitations (30 seconds)

Daytime only — nocturnal heat requires separate analysis. Correlations, not causes. 30m resolution identifies neighbourhoods, not street-level placement. Data gaps in peripheral areas.

But waiting for perfect evidence while heatwaves kill thousands is itself a policy choice.

---

## Slide 23: Closing (30 seconds)

[SLOW, DELIBERATE DELIVERY]

The evidence exists. The thresholds are identified. The priority zones are mapped.

And here's the encouraging part: every European city already possesses the data to replicate this analysis. Five open-source tools, zero proprietary dependencies. The transformation to cooler, more liveable cities is genuinely within reach.

I think Urban AI is uniquely positioned to bridge this gap between evidence and action. Your Guide covers water, air quality, curb digitization — but heat adaptation, which kills more Europeans than any other natural hazard, is still underrepresented. This framework is open, it's reproducible, and it's designed to work as a co-pilot for planners, not a replacement. I'd welcome the opportunity to develop this further — pilot cities, policy workshops, or integration into Urban AI's case study portfolio.

Thank you.

[PAUSE — wait for applause/reaction before Q&A]

---

## Slide 24: Discussion

[OPEN TO QUESTIONS — see Q&A preparation document]

---

## Timing Summary v4

| Section | Slides | Duration |
|---------|--------|----------|
| **The Problem** | 1–5 | 3:00 |
| **Data Sources** | 6–7f | 3:00 |
| **Method & Validation** | 9–10b | 2:00 |
| **Results** | 11–15 | 4:30 |
| **Policy & Impact** | 16–22 | 3:30 |
| **Close** | 23–24 | 1:00 |
| **Total content** | | **17:00** |
| Buffer | | 0:00 |
| Q&A | | 3:00 |
| **Grand total** | | **20:00** |

### Detailed Slide Timing

| Slide | Duration | Cumulative |
|-------|----------|------------|
| 1. Title | 0:20 | 0:20 |
| 2. Hook | 0:40 | 1:00 |
| 3. Translation gap | 0:40 | 1:40 |
| 4. Three gaps | 0:50 | 2:30 |
| 5. Research questions | 0:30 | 3:00 |
| 6. Study domain | 0:40 | 3:40 |
| 7. Data overview | 0:30 | 4:10 |
| 7a. GEE satellite | 0:15 | 4:25 |
| 7b. GlobalStreetscapes | 0:15 | 4:40 |
| 7c. EUBUCCO buildings | 0:15 | 4:55 |
| 7d. Urbanity networks | 0:15 | 5:10 |
| 7e. VoxCity SVF | 0:15 | 5:25 |
| 7f. 6-city voxels + toggle | 0:20 | 5:45 |
| 9. Method | 0:40 | 6:25 |
| 10. Performance | 0:30 | 6:55 |
| 10b. Cross-validation rigor | 0:20 | 7:15 |
| 11. SHAP (key slide) | 1:00 | 8:10 |
| 11b. How the blend works | 0:20 | 8:30 |
| 12. Observed vs. predicted | 0:30 | 9:00 |
| 13. Geography (side-by-side) | 0:40 | 9:40 |
| 13b. City spotlights | 0:20 | 10:00 |
| 14. 50% threshold (key slide) | 0:50 | 10:50 |
| 15. Hotspot classification | 0:30 | 11:20 |
| 16. Priority zones | 0:40 | 12:00 |
| 17. Vulnerability tiers | 0:25 | 12:25 |
| 18. Cooling potential | 0:30 | 12:55 |
| 19. City prescriptions | 0:20 | 13:15 |
| 20. Equity | 0:40 | 13:55 |
| 21. Three takeaways | 0:45 | 14:40 |
| 22. Limitations | 0:30 | 15:10 |
| 23. Closing | 0:30 | 15:40 |
| **Buffer** | **0:20** | **16:00** |
| **Q&A** | **4:00** | **20:00** |

### If constrained to 15 minutes

Slides 10b, 11b, 13b, and 19 are "quick glance" slides (15–20 seconds each) or can be moved to backup. Removing all four saves ~80 seconds.
