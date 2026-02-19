# Q&A Preparation — UrbanAI Presentation

## Strategy

The UrbanAI audience bridges academics and practitioners. Expect questions from both angles:
- **Academics** will probe methodology, validation, and limitations
- **Practitioners** will ask about implementation, transferability, and policy
- **Both** will engage with equity and governance tensions

**Response pattern:** Acknowledge the concern → Address with evidence → Pivot to what the finding still contributes.

---

## Likely Questions & Prepared Responses

### 1. "How do you know these interventions would actually produce the cooling you predict?"

**Short answer:** We don't — and that's stated explicitly. The model learns correlations from observational data, not causal effects of changing urban configurations. If tree canopy and soil moisture correlate in training data, the model can't distinguish whether cooling comes from trees or underlying soil conditions.

**Pivot:** But the predictions are conservative relative to empirical literature. Schwaab et al. document 8-12°C LST differences between trees and sealed surfaces in Central Europe. Our 1.26°C mean prediction reflects realistic implementation intensities, not theoretical maxima. What's needed now is longitudinal validation through pilot interventions with before-after monitoring.

---

### 2. "Your analysis only covers daytime. What about nocturnal heat, which drives most mortality?"

**Short answer:** Correct limitation. Landsat overpasses at 10:30 AM capture peak surface heating but miss nocturnal thermal mass release. Nocturnal UHI operates through different mechanisms — stored heat radiation, reduced ventilation, humidity trapping — and daytime priorities may not align with nocturnal ones.

**Pivot:** ECOSTRESS and geostationary platforms now offer diurnal data. This framework's architecture is designed to accommodate nocturnal targets with minimal modification — the feature engineering and spatial framework transfer directly, only the response variable changes. This is an active area I'd like to pursue further.

---

### 3. "Why XGBoost rather than deep learning or spatial models like GeoAI?"

**Short answer:** Interpretability. Gradient boosting with SHAP provides additive feature attribution — every prediction decomposes into specific contributions from impervious fraction, water proximity, vegetation. Neural networks achieve comparable accuracy on urban heat tasks but sacrifice this transparency. The goal isn't maximum accuracy — it's decision support for planners who need to understand *why* a neighbourhood is hot.

**Pivot:** Physics-informed feature engineering can match black-box performance while maintaining transparency. The R² of 0.84 with 118 interpretable features suggests we don't need to sacrifice explainability for accuracy in this domain.

---

### 4. "Athens underperforms — does this invalidate the framework for Mediterranean cities?"

**Short answer:** It identifies scope conditions, which is itself a contribution. Athens's Attica Basin creates katabatic flows, thermal inversions, and wind channelling that operate independently of built form. The model correctly learns morphological effects but can't capture terrain-driven circulation from building features alone.

**Pivot:** Barcelona (R² = 0.926) is also Mediterranean, so the issue isn't climate zone but topographic complexity. The finding tells us: morphology-centric frameworks achieve high accuracy in flat terrain and face explanatory ceilings in basin cities. Future work should stratify by topographic complexity, not just climate classification.

---

### 5. "How do you address green gentrification risks?"

**Short answer:** The framework identifies *where* to intervene but cannot prevent displacement dynamics. Anguelovski's work documents specific mechanisms: greening raises property values, landlords increase rents, long-term residents are priced out. The vulnerability weighting ensures interventions are directed toward high-need areas, but without accompanying tenure protections, we risk improving places while displacing people.

**Pivot:** This is why I frame it as "technical optimisation without equity safeguards is incomplete." The composite scoring (40% heat risk, 35% vulnerability, 25% cooling potential) operationalises a middle path, but the weighting reflects analytical judgment rather than democratic deliberation. Responsible implementation requires coupling physical interventions with rent stabilisation, community land trusts, and participatory processes.

---

### 6. "How does this compare to or improve on Local Climate Zone classification?"

**Short answer:** LCZ provides useful coarse screening — compact high-rise zones average 2.2°C warmer than open low-rise, confirming the typology. But within-zone variance approaches between-zone differences: identically classified neighbourhoods can differ by 1.5°C. LCZ categories can't capture the fine-grained heterogeneity of European historical cores where medieval streets, 19th-century boulevards, and postwar blocks create micro-scale thermal diversity.

**Pivot:** This framework complements LCZ by operating at continuous morphological scales rather than categorical classifications. The 30m grid captures block-level variation invisible to zone-based approaches. For screening, LCZ is excellent. For intervention targeting, continuous metrics are necessary.

---

### 7. "GlobalStreetscapes covers less than 50% in some cities. Isn't that a major reliability concern?"

**Short answer:** Yes. Street-level imagery systematically favours well-documented central districts while undersampling peripheral areas — precisely where vulnerability tends to concentrate. The model handles this through city-specific median imputation with binary flags, but imputation can't recover the geometric variance that makes street-view features theoretically valuable.

**Pivot:** This is actually one of the key methodological findings: comprehensive coverage of simple features (GEE satellite data, 95-99% coverage) outperforms incomplete coverage of sophisticated features (street-level imagery, 29-71%). The practical implication is that urban analytics investments should prioritise coverage completeness before measurement sophistication.

---

### 8. "What would it take to apply this framework to a new city?"

**Short answer:** All five data sources have pan-European coverage, so any European city is feasible with existing data. The pipeline is open-source. The main requirements are: (1) a Landsat summer composite for the target region, (2) EUBUCCO building data, (3) street-level imagery availability, and (4) a climate-zone classification to determine which specialist model to apply.

**Pivot:** The real question is whether the learned relationships transfer. For cities within the same climate zone as the training cities, direct application is reasonable. For novel climate contexts (Nordic, arid), recalibration with local data would be essential. The framework's modular design makes this straightforward — retrain the specialist model, keep the feature engineering.

---

### 9. "You mention 50% de-sealing — is that politically feasible in dense European historical cores?"

**Short answer:** Almost certainly not in all locations. Heritage protections, property rights, underground infrastructure, and existing land use create constraints that a quantitative framework can't capture. The 50% threshold is analytically optimal but may require incremental approaches: permeable paving during routine maintenance, green roofs during building renovation, tactical de-sealing in car parks and service roads.

**Pivot:** The non-linear response is actually the actionable insight here: 30% depaving achieves only 49% of maximum cooling, while 50% achieves 100%. This means incremental approaches that stop short of the threshold waste resources. Cities should concentrate transformative interventions in priority zones rather than distributing modest improvements everywhere. Target the headroom: car parks, industrial yards, oversized road surfaces.

---

### 10. "How does this connect to EU policy frameworks?"

**Short answer:** The EU Mission on Adaptation to Climate Change emphasises nature-based solutions and climate resilience. The European Climate Risk Assessment identifies heat as the fastest-escalating climate threat. This framework provides the diagnostic precision these policies need but lack — where to invest, what to prioritise, and how strategies should vary across contexts.

**Pivot:** There's a specific policy tension worth highlighting: EU frameworks increasingly emphasise tree planting and green infrastructure. This evidence suggests de-sealing delivers greater cooling returns per euro invested, which challenges — but doesn't contradict — the green infrastructure emphasis. Trees provide co-benefits (shade, biodiversity, air quality, stormwater) that de-sealing alone doesn't. The optimal strategy combines both, but the resource allocation should be evidence-informed rather than assumption-driven.

---

## Questions to Ask Back (if appropriate)

If the conversation allows, these questions position Gerardo as a collaborator, not just a presenter:

1. "How are cities you work with currently prioritising cooling investments? Is there alignment with what this evidence suggests?"
2. "UrbanAI's work on screenless cities is fascinating — have you explored how thermal data might be communicated through urban interfaces rather than dashboards?"
3. "What would be most useful for the practitioners in this room: the diagnostic framework, the intervention scenarios, or the priority zone methodology?"
