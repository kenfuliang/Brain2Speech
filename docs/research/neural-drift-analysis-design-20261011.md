---
title: "Design for separating neural-data shifts from encoder drift"
status: proposed
last_reviewed: 2026-10-11
owner: Brain2Speech
scope: supporting exploratory analysis design; not a frozen experiment or execution authorization
---

# Design for separating neural-data shifts from encoder drift

This note supports [I003: data-first neural-property analysis](../research-notes.md#i003--data-first-neural-property-analysis-2026-10-07). The ledger remains the decision owner; [report definitions](../research-results-report.md#2-definitions-and-reporting-principles) govern claims. **No data analysis, cleaning, training or adaptation was run.** All proposed metrics are Not measured.

## What can be shown

| Observation | Defensible initial label | Main alternative explanations |
|---|---|---|
| Raw/normalized features differ by day | Recorded neural-feature distribution shift | Recording changes, content/duration mix, normalization, task strategy |
| Same fixed mapping and encoder produce different latent distributions | Fixed-encoder latent distribution shift | Input shift or changed task sampling; not necessarily changed biological coding |
| Different encoders produce different outputs on identical probe data | Model functional drift | Optimization, adaptation, coordinate symmetries |
| Content-conditioned mapping changes after controls | Evidence consistent with recorded neural-to-content mapping drift | Unmeasured behavior or acquisition changes remain |

Biological representational drift is a stronger interpretation. Multiunit counts and power do not track identified neurons; model weights are not neural synapses. A shift can be useful for decoding research without proving a biological cause.

## Primary-source basis

MINDFUL compares reference and target feature distributions with a Gaussian mean/covariance KLD score, using reference-fit PCA and a fixed decoder. Human motor data support an association with decoding degradation. Its discussion distinguishes changes in recorded elements, recording devices and encoded information; short windows can reflect movement-content differences. This motivates a **speech-specific diagnostic**, not automatic transfer of its thresholds or effect sizes. [Harrison et al. 2024, Results/Discussion and Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC11494208/).

Gallego et al. show that different recorded neuronal populations can express stable aligned motor dynamics. Their PCA/CCA comparison controls consistent behavior, illustrating why apparently changing coordinates are not automatically changing underlying computation. This is motor evidence, not proof of invariant speech dynamics. [Gallego et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7007364/).

Representation comparisons need the **same examples**: CKA compares activation matrices for corresponding observations. Linear CKA is invariant to orthogonal transformations and isotropic scaling, but not arbitrary invertible transforms. Similarity invariance can hide changes that matter to a fixed downstream readout; high CKA is not identical decoding or biological stability. Small-sample/high-dimensional CCA has additional degeneracies. [Kornblith et al. 2019, §§2–3](https://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf).

Original LFADS stitching shares latent dynamics while permitting session-specific read-ins/read-outs. Its existence reinforces the distinction between observation-coordinate changes and latent dynamics; it does not make independently fitted latent axes directly comparable. [Pandarinath et al. 2018, Fig. 4 and Methods §1.9](https://pmc.ncbi.nlm.nih.gov/articles/PMC6380887/).

## Low-cost proposed first pass

```text
Audit channels/content/timing → matched train/development trial sets
                               ├→ recorded-feature statistics
                               └→ one fixed mapping + fixed encoder
                                                ↓
                                      shared reference-fit PCA
                                                ↓
                         day shifts vs within-day controls + decoding errors
```

1. Freeze session/trial IDs, feature units/channel correspondence and permitted data access. Analyze count and power channels separately. Fit scales/PCA on initial training only; apply identical fixed axes to every compared day. Preserve anomalous trials with flags, not automatic deletion. Report raw versus fixed-reference normalization separately from adaptive normalization.
2. Audit repeated sentences. Prefer exact same prompt across days, balancing trial duration, order/block and sample count. If unavailable, match phoneme composition/length and disclose weaker control. Transcript phones do not provide bin timing: use sentence/trial summaries initially; validated training-only forced alignment is a separate prerequisite for phoneme trajectories. Never call unlabeled bins silence without verified cues/coverage.
3. Compare between-day distances with equal-sized within-day split/repeat distances. Use trial/block resampling, not treating autocorrelated bins as independent replicates. Report bootstrap uncertainty, matched-content counts and coverage. With few sessions, trend/correlation estimates are exploratory.
4. Freeze **the complete observation mapping**, not only checkpoint file. Willett has day-specific input transforms: automatic day routing changes the map. Compare fixed trained reference adapter + shared encoder against native trained day adapters as distinct conditions. Do not silently use an untrained branch. E017 is multiday/offline-preprocessed and suitable only for retrospective descriptive analysis. A chronological claim requires a permitted initial-training-only causal encoder and the accepted acquisition-delay definition.

## Metrics and crossed-model follow-up

Initial metrics: standardized channel mean/scale shifts; covariance distance; reference-PCA projected distribution distance using regularized Gaussian KLD (declare direction/shrinkage); PCA variance retention/effective rank; and content-matched latent centroid distances relative to within-day variation. Two-dimensional plots illustrate only a fraction of the evidence. A separately fitted subspace/principal-angle sensitivity analysis must use the same channel space/rank and be labeled as distinct from the fixed-PCA view.

Relate prespecified shift scores to fixed-decoder PER/WER and content-conditioned confusion changes on permitted development trials, with uncertainty and no causal claim. A label-free drift score does not establish content-mapping drift; conversely, unchanged marginal distributions can hide changed content mappings.

Only after comparable adaptation checkpoints exist, evaluate a **crossed encoder × data-day matrix**: entries contain fixed-probe embeddings/logits and decoding metrics for encoder E_i on day D_j. Hold normalization, feature construction and adapter routing explicit. Comparing E_1(D_1) with E_2(D_2) alone confounds model and data. For model change, compare E_1(P) and E_2(P) on the exact same held-out probe P; report CKA plus raw changes/logit divergence and performance. Fit any Procrustes/CCA alignment on separate calibration examples and evaluate on disjoint probes; aligned results must accompany unaligned results. Sequential weights help ancestry, but equal-budget same-day updates and independent seeds are needed to bound optimization variability.

## Decision and claim boundary

Predominantly gain/offset shifts suggest a normalization control; stable task geometry with changing observations suggests a day-adapter/shared-dynamics test; localized channel changes suggest robustness tests. Residual content-conditioned changes motivate stronger recalibration, not a declaration of biological drift. Promotion requires a prospective protocol, matched controls, independent evaluation and the existing research reporting rules. No new experiment is accepted by this design note.
