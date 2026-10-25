---
title: "Neural drift claims in BIT, LFADS and NoMAD"
status: current
last_reviewed: 2026-10-11
owner: Brain2Speech
scope: supporting primary-paper claim audit; not a new experiment or performance result
---

# Neural drift claims in BIT, LFADS and NoMAD

The [canonical ledger](../research-notes.md) owns accepted findings and decisions; the [report definitions](../research-results-report.md#2-definitions-and-reporting-principles) govern evidence claims. This read-only literature audit changes no experiment, goal or frozen protocol.

## Short answer

**Yes, but with different scopes:** BIT explicitly discusses recording nonstationarity and probe drift; original LFADS demonstrates multi-session neural stitching and discusses potential device stability; NoMAD directly tests unsupervised stabilization across changing recording sessions. These are not interchangeable claims.

| Work | Relevant evidence | What it does not establish |
|---|---|---|
| BIT | Normalization and masked reconstruction are described as mechanisms to mitigate recording drift | Independent causal, chronological unseen-day adaptation or autonomous long-term speech stabilization |
| Original LFADS | Shared dynamics with session-specific observation mappings across separately recorded populations | Zero-shot new-day speech decoding or automatic unlabeled calibration |
| NoMAD | Unlabeled target-session alignment with fixed reference dynamics and behavior decoder | A speech-WER improvement or prospective online human speech deployment |

## BIT: explicit discussion, not an isolated drift-correction result

The audited BIT version is *Decoding inner speech with an end-to-end brain-to-text neural interface*, arXiv:2511.21740v2. Section 3.1 describes across-day normalization for Utah-array nonstationarity. Section 3.3 attributes probe-drift mitigation to temporal masking/reconstruction. Appendix A.3 says omitting normalization substantially worsens WER because of data drift; its Table 3 is a **feature/PER ablation**, not a quantified normalization-only drift ablation. Section 7 still identifies nonstationarity, plasticity and co-adaptation as open needs. [Primary paper, §§3.1/3.3/7 and Appendix A.3](https://arxiv.org/html/2511.21740v2).

Interpretation: BIT acknowledges drift and incorporates robustness mechanisms; this is not equivalent to demonstrating chronological drift correction. Appendix A.4 permits unlabeled evaluation/holdout recordings in pretraining, and §3.2 uses bidirectional attention. Its neural/text contrastive objective aligns modalities, not explicitly matching the same content across days. [Primary paper, §§3.2/3.3 and Appendix A.4](https://arxiv.org/html/2511.21740v2).

## Original LFADS: multi-session stitching is important prior art

Pandarinath et al. (2018) demonstrate one shared encoder/generator/factor representation with session-specific read-in/read-out mappings, using 44 monkey motor recording sessions spanning 162 days. Recorded populations can be disjoint. Stitched factors improve kinematic prediction compared with separate session models. Methods initialize mappings through condition-averaged PCA/regression; the demonstrated read-ins remain fixed. Thus neural-only model likelihood should not be confused with entirely task-information-free mapping initialization. [Primary paper, Fig. 4, “Stitching together data from multiple sessions” and Methods §1.9](https://pmc.ncbi.nlm.nih.gov/articles/PMC6380887/).

The Discussion proposes that stitching may improve device stability and that a causal variant could help real-time applications. Standard LFADS uses future as well as past data. This is relevant prior art for stable latent structure under changing observations, **not a demonstrated autonomous new-day speech decoder**. [Primary paper, “Overview of LFADS” and Discussion](https://pmc.ncbi.nlm.nih.gov/articles/PMC6380887/).

## NoMAD: the direct stabilization study

Karpowicz et al. (2025), *Stabilizing brain-computer interfaces through alignment of latent dynamics*, explicitly targets neural-interface instabilities. It updates the target-session neural-to-latent mapping without target behavior labels, using reference-session LFADS dynamics and a fixed behavior decoder. The source reports motor-decoding stability across 95 days of monkey wrist-force recordings and 38 days of reaching, with a separate retrospective analysis of closed-loop-collected data. Initial reference calibration is supervised; target alignment is unsupervised. These are motor R²/stability results, not brain-to-text WER. [Primary paper, Abstract, Results and Discussion](https://pmc.ncbi.nlm.nih.gov/articles/PMC12089531/).

```text
Changing recorded neurons / channel statistics
                     ↓
Session-specific observation mapping
                     ↓
Shared latent dynamics → fixed behavioral decoder
```

## Boundaries for our proposal

“Shared dynamics plus day-specific mapping” is already prior art, especially LFADS stitching and NoMAD. A speech extension must test a distinct claim—such as causal future-day WER or calibration-label efficiency—against matched controls. Encoder parameter movement alone is not evidence of biological representational drift. Recording-interface changes, task/content differences and model optimization must be distinguished.

## Access and inspection limitations

BIT's arXiv v2 primary text and original LFADS PMC text were accessible on 2026-10-11. OpenReview redirected to verification; NoMAD's direct PMC page intermittently showed verification, while its primary-paper search text exposed Abstract/Results/Discussion. The NoMAD claim here is limited to those accessible sections; no new supplementary or code audit was performed. The [existing cross-day methods note](cross-day-neural-drift-methods.md#4-latent-dynamical-models) owns its previously checked causal-sliding-window implementation discussion.
