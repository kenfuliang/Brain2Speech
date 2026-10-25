---
title: "WER local artifact and candidate-interface audit"
status: current-audit-not-performance-result
last_reviewed: 2026-10-07
owner: Brain2Speech
canonical_for: W01/W02 local prerequisite evidence for the WER investigation
---

# W01/W02 local prerequisite audit

Scope: safe read-only artifact/source inspection and synthetic accounting tests. No training, v1 collection, decoder launch or new WER collection occurred. The [WER plan](../research-plan-wer.md) owns the investigation; this is supporting evidence, not a competing report or protocol. [Machine-readable audit](../../research/runs/wer-local-audit-20261007-01.json) pins inspected sources and artifact checks.

## W01: artifact integrity and evaluation exposure

Fresh hashing confirms the stopped E008 checkpoint matches its export identity; original `logits.npz` and `predictions.json` also match their pinned hashes. Evaluation status is complete, 600 sentences / 3,682 reference words. This validates preserved inputs, not a new retraining or performance finding. [Existing independent metrics verification](../../research/runs/e008-stopped-wer-20261007-01/verification.json).

The immutable v1 manifest still has 14/5/5 chronological folds: 4,880 initial-training sentences, 1,800 development-training sentences, 2,120 final-training sentences; reserved tests are 480/200/200. The matrix status explicitly says final days opened and full comparison incomplete. Neither these final data nor E008's monitored public tests can be described as newly untouched evaluation data. Recipe training spans days that are development/final in v1; source-recipe controls therefore cannot substitute for past-days-only controls. [Manifest](../../research/data/manifest.json), [matrix state](../../research/runs/v1-comparison/status.json), [pause authority](../../research/runs/v1-comparison/pause-for-e008-wer-20261007.json).

Docker inspection returned no running containers in `colima-speechbci`; no worker was started. Free storage snapshot: 39,040,561,152 bytes (~36.36 GiB), above the 30 GiB floor but not evidence of enough headroom for arbitrary downloads/training. Recheck before each launch and budget dependencies, checkpoints, logs and temporary files.

**Decision:** preserved E008 diagnostics can be reused without retraining. Further source-recipe training needs a prospective performance agreement, checkpoint-selection/data-access contract and explicit restart direction. A new independent-evaluation claim needs audited access, not a reused monitored test set.

## W02: N-best support exists, but current outputs are top-1

```text
Preserved phoneme logits → unchanged WFST graph/search → ranked candidates + scores
                                                            ↓
                              reference labels → diagnostic-only oracle accounting
                              (labels never select operational output)
```

The actual Python binding path is `runtime/server/x86`, not only the similarly named `runtime/core` implementation:

- [`lmDecoderUtils.build_lm_decoder`](../../NeuralDecoder/neuralDecoder/utils/lmDecoderUtils.py) accepts `nbest`; `lm_decode(returnNBest=True)` returns `(sentence, ac_score, lm_score)` and resets the decoder after each trial.
- [`FinalizeSearch`](../../LanguageModelDecoder/runtime/server/x86/decoder/ctc_wfst_beam_search.cc) obtains a best path for N=1 and extracts N shortest paths from the lattice for N>1. These paths need not be N distinct normalized transcripts; report duplicates and actual counts.
- [`BrainSpeechDecoder.UpdateResult`](../../LanguageModelDecoder/runtime/server/x86/decoder/brain_speech_decoder.cc) exposes LM score and acoustic score, dividing the latter by acoustic scale. Scores are separate log-score components, not probabilities. Do not invent a new ranking formula without a specified scoring contract. Optional lattice rescoring is a separate intervention and remains disabled for the first diagnostic.
- [`lm_decoder` bindings](../../LanguageModelDecoder/runtime/server/x86/python/lm_decoder.cc) expose both scores. The [fixed worker](../../research/experiment/wfst_worker.py) sets N=1 and returns only transcripts/timing. Leave that frozen source unchanged; a new N-best worker/output identity is required.

**Measured technical evidence:** new [diagnostic accounting](../../research/reporting/nbest_metrics.py) passes 10 synthetic tests: N=1 identity, oracle monotonicity on nested candidates, exact edits, whitespace normalization, empty hypothesis versus missing candidates, duplicate transcripts, first-tie selection, corpus/per-day denominators, duplicate IDs and invalid-score rejection. An additional test executes the pinned author's Python wrapper with a fake native decoder and confirms candidate order, acoustic/LM score preservation and decode→finish→reset lifecycle; it does not execute the native search. [Tests](../../research/tests/test_nbest_metrics.py). The full current suite passes **64 tests** (exit 0). Runtime candidate extraction, oracle WER and rescoring benefit are **Not measured**. Synthetic/wrapper tests do not establish native decoder correctness or model performance. The machine-readable audit retains its earlier 10-test snapshot rather than rewriting historical evidence.

**Next diagnostic contract to freeze:** fixed checkpoint/logits/graph and current beam/scales/blank penalty; independently logged N=1 and N=50 extraction; compute N=1/10/50 curves from nested lists with fewer-than-N/duplicate accounting. Verify separate N=1 output against the existing fixed decoder before any performance interpretation; N-best lattice extraction could differ and must not be assumed identical. Select development trial identities before collection and label already-exposed data as diagnostic. Persist candidate lists and score components; recompute corpus/per-day oracle gap and error components independently. This paragraph is a proposal, not a frozen scientific protocol or a new decoder result.

**Decision:** candidate diagnosis is technically accessible without retraining. A large oracle gap would motivate selection/rescoring; a small gap requires search-coverage and neural-error analysis before attributing cause. No reference-selected oracle is ever reported as deployable WER. No baseline agreement or model-benefit threshold is inferred from this audit.

## 5. Native-decoder portability preflight — read-only update

The historical workload estimate motivates inspecting a possible native CPU implementation, not silently replacing the fixed decoder. In the actual `runtime/server/x86` build, the inspected local CMake file explicitly rejects non-Linux platforms and fetches the PyTorch 1.13.1 Linux CPU LibTorch archive. The Python binding includes Torch and performs float32 `log_softmax`, prior subtraction and blank-penalty adjustment before WFST search. Thus removing Torch or changing these numerics is not a purely packaging change. The inspected source does not establish an Apple Metal search implementation.

Local build dependencies at the standard Homebrew paths: CMake is present; OpenFST, gflags, glog and Eigen paths checked are absent. This checks those paths only, not every possible installed dependency location. The native research runtime reports **arm64 / Torch 2.14.1** and exposes its own CMake prefix; this is a different version from the original decoder's 1.13.1, so linkage/version compatibility and scoring parity remain unverified. No native build, dependency installation, download or runtime substitution was attempted.

Inspected local file hashes: `CMakeLists.txt` SHA-256 `0dea84f5387de1c6fb71149c486f6376878d470aa20a6c0950a8aebd49ea6348`; `python/lm_decoder.cc` SHA-256 `551f70cb0c1c0776c2bcd8fe2f0e626ba495958499f60a06b4b050dbb8bc717b`. These are current local source identities, not an upstream-clean revision claim; preserve existing compilation changes.

**Next gate / verdict:** native ARM CPU decoding is a candidate acceleration route, not ready or benchmarked. A separately versioned port must pin dependencies/compiler/graph and preserve interface, phoneme order, penalties, search options and scores; compare deterministic small graphs and fixed real-trial logits/transcripts against the original stack before measuring warmed throughput and memory. Broader exact-replay and score-tolerance criteria must be fixed before performance collection. Only verified compatible decoding could support a versioned execution change; the frozen v1 protocol is not altered by this preflight. No GPU or speedup claim follows from native Torch availability.
