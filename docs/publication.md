---
title: "Brain2Speech public repository scope and provenance"
status: current
last_reviewed: 2026-10-25
owner: Brain2Speech
canonical_for: public repository contents, publication boundary, and source attribution
---

# Public repository scope and provenance

This public snapshot packages existing research. It does not update scientific
findings or authorize training. The [experiment ledger](research-notes.md) owns
methods, metrics, findings, and experiment status. The
[comparative report](research-results-report.md) owns reader-facing comparisons.
Their dated observations remain dated observations.

## Included and excluded evidence

Published: source code, tests, frozen configuration and artifact manifests,
research plans, literature audits, the ledger/report, and selected report figures.
Upstream authorship and lineage are preserved, with Android signing material
removed throughout reachable published history.

Excluded: downloaded data/papers, extracted archives, model checkpoints,
virtual environments, runtime caches, and raw outputs under `research/runs/` and
`reproduction/results/`. These remain local; publication deletes none of them.
Some report figures and linked receipts refer to those excluded folders; several
figure targets were already absent from the local workspace at publication.
Missing plots are not regenerated or substituted during publication.
Their absence from this repository is not a claim that the supporting evidence
has been independently downloaded or verified by a reader.

A fresh clone needs the separately downloaded public artifacts before running
real-data experiments. Consult [the reproduction workflow](../reproduction/README.md)
and its pinned archive manifest. Tests using data, checkpoints, or optional
runtimes can also require local dependencies; source publication alone does not
establish that every experiment can run on another machine.

## Sources and licensing

- speechBCI: [fwillett/speechBCI](https://github.com/fwillett/speechBCI),
  base commit `522a9605ccf0818def80b6fd80007598e790408e`.
- BIT: [yzhang511/bit](https://github.com/yzhang511/bit), pinned Git submodule
  `research/vendor/bit-f107ae9` at
  `f107ae988d782609b78d1d7488490cca9fb7a103`. Its MIT license remains with the source.
  Initialize with `git submodule update --init --recursive` after cloning.
- Preserve existing component and third-party license notices. No new blanket
  license is applied to upstream code. The missing root speechBCI license and
  component-specific terms are documented in the canonical
  [source and reproducibility audit](research/willett-2023-speech-neuroprosthesis-reproducibility.md#licensing-and-availability-caveats).

The root README distinguishes independent results from upstream reference
claims and links to the original README at the pinned source commit. Original
upstream README versions remain in Git history. Independent measurements are
attributed in the ledger and report.

## Publication validation (2026-10-25)

- `python -m pytest research/tests -q`: **202 passed** in the existing BIT
  environment (Python 3.11, torch 2.14.1, numpy 2.4.6, transformers 4.51.3,
  pytest 9.1.1). The base v1 environment lacks transformers and cannot collect
  the BIT tests; use the appropriate optional runtime.
- Secret scanning of the proposed source commit and all 45 upstream commits
  found no credentials. This is a publication check, not a scientific audit.
- No newly added file exceeds 10 MiB; raw datasets/checkpoints and interview
  materials are excluded. Existing upstream files and notices remain intact.

These checks do not rerun training, decode the dataset, or establish new results.

## Sensitive-data review (2026-10-25)

Scope: the published tracked snapshot, the complete 46-commit history before
this documentation revision, newly added report-image metadata, and the
Brain2Speech additions relative to the pinned speechBCI base. Ignored local
research inputs and outputs were not uploaded or removed.

- Gitleaks found no credential matches in the 46-commit history. Pattern checks
  found no private interview/business materials, personal home paths, private
  network addresses, or credential files in the Brain2Speech additions.
  Report-image metadata had no detected local-path, GPS, or personal-email marker.
- The review found an inherited WeNet Android example signing keystore and
  hardcoded Gradle signing passwords; the generic secret scan did not flag them.
  On 2026-10-25, the keystore, explicit release signing configuration, and both
  passwords were removed from the current source tree. Debug signing remains
  managed locally by the Android Gradle plugin; releases are unsigned until a
  private local signing configuration is supplied. Signing-file extensions are
  ignored to prevent accidental additions. That first source-tree removal did
  not purge old commits; the subsequent history rewrite is recorded below.
  After removal, all **202 research tests passed** in 5.95 seconds in the existing
  BIT environment. Research, reproduction, NeuralDecoder, and x86 decoder source
  trees are unchanged. Signing-reference checks and the staged secret scan pass.
  The Android app was not built, and no real-data training/decoding was rerun.
- The upstream test-logit NumPy fixture is also unchanged. Upstream notebooks
  and third-party sources retain their original public paths and metadata.
  Author identities in upstream history remain attributed to their authors.
- The initial Brain2Speech publication commit exposed the author's personal
  email. With explicit user authorization, that commit is rewritten with the
  GitHub no-reply identity; subsequent commits use the same identity. Local
  evidence is preserved. Rewriting branch history cannot recall earlier clones
  or guarantee removal from GitHub caches and previously shared commit URLs.

This is a bounded repository review, not a guarantee that every possible form
of sensitive information has been detected. Published findings remain dated
research records, with the evidence-availability limitations described above.

## History cleanup and verification (2026-10-25)

The public `main` history was rewritten in an isolated mirror using
`git-filter-repo --sensitive-data-removal`. All 52 commits were processed:
the Android signing-key path was removed and its release-signing configuration,
including hardcoded passwords, was removed from historical Gradle blobs.
Original authorship and research source contents were retained. Commit IDs
changed; source pins in literature audits refer to the original external source.

Verification before the audit documentation update:

- Public refs: one branch (`main`), no tags or pull requests; no releases or
  Actions artifacts. All 2,835 reachable Git blobs were inspected for the removed
  key/reference, signing-password assignment, personal email, private key PEM,
  and private interview/home-path markers: zero matches. The original binary
  key object is absent from the sanitized reachable object set.
- Gitleaks scanned the rewritten history (44 commits with scannable changes,
  approximately 48.12 MB), with zero findings. The pinned BIT submodule snapshot
  also produced zero findings. Manual literal-password review found only the
  upstream Redis documentation's generic `auth` examples, not live credentials.
- The rewritten head tree was identical to the pre-rewrite head tree before this
  documentation update. Research, reproduction, neural decoder, and x86 decoder
  contents were therefore unchanged by history sanitation.
- A clean checkout passed 200 tests; two optimizer-parity tests require ignored
  local reference files under `reproduction/results/retraining/`. With those
  preserved files supplied locally, **all 202 tests passed in 5.79 seconds**.
  The fixtures remain excluded, and no training or real-data decoding was rerun.

**Public-object limitation, observed after the force-push:** anonymous GitHub API
requests return 404 for the current signing-key path, but 200 for its old upstream
commit and for the obsolete personal-email commit. The repository was still in
the upstream fork network at this check. Rewritten branch history alone does not
remove those public objects. Standalone-repository conversion or GitHub support
is required before claiming that those old URLs no longer expose the material.

Use a fresh clone after the rewrite. Do not merge or push old branches or tags
back into the sanitized history. Private local backups are outside this public
repository and are not included in pushes.
