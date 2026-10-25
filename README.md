# Brain2Speech

Independent brain-to-text reproduction and research by Ken-Fu Liang, based on
[Frank Willett and collaborators' speechBCI](https://github.com/fwillett/speechBCI)
at commit `522a9605ccf0818def80b6fd80007598e790408e`.

Start with the [comparative research report](docs/research-results-report.md),
[canonical experiment ledger](docs/research-notes.md), and
[reproduction instructions](reproduction/README.md).

**Public snapshot: 2026-10-25.** Research findings retain their recorded review
dates (latest canonical review: 2026-10-12); publication does not mean that
experiments have been rerun or that unfinished work has completed. The released
checkpoint replay measured **18.8756% offline WER on 600 public-test trials**
against the upstream 18.8% reference. See the ledger for the tolerance caveat,
training comparisons, negative findings, and incomplete experiments. This is
brain-to-text research; it does not reproduce the live participant experiment.

This repository publishes code, protocols, reports, and selected figures.
Datasets, downloaded papers, model weights, environments, and raw run artifacts
are excluded. Links into `research/runs/` or other ignored artifact folders refer
to preserved local evidence and will not resolve in a fresh clone. See the
[publication scope and provenance](docs/publication.md).

```mermaid
flowchart LR
    A[Public source artifacts] --> B[Verify and reproduce offline baseline]
    B --> C[Controlled research experiments]
    C --> D[Audit metrics and evidence]
    D --> E[Canonical ledger and comparative report]
```

## Upstream project: A high-performance speech neuroprosthesis

The upstream overview below is retained for attribution and setup context.
References to the original paper's results and authors describe speechBCI,
not independent Brain2Speech findings.
[![System diagram](SystemDiagram.png)](https://www.nature.com/articles/s41586-023-06377-x)

## Brain2Speech research program

Our research thesis, mandatory experiment methodology, metric definitions, iteration scorecard, findings, and next-step decisions are maintained in the canonical [Brain2Speech research notes and experiment ledger](docs/research-notes.md). Detailed paper reviews and run artifacts support that ledger but do not redefine its experiment status or decisions.

The [v1 research plan](docs/research-plan-v1.md) specifies causal LFADS-inspired Brain2Text, chronological cross-day adaptation, and sequential encoder-update analysis.

The [single comparative research report](docs/research-results-report.md) presents Willett's reference results, our reproduced baseline, and the v1 proposal using the same result format.

## Overview

This repo is associated with this [paper](https://www.nature.com/articles/s41586-023-06377-x), [dataset](https://doi.org/10.5061/dryad.x69p8czpq) and machine learning [competition](https://eval.ai/web/challenges/challenge-page/2099/overview). The code contains the RNN decoder (NeuralDecoder) and language model decoder (LanguageModelDecoder) used in the paper, and can be used to reproduce the core offline decoding results. 

The jupyter notebooks in AnalysisExamples show how to [prepare the data for decoder training](AnalysisExamples/rnn_step1_makeTFRecords.ipynb), [train the RNN decoder](AnalysisExamples/rnn_step2_trainBaselineRNN.ipynb), and [evaluate it using the language model](AnalysisExamples/rnn_step3_baselineRNNInference.ipynb). Intermediate results from these steps (.tfrecord files for training, RNN weights from my original run of this code) as well as the 3-gram and 5-gram language models we used are available [here](https://doi.org/10.5061/dryad.x69p8czpq) (in the languageModel.tar.gz, languageModel_5gram.tar.gz, and derived.tar.gz files). 

Example neural tuning analyses (e.g., classification, PSTHs) are also included in the AnalysisExamples folder ([classificationWindowSlide.ipynb](AnalysisExamples/classificationWindowSlide.ipynb), [examplePSTH.ipynb](AnalysisExamples/examplePSTH.ipynb),[naiveBayesClassification.ipynb](AnalysisExamples/naiveBayesClassification.ipynb), [tuningHeatmaps.ipynb](AnalysisExamples/tuningHeatmaps.ipynb), [exampleSaliencyMaps.ipynb](AnalysisExamples/exampleSaliencyMaps.ipynb)).

## PyTorch Implementation

We have posted a new PyTorch implementation of this original TensorFlow RNN decoder [here](https://github.com/cffan/neural_seq_decoder).

## Train/Test/CompetitionHoldOut Partitions

We have partitioned the data into a "train", "test" and "competitionHoldOut" partition (the partitioned data can be downloaded [here](https://doi.org/10.5061/dryad.x69p8czpq) as competitionData.tar.gz and has been formatted for machine learning). "test" contains the last block of each day (40 sentences), "competitionHoldOut" contains the first two (80 sentences), and "train" contains the rest. 

The transcriptions for the "competitionHoldOut" partition are redacted. See [baselineCompetitionSubmission.txt](AnalysisExamples/baselineCompetitionSubmission.txt) for an example submission file for the [competition](https://eval.ai/web/challenges/challenge-page/2099/overview) (which is generated by [this notebook](AnalysisExamples/rnn_step3_baselineRNNInference.ipynb)).

## Results

When trained on the "train" partition and evaluated on the "test" partition, our original run of the code achieved an 18.8% word error rate (RNN + 3-gram baseline) and a 13.7% word error rate (RNN + 5-gram + OPT baseline). For results on the "competitionHoldOut" partition, see the baseline word error rates [here](https://eval.ai/web/challenges/challenge-page/2099/leaderboard/4944). 

## Installation

NeuralDecoder should be installed as a python package (pip install -e .) using Python 3.9. LanguageModelDecoder needs to be compiled first and then installed as a python package (see LanguageModelDecoder/README.md). 

