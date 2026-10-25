#!/usr/bin/env python3
"""Replay the released RNN checkpoint and 3-gram LM on the labelled test split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf
from tqdm import trange as terminal_trange

from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder
from neuralDecoder.utils import lmDecoderUtils


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/data"))
    parser.add_argument("--result", type=Path, default=Path("/results/released-checkpoint-3gram.json"))
    args = parser.parse_args()

    checkpoint_dir = args.workspace / "derived" / "rnns" / "baselineRelease"
    tfrecord_dir = args.workspace / "derived" / "tfRecords"
    lm_dir = args.workspace / "languageModel"
    for required in (checkpoint_dir / "args.yaml", tfrecord_dir, lm_dir / "TLG.fst", lm_dir / "words.txt"):
        if not required.exists():
            raise FileNotFoundError(required)

    config = OmegaConf.load(checkpoint_dir / "args.yaml")
    config["loadDir"] = str(checkpoint_dir)
    config["outputDir"] = str(checkpoint_dir)
    config["mode"] = "infer"
    config["loadCheckpointIdx"] = None
    config["testDir"] = "test"
    for index in range(len(config["dataset"]["datasetProbabilityVal"])):
        config["dataset"]["datasetProbabilityVal"][index] = 1.0 if index >= 4 else 0.0
        config["dataset"]["dataDir"][index] = str(tfrecord_dir)

    decoder = lmDecoderUtils.build_lm_decoder(
        str(lm_dir), acoustic_scale=0.8, nbest=1, beam=18
    )
    tf.compat.v1.reset_default_graph()
    inference = NeuralSequenceDecoder(config).inference()
    # Upstream imports tqdm.notebook.trange, which raises outside Jupyter before
    # decoding begins. Replace only the progress renderer for headless replay.
    lmDecoderUtils.trange = terminal_trange
    decoded = lmDecoderUtils.cer_with_lm_decoder(
        decoder,
        inference,
        outputType="speech_sil",
        blankPenalty=np.log(2),
        returnCI=True,
    )
    result = {
        "target": {"wer": 0.188, "description": "released checkpoint + 3-gram public test baseline"},
        "measured": {
            "wer": jsonable(decoded["wer"]),
            "character_error_rate": jsonable(decoded["cer"]),
            "trials": len(decoded["decoded_transcripts"]),
            "mean_decode_time_ms": float(np.mean(decoded["decode_time"])),
        },
        "decoded_transcripts": decoded["decoded_transcripts"],
        "true_transcripts": decoded["true_transcripts"],
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["measured"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
