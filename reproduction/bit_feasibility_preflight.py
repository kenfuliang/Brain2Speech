"""Read-only external metadata audit; preserve a local BIT-inspired run contract.

No model download, environment mutation, training, or v1/WER interruption.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import urllib.request


def metadata(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if args.output.exists():
        raise FileExistsError('Preserve existing preflight evidence')
    model = metadata('https://huggingface.co/api/models/lmms-lab/Aero-1-Audio?blobs=true')
    code = metadata('https://api.github.com/repos/yzhang511/bit/commits/main')
    files = [{'name': f['rfilename'], 'bytes': f.get('size', 0)} for f in model['siblings']]
    model_bytes = sum(f['bytes'] for f in files)
    free = shutil.disk_usage(root).free
    floor = 30 * 1024**3
    reserve = 2 * 1024**3
    required = floor + model_bytes + reserve
    value = dict(
        recorded_at=datetime.now(timezone.utc).isoformat(),
        experiment='E009 BIT-inspired GRU-to-audio-LLM feasibility',
        status='blocked-storage' if free < required else 'preflight-storage-pass',
        storage=dict(free_bytes=free, minimum_remaining_bytes=floor,
                     model_download_bytes=model_bytes, dependency_and_output_reserve_bytes=reserve,
                     required_free_bytes=required, shortfall_bytes=max(0, required-free)),
        model=dict(repository='lmms-lab/Aero-1-Audio', revision=model['sha'], files=files),
        upstream=dict(repository='yzhang511/bit', revision=code['sha']),
        encoder=dict(checkpoint=str(root/'reproduction/results/retraining/pytorch-retrain-s17-20261007-01/latest.pt'),
                     sha256='3110a3c1e1996fc94c70c96a6fd180fe83cdd59a1c58a0240169bc1f583831d7',
                     frozen=True, representation='1024-dimensional last-GRU hidden states, not phoneme logits'),
        proposed_interface='trainable MLP from neural hidden states into audio-LLM text-embedding space; LoRA adapters',
        proposed_training=dict(seed=17, train_sentences=64, evaluation_sentences=16,
                               exclude_evaluation_prompts_from_training=True,
                               initial_steps=100, checkpoint_selection='development text loss only',
                               loss='masked target-token cross entropy; contrastive loss is a later ablation'),
        acceptance=['finite forward loss and gradients', 'projector and LoRA parameters update',
                    'frozen encoder and base LLM weights unchanged',
                    'training text loss decreases on a fixed diagnostic set',
                    'report held-out text loss before/after, even if it does not improve',
                    'zero, shuffled, and language-only controls for text loss and generated transcripts',
                    'checkpoint reload consistency'],
        caveats=['Not exact BIT reproduction: GRU replaces pretrained bidirectional Transformer',
                 'Small-subset feasibility, not low-WER or foundation-model claim',
                 'Encoder previously monitored public-test PER; downstream split is not an untouched encoder holdout',
                 'Held-out improvement and neural dependence are measured outcomes, not guaranteed'],
        training_started=False, model_downloaded=False, existing_artifacts_deleted=False,
        concurrent_wer='Do not interrupt or duplicate the E008 resident decoder',
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps({k: value[k] for k in ['experiment','status','storage','training_started']}, indent=2))


if __name__ == '__main__':
    main()
