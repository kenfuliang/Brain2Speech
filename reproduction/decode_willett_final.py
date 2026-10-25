"""Serial E017 fixed3gram scoring; legacy and standard words kept separate."""
import argparse
import json
from pathlib import Path
import time

from export_willett_final import require_idle
from decode_bit_full import worker_ids
from train_pytorch_recipe import sha, write
from research.experiment.evaluator import FixedDecoder
from research.metrics import edits, summarize
from verify_bit_full import distance


def score_text(reference, transcript, tokenization):
    assert tokenization in ('legacy-literal-space', 'standard-whitespace')
    split = (lambda text: text.split(' ')) if tokenization == 'legacy-literal-space' else (lambda text: text.split())
    words = edits(split(reference), split(transcript))
    characters = edits(list(reference), list(transcript))
    assert words['errors'] == distance(split(reference), split(transcript))
    assert characters['errors'] == distance(list(reference), list(transcript))
    return words, characters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    require_idle()
    assert not (run/'decoded-predictions.json').exists(), 'Never replace a scored run'
    audit = json.loads((run/'export-verification.json').read_text())
    identity = json.loads((run/'identity.json').read_text())
    assert audit['passed'] and audit['independent_greedy_phone_edits'] and audit['independent_ctc']
    assert audit['export_identity_sha256'] == sha(run/'identity.json')
    for name, expected in identity['artifacts_sha256'].items():
        assert sha(run/name) == expected
    rows = json.loads((run/'predictions.json').read_text())
    assert len(rows) == 600 and len({r['trial_id'] for r in rows}) == 600
    write(run/'decoder-contract.json', dict(runner_sha256=sha(Path(__file__)),
        export_audit_sha256=sha(run/'export-verification.json'), checkpoint_sha256=identity['checkpoint_sha256'],
        selection='Frozen minimum680-trial PER; no WER reselection',
        primary_word_tokenization='legacy-literal-space', secondary_word_tokenization='standard-whitespace',
        timing_scope='WFST sentence decoding excludes graph initialization and teardown',
        uncertainty='2000 session-then-trial resamples,seed101; conditional selected checkpoint,one seed'))
    write(run/'status.json', dict(status='loading-decoder', trials=600))
    started = time.monotonic()
    decoder = FixedDecoder(run/'decoder.log')
    load_seconds = time.monotonic()-started
    owned = worker_ids()
    assert len(owned) == 1
    write(run/'decoder-identity.json', dict(decoder.identity, container=owned[0], initialization_seconds=load_seconds))
    try:
        write(run/'status.json', dict(status='decoding', trials=600))
        response = decoder.decode(run)
        assert len(response['transcripts']) == len(response['decode_ms']) == 600
        standard_rows = []
        for row, text, latency in zip(rows, response['transcripts'], response['decode_ms']):
            words, characters = score_text(row['text'], text, 'legacy-literal-space')
            standard_words, _ = score_text(row['text'], text, 'standard-whitespace')
            row.update(decoded_text=text, wer=words, cer=characters, wfst_ms=latency,
                       wer_standard=standard_words)
            standard_rows.append(dict(row, wer=standard_words))
        write(run/'decoded-predictions.json', rows)
        for name, scored, tokenization in (('metrics.json', rows, 'legacy-literal-space'),
                                           ('metrics-standard.json', standard_rows, 'standard-whitespace')):
            metrics = summarize(scored, repetitions=2000)
            metrics.update(tokenization=tokenization, checkpoint_sha256=identity['checkpoint_sha256'],
                decoder=decoder.identity, decode_ms_total=sum(response['decode_ms']),
                decoder_initialization_seconds=load_seconds,
                mean_prediction_loss=sum(r['prediction_loss'] for r in rows)/600,
                error_components={field: {key: sum(row[field][key] for row in scored) for key in
                    ('errors', 'reference_units', 'substitutions', 'deletions', 'insertions')}
                    for field in ('per', 'wer', 'cer')})
            write(run/name, metrics)
    finally:
        write(run/'status.json', dict(status='waiting-owned-decoder-exit', container=owned[0]))
        decoder.close()
    for attempt in range(120):
        if owned[0] not in worker_ids():
            break
        if attempt % 6 == 0:
            print(json.dumps(dict(status='waiting-owned-decoder-exit', container=owned[0])), flush=True)
        time.sleep(5)
    assert owned[0] not in worker_ids(), 'Decoder still live: no next workload permitted'
    for name, expected in identity['artifacts_sha256'].items():
        assert sha(run/name) == expected
    write(run/'verification.json', dict(passed=True, independent_phone_audit=True,
        independent_word_character_scalar_edits=True, exports_unchanged=True, owned_decoder_exited=True,
        scored_artifact_sha256={name: sha(run/name) for name in
            ('decoded-predictions.json', 'metrics.json', 'metrics-standard.json', 'decoder-identity.json')}))
    write(run/'status.json', dict(status='complete', trials=600, v1='Still paused'))
    print('completed-600-trial-fixed3gram-scoring', flush=True)


if __name__ == '__main__':
    main()
