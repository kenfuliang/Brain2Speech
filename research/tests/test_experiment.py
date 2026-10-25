import copy
import json
import numpy as np
import pytest
import torch
from research.data import CausalFeatures
from research.models import Encoder
from research.train import write_checkpoint
from research.experiment.evaluator import evaluate, SCORING
from research.experiment.pipeline import separation
from research.experiment.analysis import paired_effect, update_diagnostics, effective_rank, calibration_savings
from research.experiment import pipeline
from research.audit import save_json, digest
from research.experiment.streaming import stream
from research.data import prepare
from research.experiment.coverage import expected_outputs, audit_collection
from research.metrics import edits, summarize
from research.experiment.figures import build_figures


class DummyDecoder:
    identity = {'scoring': SCORING, 'test_only': True}
    def decode(self, output):
        rows = json.loads((output/'predictions.json').read_text())
        return {'transcripts': [r['text'] for r in rows], 'decode_ms': [1.]*len(rows)}


@pytest.mark.parametrize('kind', ['gru','supervised','dynamics','predictive'])
def test_streaming_matches_batch_and_integrated_prefix_invariance(kind):
    raw = np.ones((32,256), np.float32)
    changed = raw.copy()
    changed[16:] *= 100
    stats = {'mean':np.zeros(256), 'variance':np.ones(256), 'samples':100}
    prepared, _ = prepare([{'raw':raw}], stats)
    altered, _ = prepare([{'raw':changed}], stats)
    model = Encoder(kind,16).eval()
    with torch.no_grad():
        whole = model(torch.tensor(prepared[0]['x'][None]))['logits'][0].numpy()
        future = model(torch.tensor(altered[0]['x'][None]))['logits'][0].numpy()
    np.testing.assert_allclose(whole[:4], future[:4], atol=1e-6)
    streamed, metrics = stream(model, prepared[0]['x'], torch.device('cpu'))
    np.testing.assert_allclose(streamed, whole, atol=1e-6, rtol=1e-5)
    assert metrics['acquisition_ms'] == 4


def test_complete_evaluator_reload_and_identity(tmp_path):
    raw = np.ones((32, 256), dtype=np.float32)
    trials = [{'id': 'day/test/0', 'session': 'day', 'text': 'hello', 'raw': raw,
               'labels': np.array([1,2])}]
    stats = CausalFeatures.fit(trials).state()
    checkpoint = tmp_path/'checkpoint.pt'
    write_checkpoint(checkpoint, Encoder('gru', 16), {'kind': 'gru', 'hidden':16}, stats)
    output = tmp_path/'evaluation'
    first = evaluate(checkpoint, trials, stats, output, torch.device('cpu'), DummyDecoder())
    assert first['wer']['corpus'] == first['cer']['corpus'] == 0
    assert first['acquisition_delay_ms'] == 4
    assert evaluate(checkpoint, trials, stats, output, torch.device('cpu'), DummyDecoder()) == first
    changed = copy.deepcopy(trials)
    changed[0]['text'] = 'different'
    with pytest.raises(ValueError, match='identity mismatch'):
        evaluate(checkpoint, changed, stats, output, torch.device('cpu'), DummyDecoder())
    assert SCORING['phone_order'] == [40,39]+list(range(39))
    assert SCORING['acoustic_scale'] == .8 and SCORING['beam'] == 18
    (output/'metrics.json').write_text('{}')
    with pytest.raises(ValueError, match='artifact changed'):
        evaluate(checkpoint, trials, stats, output, torch.device('cpu'), DummyDecoder())


def test_replay_and_calibration_separation():
    separation([{'id':'day/train/1'}], [{'id':'day/test/1'}], [{'id':'past/train/1'}])
    with pytest.raises(ValueError, match='overlap'):
        separation([{'id':'a'}], [{'id':'a'}])
    with pytest.raises(ValueError, match='reserved'):
        separation([{'id':'day/train/1'}], [{'id':'day/test/1'}], [{'id':'past/test/1'}])


def test_paired_effect_and_rank():
    control = [{'seed':s, 'session':d, 'id':d+'/0',
                'wer': {'errors':2, 'reference_units':4}} for s in [17,29,43] for d in ['a','b']]
    candidate = copy.deepcopy(control)
    for r in candidate:
        r['wer']['errors'] = 1
    result = paired_effect(control, candidate, repetitions=20)
    assert result['candidate_minus_control_macro_wer'] == -.25
    assert result['ci95'] == [-.25,-.25]
    with pytest.raises(ValueError, match='share'):
        paired_effect(control[:-1], candidate, repetitions=2)
    parent = {'layer':torch.tensor([1.,2.])}
    child = {'layer':torch.tensor([2.,3.])}
    first, vector = update_diagnostics(parent, child)
    second, _ = update_diagnostics(parent, child, vector)
    assert second['consecutive_update_cosine'] == pytest.approx(1.)
    assert effective_rank([vector, vector])['effective_rank'] == pytest.approx(1.)


def test_small_final_matrix_preserves_ancestry_and_reserved_trials(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    monkeypatch.setattr(pipeline, 'BUDGETS', [1])
    monkeypatch.setattr(pipeline, 'device_for_run', lambda: torch.device('cpu'))
    monkeypatch.setattr(pipeline, 'build_figures', lambda root,freeze:build_figures(root,freeze,qa_only=True))
    sessions = ['a','b','c','d']
    manifest = {'causal_gate':{'strict_gate':'pass'}, 'eligible_sessions':sessions,
                'folds':{'initial_train':['a'], 'development':['b'], 'final':['c','d']},
                'sessions':{s:{'train':{'trials':[{'block':1,'text':'hello','id':s+'/train/0'}]},
                               'test':{'trials':[{'block':2,'text':'hello','id':s+'/test/0'}]}} for s in sessions}}
    manifest_path = tmp_path/'research/data/manifest.json'
    save_json(manifest_path, manifest)
    source = tmp_path/'initial'
    save_json(source/'protocol.json', {'source_hashes':{}, 'manifest_sha256':digest(manifest_path)})
    raw = np.ones((32,256), np.float32)
    def loader(manifest, days, partition):
        return [{'id':s+'/'+partition+'/0','session':s,'text':'hello','labels':np.array([1,2]),
                 'raw':raw.copy()} for s in days]
    monkeypatch.setattr(pipeline, 'load_trials', loader)
    monkeypatch.setattr(pipeline.Study,'load_verified',lambda self,days,partition:loader(self.manifest,days,partition))
    stats = CausalFeatures.fit(loader(manifest,['a'],'train')).state()
    base = tmp_path/'base.pt'
    config = {'kind':'gru','hidden':16,'seed':17,'lr':.0003,'auxiliary':0.,'epochs':1}
    write_checkpoint(base, Encoder('gru',16), config, stats)
    output = tmp_path/'study'
    study = pipeline.Study(source, output, DummyDecoder())
    evaluated = []
    def fake_evaluate(checkpoint, trials, stats, output, device, decoder, condition, training_texts):
        evaluated.append([t['id'] for t in trials])
        output.mkdir(parents=True, exist_ok=True)
        np.save(output/'embeddings.npy', np.ones((len(trials),64)))
        np.savez_compressed(output/'logits.npz', **{f'trial_{i:04d}':np.zeros((8,41)) for i in range(len(trials))})
        rows = [{**{k:t[k] for k in ['id','session','text']},
                 'wer':edits(['hello'],['other']), 'cer':edits(list('hello'),list('other')),
                 'per':edits([1,2],[2,3])} for t in trials]
        metrics = {**summarize(rows,repetitions=2), 'mean_encoder_time_ms_per_trial':1.,
                   'mean_wfst_ms_per_trial':1.}
        save_json(output/'predictions.json', rows)
        save_json(output/'metrics.json', metrics)
        save_json(output/'complete.json', {'identity':{'ids':[t['id'] for t in trials]},
                  'artifact_sha256':{name:digest(output/name) for name in
                                    ['predictions.json','metrics.json','logits.npz','embeddings.npy']}})
        return metrics
    monkeypatch.setattr(pipeline, 'evaluate', fake_evaluate)
    freeze = {'manifest_sha256':digest(manifest_path), 'decoder':study.decoder.identity,
              'selected':{'gru':{'checkpoint':str(base),'config':config,
                                'adaptation_steps':{'adapter':1,'full':1}}},
              'seeds':[17], 'budgets':[1], 'wer_target':.5}
    save_json(output/'development-freeze.json', freeze)
    study.final(freeze)
    for checkpoint in output.glob('sequential/**/post.pt'):
        _, payload = pipeline.load_checkpoint(checkpoint, torch.device('cpu'))
        assert all('/train/' in i for i in payload['extra']['replay_ids'])
        assert not set(payload['extra']['replay_ids']) & set(payload['extra']['calibration_ids'])
        assert payload['extra']['parent_sha256']
    assert list(output.glob('sequential/**/trajectory.json'))
    assert any('a/test/0' in ids for ids in evaluated)
    assert (output/'comparative-summary.json').exists()
    figure_manifest = json.loads((output/'figures/figure-manifest.json').read_text())
    assert figure_manifest['figures_generated'] == 5+4*3
    names = {r['name'] for r in figure_manifest['figures']}
    assert {'calibration-adapter','calibration-full','future-days','channel-loss','neural-controls',
            'transfer-gru-adapter-1-0.0','trajectory-gru-full-1-0.25','geometry-gru-full-1-0.25'} <= names
    assert all(r['synthetic_qa'] for r in figure_manifest['figures'])
    assert json.loads((output/'collection-coverage.json').read_text())['matrix_complete']
    identity_path = next(output.glob('sequential/**/post.identity.json'))
    original_identity = json.loads(identity_path.read_text())
    save_json(identity_path, {**original_identity, 'parent_sha256':'incorrect-parent'})
    rejected = audit_collection(output, manifest, freeze)
    assert not rejected['matrix_complete'] and not rejected['checkpoint_ancestry']['ancestry_complete']
    save_json(identity_path, original_identity)
    completed = list(expected_outputs(manifest, freeze))
    missing_folder = output/completed[0][0]
    (missing_folder/'metrics.json').write_text('{}')
    audit = audit_collection(output, manifest, freeze)
    assert not audit['matrix_complete'] and audit['inconsistent']


def test_full_approved_matrix_output_accounting():
    days = [f'd{i}' for i in range(5)]
    manifest = {'folds':{'final':days,'initial_train':['a']},
                'sessions':{d:{'test':{'trials':[{'id':d+'/test/0'}]}} for d in days+['a']}}
    freeze = {'selected':{k:{} for k in ['gru','supervised','dynamics','predictive']},
              'seeds':[17,29,43], 'budgets':[5,10,25,50,100,'full']}
    outputs = list(expected_outputs(manifest, freeze))
    assert len(outputs) == 5172
    assert len({name for name,_ in outputs}) == len(outputs)


def test_zero_norm_duplicate_pair_and_calibration_target_uncertainty():
    details, vector = update_diagnostics({'bias':torch.zeros(2)}, {'bias':torch.ones(2)})
    assert details['normalized_parameter_updates']['bias'] is None
    assert details['absolute_parameter_updates']['bias']>0
    assert effective_rank([torch.zeros(2)])['cumulative_energy'] == [0.]
    row = {'seed':17,'session':'d','id':'d/0','wer':edits(['a'],['b'])}
    with pytest.raises(ValueError, match='Duplicate'):
        paired_effect([row,row],[row,row],repetitions=2)
    good = {**row,'wer':edits(['a'],['a'])}
    control = {'0':[row],'5':[row],'10':[good]}
    candidate = {'0':[row],'5':[good],'10':[good]}
    labels = {'d':{'0':0,'5':5,'10':10}}
    result = calibration_savings(control,candidate,[0,5,10],.5,labels,repetitions=10)
    assert result['nominal_support_for_2x']
    assert result['control_labels_over_candidate_labels_ci95_conditional_on_both_reaching'] == [2.,2.]
    result = calibration_savings(control,control,[0,5,10],0.,labels,repetitions=10)
    assert not result['nominal_support_for_2x']
    unreachable = {k:[row] for k in ['0','5','10']}
    result = calibration_savings(unreachable,candidate,[0,5,10],.5,labels,repetitions=10)
    assert result['both_reach_fraction'] == 0 and not result['nominal_support_for_2x']
    zero_control = {**control,'0':[good]}
    result = calibration_savings(zero_control,candidate,[0,5,10],.5,labels,repetitions=10)
    assert result['zero_label_crossing_fraction']==1 and not result['nominal_support_for_2x']


def test_verified_source_rejects_changed_raw_data(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    raw = tmp_path/'research/data/raw.mat'
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b'verified fixture')
    study = object.__new__(pipeline.Study)
    study.manifest = {'sessions':{'day':{'train':{'path':'research/data/raw.mat','sha256':digest(raw)}}}}
    study.verified_sources = set()
    monkeypatch.setattr(pipeline,'load_trials',lambda *args:['loaded'])
    assert study.load_verified(['day'],'train') == ['loaded']
    raw.write_bytes(b'changed fixture')
    study.verified_sources = set()
    with pytest.raises(ValueError,match='Frozen raw source changed'):
        study.load_verified(['day'],'train')
