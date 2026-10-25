import copy
import json
import numpy as np
import pytest
from research.audit import digest, save_json
from research.metrics import edits
from research.reporting.verification import word_edits, verify


def test_empty_word_transcript_is_all_deletions():
    assert word_edits('one two','')=={'errors':2,'substitutions':0,'deletions':2,'insertions':0,'reference_units':2}


def fixture(folder, hypothesis=''):
    logits = np.zeros((8,41))
    logits[:,40] = 10
    row = {'id':'day/test/0','session':'day','text':'one two','decoded_text':hypothesis,
           'reference_phones':[1,2],'predicted_phones':[],
           'per':edits([1,2],[]),'cer':edits(list('one two'),list(hypothesis)),
           'wer':edits('one two'.split(' '),hypothesis.split(' '))}
    save_json(folder/'predictions.json',[row])
    save_json(folder/'metrics.json',{'trials':1,'sessions':1,
              **{f:{'corpus':row[f]['errors']/row[f]['reference_units']} for f in ['per','wer','cer']}})
    np.savez_compressed(folder/'logits.npz',trial_0000=logits)
    np.save(folder/'embeddings.npy',np.ones((1,64)))
    save_json(folder/'complete.json',{'identity':{'ids':['day/test/0']},
              'artifact_sha256':{n:digest(folder/n) for n in ['predictions.json','metrics.json','logits.npz','embeddings.npy']}})


def test_verified_correction_preserves_rates_and_originals(tmp_path):
    fixture(tmp_path)
    original = digest(tmp_path/'predictions.json')
    result = verify(tmp_path,['day/test/0'])
    assert result['all_rates_unchanged'] and len(result['component_corrections'])==1
    assert result['corrected_edit_totals']['wer']['deletions']==2
    assert digest(tmp_path/'predictions.json')==original
    with pytest.raises(ValueError,match='identity'):
        verify(tmp_path,['wrong/test/0'])


def test_verified_report_rejects_changed_rates_and_artifacts(tmp_path):
    fixture(tmp_path,'one  two')
    with pytest.raises(ValueError,match='rates changed'):
        verify(tmp_path)
    fixture(tmp_path)
    (tmp_path/'metrics.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact hash'):
        verify(tmp_path)
