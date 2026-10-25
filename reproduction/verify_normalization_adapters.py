"""Independent E019 scalar phone accounting, normalization and paired bootstrap."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.audit import digest,save_json
from verify_neural_drift import scalar_distance

KEYS=['G0','Gd','D0','Dd']
CONTRASTS={'normalization_at_reference':{'D0':1,'G0':-1},
 'normalization_at_native':{'Dd':1,'Gd':-1},'adapter_at_global':{'Gd':1,'G0':-1},
 'adapter_at_day':{'Dd':1,'D0':-1},'joint_minus_baseline':{'Dd':1,'G0':-1},
 'interaction':{'Dd':1,'D0':-1,'Gd':-1,'G0':1}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run
    assert not (run/'seal.json').exists()
    p=json.loads((run/'protocol.json').read_text());rows=json.loads((run/'cohort.json').read_text())
    m=json.loads((run/'metrics.json').read_text());n=json.loads((run/'normalization.json').read_text())
    parent=Path(p['parent']);ancestry=json.loads((run/'ancestry.json').read_text())
    for name,sha in ancestry['parent_hashes'].items():assert digest(parent/name)==sha
    assert len(rows)==6000 and len({r['id'] for r in rows})==6000 and len(p['days'])==17
    assert not set(p['days']) & set(p['excluded_final_days'])
    for path,sha in p['input_hashes'].items():assert digest(path)==sha
    errors={k:[] for k in KEYS};units=[];latent={k:[] for k in KEYS}
    raw_means=defaultdict(list);raw_seconds=defaultdict(list)
    for row in rows:
        base=parent/'trials'/row['session']/f"{row['index']:04d}.npz"
        with np.load(base) as v:
            reference=v['reference'].copy()
            raw_means[row['session']].append(v['raw_mean']);raw_seconds[row['session']].append(v['raw_second'])
        units.append(len(reference))
        for key in KEYS:
            path=base if key=='G0' else run/'trials'/key/row['session']/f"{row['index']:04d}.npz"
            meta=json.loads(path.with_suffix('.json').read_text())
            assert meta['id']==row['id'] and digest(path)==meta['sha256']
            with np.load(path) as v:
                assert np.array_equal(reference,v['reference'])
                error=scalar_distance(reference.tolist(),v['phones'].tolist())
                assert error==meta['per']['errors'];errors[key].append(error)
                latent[key].append(v['latent_mean'])
                frames=int(v['hidden_length'] if key=='G0' else v['frames'])
                assert frames==(row['bins']-32)//4+1
                assert np.isfinite(v['latent_mean']).all()
            if key!='G0':
                expected=0 if key=='D0' else p['day_to_adapter'][row['session']]
                assert meta['adapter']==expected
    for day in p['days']:
        mu=np.mean(raw_means[day],axis=0)
        var=np.maximum(np.mean(raw_seconds[day],axis=0)-mu**2,1e-4)
        assert np.array_equal(mu,np.array(n['day'][day]['mean']))
        assert np.array_equal(var,np.array(n['day'][day]['variance']))
    units=np.array(units);errors={k:np.array(v) for k,v in errors.items()}
    for key in KEYS:
        assert int(errors[key].sum())==m['conditions'][key]['errors']
        assert int(units.sum())==m['conditions'][key]['reference_units']
        assert np.isclose(errors[key].sum()/units.sum(),m['conditions'][key]['corpus'])
        for day in p['days']:
            ix=np.array([i for i,r in enumerate(rows) if r['session']==day])
            assert np.isclose(errors[key][ix].sum()/units[ix].sum(),m['conditions'][key]['by_day'][day]['per'])
    rng=np.random.default_rng(101);den=np.zeros(2000);num=np.zeros((2000,4))
    for day in p['days']:
        groups=defaultdict(list)
        for i,row in enumerate(rows):
            if row['session']==day:groups[row['block']].append(i)
        indices=[np.array(groups[b]) for b in sorted(groups)]
        size=len(indices)
        counts=rng.multinomial(size,np.ones(size)/size,size=2000)
        den+=counts.dot(np.array([units[ix].sum() for ix in indices]))
        num+=counts.dot(np.array([[errors[k][ix].sum() for k in KEYS] for ix in indices]))
    boot=num/den[:,None]
    for i,key in enumerate(KEYS):
        assert np.allclose(np.quantile(boot[:,i],[.025,.975]),m['conditions'][key]['ci95'])
    for name,c in CONTRASTS.items():
        v=sum(sign*boot[:,KEYS.index(key)] for key,sign in c.items())
        assert np.allclose(np.quantile(v,[.025,.975]),m['effects'][name]['ci95'])
    with np.load(parent/'summaries.npz') as prior:
        center=prior['pca_mean'];axes=prior['pca_axes']
    for key in KEYS:
        z=(np.stack(latent[key])-center)@axes.T
        day_mean=np.stack([z[[r['session']==d for r in rows]].mean(0) for d in p['days']])
        matrix=np.sqrt(np.mean((day_mean[:,None]-day_mean[None,:])**2,axis=-1))
        assert np.allclose(matrix,m['latent'][key]['matrix'])
    assert np.array_equal(errors['G0'][:280],errors['Gd'][:280])
    original=json.loads((parent/'seal.json').read_text())
    for name,sha in original['files'].items():assert digest(parent/name)==sha
    result=dict(passed=True,trials=6000,conditions=4,scalar_distances_recomputed=24000,
        daily_moments_independently_reconstructed=True,paired_block_intervals_independently_recomputed=True,
        same_basis_latent_matrices_verified=True,adapter_routing_verified=True,
        parent_preserved_files=len(original['files']),
        source_hashes={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__))},
        metrics_sha256=digest(run/'metrics.json'),assessment='Share with retrospective/transductive caveats')
    assert not (run/'independent-verification.json').exists()
    save_json(run/'independent-verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
