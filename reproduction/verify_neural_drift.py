"""Independent, CPU-only accounting and geometry QA for a completed E018 run."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.audit import digest, save_json, normalize_text, texts_from_mat


def scalar_distance(left,right):
    table=list(range(len(right)+1))
    for i,a in enumerate(left,1):
        following=[i]
        for j,b in enumerate(right,1):
            following.append(min(table[j]+1,following[-1]+1,table[j-1]+(a!=b)))
        table=following
    return table[-1]


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args(); run=args.run
    assert not (run/'seal.json').exists(), 'Do not change completed sealed evidence'
    protocol=json.loads((run/'protocol.json').read_text())
    metrics=json.loads((run/'metrics.json').read_text())
    rows=json.loads((run/'cohort.json').read_text())
    manifest=json.loads(Path(protocol['manifest']).read_text())
    checks=[]
    for path,expected in protocol['input_hashes'].items():
        assert digest(path)==expected,path
    checks.append('all permitted source, manifest, identity and checkpoint hashes')
    assert len(rows)==6680 and len({r['id'] for r in rows})==6680
    assert all(r['partition']=='train' and r['session'] in protocol['days'] for r in rows)
    assert not set(protocol['days']) & set(protocol['excluded_final_days'])
    raw_mean=[];raw_second=[];latent=[];edits=Counter();session_errors=defaultdict(Counter)
    all_shards=set()
    for row in rows:
        path=run/'trials'/row['session']/f"{row['index']:04d}.npz"
        all_shards.add(str(path.relative_to(run)))
        meta=json.loads(path.with_suffix('.json').read_text())
        assert meta['id']==row['id'] and digest(path)==meta['sha256']
        with np.load(path) as data:
            assert data['latent_frames'].shape[0]<=4
            assert data['latent_mean'].shape==(1024,)
            assert int(data['hidden_length'])==(row['bins']-32)//4+1
            assert int(data['raw_bins'])==row['bins']
            assert all(np.isfinite(data[k]).all() for k in data.files)
            error=scalar_distance(data['reference'].tolist(),data['phones'].tolist())
            assert error==meta['per']['errors']
            units=len(data['reference'])
            edits.update(errors=error,units=units)
            session_errors[row['session']].update(errors=error,units=units)
            raw_mean.append(data['raw_mean']);raw_second.append(data['raw_second']);latent.append(data['latent_mean'])
    assert all_shards==set(json.loads((run/'shard-hashes.json').read_text()))
    assert edits['errors']==metrics['per']['errors'] and edits['units']==metrics['per']['reference_units']
    assert np.isclose(edits['errors']/edits['units'],metrics['per']['corpus'])
    for d,c in session_errors.items():
        assert np.isclose(c['errors']/c['units'],metrics['per']['by_day'][d]['per'])
    checks.append('6680 scalar edit distances, source bins, output frames, finite summaries, day/corpus PER')
    for day in protocol['days']:
        entry=manifest['sessions'][day]['train']
        mat=loadmat(ROOT/entry['path'],variable_names=['sentenceText','blockIdx'])
        text=texts_from_mat(mat['sentenceText']);blocks=mat['blockIdx'].ravel()
        day_rows=[r for r in rows if r['session']==day]
        assert len(text)==len(blocks)==len(day_rows)
        for r in day_rows:
            assert normalize_text(text[r['index']])==r['text']
            assert blocks[r['index']]==r['block']
    checks.append('all source training text/block/index metadata independently matched')
    raw_mean=np.stack(raw_mean);raw_second=np.stack(raw_second);latent=np.stack(latent)
    fit=np.array([r['session'] in protocol['initial_days'] for r in rows])
    normalization=json.loads((run/'normalization.json').read_text())
    mean=raw_mean[fit].mean(0)
    variance=np.maximum(raw_second[fit].mean(0)-mean**2,1e-4)
    assert np.array_equal(mean,np.array(normalization['mean']))
    assert np.array_equal(variance,np.array(normalization['variance']))
    checks.append('equal-trial normalization independently reconstructed from every source-derived moment')
    with np.load(run/'summaries.npz') as summary:
        assert np.array_equal(summary['raw_mean'],raw_mean)
        assert np.array_equal(summary['latent_mean'],latent)
        projected=(latent-summary['pca_mean'])@summary['pca_axes'].T
        assert np.allclose(projected,summary['latent_pca'],atol=1e-10)
        assert np.allclose(summary['pca_mean'],latent[fit].mean(0))
        assert np.allclose(summary['pca_axes']@summary['pca_axes'].T,np.eye(32),atol=1e-10)
        representations=dict(counts=(raw_mean[:,:128]-mean[:128])/np.sqrt(variance[:128]),
            power=(raw_mean[:,128:]-mean[128:])/np.sqrt(variance[128:]),latent=projected,latent16=projected[:,:16])
    checks.append('shared initial-training PCA basis and every projection')
    for name,value in representations.items():
        stats=metrics['statistics'][name]
        day_means=np.stack([value[[r['session']==d for r in rows]].mean(0) for d in protocol['days']])
        pair=np.sqrt(((day_means[:,None]-day_means[None,:])**2).mean(-1))
        assert np.allclose(pair,np.array(stats['matrix']),atol=1e-10)
        assert np.allclose(pair,pair.T) and np.allclose(np.diag(pair),0)
        within=[]
        for day in protocol['days']:
            indices=[i for i,r in enumerate(rows) if r['session']==day]
            blocks=sorted({rows[i]['block'] for i in indices})
            counts=[sum(rows[i]['block']==b for i in indices) for b in blocks]
            boundaries=[abs(sum(counts[:k])-len(indices)/2) for k in range(1,len(blocks))]
            cut=1+int(np.argmin(boundaries))
            a=value[[i for i in indices if rows[i]['block'] in blocks[:cut]]].mean(0)
            b=value[[i for i in indices if rows[i]['block'] in blocks[cut:]]].mean(0)
            within.append(float(np.sqrt(np.mean((a-b)**2))))
        assert np.allclose(within,stats['within'])
        between=np.median(pair[np.triu_indices(19,1)])
        assert np.isclose(between-np.median(within),stats['excess'])
        # Fresh block resampling from saved trial summaries, without calling the
        # analysis implementation. All day pairs share each sampled day centroid.
        rng=np.random.default_rng(101)
        boot_means=[];boot_within=[]

        def resample(indices):
            block_ids=sorted({rows[i]['block'] for i in indices})
            totals=np.array([value[[i for i in indices if rows[i]['block']==b]].sum(0) for b in block_ids])
            ns=np.array([sum(rows[i]['block']==b for i in indices) for b in block_ids])
            selected=rng.multinomial(len(block_ids),np.full(len(block_ids),1/len(block_ids)),size=2000)
            return selected.dot(totals)/selected.dot(ns)[:,None]

        for day in protocol['days']:
            indices=[i for i,r in enumerate(rows) if r['session']==day]
            blocks=sorted({rows[i]['block'] for i in indices})
            sizes=[sum(rows[i]['block']==b for i in indices) for b in blocks]
            cut=1+int(np.argmin([abs(sum(sizes[:k])-len(indices)/2) for k in range(1,len(blocks))]))
            boot_means.append(resample(indices))
            a=resample([i for i in indices if rows[i]['block'] in blocks[:cut]])
            b=resample([i for i in indices if rows[i]['block'] in blocks[cut:]])
            boot_within.append(np.sqrt(((a-b)**2).mean(1)))
        pair_samples=np.stack([np.sqrt(((boot_means[i]-boot_means[j])**2).mean(1))
            for i in range(19) for j in range(i+1,19)],axis=1)
        null_samples=np.stack(boot_within,axis=1)
        excess_samples=np.median(pair_samples,axis=1)-np.median(null_samples,axis=1)
        mean_samples=pair_samples.mean(1)-null_samples.mean(1)
        assert np.allclose(np.quantile(excess_samples,[.025,.975]),stats['excess_ci95'],atol=1e-10)
        assert np.allclose(np.quantile(mean_samples,[.025,.975]),stats['mean_excess_ci95'],atol=1e-10)
        if name in metrics['correlations']:
            per=[metrics['per']['by_day'][d]['per'] for d in protocol['days']]
            assert np.isclose(float(spearmanr(pair[0,1:],per[1:]).statistic),
                              metrics['correlations'][name]['spearman_rho'])
    checks.append('all pair distances, whole-block splits, independent 2000-draw intervals, point excesses and descriptive correlations')
    pairs=json.loads((run/'repeated-content.json').read_text())
    lookup={r['id']:i for i,r in enumerate(rows)}
    for pair in pairs['pairs']:
        i,j=lookup[pair['left']],lookup[pair['right']]
        assert rows[i]['text']==rows[j]['text']==pair['prompt']
        for name in ('counts','power','latent'):
            assert np.isclose(np.sqrt(np.mean((representations[name][i]-representations[name][j])**2)),pair['distances'][name])
    assert pairs['cross_day_prompts']==77 and pairs['within_day_prompts']==7 and pairs['interval'] is None
    checks.append('every repeated-prompt pair and sparse-control interval suppression')
    result=dict(passed=True,trials=6680,checks=checks,scalar_PER=dict(edits),
        assessment='Share with caveats',biological_cause='Not established',
        files={str(p.relative_to(run)):digest(p) for p in (run/'protocol.json',run/'metrics.json',run/'summaries.npz')})
    destination=run/'independent-verification-bootstrap.json'
    assert not destination.exists(),'Preserve existing verification'
    save_json(destination,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
