"""Read-only chart data from a training snapshot; never edits the running trainer."""
import argparse
import json
import sqlite3
from pathlib import Path
import numpy as np
from scipy.io import loadmat

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); p.add_argument('--output',type=Path); a=p.parse_args()
    root=Path(__file__).absolute().parent.parent
    # Ignore a final incomplete log line while the producer is active.
    lines=(a.run/'training.jsonl').read_bytes().splitlines(keepends=True)
    rows=[json.loads(line) for line in lines if line.endswith(b'\n')]
    assert [r['attempt'] for r in rows]==list(range(len(rows)))
    assert all(r['update_applied'] and np.isfinite(r['prediction_loss']) for r in rows)
    original=loadmat(root/'workspace/derived/rnns/baselineRelease/outputSnapshot')['perBatchData_train']
    losses=np.asarray([r['prediction_loss'] for r in rows])
    stride=max(25,int(np.ceil(len(rows)/100)))
    db=sqlite3.connect(':memory:'); db.row_factory=sqlite3.Row
    db.execute('CREATE TABLE train_log (update_number INTEGER PRIMARY KEY, session TEXT, prediction_loss REAL, learning_rate REAL, gradient_norm REAL)')
    db.execute('CREATE TABLE original_log (update_number INTEGER PRIMARY KEY, prediction_loss REAL)')
    db.executemany('INSERT INTO train_log VALUES (?,?,?,?,?)',[(i+1,r['session'],r['prediction_loss'],r['lr'],r['gradient_norm']) for i,r in enumerate(rows)])
    db.executemany('INSERT INTO original_log VALUES (?,?)',[(i+1,float(original[i,2])) for i in range(len(rows))])
    sql=f'''WITH current_trend AS (
SELECT *, AVG(prediction_loss) OVER (ORDER BY update_number ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS mean_loss,
COUNT(*) OVER (ORDER BY update_number ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS window_n,
ROW_NUMBER() OVER (PARTITION BY (update_number-1)/{stride} ORDER BY prediction_loss, update_number) AS low_rank,
ROW_NUMBER() OVER (PARTITION BY (update_number-1)/{stride} ORDER BY prediction_loss DESC, update_number) AS high_rank
FROM train_log), original_trend AS (
SELECT *, AVG(prediction_loss) OVER (ORDER BY update_number ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS mean_loss,
COUNT(*) OVER (ORDER BY update_number ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS window_n FROM original_log), chart_rows AS (
SELECT update_number, prediction_loss, 'Batch loss (sampled min/max)' AS series, 'dotted' AS stroke_style,
session, 1 AS window_updates, learning_rate, gradient_norm FROM current_trend
WHERE low_rank=1 OR high_rank=1 OR update_number IN (1,{len(rows)})
UNION ALL SELECT update_number, mean_loss, 'Ours: trailing 50 updates', 'solid', 'Mixed sessions', window_n, NULL, NULL
FROM current_trend WHERE (update_number-1)%{stride}=0 OR update_number={len(rows)}
UNION ALL SELECT update_number, mean_loss, 'Original: trailing 50 updates', 'dashed', 'Mixed sessions', window_n, NULL, NULL
FROM original_trend WHERE (update_number-1)%{stride}=0 OR update_number={len(rows)})
SELECT update_number AS "update", update_number*64.0/6640.0 AS epoch_equivalent, prediction_loss, series,
stroke_style, session, window_updates, learning_rate, gradient_norm FROM chart_rows ORDER BY series, update_number'''
    table=[dict(r) for r in db.execute(sql)]
    if a.output:
        if a.output.exists(): raise FileExistsError(a.output)
        a.output.mkdir(parents=True)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import hashlib
        def rolling(values):
            totals=np.concatenate(([0.],np.cumsum(values,dtype=float)))
            return np.asarray([(totals[i+1]-totals[max(0,i-49)])/(i+1-max(0,i-49)) for i in range(len(values))])
        x=np.arange(1,len(rows)+1)*64/6640
        current_mean=rolling(losses); original_mean=rolling(original[:len(rows),2])
        fig,ax=plt.subplots(figsize=(11.2,5.8),dpi=170)
        fig.subplots_adjust(left=.09,right=.97,bottom=.17,top=.77)
        fig.text(.09,.94,'Prediction loss vs. training exposure',fontsize=18,weight='bold',color='#222222')
        fig.text(.09,.89,f'Seed 17 • {len(rows):,} updates • {x[-1]:.1f} approximate epoch equivalents',fontsize=11,color='#555555')
        fig.text(.965,.94,'✳',ha='right',color='#2878B5',fontsize=18)
        ax.plot(x,losses,color='#BBBBBB',lw=.65,alpha=.6,label='Batch prediction loss')
        ax.plot(x,current_mean,color='#2878B5',lw=1.9,label='Ours — trailing 50-update mean')
        ax.plot(x,original_mean,color='#333333',lw=1.5,ls='--',label='Original — trailing 50-update mean')
        ax.set(xlabel='Approximate epoch equivalents (64 × updates / 6,640 training sentences)',ylabel='Prediction loss (CTC per sentence)',xlim=(0,x[-1]),ylim=(0,max(losses.max(),original_mean.max())*1.05))
        ax.grid(axis='y',color='#E7E7E7',lw=.6); ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='upper left',bbox_to_anchor=(0,1.17),frameon=False,fontsize=9,ncol=3)
        fig.text(.09,.065,'Epoch equivalents measure total sentence exposure, not unique passes: days are sampled separately.',fontsize=9,color='#555555')
        fig.text(.09,.035,'Source: active E008 training journal and released outputSnapshot. Prediction loss excludes L2; this is not WER.',fontsize=9,color='#555555')
        fig.savefig(a.output/'prediction-loss.png',facecolor='white'); plt.close(fig)
        metadata=dict(updates=len(rows),epoch_equivalents=float(x[-1]),current_last50=float(current_mean[-1]),original_last50=float(original_mean[-1]),
                      train_sentences=6640,source_paths=[str(a.run/'training.jsonl'),str(root/'workspace/derived/rnns/baselineRelease/outputSnapshot')],
                      captured_complete_log_sha256=hashlib.sha256(b''.join(line for line in lines if line.endswith(b'\n'))).hexdigest(),
                      chart='Full batch loss plus unsampled trailing-50 means; original comparison limited to matching update range',
                      epoch_caveat='Exposure equivalent, not unique dataset passes',wer=None)
        (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(json.dumps(metadata)); return
    print(json.dumps(dict(rows=table,full_logged_updates=len(rows),last_epoch_equivalent=len(rows)*64/6640,
                          current_last50=float(losses[-50:].mean()),train_sentences=6640,
                          sampling=f'Raw min/max per {stride}-update bucket, trend endpoints; averages use all logged updates',query_sql=sql,preview=table[:10])))

if __name__=='__main__': main()
