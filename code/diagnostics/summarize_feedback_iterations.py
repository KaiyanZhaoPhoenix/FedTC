"""Recompute the final matched native comparisons and per-seed accuracy plot."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils.experiment import write_json


def paired(a,b):
    assert [(r['seed'],r['identity']) for r in a]==[(r['seed'],r['identity']) for r in b]
    boot=np.random.default_rng(241).integers(len(a),size=(20000,len(a)))
    result={}
    for k in ('accuracy','worst10'):
        first=np.array([r['metrics'][k] for r in a])*100
        second=np.array([r['metrics'][k] for r in b])*100
        delta=first-second
        result[k]=dict(first=float(first.mean()),reference=float(second.mean()),diff=float(delta.mean()),
            ci95=np.percentile(delta[boot].mean(axis=1),[2.5,97.5]).tolist(),wins=int((delta>0).sum()),
            ties=int((delta==0).sum()),n=len(delta))
    return result


def main(root):
    def read(path):
        return json.loads((root/path).read_text())
    first=read('native_validated/held.json')
    confirm=read('native_confirm/held.json')
    fixed=read('native_strength/held.json')
    native=read('native_fit/held.json')
    independent=read('native_replication/held.json')
    stats={'independent_replication':read('native_replication/summary.json')}
    direct=first['direct']+confirm['direct']
    anchor=first['incumbent']+confirm['incumbent']
    stats['first_direct_vs_incumbent']=paired(first['direct'],first['incumbent'])
    stats['confirm_direct_vs_incumbent']=paired(confirm['direct'],confirm['incumbent'])
    stats['pooled_direct_vs_incumbent']=paired(direct,anchor)
    stats['pool72_all_three_batches']=paired(direct+independent['direct'][1::2],anchor+independent['incumbent'][1::2])
    stats['confirm_direct_vs_fixed_trust']=paired(confirm['direct'],fixed)
    stats['constant_reference_mix']=read('native_strength/locked.json')['mix']
    for k,v in native.items():
        stats['native_fit_'+k+'_vs_direct']=paired(v,confirm['direct'])
        stats['native_fit_'+k+'_vs_incumbent']=paired(v,native['incumbent'])
    stats['native_fit_rl_vs_immediate']=paired(native['rl'],native['immediate'])
    stats['native_fit_rl_vs_schedule']=paired(native['rl'],native['schedule'])
    stats['weights_selected']=dict(original=int(sum(m==1 for r in direct for m in r['mix'])),
            half=int(sum(m==.5 for r in direct for m in r['mix'])),
            reweighted=int(sum(m==0 for r in direct for m in r['mix'])))
    write_json(root/'final_summary.json',stats)

    fig,axes=plt.subplots(1,2,figsize=(9.3,4),layout='constrained')
    for ax,k,title in zip(axes,('accuracy','worst10'),('Global test accuracy','Worst 10% of clients')):
        groups=[first,confirm]+[{k:r[index::2] for k,r in independent.items()} for index in (0,1)]
        for j,records in enumerate(groups):
            d=np.array([(r['metrics'][k]-b['metrics'][k])*100 for r,b in
                        zip(records['direct'],records['incumbent'])])
            x=np.linspace(j-.14,j+.14,len(d))
            ax.scatter(x,d,alpha=.8,color=('#2c6884','#396d47','#845a38','#685b89')[j],s=25)
            ci=np.percentile(d[np.random.default_rng(241).integers(len(d),size=(20000,len(d)))].mean(1),[2.5,97.5])
            ax.errorbar(j+.21,d.mean(),yerr=np.array([[d.mean()-ci[0]],[ci[1]-d.mean()]]),
                        fmt='D',capsize=3,color='#272727',ms=4)
        ax.axhline(0,color='#777',linewidth=1)
        ax.set(xticks=[0,1,2,3],xticklabels=['First\n8','Confirm\n12','Pool 48\n16','Pool 72\n16'],
               xlim=(-.38,3.5),title=title,ylabel='Paired gain (percentage points)')
        ax.grid(axis='y',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(root/'native_paired_gains.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    main(parser.parse_args().root)
