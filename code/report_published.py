"""Replot verified published CSV records without checkpoints or retraining."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from verify_published import verify_archive


COLORS = {'FedAA':'#0072B2','FedTCFixed':'#444444','FedTC':'#0072B2','FedTCSAC':'#D55E00','FedTCSACGate':'#009E73'}
STYLES = {'FedAA':('--','s'),'FedTCFixed':(':','s'),'FedTC':('--','^'),'FedTCSAC':('-','o'),'FedTCSACGate':('-.','D')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    verify_archive(args.archive)
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Use a new/empty output directory; archived plots are never overwritten.')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.archive/'learning_curves.csv').open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    group_key='report' if 'report' in rows[0] else 'experiment'
    groups=list(dict.fromkeys(r[group_key] for r in rows))
    for group in groups:
        selected=[r for r in rows if r[group_key]==group]
        fig,ax=plt.subplots(figsize=(7,4.2),layout='constrained')
        for method in dict.fromkeys(r['algorithm'] for r in selected):
            values=[r for r in selected if r['algorithm']==method]
            style,marker=STYLES[method]
            color='#D55E00' if method=='FedTC' and group_key=='experiment' else COLORS[method]
            ax.plot([int(r['round']) for r in values],[float(r['accuracy_pct']) for r in values],
                    label=method,color=color,linestyle=style,marker=marker,markersize=3,markevery=max(1,len(values)//6))
        ax.set(xlabel='Federated round',ylabel='Honest-client test accuracy (%)',ylim=(0,100),
               title=f'{group} | Seed 1; raw checkpoints, no confidence bands')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        for ext in ['png','pdf']:
            fig.savefig(args.output/f'{group}_accuracy.{ext}',dpi=200,facecolor='white')
        plt.close(fig)
    print(f'Replotted {len(groups)} comparisons from verified CSV records into {args.output}')


if __name__=='__main__':
    main()
