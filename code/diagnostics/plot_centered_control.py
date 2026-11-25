"""Draw final and intermediate held-set curves from the centered-control diagnostic."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(folder):
    def read(name):
        return json.loads((folder/(name+'.json')).read_text())

    def curves(ax, groups, names, metric):
        for key, label, color, style in names:
            values = np.array([[m[metric] for m in r['curve']] for r in groups[key]])*100
            ax.plot(np.arange(values.shape[1])*5,values.mean(0),label=label,
                    color=color,linestyle=style,linewidth=1.9)
        ax.set(xlabel='Communication round',xticks=[0,10,20,30,40],xlim=(0,40))
        ax.grid(axis='y',alpha=.22)
        ax.legend(fontsize=8,loc='best',frameon=False)
        ax.spines[['top','right']].set_visible(False)

    clean = read('clean_held')
    clean['cosine-ce'] = read('clean_cosine_ce')
    scenarios = [('switch','MNIST','global_acc','Global accuracy (%)'),
                 ('longtail','CIFAR-100-LT','tail20','Tail-class accuracy (%)')]
    scenarios = [s for s in scenarios if (folder/(s[0]+'_held.json')).exists()]
    count = 1+len(scenarios)
    fig, axes = plt.subplots(1,count,figsize=(4.8*count,3.8),layout='constrained',squeeze=False)
    axes = axes[0]
    curves(axes[0],clean,[('ce','Linear, CE','#555555','--'),
        ('balanced-local','Linear, local Balanced Softmax','#b66020',':'),
        ('cosine-ce','Cosine, CE','#237376','-.'),
        ('cosine-balanced-local','Cosine, local Balanced Softmax','#1c4891','-')],'tail20')
    axes[0].set(title='Clean CIFAR-100-LT · FedAvg',ylabel='Tail-class accuracy (%)',ylim=(0,38))
    for ax,(scenario,title,metric,ylabel) in zip(axes[1:],scenarios):
        curves(ax,read(scenario+'_held'),[('FedAvg','FedAvg','#555555','--'),
            ('Krum','Krum','#8a6995',':'),('static','Validation-selected fixed','#1c4891','-'),
            ('feedback','Continuous CI controller','#b64b18','-.')],metric)
        ax.axvspan(20,40,color='#a8b5c0',alpha=.13,zorder=-1)
        ax.set(title=title+' · clean → simulated contamination',ylabel=ylabel)
    fig.savefig(folder/'held_curves.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    plot(parser.parse_args().folder)
