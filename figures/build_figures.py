"""Render archived estimates. Only unit conversion and baseline indexing are applied."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent
plt.style.use(HERE/'paper.mplstyle')
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8.5,'xtick.labelsize':9,'ytick.labelsize':9,'axes.titleweight':'bold','pdf.fonttype':42,'ps.fonttype':42})
d=json.loads((HERE/'inputs/plotted_values.json').read_text())['models']
models=['sd3','flux','qwen'];labels=['SD3.5','FLUX','Qwen'];x=np.arange(3)
BLUE,GREEN,YELLOW='#2a78d6','#1baf7a','#eda100'
def save(fig,name):
 for ext in ['pdf','png']:fig.savefig(HERE/f'{name}.{ext}',bbox_inches='tight',dpi=220)
 plt.close(fig)
def errs(rows,key='estimate',scale=1):
 y=np.array([r[key] for r in rows])*scale;b=np.array([r['ci95'] for r in rows])*scale
 assert np.all(b[:,0]<=y) and np.all(b[:,1]>=y)
 return y,np.vstack([y-b[:,0],b[:,1]-y])
def categorical(ax):ax.set(xticks=x,xticklabels=labels,xlim=(-.5,2.5));ax.set_axisbelow(True)
# The component records contain point estimates, not component CIs.
fig,axes=plt.subplots(1,2,figsize=(6.25,2.5),layout='constrained')
for arm,off,c,mark,label in [('steer',-.18,BLUE,'o','Semantic steering'),('random0',0,GREEN,'s','Random direction'),('clean',.18,YELLOW,'^','Clean')]:
 for ax,key,scale in [(axes[0],'bias',1),(axes[1],'alignment',100)]:
  ax.plot(x+off,[d[m]['components'][arm][key]*scale for m in models],ls='none',marker=mark,ms=6,color=c,label=label)
for ax in axes:categorical(ax)
axes[0].set(title='(a) Stereotype rating',ylabel='Mean rating (0–5; lower is better)',ylim=(0,5));axes[0].legend(loc='upper left',handlelength=1,borderpad=0,labelspacing=.3)
axes[1].set(title='(b) Prompt matching',ylabel='Images matching prompt (%)',ylim=(0,100))
save(fig,'baseline_components')
# Correlation versus full-population prediction, without confusing the two estimands.
fig,axes=plt.subplots(1,2,figsize=(6.25,2.6),layout='constrained')
for name,off,c,mark,label in [('G',-.09,BLUE,'o','G: direction'),('M',.09,GREEN,'s','M: response size')]:
 y,e=errs([d[m]['correlation'][name] for m in models],key='rho')
 axes[0].errorbar(x+off,y,yerr=e,fmt=mark,color=c,capsize=3,ms=6,lw=1.3,label=label)
axes[0].set(title='(a) Predicting improvement',ylabel='Spearman correlation with improvement',ylim=(-.1,.4));axes[0].axhline(0,c='#888888',lw=.7,zorder=0);axes[0].legend(loc='upper left',handlelength=1,borderpad=0)
for name,off,c,mark,label in [('B_G',-.18,BLUE,'o','B + G'),('B_M',0,GREEN,'s','B + M'),('B_G_M',.18,YELLOW,'^','B + G + M')]:
 y=[100*d[m]['prediction'][name]['MSE']/d[m]['prediction']['B']['MSE'] for m in models]
 axes[1].plot(x+off,y,ls='none',marker=mark,ms=6,color=c,label=label)
axes[1].axhline(100,c='#555555',ls='--',lw=1,zorder=0);axes[1].set(title='(b) Adding G and response size',ylabel='Test MSE / baseline MSE (%)',ylim=(88,120));axes[1].legend(loc='upper left',ncol=1,handlelength=1,borderpad=0,labelspacing=.25)
for ax in axes:categorical(ax)
save(fig,'main_results')
# Actual evaluated fractions are shown at their true x positions. No fitted curve.
fig,axes=plt.subplots(1,3,figsize=(6.25,2.55),sharey=True,layout='constrained')
for ax,m,title in zip(axes,models,labels):
 rows=d[m]['curves'];xx=np.array([r['fraction']*100 for r in rows])
 for name,c,mark in [('G',BLUE,'o'),('M',GREEN,'s')]:
  points=[r['strategies'][name] for r in rows];y,e=errs(points,key='mean_gain')
  ax.errorbar(xx,y,yerr=e,fmt=mark+'-',color=c,capsize=2,ms=6,lw=1,label=name)
 ax.axhline(d[m]['all_prompt_mean'],color=YELLOW,ls='--',lw=1.2,label='All prompts')
 ax.set(title=title,xlim=(4,105),ylim=(-10,54),xticks=[10,25,50,100],xlabel='Prompts selected (%)');ax.set_axisbelow(True)
axes[0].set_ylabel('Mean improvement (score points)')
handles,labs=axes[0].get_legend_handles_labels();ordered={label:handle for handle,label in zip(handles,labs)};fig.legend([ordered[k] for k in ['G','M','All prompts']],['G','M','All prompts'],loc='outside upper center',ncol=3,frameon=False)
save(fig,'selection_curves')
# Changes are within each selected prompt set; G and M may select different sets.
fig,axes=plt.subplots(1,2,figsize=(6.25,2.6),layout='constrained')
for name,off,c,mark in [('G',-.09,BLUE,'o'),('M',.09,GREEN,'s')]:
 for ax,key,scale in [(axes[0],'bias_reduction',1),(axes[1],'alignment_change',100)]:
  y,e=errs([d[m]['selection'][name][key] for m in models],scale=scale)
  ax.errorbar(x+off,y,yerr=e,fmt=mark,color=c,capsize=3,ms=6,lw=1.3,label=f'Top 20% by {name}')
axes[0].set(title='(a) Stereotype reduction',ylabel='Clean minus steered rating',ylim=(0,4));axes[0].legend(loc='upper left',handlelength=1,borderpad=0)
axes[1].set(title='(b) Change in prompt matching',ylabel='Steered minus clean (percentage points)',ylim=(-44,4));axes[1].axhline(0,c='#888888',lw=.7,zorder=0)
for ax in axes:categorical(ax)
save(fig,'selection_tradeoff')
print('Rendered 4 result figures from archived estimates; no refitting or resampling.')
