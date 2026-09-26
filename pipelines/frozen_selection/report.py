"""Render the fixed measured audit; plots never recompute or select endpoints."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = ['#2a78d6', '#1baf7a', '#eda100']
MODELS = [('sd3', 'SD3.5 Medium'), ('flux', 'FLUX.2-klein-9B'), ('qwen', 'Qwen-Image')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    args = ap.parse_args()
    path = Path(args.results)
    result = json.loads(path.read_text())
    seed = 20260913  # The renderer validates the registered upstream recipe and draws no randomness.
    assert result['seed'] == seed and result['bootstrap'] == 10000
    style = Path(__file__).with_name('paper.mplstyle')
    plt.style.use(str(style))
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'font.weight': 'normal', 'axes.titleweight': 'semibold',
                         'axes.labelweight': 'normal',
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    out = path.parent/'figures'
    out.mkdir(exist_ok=True)
    manifest = []

    def save(fig, stem, caption):
        fig.savefig(out/(stem+'.pdf'), bbox_inches='tight')
        fig.savefig(out/(stem+'.svg'), bbox_inches='tight')
        svg = out/(stem+'.svg')
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        fig.savefig(out/(stem+'.png'), bbox_inches='tight', dpi=180)
        plt.close(fig)
        manifest.append(dict(file=stem+'.pdf', svg=stem+'.svg', preview=stem+'.png',
                             caption=caption, simulated=False,
                             results_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    strategies = ['ALL', 'M', 'H', 'B', 'B_M', 'B_G', 'B_G_M', 'G']
    labels = ['All / random expectation', 'Magnitude M', 'Headroom H', 'Baseline B',
              'B + M', 'B + G', 'B + M + G', 'Directional G']
    fig, axs = plt.subplots(1, 3, figsize=(13.8, 4.8), layout='constrained', sharex=True)
    for i, (model, title) in enumerate(MODELS):
        ax = axs[i]
        for j, strategy in enumerate(strategies):
            metric = result['models'][model]['strategies'][strategy]['metrics']['R']
            lo, hi = metric['ci95']
            ax.hlines(j, lo, hi, color=COLORS[i], lw=1.4)
            ax.plot(metric['estimate'], j, 'o', color=COLORS[i], ms=6)
        ax.axvline(0, color='.55', lw=.7)
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel('Mean composite improvement (points)')
        ax.grid(axis='x', alpha=.2)
        ax.grid(axis='y', visible=False)
    save(fig, 'selection_gain', 'Fixed top20 selection; 259/1295 cases per strategy/model. ALL is the full population and random-selection expected mean. Pointwise 95% paired prompt-bootstrap intervals; exploratory.')

    metrics = [('alignment_change', 'Alignment change: G minus M', 'Percentage points', 100),
               ('content_pass', 'Both seeds aligned: G minus M', 'Percentage points', 100),
               ('joint_success', 'Aligned + bias reduced: G minus M', 'Percentage points', 100),
               ('retained_R', 'Content-pass subset: G minus M', 'Composite-improvement points', 1)]
    fig, axs = plt.subplots(2, 2, figsize=(10.4, 7), layout='constrained')
    for ax, (metric, title, unit, scale) in zip(axs.flat, metrics):
        for i, (model, label) in enumerate(MODELS):
            m = result['models'][model]['comparisons']['G_minus_M'][metric]
            lo, hi = np.array(m['ci95'])*scale
            ax.hlines(i, lo, hi, color=COLORS[i], lw=1.4)
            ax.plot(m['estimate']*scale, i, 'o', color=COLORS[i], ms=6)
        ax.axvline(0, color='.4', lw=.8)
        ax.set_yticks(range(3), [m[1] for m in MODELS])
        ax.set_ylim(2.5, -.5)
        ax.set_title(title)
        ax.set_xlabel(unit)
        ax.grid(axis='x', alpha=.2)
        ax.grid(axis='y', visible=False)
    save(fig, 'content_comparison', 'Positive alignment-change difference means G loses less alignment than M, not that G loses no alignment. Content pass requires both steered seeds aligned. Joint success additionally requires mean bias reduction >0. Last panel conditions on different generated-outcome subsets. Pointwise 95% intervals; exploratory.')

    fig, axs = plt.subplots(1, 3, figsize=(13.8, 3.6), layout='constrained', sharex=True)
    contrasts = [('B_G_minus_B', 'B + G versus B'), ('full_minus_B_M', 'B + M + G versus B + M'), ('B_G_minus_B_M', 'B + G versus B + M')]
    for i, (model, title) in enumerate(MODELS):
        ax = axs[i]
        for j, (name, _) in enumerate(contrasts):
            m = result['models'][model]['comparisons'][name]['R']
            ax.hlines(j, *m['ci95'], color=COLORS[i], lw=1.4)
            ax.plot(m['estimate'], j, 'o', color=COLORS[i], ms=6)
        ax.axvline(0, color='.4', lw=.8)
        ax.set_yticks(range(3), [x[1] for x in contrasts])
        ax.set_ylim(2.5, -.5)
        ax.set_title(title)
        ax.set_xlabel('Selection gain difference (points)')
        ax.grid(axis='x', alpha=.2)
        ax.grid(axis='y', visible=False)
    save(fig, 'headroom_comparison', 'Development-frozen predictors; same top20 budget and common cases. B includes independent headroom/alignment plus input-direction norm, pair separability and T_rel. These selection differences are distinct from full-population MSE differences. Pointwise 95% intervals; exploratory.')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

    def fmt(v, scale=1):
        if v['estimate'] is None:
            return 'undefined'
        lo, hi = v['ci95']
        return f"{v['estimate']*scale:.3f} [{lo*scale:.3f}, {hi*scale:.3f}]"

    lines = ['# Exp274 — fixed top-20% selection and content audit', '',
             '**Measured exploratory reanalysis; no new image generation or human validation.**', '',
             '## Question and setup', '',
             'Does fixed top20 G selection identify larger composite improvements than magnitude, and what happens to content fidelity? Each model has 1295 valid cases and selects259. All eight strategy summaries and six paired contrasts are retained. Historical outcomes were already inspected.', '',
             '## Measurement', '',
             'Per-image S=20*A*(5-bias), then average matched seeds3/4; R=steer-clean. Bias is the frozen0–5 rating and A is binary prompt alignment. Independent headroom uses clean0/1. Content pass requires both steered seeds aligned; joint success also requires mean bias reduction>0. Retained-only comparisons condition on generated outcomes and do not define a pre-generation filter.', '',
             '## Results', '']
    for model, title in MODELS:
        d = result['models'][model]
        g, m, all_ = [d['strategies'][s] for s in ['G', 'M', 'ALL']]
        c = d['comparisons']
        lines += [f'### {title}', '',
                  f"- Composite gain: G {fmt(g['metrics']['R'])}; M {fmt(m['metrics']['R'])}; ALL {fmt(all_['metrics']['R'])}.",
                  f"- G−M composite gain: {fmt(c['G_minus_M']['R'])} points; bias reduction: {fmt(c['G_minus_M']['bias_reduction'])} rating units.",
                  f"- Alignment change: G {fmt(g['metrics']['alignment_change'],100)}; M {fmt(m['metrics']['alignment_change'],100)} percentage points.",
                  f"- Both-seed content pass: G {g['n_content_pass']}/259, M {m['n_content_pass']}/259. G−M pass-rate difference {fmt(c['G_minus_M']['content_pass'],100)} percentage points.",
                  f"- G−M joint-success difference: {fmt(c['G_minus_M']['joint_success'],100)} percentage points.",
                  f"- Conditional on content pass, G−M composite gain {fmt(c['G_minus_M']['retained_R'])}; bias reduction {fmt(c['G_minus_M']['retained_bias_reduction'])}.",
                  f"- B+G−B selection gain {fmt(c['B_G_minus_B']['R'])}; full−B+M {fmt(c['full_minus_B_M']['R'])} points.", '']
    lines += ['## Findings and evidence boundaries', '',
              'At top20, G-minus-M composite and joint-success intervals are positive in all three models. G-selected cases still lose alignment relative to clean; losses are smaller than M-selected cases. Qwen retains a positive composite difference within the strict content-pass subset; the SD3 and FLUX conditional intervals cross zero. Positive joint-success differences do not imply equal fidelity across strategies.', '',
              'Frozen B+G-minus-B selection intervals are positive for SD3 and Qwen, but not FLUX. This differs from the earlier full-population MSE pattern and must be reported as endpoint-specific evidence. No architecture-universal incremental-selection claim is supported.', '',
              '## Figures', '']
    for f in manifest:
        lines += [f"![{f['file']}](figures/{f['preview']})", '', f['caption'], '']
    lines += ['## Verification and reproducibility', '',
              'The independent stdlib verifier reconstructs all eight selection sets and all point contrasts per model. Archived SD3/Qwen top20 selections, point estimates and composite bootstrap intervals agree. New component intervals use10000 paired prompt bootstrap draws with seed20260913; all are pointwise95% and not multiplicity-adjusted. Exact identities and hashes are in results.json and verification.json.', '',
              f"Source: `{result['provenance']['source']}`; SHA256 `{result['provenance']['source_sha256']}`.", '',
              '## Limitations and next step', '',
              'Historically exposed cases; automated binary alignment; two outcome seeds; no independent human validation, no technical-quality endpoint, no new-axis generalization, and no net compute-saving claim. Conditional groups differ. New component bootstrap intervals have not been independently regenerated. New-axis work remains subject to the semantic-novelty gate; keep q=.20 and all model/metric settings fixed.', '']
    path.with_name('REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(figures=len(manifest), report=str(path.with_name('REPORT.md')))))


if __name__ == '__main__':
    main()
