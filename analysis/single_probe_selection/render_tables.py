"""Format saved estimates for the manuscript; no fitting or resampling."""
import argparse
import json
from pathlib import Path


def fmt(value):
    return f'{value:.2f}'.replace('-', '$-$')


def interval(item):
    return f"{fmt(item['estimate'])} [{fmt(item['ci95'][0])}, {fmt(item['ci95'][1])}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = json.loads(args.results.read_text())
    text = r'''\paragraph{Selection with one probe.}\label{app:single-probe-selection}
We rank the same 1,295 test prompts using each of the six single probes and select the top 259. Each probe is one combination of seeds 6, 7, and 8 with the two model-specific steps in Appendix~\ref{app:implementation}. We compare its cosine score with response size measured at that same probe. Table~\ref{tab:single-probe-configurations} gives the mean improvement for every configuration under Qwen3-VL. The single-probe average is the mean of these six separately selected groups.

Table~\ref{tab:single-probe-intervals} reports this average and its paired differences from six-probe $G$ and single-probe $M$. We use 10,000 paired prompt-bootstrap draws with seed 20260913, reselecting the top 259 for every score in each draw. We average the six configuration estimates within each draw before computing the intervals. The ranges in Table~\ref{tab:cost} describe variation across configurations; the intervals below describe prompt-resampling uncertainty.

\begin{table}[!htbp]
\centering\small
\caption{Mean improvement from single-probe selection and paired differences. Brackets give 95\% prompt-bootstrap intervals.}
\label{tab:single-probe-intervals}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccc}
\toprule
Model & One-probe $G$ & Difference from six-probe $G$ & Difference from one-probe $M$ \\
\midrule
'''
    for model, name in [('sd3', 'SD3.5'), ('flux', 'FLUX'), ('qwen', 'Qwen')]:
        d = results['models'][model]
        values = [d['strategies']['G1_mean']['metrics']['R'],
                  d['contrasts']['G1_mean_minus_G6']['R'],
                  d['contrasts']['G1_mean_minus_M1_mean']['R']]
        text += name + ' & ' + ' & '.join(interval(x) for x in values) + r' \\' + '\n'
    text += r'''\bottomrule
\end{tabular}
\par\bigskip
\caption{Top-20\% mean improvement for each single probe. Brackets give 95\% prompt-bootstrap intervals.}
\label{tab:single-probe-configurations}
\begin{tabular}{lrrcc}
\toprule
Model & Probe seed & Step & Selected by direction & Selected by response size \\
\midrule
'''
    for model, name in [('sd3', 'SD3.5'), ('flux', 'FLUX'), ('qwen', 'Qwen')]:
        d = results['models'][model]['strategies']
        for j in range(6):
            g, m = d[f'G1_{j}'], d[f'M1_{j}']
            text += f"{name if j == 0 else ''} & {g['probe_seed']} & {g['probe_step']} & "
            text += interval(g['metrics']['R']) + ' & ' + interval(m['metrics']['R']) + r' \\' + '\n'
        if model != 'qwen':
            text += '\\addlinespace\n'
    text += r'''\bottomrule
\end{tabular}
\end{table}

Across the single-probe selections, mean stereotype reduction is 1.38, 1.48, and 2.71 in SD3.5, FLUX, and Qwen, with prompt-matching losses of 4.63, 10.26, and 23.46 percentage points. Relative to random prompt selection, the mean gains are 10.19, 9.69, and 18.79 points. The single-probe averages retain 87.8\%, 82.7\%, and 90.6\% of the six-probe gain above random selection. The main-text retention rates divide single-probe mean $R$ by six-probe mean $R$.

'''
    args.output.write_text(text.rstrip() + "\n")


if __name__ == '__main__':
    main()
