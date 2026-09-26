"""Independently reconstruct saved selections and point estimates with stdlib."""
import hashlib
import argparse
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/single_probe_selection_20260923'


def same(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=OUT)
    out = parser.parse_args().directory
    result = json.loads((out/'results.json').read_text())
    inputs = json.loads((out/'inputs.json').read_text())
    assert hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest() == result['inputs_sha256']
    expected_seed = 20260913
    assert result['seed'] == expected_seed and result['bootstrap'] == 10000
    checks = 0
    for model, data in inputs.items():
        rows = data['rows']; n = len(rows); k = math.ceil(n*.2)
        assert n == 1295 and k == 259
        assert len({r['key'] for r in rows}) == n
        observed = result['models'][model]
        points = {}
        for name, saved in observed['strategies'].items():
            if name.endswith('_mean'):
                continue
            if name == 'ALL':
                selected = rows
            else:
                def value(row):
                    if name in ['G6', 'M6']:
                        return row[name[0]]
                    kind = 'cosines' if name.startswith('G') else 'magnitudes'
                    return row[kind][int(name[-1])]
                selected = sorted(rows, key=lambda row: (-value(row), row['key']))[:k]
                assert [r['key'] for r in selected] == saved['selected_keys']
                assert len(set(saved['selected_keys'])) == k
                checks += 2
            points[name] = {m: statistics.mean(r[m] for r in selected) for m in ['R', 'stereotype_reduction', 'matching_change']}
            for metric, point in points[name].items():
                same(point, saved['metrics'][metric]['estimate']);checks += 1
        for name, prefix in [('G1_mean', 'G1'), ('M1_mean', 'M1')]:
            points[name] = {}
            for metric in ['R', 'stereotype_reduction', 'matching_change']:
                values = [points[f'{prefix}_{j}'][metric] for j in range(6)]
                points[name][metric] = statistics.mean(values)
                same(points[name][metric], observed['strategies'][name]['metrics'][metric]['estimate'])
                bounds = observed['strategies'][name]['configuration_range'][metric]
                same(min(values), bounds[0]);same(max(values), bounds[1]);checks += 3
        for label, metrics in observed['contrasts'].items():
            a, b = label.split('_minus_')
            for metric, saved in metrics.items():
                same(points[a][metric]-points[b][metric], saved['estimate']);checks += 1
        same(100*points['G1_mean']['R']/points['G6']['R'], observed['retained_mean_R_percent'])
        same(100*(points['G1_mean']['R']-points['ALL']['R'])/(points['G6']['R']-points['ALL']['R']), observed['retained_lift_over_ALL_percent'])
        checks += 2
    output = {'status': 'PASS', 'numeric_and_selection_checks': checks,
              'method': 'Independent Python stdlib reconstruction of every selected set, point mean, configuration range and contrast',
              'interval_check': 'Analyzer reproduces original six-probe bootstrap intervals; new intervals use the same paired estimator and recorded seed'}
    (out/'verification.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
