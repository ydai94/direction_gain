"""Independent stdlib point reconstruction and archived-bootstrap comparison."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def avg(values):
    return math.fsum(values)/len(values) if values else None


def close(a, b):
    assert (a is None and b is None) or (a is not None and b is not None and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)), (a, b)


def metrics(rows):
    r = [c['R'] for c in rows]
    ac = [c['components']['clean']['alignment'] for c in rows]
    ast = [c['components']['steer']['alignment'] for c in rows]
    bc = [c['components']['clean']['bias'] for c in rows]
    bs = [c['components']['steer']['bias'] for c in rows]
    d = [a-b for a, b in zip(bc, bs)]
    keep = [i for i, a in enumerate(ast) if a == 1]
    return dict(R=avg(r), bias_reduction=avg(d), clean_alignment=avg(ac),
                steer_alignment=avg(ast), alignment_change=avg([a-b for a, b in zip(ast, ac)]),
                clean_bias=avg(bc), steer_bias=avg(bs), content_pass=len(keep)/len(rows),
                joint_success=sum(d[i] > 0 for i in keep)/len(rows),
                retained_R=avg([r[i] for i in keep]), retained_bias_reduction=avg([d[i] for i in keep]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    args = ap.parse_args()
    root = Path(args.root)
    result = read(args.results)
    seed = 20260913  # Validate the archived estimator seed; this verifier draws no randomness.
    assert result['seed'] == seed and result['bootstrap'] == 10000
    prov = result['provenance']
    source, fits = read(root/prov['source']), read(root/prov['predictors'])
    assert sha(root/prov['source']) == prov['source_sha256']
    assert sha(root/prov['predictors']) == prov['predictors_sha256'] == source['predictors_sha256']
    assert sha(Path(__file__).with_name('analyze.py')) == prov['script_sha256']
    assert sha(Path(__file__).with_name('ANALYSIS_PLAN.md')) == prov['plan_sha256']
    checks = []
    for model, output in result['models'].items():
        cases = [c for c in source['models'][model]['cases'] if c['status'] == 'PRIMARY_VALID'
                 and all(c['S'][a] is not None for a in ['clean', 'steer', 'random0'])
                 and all(k in c for k in ['G', 'M', 'H', 'A0'])]
        cases.sort(key=lambda c: c['triplet_key'])
        assert len(cases) == output['n_valid'] == 1295
        for c in cases:
            close(c['R'], c['S']['steer']-c['S']['clean'])
        point = {}
        for name, saved in output['strategies'].items():
            def value(c):
                if name in ['G', 'M', 'H']:
                    return c[name]
                f = fits['models'][model]['fits'][name]
                return f['intercept']+math.fsum((c[k]-mu)/scale*w for k, mu, scale, w in
                       zip(f['features'], f['mean'], f['scale'], f['coef']))
            chosen = cases if name == 'ALL' else sorted(cases, key=lambda c: (-value(c), c['triplet_key']))[:math.ceil(.2*len(cases))]
            assert [c['triplet_key'] for c in chosen] == saved['selected_keys']
            assert len(chosen) == saved['n_selected']
            assert sum(c['components']['steer']['alignment'] == 1 for c in chosen) == saved['n_content_pass']
            point[name] = metrics(chosen)
            for metric, estimate in point[name].items():
                close(estimate, saved['metrics'][metric]['estimate'])
        for label, a, b in [('G_minus_M', 'G', 'M'), ('G_minus_H', 'G', 'H'),
                            ('G_minus_ALL', 'G', 'ALL'), ('B_G_minus_B', 'B_G', 'B'),
                            ('full_minus_B_M', 'B_G_M', 'B_M'), ('B_G_minus_B_M', 'B_G', 'B_M')]:
            for metric in point[a]:
                v = None if point[a][metric] is None or point[b][metric] is None else point[a][metric]-point[b][metric]
                close(v, output['comparisons'][label][metric]['estimate'])
        if model in ['sd3', 'qwen']:
            folder = 'exp273_selection' if model == 'sd3' else 'selection_qwen'
            old = next(r for r in read(root/'results'/folder/'results.json')['rows'] if r['fraction'] == .2)
            for name, entry in old['strategies'].items():
                new = output['strategies'][name]
                assert entry['selected_keys'] == new['selected_keys']
                close(entry['mean_gain'], new['metrics']['R']['estimate'])
                for a, b in zip(entry['ci95'], new['metrics']['R']['ci95']):
                    close(a, b)
            checks.append(model+': archived Exp273 identities, R estimates and bootstrap CIs match')
        checks.append(model+': all eight strategies and six contrasts independently reconstructed')
    verification = dict(status='PASS', simulated=False, checks=checks,
                        limitations='New component bootstrap intervals are not independently regenerated; point reconstruction is independent.',
                        results_sha256=sha(args.results), verifier_sha256=sha(__file__))
    Path(args.results).with_name('verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
