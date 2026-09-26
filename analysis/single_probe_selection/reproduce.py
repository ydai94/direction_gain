"""Reproduce all bootstrap estimates from the compact supplied input snapshot."""
import argparse
import hashlib
import json
from pathlib import Path

from analyze import analyze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    out = parser.parse_args().directory
    expected = json.loads((out/'results.json').read_text())
    inputs = json.loads((out/'inputs.json').read_text())
    assert hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest() == expected['inputs_sha256']
    expected_seed = 20260913
    assert expected['seed'] == expected_seed and expected['bootstrap'] == 10000
    for model in ['sd3', 'flux', 'qwen']:
        saved = expected['models'][model]
        original = {'G_M': {m: {'rho': saved['strategies'][m+'6']['rho']} for m in ['G', 'M']}}
        previous = {'strategies': {m: saved['strategies'][m+'6'] for m in ['G', 'M']}}
        reconstructed = analyze(model, inputs[model], original, previous)
        assert reconstructed == saved, model
        print(model, 'all point estimates and intervals exactly reproduced', flush=True)
    (out/'reproduction.json').write_text(json.dumps({'status': 'PASS', 'models': ['sd3', 'flux', 'qwen'],
        'draws_per_model': 10000, 'seed': expected_seed, 'comparison': 'Exact equality of all saved model results'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
