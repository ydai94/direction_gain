"""A2(b) probe: score a handful of Exp271 images with a GPT vision model using the
Qwen3-VL rubric verbatim, to (a) confirm outbound network from the node, (b) confirm
the compact prompts parse, (c) measure real token usage so the cost estimate is
measured rather than assumed. Writes a small JSON; sends no more than N images.
"""
import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("DG_ROOT", "."))
REL = ROOT / 'pipelines/heldout_prediction'
EXP = ROOT / 'results/heldout_prediction/runtime'
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'gpt-5.6-terra'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
API = 'https://api.openai.com/v1/chat/completions'


def post(payload, key):
    req = urllib.request.Request(
        API, data=json.dumps(payload).encode(),
        headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def main():
    key = os.environ['OPENAI_API_KEY']
    prompts = json.loads((REL / 'prompts.json').read_text())
    cohort = {r['triplet_key']: r for r in json.loads((REL / 'cohort.json').read_text())}
    scores = json.loads((EXP / 'sd3/scoring/scores.json').read_text())['rows']
    inputs = json.loads((EXP / 'sd3/scoring/inputs.json').read_text())['rows']
    ref = {(r['triplet_key'], r['arm'], r['seed'], r['metric']): r['value'] for r in scores}

    picked, out = [], []
    for row in inputs:
        im = row['image']
        if im['arm'] in ('clean', 'steer') and len(picked) < N:
            picked.append(im)

    for im in picked:
        case = cohort[im['triplet_key']]
        b64 = base64.b64encode((ROOT / im['path']).read_bytes()).decode()
        rec = {'triplet_key': im['triplet_key'], 'arm': im['arm'], 'seed': im['seed']}
        for metric in ('bias', 'alignment'):
            text = prompts[metric]['compact'].format(**case)
            payload = {
                'model': MODEL,
                'messages': [{'role': 'user', 'content': [
                    {'type': 'text', 'text': text},
                    {'type': 'image_url',
                     'image_url': {'url': f'data:image/png;base64,{b64}', 'detail': 'high'}}]}],
                'max_completion_tokens': 2048,
                'reasoning_effort': os.environ.get('RE', 'none'),
            }
            t = time.time()
            try:
                r = post(payload, key)
                rec[metric] = {
                    'raw': r['choices'][0]['message']['content'].strip(),
                    'usage': r.get('usage'),
                    'seconds': round(time.time() - t, 2),
                }
            except Exception as e:  # noqa: BLE001 - probe reports the failure verbatim
                body = e.read().decode()[:400] if hasattr(e, 'read') else str(e)[:400]
                rec[metric] = {'error': body, 'seconds': round(time.time() - t, 2)}
            rec[f'{metric}_qwen'] = ref.get((im['triplet_key'], im['arm'], im['seed'], metric))
        out.append(rec)
        print(json.dumps(rec), flush=True)

    dest = ROOT / 'results/second_judge/probe.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({'model': MODEL, 'n': len(out), 'rows': out}, indent=1))
    ok = [r for r in out if 'usage' in r.get('bias', {})]
    if ok:
        tin = sum(r['bias']['usage']['prompt_tokens'] + r['alignment']['usage']['prompt_tokens'] for r in ok) / len(ok)
        tout = sum(r['bias']['usage']['completion_tokens'] + r['alignment']['usage']['completion_tokens'] for r in ok) / len(ok)
        print(f'MEASURED per image: prompt_tokens={tin:.0f} completion_tokens={tout:.0f}', flush=True)


if __name__ == '__main__':
    main()
