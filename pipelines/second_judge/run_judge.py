"""Exp277 / A2(b): re-score the Exp271 images with a GPT vision judge, reusing the
Qwen3-VL rubric verbatim (bias and alignment asked as two separate calls, exactly as
the original judge was queried).

Resumable: results append to scores_<model>.jsonl and completed keys are skipped.
A hard USD ceiling stops the run rather than overspending.

Usage: python run_judge.py <model> [max_usd]
"""
import base64
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(os.environ.get("DG_ROOT", "."))
REL = ROOT / 'pipelines/heldout_prediction'
EXP = ROOT / 'results/heldout_prediction/runtime'
OUT = ROOT / 'results/second_judge'
API = 'https://api.openai.com/v1/chat/completions'
JUDGE = os.environ.get('JUDGE_MODEL', 'gpt-5.6-terra')
PRICE_IN, PRICE_OUT = 2.00 / 1e6, 12.00 / 1e6
WORKERS = int(os.environ.get('WORKERS', '24'))

lock = threading.Lock()
spent = {'usd': 0.0, 'n': 0, 'fail': 0}
stop = threading.Event()


def parse(raw, metric):
    s = raw.strip()
    if metric == 'bias' and s in list('012345'):
        return int(s)
    if metric == 'alignment' and s.lower() in ('true', 'false'):
        return s.lower() == 'true'
    return None


def call(text, b64, key):
    payload = {
        'model': JUDGE,
        'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': text},
            {'type': 'image_url',
             'image_url': {'url': f'data:image/png;base64,{b64}', 'detail': 'high'}}]}],
        'max_completion_tokens': 16,
        'reasoning_effort': 'none',
    }
    req = urllib.request.Request(
        API, data=json.dumps(payload).encode(),
        headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < 4:
                time.sleep(2 ** attempt + 1)
                continue
            raise
        except Exception:
            if attempt < 4:
                time.sleep(2 ** attempt + 1)
                continue
            raise
    raise RuntimeError('retries exhausted')


def work(job, key, prompts, cohort, fh, max_usd):
    if stop.is_set():
        return
    im, metric = job
    case = cohort[im['triplet_key']]
    try:
        b64 = base64.b64encode((ROOT / im['path']).read_bytes()).decode()
        r = call(prompts[metric]['compact'].format(**case), b64, key)
        u = r['usage']
        raw = r['choices'][0]['message']['content']
        rec = dict(triplet_key=im['triplet_key'], arm=im['arm'], seed=im['seed'], metric=metric,
                   raw=raw, value=parse(raw, metric), judge=JUDGE,
                   prompt_tokens=u['prompt_tokens'], completion_tokens=u['completion_tokens'],
                   image_sha256=im['sha256'])
        cost = u['prompt_tokens'] * PRICE_IN + u['completion_tokens'] * PRICE_OUT
    except Exception as e:  # noqa: BLE001 - record the failure and keep going
        rec = dict(triplet_key=im['triplet_key'], arm=im['arm'], seed=im['seed'], metric=metric,
                   error=str(e)[:200], judge=JUDGE)
        cost = 0.0
    with lock:
        fh.write(json.dumps(rec) + '\n')
        fh.flush()
        spent['usd'] += cost
        spent['n'] += 1
        spent['fail'] += int('error' in rec or rec.get('value') is None)
        if spent['n'] % 500 == 0:
            print(f"{spent['n']} calls  ${spent['usd']:.2f}  unparsed/failed={spent['fail']}", flush=True)
        if spent['usd'] > max_usd:
            stop.set()
            print(f"STOP: ceiling ${max_usd} reached", flush=True)


def main():
    model = sys.argv[1]
    max_usd = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    key = os.environ['OPENAI_API_KEY']
    prompts = json.loads((REL / 'prompts.json').read_text())
    cohort = {r['triplet_key']: r for r in json.loads((REL / 'cohort.json').read_text())}
    rows = json.loads((EXP / model / 'scoring/inputs.json').read_text())['rows']

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'scores_{model}.jsonl'
    done = set()
    if path.exists():
        for line in path.open():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get('value') is not None:
                done.add((d['triplet_key'], d['arm'], d['seed'], d['metric']))

    # approved scope: the three outcome arms at the two matched outcome seeds.
    # inputs.json also carries the independent baseline clean seeds 0/1, which feed the
    # learned baseline's headroom features and are not part of this judge comparison.
    OUTCOME_SEEDS = (3, 4)
    rows = [r for r in rows if r['image']['seed'] in OUTCOME_SEEDS]
    jobs = [(r['image'], m) for r in rows for m in ('bias', 'alignment')
            if (r['image']['triplet_key'], r['image']['arm'], r['image']['seed'], m) not in done]
    print(f'{model}: {len(rows)} images, {len(jobs)} calls to make, {len(done)} already done', flush=True)
    if not jobs:
        return
    t0 = time.time()
    with path.open('a') as fh, ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(lambda j: work(j, key, prompts, cohort, fh, max_usd), jobs))
    print(f"{model} finished: {spent['n']} calls  ${spent['usd']:.2f}  "
          f"unparsed/failed={spent['fail']}  {time.time() - t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
