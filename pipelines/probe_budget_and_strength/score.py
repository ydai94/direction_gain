"""Exp280 scoring: rate the alpha-sweep steered images with the paper's judge
(Qwen3-VL-30B-A3B-Instruct), same compact rubric and decoding as Exp271 score_engine,
so the resulting composite is on the same scale as the main text.

Sharded over images; each worker writes its own JSONL and is resumable.
Usage: score.py MODEL SHARD NSHARDS
"""
import fcntl
import glob
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

ROOT = Path(os.environ.get("DG_ROOT", "."))
REL = ROOT / 'pipelines/heldout_prediction'
GEN = ROOT / 'results/probe_budget_and_strength/images'
OUT = ROOT / 'results/probe_budget_and_strength/scoring'


def parse(raw, metric):
    s = raw.strip()
    if metric == 'bias' and s in list('012345'):
        return int(s)
    if metric == 'alignment' and s.lower() in ('true', 'false'):
        return s.lower() == 'true'
    raise ValueError('FORMAT_FAILURE')


def main(model, shard, nshards):
    cfg = json.loads((REL / 'judge_config.json').read_text())
    prompts = json.loads((REL / 'prompts.json').read_text())
    cohort = {r['triplet_key']: r for r in json.loads((REL / 'cohort.json').read_text())}
    images = []
    for mf in sorted(glob.glob(str(GEN / model / 'shard_*.json'))):
        images.extend(json.loads(Path(mf).read_text())['images'])
    images.sort(key=lambda im: (im['triplet_key'], im['dose'], im['seed']))
    assert images, f'no images found for {model}'
    mine = images[shard::nshards]

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f'scores_{model}_{shard:02d}.jsonl'
    lock = (OUT / f'{model}_{shard:02d}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    done = set()
    if dest.exists():
        for line in dest.open():
            d = json.loads(line)
            done.add((d['triplet_key'], d['dose'], d['seed'], d['metric']))
    todo = [(im, m) for im in mine for m in ('bias', 'alignment')
            if (im['triplet_key'], im['dose'], im['seed'], m) not in done]
    print(f'{model} shard {shard}/{nshards}: {len(mine)} images, {len(todo)} calls to make', flush=True)
    if not todo:
        return

    torch.manual_seed(cfg['seed'])
    torch.set_grad_enabled(False)
    judge = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        cfg['judge_path'], dtype=torch.bfloat16, device_map='auto').eval()
    proc = AutoProcessor.from_pretrained(cfg['judge_path'])
    t0 = time.time()
    with dest.open('a') as fh:
        for n, (im, metric) in enumerate(todo, 1):
            path = ROOT / im['path']
            question = prompts[metric]['compact'].format(**cohort[im['triplet_key']])
            attempts, value = [], None
            for _ in range(cfg['attempts']):
                msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': str(path)},
                                                     {'type': 'text', 'text': question}]}]
                batch = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                 return_dict=True, return_tensors='pt').to(judge.device)
                gen = judge.generate(**batch, max_new_tokens=cfg['max_new_tokens'], do_sample=False)
                raw = proc.batch_decode(gen[:, batch.input_ids.shape[1]:], skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)[0]
                attempts.append(raw)
                del batch, gen
                try:
                    value = parse(raw, metric)
                    break
                except ValueError:
                    question = question + ' Output only the requested value.'
            fh.write(json.dumps(dict(triplet_key=im['triplet_key'], arm='steer', dose=im['dose'],
                                     seed=im['seed'], metric=metric, value=value, raw=attempts,
                                     image_sha256=im['sha256'],
                                     judge=cfg['judge_path'].split('/')[-3])) + '\n')
            fh.flush()
            if n % 200 == 0:
                print(f'{n}/{len(todo)}  {time.time() - t0:.0f}s', flush=True)
    print(f'{model} shard {shard} done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
