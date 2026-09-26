"""Exp276: extra outcome seeds on a frozen 300-case subset, for the outcome
noise ceiling (A1-full). Clean and steered arms only, seeds 10-15.

Work-stealing: every worker repeatedly claims the next unclaimed shard, so any
number of workers on any number of partitions self-balance. Claims are stale
after STALE_S without a heartbeat, so a preempted worker's shard is retaken.
Per-image writes are atomic and resumable; re-running is idempotent.
"""
import fcntl
import json
import os
import sys
import time
from pathlib import Path

from common import ROOT, digest, read_json, write_json

SPEC = json.loads(Path(__file__).resolve().with_name('cohort300.json').read_text())
SEEDS = SPEC['new_image_seeds']
ARMS = SPEC['arms']
SHARD = 12
MIN_GB = {'qwen': 70, 'flux': 40, 'sd3': 23}
STALE_S = 3600
OUT = ROOT / 'results/seed_ceiling/runtime'


def shards():
    keys = SPEC['triplet_keys']
    return [keys[i:i + SHARD] for i in range(0, len(keys), SHARD)]


def claim(model, i):
    """Atomically take shard i; return an open heartbeat file or None."""
    d = OUT / model / 'claims'
    d.mkdir(parents=True, exist_ok=True)
    if (d / f'{i:03d}.done').exists():
        return None
    p = d / f'{i:03d}.claim'
    if p.exists() and time.time() - p.stat().st_mtime < STALE_S:
        return None
    try:
        fh = os.open(p, os.O_CREAT | os.O_RDWR)
    except OSError:
        return None
    f = os.fdopen(fh, 'r+')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        return None
    f.seek(0)
    f.truncate()
    f.write(f"{os.environ.get('SLURM_JOB_ID', 'local')} {time.time()}\n")
    f.flush()
    return f


def run_shard(model, i, keys, adapter, rows, heartbeat, deadline):
    from headroom import check_audit
    out = OUT / model
    out.mkdir(parents=True, exist_ok=True)
    mp = out / f'shard_{i:03d}.json'
    rec = read_json(mp) if mp.exists() else dict(model=model, shard=i, cases={}, images=[])
    have = {(x['triplet_key'], x['arm'], x['seed']) for x in rec['images']}
    for k in keys:
        row = rows[k]
        adapter.reset_cache()
        p = adapter.prepare(row)
        packs = {'clean': p['neutral'], 'steer': p['edited']}
        rec['cases'][k] = dict(arms=ARMS, input_direction_norm=p['input_direction_norm'],
                               pair_separability=p['pair_separability'], T_rel=p['T_rel'],
                               anchors=p['anchors'], executed_write_norm=p['executed_write_norm'])
        for arm in ARMS:
            for seed in SEEDS:
                if (k, arm, seed) in have:
                    continue
                if time.time() > deadline:
                    write_json(mp, rec)
                    return False
                path = out / f'{k}_{arm}_{seed}.png'
                tmp = path.with_suffix('.png.partial')
                t = time.perf_counter()
                image = adapter.generate(packs[arm], seed)
                check_audit(adapter.last_audit, model)
                with tmp.open('wb') as f:
                    image.save(f, format='PNG')
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)
                rec['images'].append(dict(triplet_key=k, id=row['id'], arm=arm, seed=seed,
                                          path=str(path.relative_to(ROOT)), sha256=digest(path),
                                          audit=adapter.last_audit, seconds=time.perf_counter() - t,
                                          job_id=os.environ.get('SLURM_JOB_ID', 'local')))
                have.add((k, arm, seed))
                write_json(mp, rec)
                heartbeat.seek(0)
                heartbeat.write(f"{os.environ.get('SLURM_JOB_ID','local')} {time.time()}\n")
                heartbeat.flush()
        del p, packs
        print(model, i, 'case', k, len(rec['images']), flush=True)
    expected = {(k, a, s) for k in keys for a in ARMS for s in SEEDS}
    assert {(x['triplet_key'], x['arm'], x['seed']) for x in rec['images']} == expected
    write_json(mp, rec)
    write_json(OUT / model / 'claims' / f'{i:03d}.done',
               dict(status='COMPLETE', n_images=len(rec['images']), manifest_sha256=digest(mp)))
    return True


def main(model, budget_s):
    import torch
    from adapter import FixedAdapter
    assert torch.cuda.get_device_properties(0).total_memory >= MIN_GB[model] * 1024 ** 3, \
        f"{model} needs {MIN_GB[model]} GB, got {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.0f}"
    rows = {r['triplet_key']: r for r in read_json(Path(__file__).resolve().parent / 'cohort.json')}
    deadline = time.time() + budget_s
    adapter = FixedAdapter(model)
    blocks = shards()
    done_any = 0
    while time.time() < deadline - 600:
        took = None
        for i, keys in enumerate(blocks):
            fh = claim(model, i)
            if fh is not None:
                took = (i, keys, fh)
                break
        if took is None:
            print('no unclaimed shard left', flush=True)
            break
        i, keys, fh = took
        print('claimed', model, i, flush=True)
        try:
            ok = run_shard(model, i, keys, adapter, rows, fh, deadline - 300)
            done_any += int(ok)
            if not ok:
                print('budget reached inside shard', i, flush=True)
        finally:
            fh.close()
            if not (OUT / model / 'claims' / f'{i:03d}.done').exists():
                try:
                    (OUT / model / 'claims' / f'{i:03d}.claim').unlink()
                except FileNotFoundError:
                    pass
    print('worker finished, shards completed:', done_any, flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
