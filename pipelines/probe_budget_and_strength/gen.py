"""Exp280 B4 generation: steered images at steering strengths other than the deployed one,
on the frozen 300-case subset, at the paper's outcome seeds 3 and 4.

Clean images are not regenerated -- Exp271 already holds clean seeds 3/4 for these cases,
and Exp276 holds clean seeds 10-15. Only the steered arm depends on the strength.

SD3.5 and FLUX only; Qwen's sweep is deferred (it is ~29 GPU-h on its own).

Work-stealing over shards, atomic per-image writes, resumable.
Usage: gen.py MODEL BUDGET_SECONDS
"""
import fcntl
import json
import os
import sys
import time
from pathlib import Path

from common import ROOT, digest, read_json, write_json

SPEC = json.loads(Path(__file__).resolve().with_name('cohort300.json').read_text())
PLAN = json.loads(Path(__file__).resolve().with_name('plan.json').read_text())
SEEDS = PLAN['outcome_seeds']
SHARD = 12
MIN_GB = {'qwen': 70, 'flux': 40, 'sd3': 23}
STALE_S = 3600
OUT = ROOT / 'results/probe_budget_and_strength/images'
REQUIRE_GPU = os.environ.get('EXP280_REQUIRE_GPU', 'A100')


def tag(dose):
    return ('%g' % dose).replace('.', 'p')


def shards():
    keys = SPEC['triplet_keys']
    return [keys[i:i + SHARD] for i in range(0, len(keys), SHARD)]


def claim(model, i):
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
    doses = PLAN[model]['gen_doses']
    out = OUT / model
    out.mkdir(parents=True, exist_ok=True)
    mp = out / f'shard_{i:03d}.json'
    rec = read_json(mp) if mp.exists() else dict(model=model, shard=i, doses=doses, cases={}, images=[])
    have = {(x['triplet_key'], x['dose'], x['seed']) for x in rec['images']}
    for k in keys:
        row = rows[k]
        adapter.reset_cache()
        try:
            p = adapter.prepare(row)
        except Exception as e:
            if type(e).__name__ != 'NotWritable':
                raise
            rec['cases'][k] = dict(status=f'NOT_WRITABLE:{e}')
            write_json(mp, rec)
            continue
        rec['cases'][k] = dict(status='PRIMARY_VALID', anchors=p['anchors'],
                               input_direction_norm=p['input_direction_norm'])
        for dose in doses:
            old = adapter.spec['dose']
            adapter.spec = dict(adapter.spec, dose=dose)
            try:
                pack = adapter.write(p['neutral'], p['direction'], p['anchors'])
                delta = pack['pe'].float() - p['neutral']['pe'].float()
                executed = float(delta.norm())
                del delta
            finally:
                adapter.spec = dict(adapter.spec, dose=old)
            for seed in SEEDS:
                if (k, dose, seed) in have:
                    continue
                if time.time() > deadline:
                    write_json(mp, rec)
                    return False
                path = out / f'{k}_steer_d{tag(dose)}_{seed}.png'
                tmp = path.with_suffix('.png.partial')
                t0 = time.perf_counter()
                image = adapter.generate(pack, seed)
                check_audit(adapter.last_audit, model)
                with tmp.open('wb') as f:
                    image.save(f, format='PNG')
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)
                rec['images'].append(dict(triplet_key=k, id=row['id'], arm='steer', dose=dose, seed=seed,
                                          executed_write_norm=executed,
                                          path=str(path.relative_to(ROOT)), sha256=digest(path),
                                          audit=adapter.last_audit, seconds=time.perf_counter() - t0,
                                          job_id=os.environ.get('SLURM_JOB_ID', 'local')))
                have.add((k, dose, seed))
                write_json(mp, rec)
                heartbeat.seek(0)
                heartbeat.write(f"{os.environ.get('SLURM_JOB_ID','local')} {time.time()}\n")
                heartbeat.flush()
            del pack
        del p
        print(model, i, k[:12], len(rec['images']), 'images', flush=True)
    ok = {k for k in keys if rec['cases'].get(k, {}).get('status') == 'PRIMARY_VALID'}
    expected = {(k, d, s) for k in ok for d in doses for s in SEEDS}
    assert {(x['triplet_key'], x['dose'], x['seed']) for x in rec['images']} == expected
    write_json(mp, rec)
    write_json(out / 'claims' / f'{i:03d}.done',
               dict(status='COMPLETE', n_images=len(rec['images']), manifest_sha256=digest(mp)))
    return True


def preflight(model, adapter, rows):
    """The write operator must reproduce Exp271's T_rel at the deployed strength, and
    the executed write norm must be exactly proportional to the strength. Text-encoder
    only: no sampling, no image."""
    analysis = read_json(ROOT / 'results/heldout_prediction/runtime/analysis.json')
    stored = {c['triplet_key']: c for c in analysis['models'][model]['cases']
              if c['status'] == 'PRIMARY_VALID'}
    deployed = PLAN[model]['deployed_dose']
    checked = 0
    for k in SPEC['triplet_keys']:
        if k not in stored:
            continue
        adapter.reset_cache()
        p = adapter.prepare(rows[k])
        rel = abs(p['T_rel'] - stored[k]['T_rel']) / max(abs(stored[k]['T_rel']), 1e-12)
        assert rel < 1e-6, f'write operator drift on {k[:12]}: T_rel {p["T_rel"]} vs {stored[k]["T_rel"]}'
        base = None
        for dose in [deployed] + list(PLAN[model]['gen_doses']):
            old = adapter.spec['dose']
            adapter.spec = dict(adapter.spec, dose=dose)
            try:
                pack = adapter.write(p['neutral'], p['direction'], p['anchors'])
            finally:
                adapter.spec = dict(adapter.spec, dose=old)
            n = float((pack['pe'].float() - p['neutral']['pe'].float()).norm())
            if base is None:
                base = n / dose
            assert abs(n / dose - base) / base < 1e-3, f'write norm not proportional to strength at {dose}'
            del pack
        del p
        checked += 1
        if checked >= 2:
            break
    assert checked >= 2, 'no cohort case found in stored Exp271 records'
    print(f'PREFLIGHT PASS {model}: {checked} cases, T_rel matches Exp271, '
          f'write norm proportional across {[deployed] + list(PLAN[model]["gen_doses"])}', flush=True)


def main(model, budget_s):
    import torch
    from adapter_ext import ExtAdapter
    assert PLAN[model]['gen_doses'], f'no generation doses planned for {model}'
    name = torch.cuda.get_device_name(0)
    if REQUIRE_GPU and REQUIRE_GPU not in name:
        raise AssertionError(f'Exp271 clean images came from {REQUIRE_GPU}; this node has {name}')
    have = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    assert have >= MIN_GB[model], f'{model} needs {MIN_GB[model]} GB, got {have:.0f}'
    print('gpu accepted:', name, flush=True)
    rows = {r['triplet_key']: r for r in read_json(Path(__file__).resolve().parent / 'cohort.json')}
    deadline = time.time() + budget_s
    adapter = ExtAdapter(model, PLAN[model]['deployed_timesteps'])
    if os.environ.get('EXP280_SKIP_SELFTEST') != '1':
        preflight(model, adapter, rows)
    blocks = shards()
    done = 0
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
            done += int(run_shard(model, i, keys, adapter, rows, fh, deadline - 300))
        finally:
            fh.close()
            if not (OUT / model / 'claims' / f'{i:03d}.done').exists():
                try:
                    (OUT / model / 'claims' / f'{i:03d}.claim').unlink()
                except FileNotFoundError:
                    pass
    print('worker finished, shards completed:', done, flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
