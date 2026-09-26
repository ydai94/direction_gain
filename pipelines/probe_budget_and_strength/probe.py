"""Exp280 probe grid: B1 (probe-design variants) and B4 (steering-strength sweep).

One neutral trajectory set serves every cell, because the trajectory depends only on
the neutral prompt and the image seed -- not on the probe timestep, the reference
construction, or the steering strength. Cells are therefore the union of two slices:

  B1  all timesteps at the deployed dose          -> does the probe step matter?
  B4  the deployed timesteps at every dose        -> does G survive a change of alpha?

Each cell also records a second reference construction (anti - neutral) alongside the
deployed one (anti - stereotype): no extra forward for SD3/Qwen, one extra for FLUX.
Projection vs cosine needs no new computation, since magnitude is stored per cell.

Qwen runs the deployed timesteps only. Its trajectory stops at the last probe step and
its forward audit expects that exact length, so extending it would mean rewriting a
verified path; the alternative reference is still measured for all three models.

Work-stealing over shards: any number of workers on any number of partitions
self-balance. Per-case writes are atomic and resumable.

Usage: probe.py MODEL BUDGET_SECONDS
       probe.py MODEL --selftest N     regression against stored Exp271 cells
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
SHARD = 12
MIN_GB = {'qwen': 70, 'flux': 40, 'sd3': 23}
STALE_S = 3600
OUT = (ROOT / 'results/probe_budget_and_strength'
       / (os.environ.get('EXP280_OUT_TAG') or 'probe'))
REQUIRE_GPU = os.environ.get('EXP280_REQUIRE_GPU', 'A100')
EXP271 = ROOT / 'results/heldout_prediction/runtime'


def shards():
    keys = SPEC['triplet_keys']
    return [keys[i:i + SHARD] for i in range(0, len(keys), SHARD)]


def grid(model):
    """(timestep, dose) cells: all timesteps at deployed dose, deployed timesteps at all doses."""
    p = PLAN[model]
    cells = [(t, p['deployed_dose']) for t in p['timesteps']]
    for d in p['doses']:
        if d == p['deployed_dose']:
            continue
        for t in p['deployed_timesteps']:
            cells.append((t, d))
    assert len(set(cells)) == len(cells)
    return cells


def edited_at(adapter, p, dose):
    old = adapter.spec['dose']
    adapter.spec = dict(adapter.spec, dose=dose)
    try:
        return adapter.write(p['neutral'], p['direction'], p['anchors'])
    finally:
        adapter.spec = dict(adapter.spec, dose=old)


def safe_cosine(a, b):
    import torch
    x, y = a.float().flatten(), b.float().flatten()
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('nonfinite response')
    nx, ny = float(x.norm()), float(y.norm())
    if not nx or not ny:
        return None
    v = float((x @ y) / (nx * ny))
    if not -1.00001 <= v <= 1.00001:
        raise AssertionError('cosine out of range')
    return max(-1., min(1., v))


def probe_case(adapter, row, p, model):
    """Cells for one case. Forward cost per cell: 4 (sd3/qwen) or 5 (flux)."""
    cells = []
    adapter.trajectory_audits = []
    cellplan = grid(model)
    by_dose = {d: edited_at(adapter, p, d) for _, d in cellplan}
    for seed in PLAN['probe_seeds']:
        snapshots = adapter.trajectory(p['neutral'], seed)
        for t in sorted({t for t, _ in cellplan}):
            z = snapshots[t]

            def response(pack, reference=False):
                return adapter.velocity(z, pack, p['neutral'], reference)

            v0 = response(p['neutral'])
            # FLUX swaps in reference text kwargs, so the neutral pole of an
            # alternative reference must be taken under the same convention.
            v0_ref = response(p['neutral'], True) if model == 'flux' else v0
            anti = response(p['anti'], True)
            ref_pair = anti - response(p['stereo'], True)      # deployed: anti - stereotype
            ref_anti = anti - v0_ref                            # variant:  anti - neutral
            for tt, dose in cellplan:
                if tt != t:
                    continue
                delta = response(by_dose[dose]) - v0
                cells.append(dict(
                    seed=seed, timestep=t, dose=dose,
                    cosine=safe_cosine(delta, ref_pair),
                    cosine_anti_neutral=safe_cosine(delta, ref_anti),
                    magnitude=float(delta.norm()),
                    reference_norm=float(ref_pair.norm()),
                    reference_norm_anti_neutral=float(ref_anti.norm()),
                    deployed_cell=bool(tt in PLAN[model]['deployed_timesteps']
                                       and dose == PLAN[model]['deployed_dose'])))
                del delta
            del v0, anti, ref_pair, ref_anti
        del snapshots
    return cells


def run_shard(model, i, keys, adapter, rows, heartbeat, deadline):
    out = OUT / model
    out.mkdir(parents=True, exist_ok=True)
    mp = out / f'shard_{i:03d}.json'
    rec = read_json(mp) if mp.exists() else dict(model=model, shard=i, plan=PLAN[model], cases={})
    for k in keys:
        if k in rec['cases']:
            continue
        if time.time() > deadline:
            write_json(mp, rec)
            return False
        row = rows[k]
        adapter.reset_cache()
        t0 = time.perf_counter()
        try:
            p = adapter.prepare(row)
        except Exception as e:
            if type(e).__name__ != 'NotWritable':
                raise
            rec['cases'][k] = dict(status=f'NOT_WRITABLE:{e}', cells=[])
            write_json(mp, rec)
            continue
        cells = probe_case(adapter, row, p, model)
        rec['cases'][k] = dict(status='PRIMARY_VALID', cells=cells,
                               input_direction_norm=p['input_direction_norm'],
                               pair_separability=p['pair_separability'], T_rel=p['T_rel'],
                               anchors=p['anchors'], seconds=time.perf_counter() - t0,
                               trajectory_audits=adapter.trajectory_audits,
                               job_id=os.environ.get('SLURM_JOB_ID', 'local'))
        write_json(mp, rec)
        heartbeat.seek(0)
        heartbeat.write(f"{os.environ.get('SLURM_JOB_ID','local')} {time.time()}\n")
        heartbeat.flush()
        del p
        print(model, i, k[:12], len(cells), 'cells', round(time.perf_counter() - t0, 1), 's', flush=True)
    assert set(rec['cases']) >= set(keys)
    write_json(mp, rec)
    write_json(out / 'claims' / f'{i:03d}.done',
               dict(status='COMPLETE', n_cases=len(keys), manifest_sha256=digest(mp)))
    return True


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


def build_adapter(model):
    import torch
    from adapter_ext import ExtAdapter
    # Delta v is a difference of two nearly identical velocity fields, so architecture-level
    # kernel differences are amplified: the stored Exp271 cells reproduce bit-exactly on
    # A100 and differ by up to 0.46 in cosine on H100/L40S. Pin the card before loading.
    name = torch.cuda.get_device_name(0)
    if REQUIRE_GPU and REQUIRE_GPU not in name:
        raise AssertionError(f'Exp271 cells are reproducible on {REQUIRE_GPU}; this node has {name}')
    have = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    assert have >= MIN_GB[model], f'{model} needs {MIN_GB[model]} GB, got {have:.0f}'
    print('gpu accepted:', name, flush=True)
    return ExtAdapter(model, PLAN[model]['timesteps'])


def selftest(model, n, adapter=None, rows=None):
    """Reproduce stored Exp271 cells at the deployed timesteps and dose."""
    import glob
    adapter = adapter or build_adapter(model)
    rows = rows or {r['triplet_key']: r for r in read_json(Path(__file__).resolve().parent / 'cohort.json')}
    stored = {}
    for f in sorted(glob.glob(str(EXP271 / model / 'shard_*.json'))):
        for k, v in read_json(f).get('cases', {}).items():
            cells = (v.get('probe') or {}).get('cells')
            if cells:
                stored[k] = cells
        if len(stored) >= n * 20:
            break
    keys = [k for k in SPEC['triplet_keys'] if k in stored][:n]
    assert keys, 'no overlap with stored Exp271 probe cells'
    worst_c = worst_m = 0.
    total = 0
    for k in keys:
        adapter.reset_cache()
        p = adapter.prepare(rows[k])
        got = probe_case(adapter, rows[k], p, model)
        ref = {(c['seed'], c['timestep']): c for c in stored[k]}
        checked = 0
        for c in got:
            if not c['deployed_cell']:
                continue
            o = ref.get((c['seed'], c['timestep']))
            if o is None or o.get('cosine') is None or c['cosine'] is None:
                continue
            worst_c = max(worst_c, abs(c['cosine'] - o['cosine']))
            worst_m = max(worst_m, abs(c['magnitude'] - o['magnitude']) / max(abs(o['magnitude']), 1e-9))
            checked += 1
        total += checked
        print(f'{k[:12]} checked {checked} deployed cells; running worst |dcos| {worst_c:.2e} '
              f'rel |dmag| {worst_m:.2e}', flush=True)
    assert total >= 2 * len(keys), f'only {total} deployed cells matched stored Exp271 records'
    print(f'SELFTEST {model}: {total} cells, max |dcos| {worst_c:.3e}, max rel |dmag| {worst_m:.3e}', flush=True)
    assert worst_c < 2e-3, f'cosine regression vs Exp271: {worst_c}'
    assert worst_m < 2e-2, f'magnitude regression vs Exp271: {worst_m}'
    print('SELFTEST PASS', flush=True)


def main(model, budget_s):
    adapter = build_adapter(model)
    rows = {r['triplet_key']: r for r in read_json(Path(__file__).resolve().parent / 'cohort.json')}
    # Gate every worker on reproducing the stored Exp271 cells before it writes anything:
    # the probe timesteps are configurable here, so a worker that cannot reproduce the
    # deployed cells must not contribute cells to the pooled result.
    if os.environ.get('EXP280_SKIP_SELFTEST') != '1':
        selftest(model, 1, adapter=adapter, rows=rows)
    deadline = time.time() + budget_s
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
            ok = run_shard(model, i, keys, adapter, rows, fh, deadline - 300)
            done += int(ok)
        finally:
            fh.close()
            if not (OUT / model / 'claims' / f'{i:03d}.done').exists():
                try:
                    (OUT / model / 'claims' / f'{i:03d}.claim').unlink()
                except FileNotFoundError:
                    pass
    print('worker finished, shards completed:', done, flush=True)


if __name__ == '__main__':
    if sys.argv[2] == '--selftest':
        selftest(sys.argv[1], int(sys.argv[3]))
    else:
        main(sys.argv[1], int(sys.argv[2]))
