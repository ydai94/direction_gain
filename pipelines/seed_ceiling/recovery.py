"""Exp276 fixed-shard generation; scientific adapter is unchanged from Exp271."""
import argparse
import datetime
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from common import CODE, ROOT, digest, read_json, write_json

MODELS = ('qwen', 'sd3', 'flux')
BASE = ROOT / 'results/seed_ceiling/recovery_v2'
MIN_GIB = {'qwen': 70, 'sd3': 23, 'flux': 40}


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def release_binding():
    frozen = read_json(CODE / 'release.json')
    for name, sha in frozen['files'].items():
        if digest(CODE / name) != sha:
            raise RuntimeError('release changed: ' + name)
    for name, sha in frozen['dependencies'].items():
        if digest(ROOT / name) != sha:
            raise RuntimeError('dependency changed: ' + name)
    for asset in frozen.get('model_assets', []):
        p = Path(asset['path'])
        if not p.is_file() or p.stat().st_size != asset['size']:
            raise RuntimeError('missing/changed model asset: ' + str(p))
        if 'sha256' in asset and digest(p) != asset['sha256']:
            raise RuntimeError('model configuration changed: ' + str(p))
    return digest(CODE / 'release.json')


def preflight():
    binding = release_binding()
    spec = read_json(CODE / 'cohort300.json')
    rows = read_json(CODE / 'cohort.json')
    keys = spec['triplet_keys']
    if len(keys) != 300 or len(set(keys)) != 300:
        raise ValueError('expected 300 unique selected keys')
    if spec['arms'] != ['clean', 'steer'] or spec['new_image_seeds'] != list(range(10, 16)):
        raise ValueError('registered arms/seeds changed')
    by_key = {r['triplet_key']: r for r in rows}
    if len(by_key) != len(rows) or not set(keys) <= set(by_key):
        raise ValueError('duplicate or missing source cases')
    for k in keys:
        for field in ('id', 'prompt_neutral', 'prompt_stereotype', 'prompt_anti_stereotype'):
            if field not in by_key[k] or by_key[k][field] in (None, ''):
                raise ValueError(f'missing {field} for {k}')
    protocol = read_json(CODE / 'protocol.json')['settings']
    expected = {'qwen': (50, 4., 2., 'normalized_cfg'),
                'sd3': (28, 4., 2., 'linear_cfg'),
                'flux': (8, None, 1., 'guidance_distilled_native')}
    for model, values in expected.items():
        if tuple(protocol[model][x] for x in ('steps', 'guidance_scale', 'dose', 'guidance_mode')) != values:
            raise ValueError('generation protocol changed: ' + model)
    return dict(status='PASS_STATIC', binding=binding, n_cases=300, shards_per_model=25,
                images_per_model=3600, scientific_equivalence_not_certified=True)


def selected(stage, shard):
    keys = read_json(CODE / 'cohort300.json')['triplet_keys']
    if stage == 'smoke':
        if shard != 0:
            raise ValueError('smoke uses shard 0')
        return keys[:1], [10]
    if not 0 <= shard < 25:
        raise ValueError('shard outside 0..24')
    return keys[12 * shard:12 * (shard + 1)], list(range(10, 16))


def expected(stage, shard):
    keys, seeds = selected(stage, shard)
    return {(k, arm, seed) for k in keys for arm in ('clean', 'steer') for seed in seeds}


def audit_check(audit, model):
    recipe = read_json(CODE / 'protocol.json')['settings'][model]
    want = dict(kind='generate', forward_calls={'qwen': 100, 'sd3': 28, 'flux': 8}[model],
                batch_size=2 if model == 'sd3' else 1, unconditional_executed=model != 'flux',
                guidance_mode=recipe['guidance_mode'], guidance_scale=recipe['guidance_scale'],
                snapshot_convention='actual_pre_forward', indices=recipe['indices'])
    if any(audit.get(k) != v for k, v in want.items()):
        raise ValueError('generation audit mismatch')


def key_of(row):
    return (row['triplet_key'], row['arm'], row['seed'])


def validate_image(row, model, folder):
    from PIL import Image
    p = ROOT / row['path']
    if p.parent.resolve() != folder.resolve():
        raise ValueError('image outside shard directory')
    audit_check(row['audit'], model)
    if not p.is_file() or digest(p) != row['sha256']:
        raise ValueError('missing image or SHA256 mismatch')
    with Image.open(p) as im:
        if im.size != (1024, 1024) or im.format != 'PNG':
            raise ValueError('invalid image dimensions/format')
        im.verify()


def folder_for(model, stage, shard):
    return BASE / stage / model / f'shard_{shard:03d}'


def load_record(folder, model, stage, shard, binding):
    path = folder / 'manifest.json'
    wanted = dict(model=model, stage=stage, shard=shard, release_sha256=binding)
    rec = read_json(path) if path.exists() else dict(**wanted, cases={}, images=[], recovery_events=[])
    if any(rec.get(k) != v for k, v in wanted.items()):
        raise ValueError('manifest binding mismatch')
    seen = [key_of(r) for r in rec['images']]
    if len(set(seen)) != len(seen) or not set(seen) <= expected(stage, shard):
        raise ValueError('duplicate or unexpected image combination')
    return rec


def quarantine(path, folder, reason, rec):
    target = folder / 'quarantine' / f'{time.time_ns()}_{path.name}'
    target.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        os.replace(path, target)
    rec['recovery_events'].append(dict(time=stamp(), path=str(path.relative_to(ROOT)),
                                       reason=reason, quarantine=str(target.relative_to(ROOT))))


def repair_record(rec, folder, model):
    good = []
    for row in rec['images']:
        # A changed audit or path is a protocol fault, not an image to silently repair.
        audit_check(row['audit'], model)
        p = ROOT / row['path']
        if p.parent.resolve() != folder.resolve():
            raise ValueError('image outside shard directory')
        try:
            validate_image(row, model, folder)
        except (ValueError, OSError) as exc:
            quarantine(p, folder, str(exc), rec)
        else:
            good.append(row)
    rec['images'] = good
    referenced = {str((ROOT / r['path']).resolve()) for r in good}
    for p in sorted(folder.glob('*.png*')):
        if str(p.resolve()) not in referenced:
            quarantine(p, folder, 'orphan_or_partial', rec)
    write_json(folder / 'manifest.json', rec)


def verify_shard(model, stage, shard, binding=None):
    binding = binding or release_binding()
    folder = folder_for(model, stage, shard)
    rec = load_record(folder, model, stage, shard, binding)
    if {key_of(r) for r in rec['images']} != expected(stage, shard):
        raise ValueError('incomplete image coverage')
    for row in rec['images']:
        validate_image(row, model, folder)
    return dict(status='VERIFIED_COMPLETE', model=model, stage=stage, shard=shard,
                n_images=len(rec['images']), release_sha256=binding,
                manifest_sha256=digest(folder / 'manifest.json'))


def owner_job():
    job = os.environ.get('SLURM_ARRAY_JOB_ID', os.environ.get('SLURM_JOB_ID', 'local'))
    task = os.environ.get('SLURM_ARRAY_TASK_ID')
    return f'{job}_{task}' if task is not None else job


def acquire(folder):
    lock = folder / 'writer.lock'
    lock.mkdir()  # Atomic across nodes; only controller may clear a terminal owner's lock.
    write_json(lock / 'owner.json', dict(job_id=owner_job(), pid=os.getpid(), time=stamp()))
    return lock


def provenance(adapter, folder, binding):
    import torch
    packages = {}
    for name in ('torch', 'diffusers', 'transformers', 'accelerate', 'numpy', 'Pillow', 'safetensors'):
        packages[name] = importlib.metadata.version(name)
    sources = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            p = Path(filename).resolve()
            if p.is_file() and p.suffix == '.py' and ROOT in p.parents:
                sources[str(p.relative_to(ROOT))] = digest(p)
    dest = folder / 'provenance' / f'{owner_job()}_{time.time_ns()}'
    for name in sources:
        dst = dest / 'code' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dst)
    pipe_source = Path(inspect.getfile(type(adapter.pipe)))
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pipe_source, dest / pipe_source.name)
    try:
        driver = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                          '--format=csv,noheader'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        driver = 'unavailable'
    result = dict(job_id=owner_job(), time=stamp(), release_sha256=binding, packages=packages,
                  gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                  gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                  driver=driver, source_sha256=sources, pipeline_sha256=digest(pipe_source))
    write_json(dest / 'environment.json', result)
    return result


def generate(model, stage, shard, budget):
    static = preflight()
    binding = static['binding']
    folder = folder_for(model, stage, shard)
    folder.mkdir(parents=True, exist_ok=True)
    lock = acquire(folder)
    started = time.monotonic()
    try:
        rec = load_record(folder, model, stage, shard, binding)
        repair_record(rec, folder, model)
        have = {key_of(r) for r in rec['images']}
        if have == expected(stage, shard):
            write_json(folder / 'complete.json', verify_shard(model, stage, shard, binding))
            return 0
        (folder / 'complete.json').unlink(missing_ok=True)
        import torch
        from adapter import FixedAdapter
        if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < MIN_GIB[model] * 1024**3:
            raise RuntimeError('insufficient GPU memory')
        adapter = FixedAdapter(model)
        env = provenance(adapter, folder, binding)
        keys, seeds = selected(stage, shard)
        rows = {r['triplet_key']: r for r in read_json(CODE / 'cohort.json')}
        for key in keys:
            if all((key, a, s) in have for a in ('clean', 'steer') for s in seeds):
                continue
            adapter.reset_cache()
            pack = adapter.prepare(rows[key])
            fields = ('input_direction_norm', 'pair_separability', 'T_rel', 'anchors', 'executed_write_norm')
            meta = {f: pack[f] for f in fields}
            if not all(math.isfinite(meta[f]) for f in fields if f != 'anchors'):
                raise ValueError('nonfinite preparation')
            previous = rec['cases'].get(key)
            if previous is not None and previous['anchors'] != meta['anchors']:
                raise ValueError('resume changed anchors')
            rec['cases'][key] = meta
            rec.setdefault('preparation_history', []).append(dict(triplet_key=key, job_id=owner_job(), **meta))
            for arm in ('clean', 'steer'):
                for seed in seeds:
                    if (key, arm, seed) in have:
                        continue
                    if time.monotonic() - started > budget - 300:
                        write_json(folder / 'manifest.json', rec)
                        print('INCOMPLETE_BUDGET', model, shard, len(have), flush=True)
                        return 75
                    path = folder / f'{key}_{arm}_{seed}.png'
                    temp = path.with_suffix(f'.{owner_job()}.png.partial')
                    t0 = time.monotonic()
                    image = adapter.generate(pack['neutral' if arm == 'clean' else 'edited'], seed)
                    audit_check(adapter.last_audit, model)
                    with temp.open('wb') as f:
                        image.save(f, format='PNG'); f.flush(); os.fsync(f.fileno())
                    os.replace(temp, path)
                    item = dict(triplet_key=key, id=rows[key]['id'], arm=arm, seed=seed,
                                path=str(path.relative_to(ROOT)), sha256=digest(path),
                                audit=adapter.last_audit, seconds=time.monotonic()-t0,
                                job_id=owner_job(), gpu=env['gpu'])
                    validate_image(item, model, folder)
                    rec['images'].append(item)
                    have.add((key, arm, seed))
                    write_json(folder / 'manifest.json', rec)
                    print(json.dumps(dict(model=model, shard=shard, done=len(have),
                                          expected=len(expected(stage, shard)))), flush=True)
            del pack
        release_binding()
        result = verify_shard(model, stage, shard, binding)
        result.update(job_id=owner_job(), elapsed_seconds=time.monotonic()-started, time=stamp())
        write_json(folder / 'complete.json', result)
        print(json.dumps(result), flush=True)
        return 0
    finally:
        shutil.rmtree(lock)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['preflight', 'generate', 'verify'])
    ap.add_argument('--model', choices=MODELS)
    ap.add_argument('--stage', choices=['smoke', 'formal'], default='formal')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--budget', type=int, default=3600)
    args = ap.parse_args()
    if args.action == 'preflight':
        print(json.dumps(preflight())); return 0
    if args.model is None:
        ap.error('--model required')
    if args.action == 'verify':
        print(json.dumps(verify_shard(args.model, args.stage, args.shard))); return 0
    return generate(args.model, args.stage, args.shard, args.budget)


if __name__ == '__main__':
    raise SystemExit(main())
