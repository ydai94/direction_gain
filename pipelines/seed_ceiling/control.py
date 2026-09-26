"""Bounded Exp276 smoke -> pilot -> fixed arrays -> verification controller."""
import argparse
import fcntl
import json
import math
import os
import subprocess
import time
from common import CODE, ROOT, read_json, write_json
from recovery import BASE, MODELS, folder_for, preflight, release_binding, stamp, verify_shard

SITE_SBATCH_ARGS = os.environ.get('DG_SBATCH_ARGS', '').split()
_GPU = os.environ.get('DG_GPU_PARTITIONS', 'gpu')
PARTITIONS = {'qwen': _GPU, 'sd3': _GPU, 'flux': _GPU}
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED'}


def note(text):
    with (ROOT/'results/REPORT.md').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write('\n\n### Exp276 recovery '+stamp()+'\n'+text+'\n')
        f.flush(); os.fsync(f.fileno())


def save(state):
    write_json(BASE/'control'/(state['model']+'.json'), state)


def submit(argv, state, role):
    command = ['sbatch', '--parsable', *argv]
    state.setdefault('submission_intents', []).append(dict(time=stamp(), role=role, command=command))
    save(state)
    job = subprocess.check_output(command, text=True, cwd=ROOT).strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('unexpected sbatch response: '+job)
    state.setdefault('jobs', []).append(dict(id=job, role=role, command=command, time=stamp()))
    try:
        save(state)
    except Exception:
        subprocess.run(['scancel', job], check=False)
        raise
    return job


def schedule(state, shards, minutes):
    model, phase = state['model'], state['phase']
    if not shards:
        raise ValueError('empty shard submission')
    stage = 'smoke' if phase == 'smoke' else 'formal'
    wall = f'{minutes//60:02d}:{minutes%60:02d}:00'
    job = submit(['--partition='+PARTITIONS[model], *SITE_SBATCH_ARGS,
                  '--time='+wall, '--array='+','.join(map(str, shards))+'%4',
                  '--job-name=e276r_'+model+'_'+phase,
                  str(CODE/'worker.slurm'), model, stage, str(minutes*60-60)], state, 'generation')
    state['active'] = dict(job=job, shards=shards, stage=stage, minutes=minutes)
    save(state)
    watcher = submit(['--dependency=afterany:'+job, '--job-name=e276v_'+model,
                      str(CODE/'cpu.slurm'), model], state, 'controller')
    state['watcher'] = watcher; save(state)
    note(f"Generation {model}/{phase}: array {job}, shards {shards}, walltime {wall}, "
         f"CPU verifier/controller {watcher}; release {state['release_sha256']}. "
         'Submission is not completion; no efficacy result.')


def accounting(job):
    text = subprocess.check_output(['sacct', '-j', job, '-X', '-n', '-P',
                                    '--format=JobID,State,ExitCode,ElapsedRaw'], text=True)
    rows = {}
    for line in text.splitlines():
        p = line.split('|')
        if len(p) >= 4:
            rows[p[0]] = dict(state=p[1].split()[0], exit_code=p[2], elapsed=p[3])
    return rows


def stop(state, reason):
    state.update(status='STOPPED', reason=reason, updated=stamp()); save(state)
    note(f"Recovery {state['model']} STOPPED: {reason}. No downstream expansion or success claim.")
    print(json.dumps(state), flush=True)


def clear_terminal_lock(folder, task):
    lock = folder/'writer.lock'
    if not lock.exists():
        return
    owner = read_json(lock/'owner.json') if (lock/'owner.json').exists() else None
    if owner is not None and owner['job_id'] != task:
        raise RuntimeError('lock belongs to a different task')
    # Called only after accounting confirms this exact task is terminal.
    dest = folder/'quarantine'/('lock_'+str(time.time_ns()))
    dest.parent.mkdir(parents=True, exist_ok=True); os.replace(lock, dest)


def complete_global():
    for model in MODELS:
        p = BASE/'control'/(model+'.json')
        if not p.exists() or read_json(p).get('status') != 'COMPLETE':
            return
    binding = release_binding()
    records = [verify_shard(m, 'formal', i, binding) for m in MODELS for i in range(25)]
    result = dict(status='VERIFIED_GENERATION_COMPLETE', time=stamp(), n_images=sum(r['n_images'] for r in records),
                  release_sha256=binding, shards=records, scoring_and_noise_ceiling_analysis='NOT_RUN')
    if result['n_images'] != 10800:
        raise ValueError('global image count')
    write_json(BASE/'FINAL_VERIFICATION.json', result)
    note('Generation VERIFIED COMPLETE:75 shards/10800 images; hashes, decoding, dimensions and '
         'forward audits checked. Scoring and noise-ceiling analysis NOT RUN. '
         f"Evidence:{BASE/'FINAL_VERIFICATION.json'}. Lesson: verify config closure before GPU fan-out.")


def advance(model, start=False):
    preflight()
    control = BASE/'control'; control.mkdir(parents=True, exist_ok=True)
    with (control/(model+'.lock')).open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
        path = control/(model+'.json')
        if start:
            if path.exists():
                raise RuntimeError('already initialized; inspect journal rather than resubmit')
            state = dict(model=model, status='ACTIVE', phase='smoke', release_sha256=release_binding(),
                         created=stamp(), retries={}, progress={}, no_progress={}, jobs=[])
            save(state); schedule(state, [0], 45); return
        state = read_json(path)
        if state['status'] != 'ACTIVE':
            print(state['status']); return
        if state['release_sha256'] != release_binding():
            return stop(state, 'release changed')
        active = state['active']; job = active['job']; rows = accounting(job)
        for _ in range(10):
            if all(f'{job}_{i}' in rows and rows[f'{job}_{i}']['state'] not in ACTIVE for i in active['shards']):
                break
            time.sleep(2); rows = accounting(job)
        write_json(control/f'accounting_{job}.json', rows)
        retry = []
        for shard in active['shards']:
            task = f'{job}_{shard}'; row = rows.get(task)
            if row is None or row['state'] in ACTIVE:
                return stop(state, 'accounting missing or task still active:'+task)
            folder = folder_for(model, active['stage'], shard)
            if folder.exists():
                clear_terminal_lock(folder, task)
            try:
                result = verify_shard(model, active['stage'], shard)
            except (ValueError, FileNotFoundError, OSError) as exc:
                eligible = row['state'] in {'PREEMPTED', 'TIMEOUT'} or row['exit_code'] == '75:0'
                if not eligible:
                    return stop(state, f'{task} {row}: {exc}; diagnosis required')
                key = active['stage']+':'+str(shard); mp = folder/'manifest.json'
                count = len(read_json(mp)['images']) if mp.exists() else 0
                previous = state['progress'].get(key, 0)
                stalled = state['no_progress'].get(key, 0)+1 if count <= previous else 0
                state['progress'][key] = count; state['no_progress'][key] = stalled
                n = state['retries'].get(key, 0)
                if n >= 3 or stalled >= 2:
                    return stop(state, f'{task}: retry limit or two attempts without progress')
                state['retries'][key] = n+1; retry.append(shard)
            else:
                result.update(verified_at=stamp(), terminal_accounting=row)
                old = read_json(folder/'complete.json') if (folder/'complete.json').exists() else {}
                write_json(folder/'complete.json', {**old, **result})
                if row['state'] not in {'COMPLETED', 'PREEMPTED', 'TIMEOUT'} and row['exit_code'] != '75:0':
                    return stop(state, f'{task}: valid images but unexplained terminal state {row}')
        save(state)
        if retry:
            schedule(state, retry, active['minutes']); return
        if state['phase'] == 'smoke':
            state['phase'] = 'pilot'; save(state); schedule(state, [0], 230); return
        if state['phase'] == 'pilot':
            pilot_jobs = [j['id'] for j in state['jobs'] if j['role']=='generation'
                          and 'e276r_'+model+'_pilot' in ' '.join(j['command'])]
            elapsed = sum(int(v['elapsed']) for j in pilot_jobs for k,v in accounting(j).items() if '_' in k)
            minutes = min(230, max(20, 10*math.ceil((1.5*elapsed+600)/600)))
            state.update(phase='formal', calibrated_minutes=minutes, pilot_elapsed_seconds=elapsed)
            save(state); schedule(state, list(range(1,25)), minutes); return
        verified = [verify_shard(model, 'formal', i) for i in range(25)]
        state.update(status='COMPLETE', completed=stamp(), n_images=sum(r['n_images'] for r in verified))
        save(state)
        note(f"Exp276 {model}:25 shards/{state['n_images']} images verified. Scoring/analysis not run.")
        with (control/'finalize.lock').open('a') as final_lock:
            fcntl.flock(final_lock, fcntl.LOCK_EX)
            if not (BASE/'FINAL_VERIFICATION.json').exists():
                complete_global()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('model', choices=MODELS)
    ap.add_argument('--start', action='store_true'); args = ap.parse_args()
    advance(args.model, args.start)
