"""Small immutable bindings and atomic metadata persistence for Exp267."""
import hashlib
import json
import os
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('DG_ROOT', str(CODE.parents[1])))
CONFIG = json.loads((CODE / 'config.json').read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def stable_seed(label):
    return int.from_bytes(hashlib.sha256(f"Exp267:{CONFIG['seed']}:{label}".encode()).digest()[:8], 'little') % (2**31)


def binding():
    return {p.name:digest(p) for p in sorted(CODE.iterdir()) if p.is_file()}
