from pathlib import Path
import hashlib
import json
import math
import os

ROOT = Path(__file__).resolve().parents[1]


def workspace_path(value, *, exists=False):
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("path_outside_workspace")
    if exists and not path.exists():
        raise FileNotFoundError(str(path))
    return path


def sha256(path):
    digest = hashlib.sha256()
    with workspace_path(path, exists=True).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = workspace_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def finite(value, name):
    if not math.isfinite(float(value)):
        raise ValueError("non_finite_" + name)
    return float(value)
