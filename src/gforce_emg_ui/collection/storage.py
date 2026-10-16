"""Collection metadata I/O: atomic JSON files written on one background thread,
safe folder names, and the software revision.

All writes go through IoWorker (one thread, jobs run in submission order) so
the GUI and SDK callback threads never wait on the disk. Each write goes to a
temporary file in the same folder, is fsync'ed, then atomically replaces the
target: a reader sees the old or the new file, never a partial one.
"""

import json
import math
import os
import re
import subprocess
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")       # participant / session IDs
KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")                # object / posture IDs
RESERVED_IDS = {"simulated"}      # folder used for simulated attempts


def valid_id(value: str) -> bool:
    return bool(ID_RE.match(value or "")) and value.lower() not in RESERVED_IDS


def valid_key(value: str) -> bool:
    return bool(KEY_RE.match(value or ""))


def jsonable(obj: Any) -> Any:
    """Plain JSON types; NaN/inf become null (strict JSON, 'unknown')."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        items = sorted(obj) if isinstance(obj, set) else obj
        return [jsonable(v) for v in items]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def dumps(obj: Any) -> str:
    return json.dumps(jsonable(obj), indent=2, allow_nan=False) + "\n"


def atomic_write_text(path: Path, text: str):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    try:                                  # persist the rename itself (POSIX only)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


class IoWorker:
    """One background thread for collection metadata; jobs run in order."""

    def __init__(self):
        self._ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="collection-io")

    def submit(self, fn, *args, **kwargs) -> Future:
        return self._ex.submit(fn, *args, **kwargs)

    def write_json(self, path: Path, obj: Any) -> Future:
        """Serialize now (snapshot of obj), write atomically on the worker."""
        try:
            text = dumps(obj)
        except Exception as e:            # report like a failed write
            f: Future = Future()
            f.set_exception(e)
            return f
        return self._ex.submit(atomic_write_text, Path(path), text)

    def shutdown(self, wait: bool = True):
        self._ex.shutdown(wait=wait)


def s_tag(s: float) -> str:
    """Command value for folder names: 0.0 -> s0p0, 0.5 -> s0p5, 1.0 -> s1p0."""
    return f"s{s:.1f}".replace(".", "p").replace("-", "m")


def attempt_name(object_id: str, target_label: str, s: float, trial_tag: str,
                 attempt_number: int, attempt_uid: str, role: str = "development") -> str:
    """Name passed to Recorder.start (which prefixes the timestamp), e.g.
    obj03_high_s1p0_t042_a01_3fa2b1c4. Non-development roles are prefixed."""
    parts = [] if role == "development" else [role]
    parts += [object_id, target_label, s_tag(s), trial_tag, f"a{attempt_number:02d}", attempt_uid]
    name = "_".join(parts)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"unsafe attempt folder name {name!r}")
    return name


def ensure_within(root: Path, path: Path) -> Path:
    """Resolve path and require it to be inside root (no traversal / symlink escape)."""
    root_r, path_r = Path(root).resolve(), Path(path).resolve()
    path_r.relative_to(root_r)            # raises ValueError if outside
    return path_r


_GIT_CACHE: Optional[dict] = None


def git_info(path: Optional[Path] = None) -> dict:
    """Revision and dirty flag of the checkout this package runs from (if any).

    Only `git rev-parse HEAD` and `git status --porcelain` are run; nothing else
    (no remotes, user names or environment) is collected. dirty = tracked files
    differ from HEAD (untracked files ignored). Unknown -> None.
    """
    global _GIT_CACHE
    if _GIT_CACHE is not None and path is None:
        return dict(_GIT_CACHE)
    where = Path(path) if path is not None else Path(__file__).resolve().parent
    out = {"revision": None, "dirty": None}
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=where, capture_output=True,
                             text=True, timeout=5, env=env)
        if rev.returncode == 0:
            out["revision"] = rev.stdout.strip()
            st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                cwd=where, capture_output=True, text=True, timeout=10, env=env)
            if st.returncode == 0:
                out["dirty"] = bool(st.stdout.strip())
    except Exception:
        pass
    if path is None:
        _GIT_CACHE = dict(out)
    return out
