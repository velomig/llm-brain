"""Small shared helpers with no brain-module dependencies."""
from __future__ import annotations

import fcntl
import hashlib
import json
import re
import threading
from contextlib import contextmanager
from pathlib import Path


_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.Lock] = {}


def clean_text(value, limit: int = 240) -> str:
    """Whitespace-normalized text capped at `limit`, cut at a word boundary.

    A hard slice can end mid-word and hand the model a fragment that reads like
    a complete statement. When truncation happens, cut back to the last word
    boundary (when one is reasonably close) and append an ellipsis so the
    reader — human or model — can SEE that the text was cut.
    """
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if limit <= 0 or len(text) <= limit:
        return text
    cut = text[: max(1, limit - 1)]
    space = cut.rfind(" ")
    if space >= int(limit * 0.6):
        cut = cut[:space]
    return cut.rstrip() + "…"


@contextmanager
def file_lock(lock_file: Path):
    """Exclusive cross-process lock via fcntl.flock on a dedicated lock file.

    The one shared implementation behind every state module's `_state_lock()`:
    flock serializes the web server against background jobs (reflect, sleep,
    will) across processes. Callers pair it with a module-level threading.Lock
    so threads within one process queue on that instead of churning file
    descriptors here. Blocks until the lock is granted; always unlocks and
    closes, even when the guarded block raises.
    """
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_file.open("a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


@contextmanager
def path_lock(target: Path, lock_dir: Path, prefix: str = "path"):
    """Serialize a target-file transaction across threads and processes.

    Atomic replacement prevents torn files but does not prevent lost updates:
    two writers can read the same original, independently replace it, and both
    report success while the later replacement discards the earlier change.
    A per-target thread lock plus a hashed flock file protects the whole
    read/validate/write transaction without placing lock artifacts beside the
    user-authored target file.
    """
    key = str(Path(target).resolve(strict=False))
    with _PATH_LOCKS_GUARD:
        thread_lock = _PATH_LOCKS.setdefault(key, threading.Lock())
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    lock_file = Path(lock_dir) / f".neural_map_{prefix}_{digest}.lock"
    with thread_lock, file_lock(lock_file):
        yield


def safe_json(raw: str) -> dict:
    """Parse a model's JSON reply, tolerating code fences and surrounding prose.

    Returns {} for anything that isn't a JSON object — the callers all treat a
    failed parse as "the model produced nothing usable this pass" and fall back.
    """
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
