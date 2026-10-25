"""Small shared helpers with no brain-module dependencies."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path


_FILE_LOCKS_GUARD = threading.Lock()
_FILE_LOCKS: dict[str, threading.RLock] = {}
_HELD_FILE_LOCKS = threading.local()


def _reset_child_locks() -> None:
    # A forked child owns none of the parent's critical sections.
    global _FILE_LOCKS_GUARD, _FILE_LOCKS, _HELD_FILE_LOCKS
    _FILE_LOCKS_GUARD = threading.Lock()
    _FILE_LOCKS = {}
    _HELD_FILE_LOCKS = threading.local()


os.register_at_fork(after_in_child=_reset_child_locks)


def replace_private(path: Path, text: str | bytes) -> None:
    """Atomically replace `path` with `text`, keeping the file owner-only.

    Path.write_text on the temp file applies the process umask (typically
    producing 0644), so every atomic update of a private file silently widens
    its permissions. These are trusted prompt/calibration files with personal
    content: the replacement is always created 0600, which also heals a file
    that a previous update already widened.
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(text.encode("utf-8") if isinstance(text, str) else text)
        os.replace(tmp, path)
    except OSError:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


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
    """Reentrant thread and process lock, held through nested transactions."""
    key = str(Path(lock_file).resolve(strict=False))
    with _FILE_LOCKS_GUARD:
        thread_lock = _FILE_LOCKS.setdefault(key, threading.RLock())
    with thread_lock:
        held = getattr(_HELD_FILE_LOCKS, "keys", None)
        if held is None:
            held = _HELD_FILE_LOCKS.keys = set()
        if key in held:
            yield
            return
        with _flock(Path(key)):
            held.add(key)
            try:
                yield
            finally:
                held.remove(key)


@contextmanager
def _flock(lock_file: Path):
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
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    lock_file = Path(lock_dir) / f".neural_map_{prefix}_{digest}.lock"
    with file_lock(lock_file):
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
