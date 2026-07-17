"""Small shared helpers with no brain-module dependencies."""
from __future__ import annotations

import fcntl
import json
import re
from contextlib import contextmanager
from pathlib import Path


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
