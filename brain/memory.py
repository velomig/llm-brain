"""The Memory Unit: a structured object, not just a string.

A memory is a piece of text plus metadata (when, what kind, how important).
This module also owns the time-decay math for the "forgetting curve".
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


# Metadata schema version stamped on every memory. Bump this whenever the shape
# of the metadata dict changes (a new field, a changed default), then teach
# `repaired_metadata` how to upgrade older records and run `migrate.py`.
#   v1 — first versioned schema: category, importance, date, access_count,
#        optional source, plus situational last_accessed/superseded/due/status.
#   v2 — source_type and pinned become first-class metadata for provenance and
#        safe forgetting controls.
SCHEMA_VERSION = 2


def clamp_importance(value: int | float | str | None) -> int:
    try:
        parsed = int(value if value is not None else 3)
    except (TypeError, ValueError):
        parsed = 3
    return min(5, max(1, parsed))


def _as_int(value: int | float | str | None, default: int) -> int:
    try:
        return int(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def repaired_metadata(meta: dict | None) -> dict | None:
    """Upgrade a stored metadata dict to the current schema, or None if it's
    already current (so callers can skip a no-op write).

    Idempotent and behavior-preserving: it stamps `schema_version` and makes the
    defaults the read paths already assume (access_count, importance, category)
    explicit on disk. It deliberately does NOT invent a `date` (that would fake
    recency) or touch situational fields (source, superseded, due, status,
    last_accessed) — those are absent by design, not by age.
    """
    meta = meta or {}
    if _as_int(meta.get("schema_version"), 0) >= SCHEMA_VERSION:
        return None
    upgraded = dict(meta)
    upgraded["schema_version"] = SCHEMA_VERSION
    upgraded["access_count"] = _as_int(meta.get("access_count"), 0)
    upgraded["importance"] = clamp_importance(meta.get("importance", 3))
    upgraded.setdefault("category", "conversation")
    upgraded.setdefault("source_type", "legacy")
    upgraded.setdefault("pinned", False)
    return upgraded


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryUnit:
    text: str
    category: str = "conversation"
    importance: int = 3                       # 1 (trivial) .. 5 (critical)
    date: str = field(default_factory=now_iso)
    source: str | None = None                 # provenance, e.g. an ingested file path
    access_count: int = 0                      # times recalled to answer (reinforcement)
    extra: dict | None = None                  # extra metadata (e.g. reminder due date)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def metadata(self) -> dict:
        # ChromaDB metadata values must be str/int/float/bool (and non-null).
        meta = {
            "schema_version": SCHEMA_VERSION,
            "category": self.category,
            "importance": clamp_importance(self.importance),
            "date": self.date,
            "access_count": int(self.access_count),
            "source_type": "user_entered",
            "pinned": False,
        }
        if self.source:
            meta["source"] = self.source
        for key, val in (self.extra or {}).items():
            if isinstance(val, (str, int, float, bool)) and val is not None:
                meta[key] = val
        return meta


def age_days(date_iso: str | None) -> float | None:
    """Whole-day age of a memory, or None if the date is missing/unparseable."""
    if not date_iso:
        return None
    try:
        then = datetime.fromisoformat(date_iso)
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - then).total_seconds() / 86400.0)


def recency_score(date_iso: str | None, halflife_days: float, now: datetime | None = None) -> float:
    """Exponential decay in [0, 1]. Today ≈ 1.0, one half-life ago = 0.5, etc.

    This is the "Time Decay Factor" from the blueprint — older memories fade
    but never vanish, so a strong semantic match can still surface them.

    Pass one `now` snapshot when scoring a batch: with a fresh now() per call,
    memories sharing the SAME date get microsecond-different scores, which
    defeats the retriever's exact tie detection (tied values share a rank).
    """
    if not date_iso:
        return 0.0
    try:
        then = datetime.fromisoformat(date_iso)
    except ValueError:
        return 0.0
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    age_days = max(0.0, (now - then).total_seconds() / 86400.0)
    return math.pow(0.5, age_days / halflife_days)
