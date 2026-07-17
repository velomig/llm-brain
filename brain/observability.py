"""Bounded cognitive traces for seeing through layered orchestration.

Fallback logs show failures. Traces show the ordinary path too: which layers ran,
which skipped, which safety guards fired, and what final route the turn took.
They intentionally avoid storing full user text or memory text.
"""
from __future__ import annotations

from collections import deque
import contextvars
from datetime import datetime, timezone
import hashlib
import uuid

_RECENT: deque[dict] = deque(maxlen=100)
_ACTIVE_TRACE_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "neural_map_active_trace_id", default=None
)
_ACTIVE_TRACE_KIND: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "neural_map_active_trace_kind", default=None
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(text: str | None) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _latency_delta_ms(started_at: str | None, finished_at: str | None) -> float | None:
    if not started_at or not finished_at:
        return None
    try:
        started = datetime.fromisoformat(started_at)
        finished = datetime.fromisoformat(finished_at)
        delta = finished - started
    except (TypeError, ValueError):
        return None
    return max(0.0, round(delta.total_seconds() * 1000, 3))


def _metrics(trace: dict) -> dict:
    event_count = len(trace.get("events") or [])
    return {
        "latency_delta_ms": _latency_delta_ms(trace.get("started_at"), trace.get("finished_at")),
        "event_count": event_count,
        "complexity_score": event_count,
    }


def active_trace_fields() -> dict:
    trace_id = _ACTIVE_TRACE_ID.get()
    if not trace_id:
        return {}
    fields = {"trace_id": trace_id}
    trace_kind = _ACTIVE_TRACE_KIND.get()
    if trace_kind:
        fields["trace_kind"] = trace_kind
    return fields


def start(kind: str, *, model: str | None = None, user_text: str | None = None, **fields: object) -> dict:
    trace = {
        "id": uuid.uuid4().hex[:12],
        "kind": kind,
        "started_at": _now(),
        "finished_at": None,
        "status": "running",
        "model": model,
        "input": {
            "chars": len(user_text or ""),
            "sha256": _fingerprint(user_text),
        },
        "fields": {k: v for k, v in fields.items() if v is not None},
        "events": [],
    }
    _ACTIVE_TRACE_ID.set(trace["id"])
    _ACTIVE_TRACE_KIND.set(trace["kind"])
    event(trace, "trace", "started")
    return trace


def event(trace: dict | None, layer: str, status: str, **fields: object) -> None:
    if trace is None:
        return
    trace.setdefault("events", []).append(
        {
            "time": _now(),
            "layer": layer,
            "status": status,
            "fields": {k: v for k, v in fields.items() if v is not None},
        }
    )


def finish(trace: dict | None, status: str = "ok", **fields: object) -> dict | None:
    if trace is None:
        return None
    trace["finished_at"] = _now()
    trace["status"] = status
    if fields:
        trace.setdefault("fields", {}).update({k: v for k, v in fields.items() if v is not None})
    event(trace, "trace", "finished", final_status=status)
    trace["metrics"] = _metrics(trace)
    _RECENT.append(trace)
    if _ACTIVE_TRACE_ID.get() == trace.get("id"):
        _ACTIVE_TRACE_ID.set(None)
        _ACTIVE_TRACE_KIND.set(None)
    return trace


def recent(limit: int = 50) -> list[dict]:
    return list(_RECENT)[-limit:][::-1]
