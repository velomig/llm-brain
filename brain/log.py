"""Structured logging for swallowed fallback events.

Much of the chat pipeline deliberately catches exceptions so a degraded
subsystem — embeddings down, a flaky tool call, a bad consolidation pass — never
breaks the conversation. That is humane, but it makes debugging hard because the
failure vanishes silently. Every such fallback calls `fallback()` here, turning
the swallowed error into ONE structured log line on the "neural_map" logger
instead of disappearing.

Each line looks like:
    2026-06-28 13:40:01 WARNING neural_map event=graph_failed phase=recall error="OperationalError: ..."

Configure with env vars:
  NEURAL_MAP_LOG_LEVEL  (default WARNING) — set DEBUG to also attach tracebacks
  NEURAL_MAP_LOG_FILE   (optional)        — write to a file instead of stderr
"""
from __future__ import annotations

import logging
import os
import sys
from collections import deque
from datetime import datetime, timezone

logger = logging.getLogger("neural_map")
_RECENT: deque[dict] = deque(maxlen=200)


def _configure() -> None:
    # Idempotent: importing this module from many places must not stack handlers.
    if getattr(logger, "_neural_map_configured", False):
        return
    level_name = os.getenv("NEURAL_MAP_LOG_LEVEL", "WARNING").upper()
    logger.setLevel(getattr(logging, level_name, logging.WARNING))
    log_file = os.getenv("NEURAL_MAP_LOG_FILE")
    handler: logging.Handler = (
        logging.FileHandler(log_file) if log_file else logging.StreamHandler(sys.stderr)
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    # Don't also bubble to the root logger (uvicorn configures its own), which
    # would double-print every line.
    logger.propagate = False
    logger._neural_map_configured = True  # type: ignore[attr-defined]


_configure()


def _render(value: object) -> str:
    text = str(value)
    if text == "" or any(ch.isspace() for ch in text):
        return '"' + text.replace('"', "'") + '"'
    return text


def fallback(
    event: str,
    exc: BaseException | None = None,
    *,
    level: int = logging.WARNING,
    **fields: object,
) -> None:
    """Record one swallowed failure as a structured log line.

    event   stable identifier, e.g. "tool_calling_failed", "embedding_unavailable".
    exc     the exception being swallowed; its type and message are included.
    level   log level (default WARNING; use DEBUG for routine/expected fallbacks).
    fields  extra key=value context, e.g. model="gemma4:latest", phase="recall".

    Full tracebacks are attached only when the logger is at DEBUG, so ordinary
    runs stay at one line per event while deep debugging still gets the stack.
    """
    try:
        from brain import observability

        for key, val in observability.active_trace_fields().items():
            fields.setdefault(key, val)
    except Exception:
        pass

    parts = [f"event={event}"]
    for key, val in fields.items():
        if val is not None:
            parts.append(f"{key}={_render(val)}")
    if exc is not None:
        parts.append(f"error={_render(f'{type(exc).__name__}: {exc}')}")
    _RECENT.append(
        {
            "time": datetime.now(timezone.utc).isoformat(),
            "level": logging.getLevelName(level),
            "event": event,
            "fields": {k: v for k, v in fields.items() if v is not None},
            "error": f"{type(exc).__name__}: {exc}" if exc is not None else None,
        }
    )
    want_trace = exc is not None and logger.isEnabledFor(logging.DEBUG)
    logger.log(level, " ".join(parts), exc_info=exc if want_trace else None)


def recent(limit: int = 100) -> list[dict]:
    """Recent fallback events captured in this process, newest first."""
    return list(_RECENT)[-limit:][::-1]
