"""Approval-gated self-model lesson recording.

A lesson is a standing behavioral rule injected into every prompt ("avoid
repeating listed mistakes"), so it carries more authority than ordinary state.
This tool is WRITE_SAFE: calling it queues a pending action, and only an
explicit approval in the Actions panel (or chat) commits the lesson to
self_model.json — a misread turn can propose a false rule, but it cannot
self-commit one.
"""
from __future__ import annotations

from brain import self_model


def record_lesson(args: dict) -> dict:
    text = str((args or {}).get("text", "")).strip()
    if not text:
        return {"error": "lesson text is required"}
    source = str((args or {}).get("source", "") or "").strip() or None
    return self_model.add_lesson(text, source=source)
