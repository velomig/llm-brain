"""Small contracts for optional cognitive layers.

The chat pipeline treats layers like metacognition as advisors. Their return
values must be boring and explicit so orchestration can degrade gracefully when
an advisor skips, exhausts its budget, or fails.
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict

LayerStatus = Literal["ok", "skipped", "exhausted", "failed"]


class LayerResult(TypedDict, total=False):
    status: LayerStatus
    reason: str
    id: str | None
    stored: int
    bias: str


def result(status: LayerStatus, reason: str | None = None, **fields: Any) -> LayerResult:
    out: LayerResult = {"status": status}
    if reason:
        out["reason"] = reason
    out.update(fields)
    return out


def normalize(raw: Any) -> LayerResult:
    """Normalize current and legacy layer responses into the canonical shape."""
    if isinstance(raw, str):
        if raw.strip().upper() == "MAX_DEPTH_REACHED":
            return result("exhausted", "max_reflection_depth")
        return result("failed", "unexpected_string_result")
    if not isinstance(raw, dict):
        return result("failed", "unexpected_result_type")

    status = str(raw.get("status") or "").strip().lower()
    reason = str(raw.get("reason") or raw.get("skipped") or "").strip()
    legacy = str(raw.get("skipped") or raw.get("reason") or "").strip()

    if status == "max_depth_reached" or str(raw.get("status") or "").strip().upper() == "MAX_DEPTH_REACHED":
        status = "exhausted"
        reason = reason or "max_reflection_depth"
    elif legacy == "max_reflection_depth":
        status = "exhausted"
        reason = reason or legacy
    elif status not in {"ok", "skipped", "exhausted", "failed"}:
        status = "ok" if raw.get("stored") else "skipped"
        reason = reason or ("no_action" if status == "skipped" else "")

    out = dict(raw)
    out["status"] = status
    if reason:
        out["reason"] = reason
    return out  # type: ignore[return-value]


def is_exhausted(raw: Any) -> bool:
    return normalize(raw).get("status") == "exhausted"
