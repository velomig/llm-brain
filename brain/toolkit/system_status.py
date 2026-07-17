"""Narrow read-only status views for protected Neural Map runtime state."""
from __future__ import annotations

from datetime import datetime

import config
from brain import will


def _local_timestamp(value: object) -> str | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed.astimezone().isoformat()


def will_status(args: dict) -> dict:
    """Return an authoritative receipt without exposing memory_store files."""
    state = will.state()
    return {
        "source": "will_runtime_state",
        "enabled": config.WILL_ENABLED,
        "last_run": state.get("last_run"),
        "last_run_local": _local_timestamp(state.get("last_run")),
        "last_result": state.get("last_result"),
        "last_outreach_at": state.get("last_outreach_at"),
        "outreach_today": int(state.get("outreach_today") or 0),
        "active_cooldowns": len(state.get("cooldowns") or {}),
        "guidance": "This runtime state, not audit_log.md or git history, records scheduled Will execution.",
    }
