"""Facade for Neural Map tool calling.

The implementations live in ``brain.toolkit`` modules. This file preserves the
existing ``from brain import tools`` import surface while keeping dispatch,
schemas, policy, and pending-action approval in one small place.
"""
from __future__ import annotations

from brain.toolkit import pending
from brain.toolkit.basic import calculate as _calculate_impl
from brain.toolkit.registry import (
    DESTRUCTIVE,
    EXTERNAL_SEND,
    EXTERNAL_WRITE,
    READ_ONLY,
    WRITE_SAFE,
    active_tool_specs,
    approve_pending,
    dispatch,
    execute,
    list_audit,
    list_permissions,
    policy_for,
    reject_pending,
    requires_approval,
    set_permission,
    undo_pending,
)

TOOL_SPECS = active_tool_specs()


def _calculate(expression: str) -> dict:
    """Backward-compatible helper for older tests/callers."""
    return _calculate_impl({"expression": expression})


def list_pending(include_resolved: bool = False) -> list[dict]:
    return pending.list_actions(include_resolved=include_resolved)


def split_used(used: list[dict]) -> tuple[dict | None, dict | None]:
    """Map executed tool calls back into the live_data / web_data shape the UI shows."""
    live_data = web_data = None
    for call in used:
        if call.get("tool") == "get_stock_quotes":
            live_data = call.get("result")
        elif call.get("tool") == "web_search":
            web_data = call.get("result")
    return live_data, web_data
