"""Persistent pending-action queue for risky tool calls."""
from __future__ import annotations

import json
import contextvars
from contextlib import contextmanager
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import config
from brain.memory import now_iso
from brain.utils import file_lock

PENDING_FILE = config.DB_DIR / ".neural_map_pending_actions.json"
_LOCK = threading.Lock()
_TASK_CONTEXT = contextvars.ContextVar("pending_task_context", default=None)


@contextmanager
def task_context(task_id: str, attempt_id: str):
    token = _TASK_CONTEXT.set({"task_id": task_id, "attempt_id": attempt_id})
    try:
        yield
    finally:
        _TASK_CONTEXT.reset(token)



def _state_lock():
    """Cross-process exclusive lock on the pending-actions file.

    The threading.Lock alone only serializes threads within one process; with
    multiple Uvicorn workers sharing the file, two approvers could both read
    ``pending``, both claim, and both execute a non-idempotent side effect.
    Every read-modify-write below must hold this flock too."""
    return file_lock(PENDING_FILE.with_name(".neural_map_pending_actions.lock"))


def _empty() -> dict:
    return {"actions": []}


def _json_safe(value):
    """Make tool output durable even when a library returns bytes/Path/etc."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _read() -> dict:
    try:
        data = json.loads(PENDING_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return _empty()
    return data if isinstance(data, dict) and isinstance(data.get("actions"), list) else _empty()


def _write(data: dict) -> None:
    PENDING_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PENDING_FILE.with_name(f"{PENDING_FILE.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(_json_safe(data), ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(PENDING_FILE)


def _parsed_at(value) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value or ""))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _recover_stale_unlocked(data: dict) -> bool:
    now = datetime.now(timezone.utc)
    changed = False
    for action in data.get("actions") or []:
        if action.get("status") != "approving":
            continue
        started = _parsed_at(action.get("approval_started_at") or action.get("created_at"))
        if started is None or (now - started).total_seconds() <= config.PENDING_APPROVAL_LEASE_SECONDS:
            continue
        action["status"] = "failed"
        action["resolved_at"] = now_iso()
        action["result"] = {
            "error": "approval execution was interrupted or exceeded its lease",
            "recovered_stale_claim": True,
        }
        changed = True
    return changed


def create(
    tool: str,
    args: dict,
    policy: str,
    reason: str | None = None,
    preview: dict | None = None,
    snapshot: dict | None = None,
    taint: list[str] | None = None,
) -> dict:
    with _LOCK, _state_lock():
        data = _read()
        action = {
            "id": f"act_{uuid.uuid4().hex[:12]}",
            "tool": tool,
            "args": args or {},
            "policy": policy,
            "reason": reason or f"{policy} tool requires approval",
            "preview": preview or {},
            "snapshot": snapshot or {},
            # Untrusted content that was in context when the model proposed this
            # action (retrieved memories, web results, file reads). A tainted
            # action cannot be approved via the blind chat shortcut — only from
            # the Actions panel, where the preview is visible.
            "taint_sources": list(taint or []),
            "status": "pending",
            "created_at": now_iso(),
        }
        action.update(_TASK_CONTEXT.get() or {})
        data["actions"].append(action)
        _write(data)
        return action


def list_actions(include_resolved: bool = False) -> list[dict]:
    with _LOCK, _state_lock():
        data = _read()
        if _recover_stale_unlocked(data):
            _write(data)
        actions = list(data.get("actions") or [])
    if not include_resolved:
        actions = [a for a in actions if a.get("status") == "pending"]
    actions.sort(key=lambda a: a.get("created_at") or "", reverse=True)
    return actions


def get(action_id: str) -> dict | None:
    with _LOCK, _state_lock():
        data = _read()
        if _recover_stale_unlocked(data):
            _write(data)
        for action in data.get("actions") or []:
            if action.get("id") == action_id:
                return dict(action)
    return None


def claim(action_id: str, patch: dict | None = None) -> dict | None:
    """Atomically transition a pending action to ``approving`` under the lock.

    Only the caller that observes the action still ``pending`` wins the claim and
    gets the action back; a racing approver sees a non-pending status and gets
    ``None``. This is the compare-and-set that makes approval single-shot: the
    status check and the state change happen in one critical section, so two
    concurrent approvals of the same action can no longer both execute the
    (possibly external, non-idempotent) side effect. The flock extends the
    guarantee across processes: without it, two Uvicorn workers could each
    read ``pending`` and both win."""
    with _LOCK, _state_lock():
        data = _read()
        for action in data.get("actions") or []:
            if action.get("id") == action_id:
                if action.get("status") != "pending":
                    return None
                action["status"] = "approving"
                if patch:
                    action.update(patch)
                _write(data)
                return dict(action)
    return None


def update_action(action_id: str, patch: dict) -> dict | None:
    with _LOCK, _state_lock():
        data = _read()
        found = None
        for action in data.get("actions") or []:
            if action.get("id") == action_id:
                action.update(patch)
                found = dict(action)
                break
        if found is not None:
            _write(data)
        return found


def resolve(
    action_id: str,
    status: str,
    result: dict | None = None,
    expected_status: str | None = None,
) -> dict | None:
    """Set a terminal status; with ``expected_status``, only from that state.

    The guard is checked in the same critical section as the write, so e.g.
    reject-if-still-pending cannot clobber an action a racing approver already
    executed (which would strand a completed side effect as ``rejected`` and
    make it un-undoable). Returns None when the action is missing or the
    guard fails — callers that need to tell those apart re-``get`` after."""
    with _LOCK, _state_lock():
        data = _read()
        found = None
        for action in data.get("actions") or []:
            if action.get("id") == action_id:
                if expected_status is not None and action.get("status") != expected_status:
                    return None
                action["status"] = status
                action["resolved_at"] = now_iso()
                if result is not None:
                    action["result"] = result
                found = dict(action)
                break
        if found is not None:
            _write(data)
        return found


def record_undo(action_id: str, result: dict) -> dict | None:
    return update_action(action_id, {"undo_result": result, "undone_at": now_iso()})
