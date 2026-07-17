"""Persistent goal stack, world model, and belief ledger.

This is the "current state" layer beside long-term memory. Memories preserve
history; cognitive state records what the system should currently treat as
active goals, current facts, and sourced beliefs.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone

import config
from brain.utils import file_lock
from brain.memory import clamp_importance, now_iso

SCHEMA_VERSION = 1
GOAL_STATUSES = {"active", "blocked", "done", "dropped"}
BELIEF_STATUSES = {"active", "stale", "rejected"}
WORLD_STATUSES = {"current", "stale", "archived"}

_STATE_FILE = config.DB_DIR / ".neural_map_cognitive_state.json"
_STATE_THREAD_LOCK = threading.Lock()


def _state_lock():
    """Cross-process exclusive lock on this module's state file."""
    return file_lock(_STATE_FILE.with_name(".neural_map_cognitive_state.lock"))


def _empty_state() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": None,
        "goals": [],
        "world": [],
        "beliefs": [],
    }


def _clamp_confidence(value) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.5
    return round(min(1.0, max(0.0, parsed)), 4)


def _clean_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        return []
    out: list[str] = []
    for item in values:
        text = str(item).strip()
        if text:
            out.append(text[:500])
    return out[:20]


def _dict_rows(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _expired(value: str | None, now: datetime | None = None) -> bool:
    expires = _parse_dt(value)
    return bool(expires and expires < (now or datetime.now(timezone.utc)))


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _repair_goal(raw: dict) -> dict | None:
    if not isinstance(raw, dict):
        return None
    text = str(raw.get("text") or raw.get("goal") or "").strip()
    if not text:
        return None
    status = str(raw.get("status") or "active").strip().lower()
    if status not in GOAL_STATUSES:
        status = "active"
    created_at = raw.get("created_at") or now_iso()
    updated_at = raw.get("updated_at") or created_at
    return {
        "id": str(raw.get("id") or _new_id("goal"))[:80],
        "text": text[:1000],
        "status": status,
        "priority": clamp_importance(raw.get("priority", 3)),
        "due": raw.get("due"),
        "evidence": _clean_list(raw.get("evidence")),
        "next_action": str(raw.get("next_action") or "").strip()[:1000],
        "blocked_by": _clean_list(raw.get("blocked_by")),
        "created_at": created_at,
        "updated_at": updated_at,
    }


def _repair_world(raw: dict) -> dict | None:
    if not isinstance(raw, dict):
        return None
    key = str(raw.get("key") or "").strip()
    value = str(raw.get("value") or "").strip()
    if not key or not value:
        return None
    status = str(raw.get("status") or "current").strip().lower()
    if status not in WORLD_STATUSES:
        status = "current"
    created_at = raw.get("created_at") or now_iso()
    updated_at = raw.get("updated_at") or created_at
    return {
        "id": str(raw.get("id") or _new_id("world"))[:80],
        "key": key[:160],
        "value": value[:2000],
        "status": status,
        "confidence": _clamp_confidence(raw.get("confidence", 0.7)),
        "source": str(raw.get("source") or "").strip()[:500],
        "updated_at": updated_at,
        "created_at": created_at,
        "expires": raw.get("expires"),
    }


def _repair_belief(raw: dict) -> dict | None:
    if not isinstance(raw, dict):
        return None
    claim = str(raw.get("claim") or "").strip()
    if not claim:
        return None
    status = str(raw.get("status") or "active").strip().lower()
    if status not in BELIEF_STATUSES:
        status = "active"
    created_at = raw.get("created_at") or now_iso()
    last_verified = raw.get("last_verified") or raw.get("updated_at") or created_at
    return {
        "id": str(raw.get("id") or _new_id("belief"))[:80],
        "claim": claim[:2000],
        "status": status,
        "confidence": _clamp_confidence(raw.get("confidence", 0.5)),
        "source": str(raw.get("source") or "").strip()[:500],
        "last_verified": last_verified,
        "created_at": created_at,
        "expires": raw.get("expires"),
    }


def _repair_state(data) -> dict:
    state = _empty_state()
    if not isinstance(data, dict):
        return state
    state["schema_version"] = SCHEMA_VERSION
    state["updated_at"] = data.get("updated_at")
    state["goals"] = [g for g in (_repair_goal(x) for x in _dict_rows(data.get("goals"))) if g]
    state["world"] = [w for w in (_repair_world(x) for x in _dict_rows(data.get("world"))) if w]
    state["beliefs"] = [b for b in (_repair_belief(x) for x in _dict_rows(data.get("beliefs"))) if b]
    return state


def _read_state_unlocked() -> dict:
    try:
        data = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return _empty_state()
    return _repair_state(data)


def _write_state_unlocked(state: dict) -> None:
    state = _repair_state({**state, "updated_at": now_iso()})
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STATE_FILE.with_name(f"{_STATE_FILE.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(state, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(tmp, _STATE_FILE)


def state(include_inactive: bool = True) -> dict:
    with _STATE_THREAD_LOCK:
        with _state_lock():
            current = _read_state_unlocked()
    if include_inactive:
        return annotate(current)
    filtered = {
        **current,
        "goals": [g for g in current["goals"] if g["status"] in {"active", "blocked"}],
        "world": [w for w in current["world"] if w["status"] == "current"],
        "beliefs": [b for b in current["beliefs"] if b["status"] == "active"],
    }
    return annotate(filtered)


def annotate(current: dict) -> dict:
    now = datetime.now(timezone.utc)
    out = _repair_state(current)
    for row in out["world"]:
        row["expired"] = _expired(row.get("expires"), now)
    for row in out["beliefs"]:
        row["expired"] = _expired(row.get("expires"), now)
    return out


def replace_state(new_state: dict) -> dict:
    with _STATE_THREAD_LOCK:
        with _state_lock():
            fixed = _repair_state(new_state)
            _write_state_unlocked(fixed)
            return _read_state_unlocked()


def clear() -> dict:
    return replace_state(_empty_state())


def upsert_goal(
    text: str,
    goal_id: str | None = None,
    status: str = "active",
    priority: int = 3,
    due: str | None = None,
    evidence: list[str] | None = None,
    next_action: str | None = None,
    blocked_by: list[str] | None = None,
) -> dict:
    with _STATE_THREAD_LOCK:
        with _state_lock():
            current = _read_state_unlocked()
            existing = next((g for g in current["goals"] if goal_id and g["id"] == goal_id), None)
            record = _repair_goal(
                {
                    **(existing or {}),
                    "id": goal_id or (existing or {}).get("id") or _new_id("goal"),
                    "text": text,
                    "status": status,
                    "priority": priority,
                    "due": due,
                    "evidence": evidence,
                    "next_action": next_action,
                    "blocked_by": blocked_by,
                    "created_at": (existing or {}).get("created_at") or now_iso(),
                    "updated_at": now_iso(),
                }
            )
            if record is None:
                raise ValueError("goal text is required")
            current["goals"] = [g for g in current["goals"] if g["id"] != record["id"]]
            current["goals"].append(record)
            _write_state_unlocked(current)
            return record


def set_goal_status(goal_id: str, status: str, blocked_by: list[str] | None = None) -> dict | None:
    status = status.strip().lower()
    if status not in GOAL_STATUSES:
        raise ValueError(f"invalid goal status: {status}")
    with _STATE_THREAD_LOCK:
        with _state_lock():
            current = _read_state_unlocked()
            for goal in current["goals"]:
                if goal["id"] == goal_id:
                    goal["status"] = status
                    if blocked_by is not None:
                        goal["blocked_by"] = _clean_list(blocked_by)
                    goal["updated_at"] = now_iso()
                    _write_state_unlocked(current)
                    return goal
    return None


def upsert_world_item(
    key: str,
    value: str,
    item_id: str | None = None,
    confidence: float = 0.7,
    source: str | None = None,
    expires: str | None = None,
    status: str = "current",
) -> dict:
    with _STATE_THREAD_LOCK:
        with _state_lock():
            current = _read_state_unlocked()
            existing = next((w for w in current["world"] if (item_id and w["id"] == item_id) or w["key"] == key), None)
            record = _repair_world(
                {
                    **(existing or {}),
                    "id": item_id or (existing or {}).get("id") or _new_id("world"),
                    "key": key,
                    "value": value,
                    "confidence": confidence,
                    "source": source,
                    "expires": expires,
                    "status": status,
                    "created_at": (existing or {}).get("created_at") or now_iso(),
                    "updated_at": now_iso(),
                }
            )
            if record is None:
                raise ValueError("world item key and value are required")
            current["world"] = [w for w in current["world"] if w["id"] != record["id"] and w["key"] != record["key"]]
            current["world"].append(record)
            _write_state_unlocked(current)
            return record


def upsert_belief(
    claim: str,
    belief_id: str | None = None,
    confidence: float = 0.5,
    source: str | None = None,
    last_verified: str | None = None,
    expires: str | None = None,
    status: str = "active",
) -> dict:
    with _STATE_THREAD_LOCK:
        with _state_lock():
            current = _read_state_unlocked()
            existing = next((b for b in current["beliefs"] if belief_id and b["id"] == belief_id), None)
            record = _repair_belief(
                {
                    **(existing or {}),
                    "id": belief_id or (existing or {}).get("id") or _new_id("belief"),
                    "claim": claim,
                    "confidence": confidence,
                    "source": source,
                    "last_verified": last_verified or now_iso(),
                    "expires": expires,
                    "status": status,
                    "created_at": (existing or {}).get("created_at") or now_iso(),
                }
            )
            if record is None:
                raise ValueError("belief claim is required")
            current["beliefs"] = [b for b in current["beliefs"] if b["id"] != record["id"]]
            current["beliefs"].append(record)
            _write_state_unlocked(current)
            return record


def prompt_snapshot() -> dict:
    if not config.COGNITIVE_STATE_ENABLED:
        return {"enabled": False}
    current = state(include_inactive=False)
    goals = sorted(
        current["goals"],
        key=lambda g: (g["status"] != "active", -int(g.get("priority") or 3), g.get("due") or "9999"),
    )
    world = sorted(
        [w for w in current["world"] if not w.get("expired")],
        key=lambda w: (-(w.get("confidence") or 0.0), w.get("key") or ""),
    )
    beliefs = sorted(
        [b for b in current["beliefs"] if not b.get("expired")],
        key=lambda b: (-(b.get("confidence") or 0.0), b.get("claim") or ""),
    )
    return {
        "enabled": True,
        "active_goals": goals[: config.COGNITIVE_STATE_MAX_GOALS],
        "world_model": world[: config.COGNITIVE_STATE_MAX_WORLD],
        "belief_ledger": beliefs[: config.COGNITIVE_STATE_MAX_BELIEFS],
    }
