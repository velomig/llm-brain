"""Optional initiative layer: soft follow-ups, open loops, and daily nudges.

INITIATIVE LAYER / ROLLBACK NOTE:
This module is intentionally separate from reminders and retrieval. To regress to
the previous reactive mode, set NEURAL_MAP_INITIATIVE=0 or use mode "quiet"; the
stored initiative records remain ordinary category="initiative" memories but are
not injected into prompts or shown by /api/initiative.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone

import config
from brain.utils import file_lock
from brain import arbiter, log, store
from brain.memory import MemoryUnit, clamp_importance, now_iso

MODES = ("quiet", "helpful", "active", "coach")
_MODE_MIN_PRIORITY = {"quiet": 99, "helpful": 4, "active": 3, "coach": 2}
# "helpful" stays deliberately modest (at most 2 at a time) but still respects a
# LOWER configured global cap, so MAX_ACTIVE_INITIATIVES=1 (or 0) applies everywhere.
_MODE_CAP = {
    "quiet": 0,
    "helpful": min(2, config.MAX_ACTIVE_INITIATIVES),
    "active": config.MAX_ACTIVE_INITIATIVES,
    "coach": config.MAX_ACTIVE_INITIATIVES,
}
_MODE_DAILY_CAP = {
    "quiet": 0,
    "helpful": config.INITIATIVE_MAX_CONTACTS_HELPFUL,
    "active": config.INITIATIVE_MAX_CONTACTS_ACTIVE,
    "coach": config.INITIATIVE_MAX_CONTACTS_COACH,
}
_STATE_FILE = config.DB_DIR / ".neural_map_initiative_state.json"
_STATE_THREAD_LOCK = threading.Lock()


def _state_lock():
    """Cross-process exclusive lock on this module's state file."""
    return file_lock(_STATE_FILE.with_name(".neural_map_initiative_state.lock"))


def normalize_mode(mode: str | None = None) -> str:
    value = (mode or config.INITIATIVE_DEFAULT_MODE or "helpful").lower()
    return value if value in MODES else "helpful"


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _expired(meta: dict, now: datetime | None = None) -> bool:
    expires = _parse_date(meta.get("expires_at"))
    return bool(expires and expires < (now or datetime.now(timezone.utc)))


def _priority(meta: dict) -> int:
    return clamp_importance(meta.get("priority", meta.get("importance", 3)))


def _metadata_list(meta: dict, key: str) -> list[str]:
    try:
        raw = json.loads(meta.get(key) or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    return [str(value) for value in raw] if isinstance(raw, list) else []


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _empty_state() -> dict:
    return {
        "date": _today(),
        "contacts_today": 0,
        "dismissals_today": 0,
        "last_contact_at": None,
        "quiet_until": None,
        "kind_cooldowns": {},
    }


def _read_state_unlocked() -> dict:
    try:
        data = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return _empty_state()
    if not isinstance(data, dict):
        return _empty_state()
    state = {**_empty_state(), **data}
    if state.get("date") != _today():
        state["date"] = _today()
        state["contacts_today"] = 0
        state["dismissals_today"] = 0
    return state


def _write_state_unlocked(state: dict) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STATE_FILE.with_name(f"{_STATE_FILE.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(state, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(tmp, _STATE_FILE)


def _read_state() -> dict:
    with _STATE_THREAD_LOCK:
        with _state_lock():
            return _read_state_unlocked()


def _write_state(state: dict) -> None:
    with _STATE_THREAD_LOCK:
        with _state_lock():
            _write_state_unlocked(state)


def _in_future(value: str | None) -> bool:
    when = _parse_date(value)
    return bool(when and when > datetime.now(timezone.utc))


def _budget_allows(mode: str, state: dict) -> bool:
    """Interruption budget gate.

    INITIATIVE LAYER: this is the deeper "don't be noisy" control. Delete the
    state JSON or set mode=quiet to return to fully reactive behavior.
    """
    if mode == "quiet":
        return False
    if _in_future(state.get("quiet_until")):
        return False
    return int(state.get("contacts_today") or 0) < _MODE_DAILY_CAP[mode]


def active_initiatives(
    model: str | None = None,
    mode: str | None = None,
    respect_budget: bool = True,
) -> list[dict]:
    """Active initiative candidates, filtered by user-selected initiative mode."""
    if not config.INITIATIVE_ENABLED:
        return []
    mode = normalize_mode(mode)
    state = _read_state()
    if respect_budget and not _budget_allows(mode, state):
        return []
    cap = _MODE_CAP[mode]
    if cap <= 0:
        return []
    minimum = _MODE_MIN_PRIORITY[mode]
    rows: list[dict] = []
    now = datetime.now(timezone.utc)
    cooldowns = state.get("kind_cooldowns") or {}
    for rec in store.list_records_all(model):
        meta = rec["meta"] or {}
        if meta.get("category") != config.INITIATIVE_CATEGORY:
            continue
        if meta.get("status") not in (None, "open"):
            continue
        if _expired(meta, now):
            continue
        kind = meta.get("kind") or "nudge"
        if _in_future(cooldowns.get(kind)):
            continue
        priority = _priority(meta)
        if priority < minimum:
            continue
        rows.append(
            {
                "id": rec["id"],
                "text": rec["text"],
                "kind": kind,
                "priority": priority,
                "date": meta.get("date"),
                "expires_at": meta.get("expires_at"),
                "origin": meta.get("origin"),
                "source_type": meta.get("source_type"),
                "grounding_status": meta.get("grounding_status"),
                "grounding_channels": _metadata_list(meta, "grounding_channels_json"),
                "evidence_ids": _metadata_list(meta, "evidence_ids_json"),
                "will_decision_id": meta.get("will_decision_id"),
                "will_initiative_id": rec["id"] if meta.get("will_decision_id") else None,
                "decision_mode": meta.get("decision_mode"),
                "experimental": bool(meta.get("experimental")),
                "non_top_selected": bool(meta.get("non_top_selected")),
                "decision_score": meta.get("decision_score"),
                "selection_reason": meta.get("selection_reason"),
                "will_outcome": meta.get("will_outcome"),
                "mode": rec["mode"],
            }
        )
    rows.sort(key=lambda r: (r["priority"], r.get("date") or ""), reverse=True)
    return arbiter.mediate_initiatives(rows)[:cap]


def note_shown(items: list[dict] | None, mode: str | None = None) -> None:
    """Record that initiative was actually surfaced to the user.

    The UI/API call this only after a non-empty list is returned. A one-hour
    debounce keeps page refreshes from burning the whole daily budget.
    """
    if not items or normalize_mode(mode) == "quiet":
        return
    with _STATE_THREAD_LOCK:
        with _state_lock():
            state = _read_state_unlocked()
            last = _parse_date(state.get("last_contact_at"))
            now = datetime.now(timezone.utc)
            if last and (now - last).total_seconds() < 3600:
                return
            state["contacts_today"] = int(state.get("contacts_today") or 0) + 1
            state["last_contact_at"] = now.isoformat()
            _write_state_unlocked(state)


def dismiss(ids: list[str], model: str | None = None) -> int:
    """Mark initiative records dismissed so the assistant learns to back off."""
    requested_ids = list(dict.fromkeys(ids))
    if not requested_ids:
        return 0
    with _STATE_THREAD_LOCK:
        with _state_lock():
            records = store.get_by_ids(requested_ids, model=model)
            open_ids: list[str] = []
            open_records = []
            for cid in requested_ids:
                rec = records.get(cid)
                meta = (rec or {}).get("meta") or {}
                if meta.get("category") == config.INITIATIVE_CATEGORY and meta.get("status") in (None, "open"):
                    open_ids.append(cid)
                    open_records.append(rec)
            if not open_ids:
                return 0
            dismissed = store.update_metadata_all(open_ids, {"status": "dismissed", "dismissed_at": now_iso()}, model=model)
            if dismissed <= 0:
                return 0
            state = _read_state_unlocked()
            cooldowns = dict(state.get("kind_cooldowns") or {})
            until = (datetime.now(timezone.utc) + timedelta(days=config.INITIATIVE_KIND_COOLDOWN_DAYS)).isoformat()
            for rec in open_records:
                kind = (rec.get("meta") or {}).get("kind") or "nudge"
                cooldowns[kind] = until
            state["dismissals_today"] = int(state.get("dismissals_today") or 0) + dismissed
            state["kind_cooldowns"] = cooldowns
            if state["dismissals_today"] >= config.INITIATIVE_QUIET_AFTER_DISMISSALS:
                state["quiet_until"] = until
            _write_state_unlocked(state)
            return dismissed


def state() -> dict:
    """Current interruption-budget state, exposed for diagnostics/tests."""
    return _read_state()


def reset() -> dict:
    _write_state(_empty_state())
    return state()


def unit_from_outreach(item: dict, default_kind: str = "nudge") -> MemoryUnit | None:
    """Turn reflection JSON into a stored initiative memory.

    Accepted shapes are deliberately loose so older prompts can emit
    {"message": "..."} or {"question": "..."} while newer prompts can include
    kind/priority/expires_in_days. Invalid/empty items are ignored.
    """
    if not isinstance(item, dict):
        return None
    text = (item.get("message") or item.get("question") or item.get("text") or "").strip()
    if not text:
        return None
    kind = (item.get("kind") or item.get("type") or default_kind or "nudge").strip()[:40]
    priority = clamp_importance(item.get("priority", item.get("importance", 3)))
    ttl = item.get("expires_in_days", config.INITIATIVE_DEFAULT_TTL_DAYS)
    try:
        ttl_days = max(1, min(90, int(ttl)))
    except (TypeError, ValueError):
        ttl_days = config.INITIATIVE_DEFAULT_TTL_DAYS
    expires_at = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()
    return MemoryUnit(
        text=text,
        category=config.INITIATIVE_CATEGORY,
        importance=priority,
        extra={
            "kind": kind,
            "priority": priority,
            "status": "open",
            "expires_at": expires_at,
            "source_type": "generated_initiative",
        },
    )


def active_preferences(model: str | None = None, limit: int = 6, mode: str | None = None) -> list[dict]:
    """Learned interaction-style preferences, newest/highest-importance first.

    Part of the initiative layer: with NEURAL_MAP_INITIATIVE=0 the rollback is
    fully reactive, so learned preferences are not surfaced (mirrors
    active_initiatives). The stored records remain for the Memory Manager."""
    if not config.INITIATIVE_ENABLED or normalize_mode(mode) == "quiet":
        return []
    rows = []
    for rec in store.list_records_all(model):
        meta = rec["meta"] or {}
        if meta.get("category") != config.PREFERENCE_CATEGORY:
            continue
        rows.append(
            {
                "id": rec["id"],
                "text": rec["text"],
                "importance": _priority(meta),
                "date": meta.get("date"),
            }
        )
    rows.sort(key=lambda r: (r["importance"], r.get("date") or ""), reverse=True)
    return rows[:limit]


def briefing(model: str | None = None, mode: str | None = None, respect_budget: bool = False) -> dict:
    """Deterministic daily briefing: reminders + initiatives + style preferences.

    Explicit briefings are user-requested, so they bypass the unsolicited-contact
    budget by default. Pass respect_budget=True for scheduled/ambient display.
    """
    from brain import reflect

    reminders = reflect.active_reminders(model=model)
    initiatives = active_initiatives(model=model, mode=mode, respect_budget=respect_budget)
    mediated = arbiter.mediate(reminders, initiatives)
    reminders = mediated["reminders"]
    initiatives = mediated["initiatives"]
    prefs = active_preferences(model=model, mode=mode)
    try:
        # Lazy import avoids the module-level Will -> initiative dependency.
        from brain import will

        experiment_reflections = will.interesting_reflections(model=model)
    except Exception as exc:
        log.fallback("will_reflections_failed", exc, model=model, phase="briefing")
        experiment_reflections = []
    conflicts = [i for i in initiatives if i.get("kind") in {"conflict", "arbiter_conflict"}]
    grounding_reviews = [i for i in initiatives if i.get("kind") == "arbiter_grounding"]
    open_loops = [i for i in initiatives if i.get("kind") in {"open_loop", "follow_up"}]
    arbiter_conflicts = [i for i in conflicts if i.get("kind") == "arbiter_conflict"]
    suggested = None
    if arbiter_conflicts:
        suggested = arbiter_conflicts[0].get("text")
    elif reminders:
        suggested = reminders[0].get("text")
    elif conflicts:
        suggested = conflicts[0].get("text")
    elif grounding_reviews:
        suggested = grounding_reviews[0].get("text")
    elif initiatives:
        suggested = initiatives[0].get("text")
    return {
        "mode": normalize_mode(mode),
        "reminders": reminders,
        "initiatives": initiatives,
        "conflicts": conflicts,
        "open_loops": open_loops,
        "grounding_reviews": grounding_reviews,
        "preferences": prefs,
        "experiment_reflections": experiment_reflections,
        "suggested_next_action": suggested,
        "state": _read_state(),
    }
