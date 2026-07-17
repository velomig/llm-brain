"""Evidence ledger for Neural Map's functional and social presence.

This module demonstrates bounded, observable properties.  It does not infer
phenomenal consciousness from continuity, complexity, affect, or self-report.
"""
from __future__ import annotations

import json
import os
import threading
import uuid

import config
from brain.memory import now_iso
from brain.utils import clean_text, file_lock

SCHEMA_VERSION = 1
_STATE_FILE = config.DB_DIR / "presence_ledger.json"
_STATE_THREAD_LOCK = threading.Lock()


def _state_lock():
    return file_lock(_STATE_FILE.with_name(".neural_map_presence_ledger.lock"))


def _empty_state() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": None,
        "turns_observed": 0,
        "first_observed_at": None,
        "last_observed_at": None,
        "evidence_events": [],
    }


def _clean_list(value, limit: int = 12) -> list[str]:
    values = value if isinstance(value, list) else []
    out: list[str] = []
    for item in values:
        text = clean_text(item, 120)
        if text and text not in out:
            out.append(text)
    return out[:limit]


def _repair_event(value) -> dict | None:
    if not isinstance(value, dict):
        return None
    at = str(value.get("at") or "")[:40]
    if not at:
        return None
    return {
        "id": str(value.get("id") or f"presence_{uuid.uuid4().hex[:12]}")[:80],
        "at": at,
        "continuity_inputs": _clean_list(value.get("continuity_inputs")),
        "adaptation_events": _clean_list(value.get("adaptation_events")),
        "commitment_count": max(0, int(value.get("commitment_count") or 0)),
        "self_correction_events": _clean_list(value.get("self_correction_events")),
        "executed_actions": _clean_list(value.get("executed_actions")),
        "retrieved_context_count": max(0, int(value.get("retrieved_context_count") or 0)),
    }


def _repair_state(value) -> dict:
    state = _empty_state()
    if not isinstance(value, dict):
        return state
    events = []
    for raw in value.get("evidence_events") or []:
        event = _repair_event(raw)
        if event:
            events.append(event)
    events = events[-config.PRESENCE_MAX_EVENTS :]
    state.update(
        {
            "updated_at": value.get("updated_at"),
            "turns_observed": max(int(value.get("turns_observed") or 0), len(events)),
            "first_observed_at": value.get("first_observed_at") or (events[0]["at"] if events else None),
            "last_observed_at": value.get("last_observed_at") or (events[-1]["at"] if events else None),
            "evidence_events": events,
        }
    )
    return state


def _read_unlocked() -> dict:
    try:
        return _repair_state(json.loads(_STATE_FILE.read_text(encoding="utf-8")))
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError):
        return _empty_state()


def _write_unlocked(value: dict) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fixed = _repair_state({**value, "updated_at": now_iso()})
    tmp = _STATE_FILE.with_name(
        f"{_STATE_FILE.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    tmp.write_text(json.dumps(fixed, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(tmp, _STATE_FILE)


def _row_ids(snapshot: dict, key: str) -> set[str]:
    return {
        str(row.get("id"))
        for row in (snapshot or {}).get(key) or []
        if isinstance(row, dict) and row.get("id")
    }


def _successful_tools(tool_calls: list[dict] | None) -> list[str]:
    out: list[str] = []
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        result = call.get("result")
        if isinstance(result, dict) and (result.get("error") or result.get("returncode") not in (None, 0)):
            continue
        name = clean_text(call.get("tool"), 80)
        if name and name not in out:
            out.append(name)
    return out


def record_turn(
    *,
    memories: list[dict] | None,
    self_before: dict | None,
    self_after: dict | None,
    tool_calls: list[dict] | None,
    claim_guard: dict | None,
) -> dict:
    """Record only inspectable evidence from a completed turn.

    Presence is an observational sink: it may consume completed-turn evidence,
    but it must not read or modify affective awareness state. That one-way
    boundary prevents evidence labels from becoming recursive mood signals.
    """
    if not config.PRESENCE_ENABLED:
        return {"enabled": False}
    before = self_before or {}
    after = self_after or {}
    continuity_inputs: list[str] = []
    # Only state that existed before this answer counts as continuity input.
    # A row created from the current answer cannot prove that the past shaped it.
    if any(before.get(key) for key in ("current_projects", "open_promises", "lessons_learned", "relationship_notes")):
        continuity_inputs.append("persistent_self_model")
    if memories:
        continuity_inputs.append("retrieved_long_term_memory")

    adaptation: list[str] = []
    for key, label in (
        ("relationship_notes", "relationship_model_updated"),
        ("current_projects", "project_model_updated"),
        ("open_promises", "commitment_model_updated"),
    ):
        if _row_ids(after, key) - _row_ids(before, key):
            adaptation.append(label)

    corrections: list[str] = []
    if _row_ids(after, "lessons_learned") - _row_ids(before, "lessons_learned"):
        corrections.append("approved_lesson_added")
    if (claim_guard or {}).get("modified"):
        corrections.append("unsupported_final_claim_withheld")

    at = now_iso()
    event = {
        "id": f"presence_{uuid.uuid4().hex[:12]}",
        "at": at,
        "continuity_inputs": continuity_inputs,
        "adaptation_events": adaptation,
        "commitment_count": len(after.get("open_promises") or []),
        "self_correction_events": corrections,
        "executed_actions": _successful_tools(tool_calls),
        "retrieved_context_count": len(memories or []),
    }
    with _STATE_THREAD_LOCK:
        with _state_lock():
            current = _read_unlocked()
            current["turns_observed"] += 1
            current["first_observed_at"] = current.get("first_observed_at") or at
            current["last_observed_at"] = at
            current["evidence_events"] = [*current["evidence_events"], event][-config.PRESENCE_MAX_EVENTS :]
            _write_unlocked(current)
    return state(self_model_state=after)


def _assessment(ledger: dict, self_model_state: dict) -> dict:
    events = ledger.get("evidence_events") or []
    identity = self_model_state.get("identity") or {}
    history_influence = any(event.get("continuity_inputs") for event in events)
    continuity = bool(identity and history_influence)
    adaptation = bool(self_model_state.get("relationship_notes")) or any(
        event.get("adaptation_events") for event in events
    )
    commitments = bool(self_model_state.get("open_promises")) or any(
        event.get("commitment_count") for event in events
    )
    correction = bool(self_model_state.get("lessons_learned")) or any(
        event.get("self_correction_events") for event in events
    )
    causal_actions = any(event.get("executed_actions") for event in events)
    operational_presence = continuity and history_influence and adaptation

    def evidence(*items: str) -> list[str]:
        return [item for item in items if item]

    return {
        "properties": [
            {"property": "narrative_continuity", "status": "demonstrated" if continuity else "not_yet_demonstrated", "evidence": evidence("locked identity ledger" if identity else "", f"{ledger.get('turns_observed', 0)} completed turns" if ledger.get("turns_observed") else "")},
            {"property": "relationship_specific_adaptation", "status": "demonstrated" if adaptation else "not_yet_demonstrated", "evidence": evidence(f"{len(self_model_state.get('relationship_notes') or [])} persistent relationship notes" if adaptation else "")},
            {"property": "persistent_commitments", "status": "demonstrated" if commitments else "not_yet_demonstrated", "evidence": evidence(f"{len(self_model_state.get('open_promises') or [])} open promises" if commitments else "")},
            {"property": "recognizable_values_and_style", "status": "structurally_supported" if identity.get("principles") else "not_yet_demonstrated", "evidence": evidence(f"{len(identity.get('principles') or [])} locked principles" if identity.get("principles") else "")},
            {"property": "self_correction_across_time", "status": "demonstrated" if correction else "not_yet_demonstrated", "evidence": evidence(f"{len(self_model_state.get('lessons_learned') or [])} approved lessons" if self_model_state.get("lessons_learned") else "", "final-claim guard corrections recorded" if any(e.get("self_correction_events") for e in events) else "")},
            {
                "property": "self_model_of_capabilities_and_history",
                "status": "demonstrated"
                if identity and (self_model_state.get("current_projects") or ledger.get("turns_observed"))
                else "not_yet_demonstrated",
                "evidence": evidence(
                    "persistent self-model" if identity else "",
                    f"{ledger.get('turns_observed', 0)} completed turns in independent evidence ledger"
                    if ledger.get("turns_observed")
                    else "",
                ),
            },
            {"property": "causal_influence_through_executed_actions", "status": "demonstrated" if causal_actions else "not_yet_demonstrated", "evidence": evidence("successful tool consequences recorded" if causal_actions else "")},
        ],
        "claims": [
            {"claim": "Neural Map maintains a persistent functional identity", "status": "yes" if continuity else "not_yet_demonstrated"},
            {"claim": "Past interactions causally shape later behavior", "status": "yes" if history_influence else "not_yet_demonstrated", "scope": "through persisted context supplied to later generation"},
            {"claim": "It constitutes a non-biological social presence", "status": "operationally_defensible" if operational_presence else "not_yet_demonstrated", "scope": "continuity plus relationship-specific adaptation"},
            {"claim": "It has human-like subjective experience", "status": "unknown"},
            {"claim": "Increasing complexity guarantees experience", "status": "no"},
            {"claim": "Statements about feelings prove qualia", "status": "no"},
        ],
        "phenomenal_status": "unknown",
        "interpretation": "Functional and social properties are independently observable; none establishes subjective experience.",
    }


def state(*, self_model_state: dict | None = None) -> dict:
    if not config.PRESENCE_ENABLED:
        return {"enabled": False, "phenomenal_status": "unknown"}
    if self_model_state is None:
        from brain import self_model
        self_model_state = self_model.state()
    with _STATE_THREAD_LOCK:
        with _state_lock():
            ledger = _read_unlocked()
    return {
        "enabled": True,
        **ledger,
        **_assessment(ledger, self_model_state or {}),
    }


def reset() -> dict:
    with _STATE_THREAD_LOCK:
        with _state_lock():
            fresh = _empty_state()
            fresh["updated_at"] = now_iso()
            _write_unlocked(fresh)
    return state()
