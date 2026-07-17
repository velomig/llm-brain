"""Persistent self-model for autobiographical continuity.

This is not a claim of sentience. It is a compact continuity ledger: what the
assistant is for, which projects are active, what it has promised, and which
mistakes it should avoid repeating.
"""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone

import config
from brain.utils import clean_text, file_lock
from brain.memory import now_iso

SCHEMA_VERSION = 1
_STATE_FILE = config.DB_DIR / "self_model.json"
_STATE_THREAD_LOCK = threading.Lock()


def _state_lock():
    """Cross-process exclusive lock on this module's state file."""
    return file_lock(_STATE_FILE.with_name(".neural_map_self_model.lock"))


def _default_principles() -> list[str]:
    return [
        "Be useful, grounded, and transparent about uncertainty.",
        "Prefer current user intent and trusted current facts over stale memory.",
        "Preserve user agency; ask before risky or irreversible changes.",
        "Follow through on promised implementation and verification work.",
    ]


def _empty_state() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": None,
        "identity": {
            "name": "Will",
            "role": "Neural Map's local memory companion, coding partner, and reflective assistant",
            "principles": _default_principles(),
        },
        "current_projects": [],
        "open_promises": [],
        "lessons_learned": [],
        "relationship_notes": [],
        # Retired active-work entries (projects/promises past their TTL). Kept
        # out of the prompt snapshot; visible in /api/self_model.
        "archive": [],
    }


def _clean_text(value, limit: int = 280) -> str:
    # Word-boundary truncation with a visible ellipsis (see utils.clean_text):
    # rows feed the prompt, so a silent mid-sentence cut would read as a
    # complete — but wrong — statement.
    return clean_text(value, limit)


def _row(text: str, kind: str, source: str | None = None) -> dict:
    return {
        "id": f"{kind}_{uuid.uuid4().hex[:12]}",
        "text": _clean_text(text),
        "kind": kind,
        "source": _clean_text(source, 120),
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }


def _repair_rows(value, kind: str, limit: int | None = None) -> list[dict]:
    rows = value if isinstance(value, list) else []
    fixed: list[dict] = []
    for item in rows:
        if isinstance(item, str):
            item = _row(item, kind)
        if not isinstance(item, dict):
            continue
        text = _clean_text(item.get("text"))
        if not text:
            continue
        fixed.append(
            {
                "id": str(item.get("id") or f"{kind}_{uuid.uuid4().hex[:12]}")[:80],
                "text": text,
                "kind": str(item.get("kind") or kind)[:40],
                "source": _clean_text(item.get("source"), 120),
                "created_at": item.get("created_at") or now_iso(),
                "updated_at": item.get("updated_at") or item.get("created_at") or now_iso(),
            }
        )
    return fixed[-(limit or config.SELF_MODEL_MAX_ITEMS) :]


def _is_stale(row: dict, now: datetime, ttl_days: float) -> bool:
    """True when the row's last confirmation is older than the TTL.

    ttl_days <= 0 disables expiry. An unparseable timestamp keeps the row —
    losing state to a formatting bug would be worse than one stale line."""
    if ttl_days <= 0:
        return False
    stamp = str(row.get("updated_at") or row.get("created_at") or "")
    try:
        then = datetime.fromisoformat(stamp)
    except ValueError:
        return False
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (now - then).total_seconds() > ttl_days * 86400.0


def _repair_state(data) -> dict:
    state = _empty_state()
    if not isinstance(data, dict):
        return state
    state["schema_version"] = SCHEMA_VERSION
    state["updated_at"] = data.get("updated_at")
    # Identity is LOCKED to the code constants above. There is deliberately no
    # runtime path — tool, API, or state-file edit — that can rewrite the
    # assistant's name, role, or principles: whatever the file claims is
    # ignored here and healed on the next write, so a drifted or tampered
    # self_model.json cannot change the persona injected into every prompt.
    # Changing identity is a reviewed source-code change (human by construction).
    state["current_projects"] = _repair_rows(data.get("current_projects"), "project")
    state["open_promises"] = _repair_rows(data.get("open_promises"), "promise")
    state["lessons_learned"] = _repair_rows(data.get("lessons_learned"), "lesson")
    state["relationship_notes"] = _repair_rows(data.get("relationship_notes"), "relationship")
    # Lifecycle sweep: projects/promises are ACTIVE work, and _append_unique
    # re-stamps updated_at whenever a turn re-confirms one — so an entry whose
    # updated_at aged past the TTL is finished or abandoned. Move it to the
    # archive instead of letting it squat in one of the few prompt slots.
    # Lessons and relationship notes are identity, not workload: no TTL.
    archive = _repair_rows(data.get("archive"), "archived", limit=config.SELF_MODEL_MAX_ARCHIVE)
    now = datetime.now(timezone.utc)
    for key in ("current_projects", "open_promises"):
        active: list[dict] = []
        for row in state[key]:
            if _is_stale(row, now, config.SELF_MODEL_ACTIVE_TTL_DAYS):
                archive.append({**row, "kind": f"archived_{row.get('kind') or 'entry'}"[:40]})
            else:
                active.append(row)
        state[key] = active
    state["archive"] = archive[-config.SELF_MODEL_MAX_ARCHIVE:]
    return state


def _read_state_unlocked() -> dict:
    try:
        data = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return _empty_state()
    return _repair_state(data)


def _write_state_unlocked(state: dict) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fixed = _repair_state({**state, "updated_at": now_iso()})
    tmp = _STATE_FILE.with_name(f"{_STATE_FILE.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(fixed, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(tmp, _STATE_FILE)


def state() -> dict:
    if not config.SELF_MODEL_ENABLED:
        return {"enabled": False}
    with _STATE_THREAD_LOCK:
        with _state_lock():
            current = _read_state_unlocked()
    return {"enabled": True, **current}


def reset() -> dict:
    with _STATE_THREAD_LOCK:
        with _state_lock():
            fresh = _empty_state()
            fresh["updated_at"] = now_iso()
            _write_state_unlocked(fresh)
    return state()


def add_lesson(text: str, source: str | None = None) -> dict:
    """Commit one lesson to the ledger. Called by the APPROVED record_lesson
    tool — never directly from turn interpretation (see update_from_turn)."""
    text = _clean_text(text)
    if not text:
        return {"error": "lesson text is required"}
    with _STATE_THREAD_LOCK:
        with _state_lock():
            current = _read_state_unlocked()
            current["lessons_learned"] = _append_unique(current["lessons_learned"], text, "lesson", source or "approved")
            row = next(
                (r for r in current["lessons_learned"] if r.get("text", "").lower() == text.lower()),
                None,
            )
            _write_state_unlocked(current)
    return {"ok": True, "id": (row or {}).get("id"), "text": text, "kind": "lesson"}


def remove_lesson(lesson_id: str) -> dict:
    """Remove a lesson by id (the undo path for an approved record_lesson)."""
    lesson_id = str(lesson_id or "").strip()
    if not lesson_id:
        return {"error": "lesson id is required"}
    with _STATE_THREAD_LOCK:
        with _state_lock():
            current = _read_state_unlocked()
            before = len(current["lessons_learned"])
            current["lessons_learned"] = [r for r in current["lessons_learned"] if r.get("id") != lesson_id]
            removed = before - len(current["lessons_learned"])
            if removed:
                _write_state_unlocked(current)
    if not removed:
        return {"error": "lesson not found"}
    return {"ok": True, "removed": [lesson_id]}


def _append_unique(rows: list[dict], text: str, kind: str, source: str | None = None) -> list[dict]:
    text = _clean_text(text)
    if not text:
        return rows
    normalized = text.lower()
    existing = next((r for r in rows if r.get("text", "").lower() == normalized), None)
    if existing:
        # Re-confirmation refreshes the TTL clock AND moves the row to the end,
        # so the cap below evicts the least-recently-confirmed entry instead of
        # whatever happened to be inserted first.
        existing["updated_at"] = now_iso()
        rows = [r for r in rows if r is not existing] + [existing]
        return rows[-config.SELF_MODEL_MAX_ITEMS :]
    return [*rows, _row(text, kind, source)][-config.SELF_MODEL_MAX_ITEMS :]


def _project_candidates(user_text: str, reply_text: str) -> list[str]:
    blob = f"{user_text}\n{reply_text}".lower()
    projects: list[str] = []
    if any(term in blob for term in ("conscious", "consciousness", "sesame", "awareness", "presence")):
        projects.append("Continue the Neural Map presence and awareness upgrade.")
    if any(term in blob for term in ("portfolio", "fndx", "fndy", "fndz", "source of truth")):
        projects.append("Keep portfolio analysis anchored to current pinned memories and Source of Truth.")
    if any(term in blob for term in ("server", "port 8000", "restart", "browser", "web ui")):
        projects.append("Keep the local Neural Map web app stable and easy to resume.")
    return projects


_PROMISE_RE = re.compile(
    r"\b(?:I(?:'ll| will| am going to)|I’m going to|I'm going to)\s+([^.!?\n]{8,180})",
    re.IGNORECASE,
)


def _promise_candidates(reply_text: str) -> list[str]:
    out: list[str] = []
    for match in _PROMISE_RE.finditer(reply_text or ""):
        text = _clean_text("I will " + match.group(1).rstrip(" ."))
        if text:
            out.append(text)
    return out[:3]


def _lesson_candidates(user_text: str, reply_text: str) -> list[str]:
    blob = f"{user_text}\n{reply_text}".lower()
    lessons: list[str] = []
    if any(term in blob for term in ("wrong", "off", "stale", "old data", "doesn't", "does not", "isn't", "not access")):
        lessons.append("When behavior looks wrong or stale, inspect actual retrieval/tool traces and source priority before explaining.")
    if "model returned no final text" in blob or "no final text" in blob:
        lessons.append("After tools run, force a final answer from tool results instead of surfacing raw tool-loop failure.")
    if "source of truth" in blob and any(term in blob for term in ("update", "old", "stale", "current")):
        lessons.append("Treat Source of Truth as editable calibration and keep it aligned with newer pinned factual notes.")
    return lessons


def _relationship_candidates(user_text: str) -> list[str]:
    text = user_text.lower()
    notes: list[str] = []
    if any(term in text for term in ("consciousness", "conscious", "biology", "sesame")):
        notes.append("The user is exploring non-biological consciousness and wants Neural Map to feel continuous while staying grounded.")
    if "portfolio" in text:
        notes.append("The user cares about accurate portfolio analysis and notices stale financial data quickly.")
    return notes


def _lesson_already_tracked(text: str, current: dict) -> bool:
    """Committed, pending, OR previously rejected — all mean "do not re-propose".

    A rejected proposal counting as tracked is what keeps the capture path from
    nagging with the same lesson every turn after the user said no."""
    from brain import tools

    normalized = _clean_text(text).lower()
    if any(r.get("text", "").lower() == normalized for r in current.get("lessons_learned") or []):
        return True
    for action in tools.list_pending(include_resolved=True):
        if action.get("tool") != "record_lesson":
            continue
        if _clean_text((action.get("args") or {}).get("text")).lower() == normalized:
            return True
    return False


def update_from_turn(user_text: str, reply_text: str, *, thinking: dict | None = None) -> dict:
    if not config.SELF_MODEL_ENABLED:
        return state()
    with _STATE_THREAD_LOCK:
        with _state_lock():
            current = _read_state_unlocked()
            for text in _project_candidates(user_text, reply_text):
                current["current_projects"] = _append_unique(current["current_projects"], text, "project", "turn")
            for text in _promise_candidates(reply_text):
                current["open_promises"] = _append_unique(current["open_promises"], text, "promise", "assistant_reply")
            for text in _relationship_candidates(user_text):
                current["relationship_notes"] = _append_unique(current["relationship_notes"], text, "relationship", "user_turn")
            if thinking and thinking.get("tools_used"):
                current["current_projects"] = _append_unique(
                    current["current_projects"],
                    "Explain tool results clearly and verify user-facing changes after acting.",
                    "project",
                    "tools_used",
                )
            _write_state_unlocked(current)
    # Lessons are standing behavioral rules, so they get action-package rigor:
    # a turn may only PROPOSE one (record_lesson is approval-gated), never
    # commit it. Queued outside the state lock — the pending queue has its own.
    candidates = _lesson_candidates(user_text, reply_text)
    if candidates:
        snapshot = state()
        for text in candidates:
            try:
                from brain import tools

                if _lesson_already_tracked(text, snapshot):
                    continue
                tools.dispatch("record_lesson", {"text": text, "source": "turn_heuristic"}, taint=["turn_interpretation"])
            except Exception as exc:
                from brain import log

                log.fallback("lesson_proposal_failed", exc, phase="self_model")
    return state()


def prompt_snapshot(current: dict | None = None) -> dict:
    if not config.SELF_MODEL_ENABLED:
        return {"enabled": False}
    current = current or state()
    return {
        "enabled": True,
        "identity": current.get("identity") or _empty_state()["identity"],
        "current_projects": current.get("current_projects") or [],
        "open_promises": current.get("open_promises") or [],
        "lessons_learned": current.get("lessons_learned") or [],
        "relationship_notes": current.get("relationship_notes") or [],
        "guidance": (
            "Use this as continuity and follow-through context only. Do not claim literal sentience. "
            "Honor open promises when relevant and avoid repeating listed mistakes."
        ),
    }
