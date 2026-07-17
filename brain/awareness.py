"""Persistent awareness synthesis.

This is not literal consciousness. It is the app's compact, carried "what am I
attending to right now?" layer: current focus, stance, open loops, and a next
best action derived from existing state. It gives the UI and prompt a stable
presence without adding another expensive LLM call.
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
_STATE_FILE = config.DB_DIR / "awareness_state.json"
_STATE_THREAD_LOCK = threading.Lock()


def _state_lock():
    """Cross-process exclusive lock on this module's state file."""
    return file_lock(_STATE_FILE.with_name(".neural_map_awareness_state.lock"))


def _empty_state() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": None,
        "current_focus": "idle",
        "stance": "neutral",
        "energy": _energy_baseline(),
        "confidence": 0.5,
        "curiosity": 0.5,
        "open_loops": [],
        # text -> turns carried WITHOUT a live source re-confirming it (0 =
        # confirmed this turn). Orphans expire past AWARENESS_LOOP_TTL_TURNS.
        "open_loop_meta": {},
        "last_user_intent": None,
        "last_user_seen": None,
        "last_tools": [],
        "next_best_action": "wait for the next user request",
        "reason": "neutral baseline",
        "recent_turns": [],
    }


def _clamp(value, lo: float = 0.0, hi: float = 1.0, default: float = 0.5) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return round(min(hi, max(lo, parsed)), 4)


def _energy_baseline() -> float:
    """Configured neutral set point, bounded to the persisted scalar range."""
    return _clamp(config.AWARENESS_ENERGY_BASELINE, default=0.5)


def _clean_text(value, limit: int = 240) -> str:
    # Word-boundary truncation with a visible ellipsis (see utils.clean_text):
    # these fields feed the prompt, so a silent mid-sentence cut would read as
    # a complete — but wrong — statement.
    return clean_text(value, limit)


def _clean_list(value, limit: int = 5) -> list[str]:
    rows = value if isinstance(value, list) else []
    out: list[str] = []
    for item in rows:
        text = _clean_text(item, 240)
        if text and text not in out:
            out.append(text)
    return out[:limit]


def _repair_state(data) -> dict:
    state = _empty_state()
    if not isinstance(data, dict):
        return state
    state["schema_version"] = SCHEMA_VERSION
    state["updated_at"] = data.get("updated_at")
    state["current_focus"] = _clean_text(data.get("current_focus") or state["current_focus"], 120)
    state["stance"] = _clean_text(data.get("stance") or state["stance"], 80)
    state["energy"] = _clamp(data.get("energy"), default=_energy_baseline())
    state["confidence"] = _clamp(data.get("confidence"), default=0.5)
    state["curiosity"] = _clamp(data.get("curiosity"), default=0.5)
    state["open_loops"] = _clean_list(data.get("open_loops"), config.AWARENESS_MAX_OPEN_LOOPS)
    meta_raw = data.get("open_loop_meta") if isinstance(data.get("open_loop_meta"), dict) else {}
    state["open_loop_meta"] = {}
    for text in state["open_loops"]:
        try:
            carried = int(meta_raw.get(text, 0))
        except (TypeError, ValueError):
            carried = 0
        state["open_loop_meta"][text] = max(0, carried)
    state["last_user_intent"] = _clean_text(data.get("last_user_intent"), 240) or None
    state["last_user_seen"] = data.get("last_user_seen")
    state["last_tools"] = _clean_list(data.get("last_tools"), 12)
    state["next_best_action"] = _clean_text(data.get("next_best_action") or state["next_best_action"], 240)
    state["reason"] = _clean_text(data.get("reason") or state["reason"], 240)
    turns = data.get("recent_turns") if isinstance(data.get("recent_turns"), list) else []
    state["recent_turns"] = [
        {
            "at": row.get("at"),
            "user": _clean_text(row.get("user"), 400),
            "assistant": _clean_text(row.get("assistant"), 400),
            "path": _clean_text(row.get("path"), 40),
        }
        for row in turns
        if isinstance(row, dict) and (row.get("user") or row.get("assistant"))
    ][-config.AWARENESS_MAX_RECENT_TURNS :]
    return state


def _read_state_unlocked() -> dict:
    try:
        data = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return _empty_state()
    return _repair_state(data)


def _write_state_unlocked(state: dict) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STATE_FILE.with_name(f"{_STATE_FILE.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(_repair_state(state), sort_keys=True, indent=2), encoding="utf-8")
    os.replace(tmp, _STATE_FILE)


def state() -> dict:
    if not config.AWARENESS_ENABLED:
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


def _focus_from_text(user_text: str, mems: list[dict] | None) -> str:
    text = user_text.lower()
    if any(term in text for term in ("portfolio", "schd", "fxaix", "sgov", "shares", "allocation")):
        return "portfolio accuracy"
    if any(term in text for term in ("memory", "source of truth", "pinned", "recall")):
        return "memory reliability"
    if any(term in text for term in ("code", "bug", "fix", "test", "server", "browser")):
        return "code repair"
    if any(term in text for term in ("conscious", "consciousness", "sesame", "presence", "aware")):
        return "awareness design"
    if mems:
        top = next((m for m in mems if (m.get("category") or "").strip()), None)
        if top:
            return f"{top.get('category')} recall"
    return _clean_text(user_text, 80) or "conversation"


def _intent_from_text(user_text: str) -> str:
    text = user_text.strip()
    low = text.lower()
    if low.startswith(("/trim", "/clear", "/context")):
        return "manage short-term chat context"
    if "?" in text or low.startswith(("how", "what", "why", "can", "is there")):
        return "answer a question"
    if any(term in low for term in ("fix", "make", "add", "implement", "do it", "lets do it", "let's do it")):
        return "make a code change"
    return _clean_text(text, 120) or "continue conversation"


def _stance(emotion_state: dict | None, thinking: dict | None) -> str:
    labels = (emotion_state or {}).get("labels") or []
    if isinstance(labels, list) and labels:
        base = ", ".join(str(x) for x in labels[:2])
    else:
        base = "neutral"
    path = (thinking or {}).get("path")
    if path == "system2":
        return f"{base}, deliberate"
    if path == "standard":
        return f"{base}, focused"
    return base


def _scalar_from_emotion(emotion_state: dict | None, key: str, default: float = 0.5) -> float:
    if not isinstance(emotion_state, dict):
        return default
    try:
        value = float(emotion_state.get(key, 0.0))
    except (TypeError, ValueError):
        return default
    return _clamp((value + 1.0) / 2.0, default=default)


def _open_loops(
    reminders: list[dict] | None,
    initiatives: list[dict] | None,
    cognitive_snapshot: dict | None,
    previous: list[str],
    previous_meta: dict | None = None,
) -> tuple[list[str], dict[str, int]]:
    """Current open loops plus a carry counter per loop.

    Reminders, initiatives, and goals are re-fetched live every turn, so any
    loop they produce is confirmed (carry 0) and sorts first. A previous loop
    with no live source this turn is an orphan: it is carried at reduced
    priority and RETIRED once unconfirmed for AWARENESS_LOOP_TTL_TURNS turns —
    otherwise a dead task recycles forever and, at the list cap, blocks newer
    loops from ever appearing.
    """
    fresh: list[str] = []
    for row in reminders or []:
        text = _clean_text(row.get("text") or row.get("action"), 180)
        if text:
            fresh.append(text)
    for row in initiatives or []:
        text = _clean_text(row.get("text"), 180)
        if text:
            fresh.append(text)
    for row in (cognitive_snapshot or {}).get("active_goals") or []:
        action = _clean_text(row.get("next_action") or row.get("text"), 180)
        if action:
            fresh.append(action)

    previous_meta = previous_meta or {}
    ttl = max(0, config.AWARENESS_LOOP_TTL_TURNS)
    confirmed = set(_clean_list(fresh, len(fresh) or 1))
    loops = list(fresh)
    for text in previous:
        if text in confirmed:
            continue   # re-confirmed by a live source; the fresh copy leads
        try:
            carried = max(0, int(previous_meta.get(text, 0))) + 1
        except (TypeError, ValueError):
            carried = 1
        if carried > ttl:
            continue   # orphan past its TTL: retire instead of recycling
        loops.append(text)

    cleaned = _clean_list(loops, config.AWARENESS_MAX_OPEN_LOOPS)
    meta: dict[str, int] = {}
    for text in cleaned:
        if text in confirmed:
            meta[text] = 0
        else:
            try:
                meta[text] = max(0, int(previous_meta.get(text, 0))) + 1
            except (TypeError, ValueError):
                meta[text] = 1
    return cleaned, meta


def _smoothed_energy(previous, *, loops, tools, mems) -> float:
    """Energy eases toward activity and returns to a neutral set point.

    Activity is a transient positive deviation from AWARENESS_ENERGY_BASELINE.
    With no activity the baseline itself becomes the target, providing the
    homeostatic restoring force that a free-running EMA lacks. The carried
    value moves AWARENESS_ENERGY_SMOOTHING of the remaining distance per turn;
    1.0 restores instantaneous target tracking."""
    baseline = _energy_baseline()
    target = _clamp(
        baseline
        + 0.12 * bool(loops)
        + 0.08 * bool(tools)
        + 0.05 * bool(mems)
    )
    smoothing = min(1.0, max(0.0, config.AWARENESS_ENERGY_SMOOTHING))
    previous = _clamp(previous, default=baseline)
    return _clamp(previous + (target - previous) * smoothing)


def _next_action(focus: str, open_loops: list[str], tools: list[str], reply_text: str) -> str:
    if tools:
        return "explain the tool results and verify the answer with the user"
    if open_loops:
        return open_loops[0]
    if "?" in reply_text:
        return "wait for the user's answer"
    if focus in {"portfolio accuracy", "memory reliability"}:
        return "keep current facts pinned and Source of Truth aligned"
    return "wait for the next user request"


def update_from_turn(
    user_text: str,
    reply_text: str,
    *,
    mems: list[dict] | None = None,
    thinking: dict | None = None,
    reminders: list[dict] | None = None,
    initiatives: list[dict] | None = None,
    emotion_state: dict | None = None,
    cognitive_snapshot: dict | None = None,
) -> dict:
    if not config.AWARENESS_ENABLED:
        return state()
    tools = [str(x) for x in (thinking or {}).get("tools_used") or [] if x]
    focus = _focus_from_text(user_text, mems)
    now = now_iso()
    with _STATE_THREAD_LOCK:
        with _state_lock():
            current = _read_state_unlocked()
            loops, loop_meta = _open_loops(
                reminders,
                initiatives,
                cognitive_snapshot,
                current.get("open_loops") or [],
                current.get("open_loop_meta") or {},
            )
            updated = {
                **current,
                "updated_at": now,
                "current_focus": focus,
                "stance": _stance(emotion_state, thinking),
                "energy": _smoothed_energy(current.get("energy"), loops=loops, tools=tools, mems=mems),
                "confidence": _scalar_from_emotion(emotion_state, "confidence", 0.5),
                "curiosity": _scalar_from_emotion(emotion_state, "curiosity", 0.5),
                "open_loops": loops,
                "open_loop_meta": loop_meta,
                "last_user_intent": _intent_from_text(user_text),
                "last_user_seen": now,
                "last_tools": tools,
                "next_best_action": _next_action(focus, loops, tools, reply_text),
                "reason": f"updated after turn: {_clean_text(user_text, 120)}",
                "recent_turns": [
                    *(current.get("recent_turns") or []),
                    {
                        "at": now,
                        "user": _clean_text(user_text, 400),
                        "assistant": _clean_text(reply_text, 400),
                        "path": _clean_text((thinking or {}).get("path"), 40),
                    },
                ],
            }
            _write_state_unlocked(updated)
    return state()


def prompt_snapshot(current: dict | None = None) -> dict:
    if not config.AWARENESS_ENABLED:
        return {"enabled": False}
    current = current or state()
    return {
        "enabled": True,
        "current_focus": current.get("current_focus"),
        "stance": current.get("stance"),
        "energy": current.get("energy"),
        "confidence": current.get("confidence"),
        "curiosity": current.get("curiosity"),
        "open_loops": current.get("open_loops") or [],
        "last_user_intent": current.get("last_user_intent"),
        "next_best_action": current.get("next_best_action"),
        "guidance": (
            "Use as continuity/presence context only. Do not claim literal sentience. "
            "Let it shape attention, follow-through, and concise self-monitoring."
        ),
    }
