"""Homeostatic coupling layer: a derived operating posture ("synthesis engine").

Cognitive state stores isolated facts; attention tracks isolated salience.
Nothing connects them, so "clinical hours doubled" and "how deep should the
portfolio analysis go" only meet inside the model's judgment, per turn, if at
all. This layer couples them mechanically: a few named variables in [0, 1]
where sustained pressure on a sensed variable (user_load) automatically
re-weights derived behavior dials (portfolio_intensity, proactivity) through
explicit rules — the way a glucose drop IS the metabolic shift, not a fact
retrieved and then deliberated.

Design constraints, in priority order:
  * Deterministic — sensors read existing layers (attention domains, cognitive
    state world items); no LLM calls, no randomness.
  * Attributable — every rule firing/release lands in a bounded adjustments
    log and an observability event; no silent behavior drift.
  * Governed — coupling rules are declarative data. Defaults ship here;
    replacements arrive only through the API (a deliberate user action), never
    by inference. Attention salience alone is scaled so it can never fire a
    rule without corroborating world-model evidence.
  * Stable — rules activate at a threshold and release at a lower one
    (hysteresis), and variables move by smoothing, so a noisy signal cannot
    make the posture oscillate turn to turn.

This is a derived policy layer, not a truth source: it never writes memories
or cognitive state, and its posture never overrides Source of Truth, current
user instructions, or retrieved evidence.
"""
from __future__ import annotations

import json
import os
import re
import threading
import uuid

import config
from brain.utils import file_lock
from brain.memory import now_iso

SCHEMA_VERSION = 1

# Sensed variables track the world and are written only by their sensors.
# Derived variables are behavior dials that rules push away from a baseline.
VARIABLES: dict[str, dict] = {
    "user_load": {
        "kind": "sensed",
        "baseline": 0.0,
        "description": "How loaded the user currently is (clinical hours, work, school).",
    },
    "portfolio_intensity": {
        "kind": "derived",
        "baseline": 0.7,
        "description": "How much discretionary portfolio analysis to favor over rule-based stability.",
    },
    "proactivity": {
        "kind": "derived",
        "baseline": 0.6,
        "description": "How much unprompted suggestion/contact pressure is appropriate.",
    },
}
SENSED = tuple(name for name, spec in VARIABLES.items() if spec["kind"] == "sensed")
DERIVED = tuple(name for name, spec in VARIABLES.items() if spec["kind"] == "derived")

# While a rule is active its effect is weight * (source - release), so the
# effect fades continuously to zero as the source approaches the release point
# instead of snapping off at the activation threshold.
DEFAULT_RULES: list[dict] = [
    {
        "id": "load_caps_portfolio_intensity",
        "source": "user_load",
        "target": "portfolio_intensity",
        "weight": -1.2,
        "threshold": 0.5,
        "release": 0.4,
        "note": "High clinical/work load shifts portfolio work toward rule-based stability.",
    },
    {
        "id": "load_caps_proactivity",
        "source": "user_load",
        "target": "proactivity",
        "weight": -0.8,
        "threshold": 0.65,
        "release": 0.5,
        "note": "Very high load means fewer, higher-value proactive contacts.",
    },
]

_POSTURE_GUIDANCE = {
    "portfolio_intensity": {
        "expansive": "normal depth: discretionary analysis and rebalancing proposals are fine",
        "balanced": "medium depth: prefer straightforward analysis; flag rather than propose complex changes",
        "conservative": "favor rule-based stability; defer complex or discretionary portfolio moves",
    },
    "proactivity": {
        "expansive": "proactive suggestions welcome at the normal cadence",
        "balanced": "raise only clearly useful proactive items",
        "conservative": "hold proactive items unless urgent or time-critical",
    },
}

# Attention salience is corroborating evidence, not proof of workload — talking
# about nursing a lot is not working 40 hours. Scaled below every default rule
# threshold so attention alone can never fire a rule; world-model evidence
# (an actual hours/schedule fact) is required to cross.
_ATTENTION_LOAD_DOMAINS = ("nursing", "benefits")
_ATTENTION_LOAD_SCALE = 0.45

_LOAD_KEY = re.compile(r"clinic|work|shift|rotation|schedule|semester|hours|load|study", re.I)
_HOURS = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*(?:hours?|hrs?)\b", re.I)
_QUALITATIVE_LOAD = (
    ("overtime", 1.0),
    ("double shift", 1.0),
    ("full-time", 0.9),
    ("full time", 0.9),
    ("part-time", 0.5),
    ("part time", 0.5),
    ("time off", 0.1),
    ("on break", 0.1),
)

_STATE_FILE = config.DB_DIR / "homeostasis_state.json"
_STATE_THREAD_LOCK = threading.Lock()


def _state_lock():
    """Cross-process exclusive lock on this module's state file."""
    return file_lock(_STATE_FILE.with_name(".neural_map_homeostasis_state.lock"))


def _clamp(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return round(min(1.0, max(0.0, parsed)), 4)


def _empty_state() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": None,
        "variables": {name: spec["baseline"] for name, spec in VARIABLES.items()},
        "active_rules": [],
        "custom_rules": None,
        "adjustments": [],
        "last_signals": {},
        "reason": "neutral baseline",
    }


def _repair_rule(raw: object, seen_ids: set[str]) -> dict:
    """Validate one coupling rule; raises ValueError so a bad API payload is rejected whole."""
    if not isinstance(raw, dict):
        raise ValueError("each rule must be an object")
    rule_id = str(raw.get("id") or "").strip()[:80]
    if not rule_id:
        raise ValueError("rule id is required")
    if rule_id in seen_ids:
        raise ValueError(f"duplicate rule id: {rule_id}")
    source = str(raw.get("source") or "").strip()
    if source not in SENSED:
        raise ValueError(f"rule source must be a sensed variable ({', '.join(SENSED)}): {rule_id}")
    target = str(raw.get("target") or "").strip()
    if target not in DERIVED:
        raise ValueError(f"rule target must be a derived variable ({', '.join(DERIVED)}): {rule_id}")
    try:
        weight = round(float(raw.get("weight")), 4)
    except (TypeError, ValueError):
        raise ValueError(f"rule weight must be a number: {rule_id}")
    if not -2.0 <= weight <= 2.0 or weight == 0.0:
        raise ValueError(f"rule weight must be in [-2, 2] and nonzero: {rule_id}")
    threshold = _clamp(raw.get("threshold", 0.5))
    raw_release = raw.get("release")
    release = _clamp(raw_release) if raw_release is not None else _clamp(max(0.0, threshold - 0.1))
    if release > threshold:
        raise ValueError(f"rule release must be <= threshold (hysteresis): {rule_id}")
    seen_ids.add(rule_id)
    return {
        "id": rule_id,
        "source": source,
        "target": target,
        "weight": weight,
        "threshold": threshold,
        "release": release,
        "note": str(raw.get("note") or "").strip()[:240],
    }


def _validate_rules(raw: object) -> list[dict]:
    if not isinstance(raw, list):
        raise ValueError("rules must be a list")
    if len(raw) > config.HOMEOSTASIS_MAX_RULES:
        raise ValueError(f"at most {config.HOMEOSTASIS_MAX_RULES} rules allowed")
    seen: set[str] = set()
    return [_repair_rule(rule, seen) for rule in raw]


def _repair_state(data: object) -> dict:
    state = _empty_state()
    if not isinstance(data, dict):
        return state
    raw_vars = data.get("variables") if isinstance(data.get("variables"), dict) else {}
    state["updated_at"] = data.get("updated_at")
    state["variables"] = {
        name: _clamp(raw_vars.get(name, spec["baseline"]), spec["baseline"])
        for name, spec in VARIABLES.items()
    }
    try:
        custom = _validate_rules(data.get("custom_rules")) if data.get("custom_rules") is not None else None
    except ValueError:
        custom = None  # a corrupted rules payload falls back to the shipped defaults
    state["custom_rules"] = custom
    rule_ids = {rule["id"] for rule in (custom if custom is not None else DEFAULT_RULES)}
    state["active_rules"] = sorted(
        {str(rid) for rid in (data.get("active_rules") or []) if str(rid) in rule_ids}
    )
    state["adjustments"] = [
        row for row in (data.get("adjustments") or []) if isinstance(row, dict)
    ][-config.HOMEOSTASIS_MAX_ADJUSTMENTS:]
    state["last_signals"] = data.get("last_signals") if isinstance(data.get("last_signals"), dict) else {}
    state["reason"] = str(data.get("reason") or state["reason"])[:240]
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


def _rules_for(state: dict) -> list[dict]:
    custom = state.get("custom_rules")
    return [dict(rule) for rule in (custom if custom is not None else DEFAULT_RULES)]


def _level(value: float) -> str:
    if value >= 0.6:
        return "expansive"
    if value >= 0.35:
        return "balanced"
    return "conservative"


def _posture_levels(variables: dict) -> dict[str, str]:
    return {name: _level(variables.get(name, VARIABLES[name]["baseline"])) for name in DERIVED}


# --- sensors -------------------------------------------------------------------

def _attention_load(attention_state: dict | None) -> float:
    domains = (attention_state or {}).get("domains") or {}
    top = max((_clamp(domains.get(domain)) for domain in _ATTENTION_LOAD_DOMAINS), default=0.0)
    return _clamp(top * _ATTENTION_LOAD_SCALE)


def _item_load(text: str) -> float | None:
    hours = [float(m.group(1)) for m in _HOURS.finditer(text)]
    if hours:
        full = max(1.0, config.HOMEOSTASIS_FULL_LOAD_HOURS)
        return _clamp(max(hours) / full)
    lowered = text.lower()
    for term, level in _QUALITATIVE_LOAD:
        if term in lowered:
            return level
    return None


def _world_load(world_items: list[dict] | None) -> float:
    best = 0.0
    for item in world_items or []:
        if not isinstance(item, dict) or item.get("expired"):
            continue
        key = str(item.get("key") or "")
        if not _LOAD_KEY.search(key):
            continue
        level = _item_load(f"{key} {item.get('value') or ''}")
        if level is None:
            continue
        confidence = _clamp(item.get("confidence", 0.7), 0.7)
        best = max(best, level * confidence)
    return _clamp(best)


def _sense(attention_state: dict | None, world_items: list[dict] | None) -> dict[str, dict]:
    from_attention = _attention_load(attention_state)
    from_world = _world_load(world_items)
    return {
        "user_load": {
            "attention": from_attention,
            "world": from_world,
            "combined": max(from_attention, from_world),
        }
    }


# --- state API -------------------------------------------------------------------

def state() -> dict:
    if not config.HOMEOSTASIS_ENABLED:
        return {"enabled": False}
    with _STATE_THREAD_LOCK:
        with _state_lock():
            current = _read_state_unlocked()
    current["posture_levels"] = _posture_levels(current["variables"])
    return {"enabled": True, **current}


def reset() -> dict:
    with _STATE_THREAD_LOCK:
        with _state_lock():
            fresh = _empty_state()
            fresh["updated_at"] = now_iso()
            _write_state_unlocked(fresh)
    return state()


def effective_rules() -> list[dict]:
    with _STATE_THREAD_LOCK:
        with _state_lock():
            return _rules_for(_read_state_unlocked())


def set_rules(rules: list[dict] | None) -> dict:
    """Replace the coupling rules (None restores the shipped defaults).

    This is the only way rules change — a deliberate API call, never inference —
    and the replacement itself is recorded in the adjustments log."""
    validated = _validate_rules(rules) if rules is not None else None
    with _STATE_THREAD_LOCK:
        with _state_lock():
            current = _read_state_unlocked()
            current["custom_rules"] = validated
            rule_ids = {rule["id"] for rule in _rules_for(current)}
            current["active_rules"] = [rid for rid in current["active_rules"] if rid in rule_ids]
            current["adjustments"].append(
                {
                    "at": now_iso(),
                    "event": "rules_replaced",
                    "note": (
                        "restored default rules"
                        if validated is None
                        else "custom rules set: " + ", ".join(sorted(rule_ids))
                    )[:240],
                }
            )
            current["reason"] = "coupling rules replaced via API"
            current["updated_at"] = now_iso()
            _write_state_unlocked(current)
    return state()


def update_from_turn(
    *,
    attention_state: dict | None = None,
    world_items: list[dict] | None = None,
) -> dict:
    """Sense, propagate coupling rules, persist; returns state plus this turn's rule events."""
    if not config.HOMEOSTASIS_ENABLED:
        return state()
    if attention_state is None:
        from brain import attention

        try:
            attention_state = attention.state()
        except Exception:
            attention_state = None
    if world_items is None:
        from brain import cognitive_state

        try:
            world_items = cognitive_state.state(include_inactive=False).get("world") or []
        except Exception:
            world_items = []
    signals = _sense(attention_state, world_items)
    now = now_iso()
    with _STATE_THREAD_LOCK:
        with _state_lock():
            current = _read_state_unlocked()
            variables = dict(current["variables"])
            blend = _clamp(config.HOMEOSTASIS_SIGNAL_BLEND, 0.35)
            # Sensed variables track their signal level (not accumulate it):
            # move partway toward the reading each turn, so one mention cannot
            # spike load but a sustained fact converges within a few turns.
            for name in SENSED:
                reading = signals.get(name, {}).get("combined", 0.0)
                variables[name] = _clamp(variables[name] + (reading - variables[name]) * blend)

            active = set(current["active_rules"])
            events: list[dict] = []
            effects = {name: 0.0 for name in DERIVED}
            for rule in _rules_for(current):
                source_value = variables.get(rule["source"], 0.0)
                was_active = rule["id"] in active
                if was_active and source_value < rule["release"]:
                    active.discard(rule["id"])
                    events.append({"event": "released", "rule": rule, "source_value": source_value})
                elif not was_active and source_value >= rule["threshold"]:
                    active.add(rule["id"])
                    events.append({"event": "fired", "rule": rule, "source_value": source_value})
                if rule["id"] in active:
                    effects[rule["target"]] += rule["weight"] * max(0.0, source_value - rule["release"])

            smoothing = _clamp(config.HOMEOSTASIS_SMOOTHING, 0.4)
            before = dict(variables)
            for name in DERIVED:
                target = _clamp(VARIABLES[name]["baseline"] + effects[name])
                variables[name] = _clamp(variables[name] + (target - variables[name]) * smoothing)

            turn_events = [
                {
                    "at": now,
                    "event": ev["event"],
                    "rule": ev["rule"]["id"],
                    "note": ev["rule"]["note"],
                    "source": ev["rule"]["source"],
                    "source_value": round(ev["source_value"], 4),
                    "target": ev["rule"]["target"],
                    "target_before": before[ev["rule"]["target"]],
                    "target_after": variables[ev["rule"]["target"]],
                }
                for ev in events
            ]
            updated = {
                **current,
                "updated_at": now,
                "variables": variables,
                "active_rules": sorted(active),
                "adjustments": (current["adjustments"] + turn_events)[-config.HOMEOSTASIS_MAX_ADJUSTMENTS:],
                "last_signals": signals,
                "reason": (
                    "; ".join(f"{ev['event']} {ev['rule']}" for ev in turn_events)
                    or "steady update from turn"
                )[:240],
            }
            _write_state_unlocked(updated)
    return {**state(), "turn_events": turn_events}


def gate_initiatives(items: list[dict] | None) -> tuple[list[dict], int]:
    """Trim proactive initiative candidates when the proactivity posture is conservative.

    Only suggestions are trimmed — reminders and arbiter mediation items always
    pass through — and the caller logs the dropped count, so suppression stays
    visible instead of becoming silent behavior drift."""
    rows = list(items or [])
    if not config.HOMEOSTASIS_ENABLED or not rows:
        return rows, 0
    current = state()
    if _level(current["variables"].get("proactivity", 1.0)) != "conservative":
        return rows, 0

    def keep(row: dict) -> bool:
        if str(row.get("id") or "").startswith("arbiter:"):
            return True
        try:
            priority = int(row.get("priority", row.get("importance", 3)))
        except (TypeError, ValueError):
            priority = 3
        return priority >= 5

    kept = [row for row in rows if keep(row)]
    return kept, len(rows) - len(kept)


def prompt_snapshot(current: dict | None = None) -> dict:
    if not config.HOMEOSTASIS_ENABLED:
        return {"enabled": False}
    current = current or state()
    variables = current.get("variables") or {}
    posture = []
    for name in DERIVED:
        level = _level(variables.get(name, VARIABLES[name]["baseline"]))
        posture.append(
            {
                "variable": name,
                "value": variables.get(name),
                "level": level,
                "guidance": _POSTURE_GUIDANCE[name][level],
            }
        )
    active_ids = set(current.get("active_rules") or [])
    rules = current.get("custom_rules") if current.get("custom_rules") is not None else DEFAULT_RULES
    return {
        "enabled": True,
        "variables": variables,
        "posture": posture,
        "active_rules": [
            {"id": rule["id"], "note": rule["note"]} for rule in rules if rule["id"] in active_ids
        ],
        "recent_adjustments": (current.get("adjustments") or [])[-3:],
        "guidance": (
            "Derived operating posture coupling tracked life load to working style. "
            "Use it only to modulate depth of discretionary analysis and proactive pacing. "
            "It is not evidence and never overrides Source of Truth, current instructions, "
            "or retrieved facts."
        ),
    }
