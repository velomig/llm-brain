"""Deterministic dynamic attention router.

Attention is a lightweight focus layer: it tracks which life/system domains are
currently salient and uses those scores to rank already-retrieved context. It is
not a source of truth and does not create relevance by itself.
"""
from __future__ import annotations

import json
import os
import re
import threading
import uuid

import config
from brain.utils import file_lock
from brain import evolution
from brain.memory import now_iso

SCHEMA_VERSION = 1
DOMAINS = ("finance", "nursing", "tax_home", "benefits", "system_maintenance", "memory", "agency")
_STATE_FILE = config.DB_DIR / "attention_state.json"
_STATE_THREAD_LOCK = threading.Lock()

_DOMAIN_TERMS = {
    "finance": {
        "allocation", "coin", "buy", "exco", "dividend", "etf", "fidelity", "finance",
        "fndy", "holding", "market", "money", "portfolio", "price", "fndx", "fndz",
        "shares", "acme", "stock", "trade", "ticker",
    },
    "nursing": {
        "clinical", "exam", "lab", "medication", "nclex", "nurse", "nursing", "patient",
        "rn", "school", "semester", "study",
    },
    "tax_home": {
        "cpa", "housing", "relocation", "tax", "taxes", "tax-home",
        "taxhome", "travel", "traveling",
    },
    "benefits": {
        "benefit", "benefits", "disability", "medicare", "ssa", "ssdi", "trial", "va",
        "veteran", "work", "workload",
    },
    "system_maintenance": {
        "api", "app", "bug", "code", "config", "fix", "health", "implement", "python",
        "server", "test", "tests", "tool", "web",
    },
    "memory": {
        "context", "fact", "memory", "memories", "pinned", "precision", "recall",
        "remember", "retrieval", "source", "truth",
    },
    "agency": {
        "action", "agency", "approve", "attention", "autonomy", "package", "proactive",
        "review", "trigger",
    },
}

_CATEGORY_DOMAIN_HINTS = {
    "initiative": "agency",
    "reminder": "agency",
    "user_preference": "memory",
    "action_package": "agency",
}
_PROTECTED_MEMORY_DOMAINS = {"finance", "nursing"}


def _state_lock():
    """Cross-process exclusive lock on this module's state file."""
    return file_lock(_STATE_FILE.with_name(".neural_map_attention_state.lock"))


def _empty_state() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "enabled": config.ATTENTION_ENABLED,
        "updated_at": None,
        "domains": {domain: 0.0 for domain in DOMAINS},
        "focused_domain": "general",
        "reason": "neutral baseline",
        "last_signals": {},
    }


def _clamp(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return round(min(1.0, max(0.0, parsed)), 4)


def _tok(text: str) -> list[str]:
    return re.findall(r"[\w-]+", text.lower())


def _text_for_item(item: dict | None) -> str:
    item = item or {}
    return " ".join(
        str(part or "")
        for part in (
            item.get("text"),
            item.get("category"),
            item.get("kind"),
            item.get("source"),
            item.get("origin"),
        )
    )


def _repair_state(data: object) -> dict:
    state = _empty_state()
    if not isinstance(data, dict):
        return state
    raw_domains = data.get("domains") if isinstance(data.get("domains"), dict) else {}
    state["updated_at"] = data.get("updated_at")
    state["domains"] = {domain: _clamp(raw_domains.get(domain), 0.0) for domain in DOMAINS}
    state["focused_domain"] = _focused_domain(state["domains"])
    state["reason"] = str(data.get("reason") or state["reason"])[:240]
    state["last_signals"] = {
        domain: _clamp(value, 0.0)
        for domain, value in (data.get("last_signals") or {}).items()
        if domain in DOMAINS
    }
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


def _focused_domain(domains: dict[str, float]) -> str:
    if not domains:
        return "general"
    domain, score = max(domains.items(), key=lambda item: (item[1], item[0]))
    return domain if score >= config.ATTENTION_FOCUS_THRESHOLD else "general"


def _domain_terms() -> dict[str, set[str]]:
    """Built-in domain vocabulary plus approved learned terms.

    The static _DOMAIN_TERMS table is the router's blind spot: a topic outside
    it scores "general" forever. Learned terms arrive only through the evolution
    propose→approve pipeline (evolution.learned_attention_terms), so vocabulary
    grows with human sign-off and reverts with evolution snapshots/rollback.
    Fetch once per scoring pass and pass down — it reads the evolution state
    file under a lock."""
    merged = {domain: set(terms) for domain, terms in _DOMAIN_TERMS.items()}
    try:
        learned = evolution.learned_attention_terms()
    except Exception:
        learned = {}
    for domain, terms in (learned or {}).items():
        if domain in merged:
            merged[domain].update(term for term in terms if isinstance(term, str))
    return merged


def _score_text(text: str, domain_terms: dict[str, set[str]] | None = None) -> dict[str, float]:
    domain_terms = domain_terms if domain_terms is not None else _domain_terms()
    tokens = set(_tok(text))
    if not tokens:
        return {domain: 0.0 for domain in DOMAINS}
    scores: dict[str, float] = {}
    for domain, terms in domain_terms.items():
        hits = len(tokens & terms)
        scores[domain] = _clamp(hits / max(3, min(10, len(terms))), 0.0)
    return scores


def _merge_signal(target: dict[str, float], text: str, weight: float, domain_terms: dict[str, set[str]] | None = None) -> None:
    for domain, score in _score_text(text, domain_terms).items():
        target[domain] = min(1.0, target.get(domain, 0.0) + score * weight)


def score_domains(
    user_text: str,
    *,
    mems: list[dict] | None = None,
    reminders: list[dict] | None = None,
    initiatives: list[dict] | None = None,
    tools: list[str] | None = None,
) -> dict[str, float]:
    signal = {domain: 0.0 for domain in DOMAINS}
    domain_terms = _domain_terms()
    _merge_signal(signal, user_text, 1.0, domain_terms)
    for row in mems or []:
        memory_scores = _score_text(_text_for_item(row), domain_terms)
        for domain, score in memory_scores.items():
            if domain in _PROTECTED_MEMORY_DOMAINS and signal.get(domain, 0.0) <= 0:
                continue
            signal[domain] = min(1.0, signal.get(domain, 0.0) + score * config.ATTENTION_MEMORY_SIGNAL_WEIGHT)
        category = str((row or {}).get("category") or "").lower()
        hinted = _CATEGORY_DOMAIN_HINTS.get(category)
        if hinted:
            signal[hinted] = min(1.0, signal[hinted] + 0.08)
    for row in reminders or []:
        _merge_signal(signal, _text_for_item(row), config.ATTENTION_PROACTIVE_SIGNAL_WEIGHT, domain_terms)
    for row in initiatives or []:
        _merge_signal(signal, _text_for_item(row), config.ATTENTION_PROACTIVE_SIGNAL_WEIGHT, domain_terms)
    for tool in tools or []:
        if tool in {"get_stock_quotes"}:
            signal["finance"] = min(1.0, signal["finance"] + 0.12)
        if tool in {"read_file", "write_file", "run_tests", "git_diff", "git_status"}:
            signal["system_maintenance"] = min(1.0, signal["system_maintenance"] + 0.12)
    return {domain: _clamp(score, 0.0) for domain, score in signal.items()}


def state() -> dict:
    if not config.ATTENTION_ENABLED:
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


def update_from_turn(
    user_text: str,
    *,
    mems: list[dict] | None = None,
    reminders: list[dict] | None = None,
    initiatives: list[dict] | None = None,
    tools: list[str] | None = None,
) -> dict:
    if not config.ATTENTION_ENABLED:
        return state()
    signals = score_domains(
        user_text,
        mems=mems,
        reminders=reminders,
        initiatives=initiatives,
        tools=tools,
    )
    decay = _clamp(config.ATTENTION_DECAY, 0.85)
    blend = _clamp(config.ATTENTION_SIGNAL_BLEND, 0.5)
    now = now_iso()
    with _STATE_THREAD_LOCK:
        with _state_lock():
            current = _read_state_unlocked()
            # Momentum blend, not max(): each turn's signal is ADDED at reduced
            # weight onto the decayed score, so one off-topic message cannot
            # instantly displace an established focus — a genuine topic shift
            # wins after ~2-3 sustained turns as the old focus decays and the
            # new one accumulates. An unsignaled domain still decays exactly as
            # before (old * decay + 0), and a cold start still focuses on the
            # first clear message because blend * signal comfortably clears
            # ATTENTION_FOCUS_THRESHOLD.
            domains = {
                domain: _clamp(
                    current["domains"].get(domain, 0.0) * decay
                    + signals.get(domain, 0.0) * blend
                )
                for domain in DOMAINS
            }
            updated = {
                **current,
                "updated_at": now,
                "domains": domains,
                "focused_domain": _focused_domain(domains),
                "reason": f"updated from current turn: {user_text[:160]}",
                "last_signals": signals,
            }
            _write_state_unlocked(updated)
    return state()


def item_domains(item: dict, domain_terms: dict[str, set[str]] | None = None) -> dict[str, float]:
    text = _text_for_item(item)
    scores = _score_text(text, domain_terms)
    category = str((item or {}).get("category") or "").lower()
    hinted = _CATEGORY_DOMAIN_HINTS.get(category)
    if hinted:
        scores[hinted] = min(1.0, scores[hinted] + 0.2)
    return scores


def attention_score(item: dict, current: dict | None = None, domain_terms: dict[str, set[str]] | None = None) -> float:
    if not config.ATTENTION_ENABLED:
        return 0.0
    current = current or state()
    domains = current.get("domains") or {}
    scores = item_domains(item, domain_terms)
    return round(max((domains.get(domain, 0.0) * score for domain, score in scores.items()), default=0.0), 4)


def rank_memories(user_text: str, mems: list[dict] | None) -> list[dict]:
    rows = list(mems or [])
    if not config.ATTENTION_ENABLED or len(rows) < 2:
        return rows
    current = state()
    turn_signal = score_domains(user_text)
    domain_terms = _domain_terms()
    allowed_protected = {
        domain for domain in _PROTECTED_MEMORY_DOMAINS if turn_signal.get(domain, 0.0) > 0
    }

    def static_attention(row):
        scores = item_domains(row, domain_terms)
        return round(
            max(
                (
                    current.get("domains", {}).get(domain, 0.0) * score
                    for domain, score in scores.items()
                    if domain not in _PROTECTED_MEMORY_DOMAINS or domain in allowed_protected
                ),
                default=0.0,
            ),
            4,
        )

    def key(pair):
        idx, row = pair
        static = static_attention(row)
        turn = max(
            turn_signal.get(domain, 0.0) * score
            for domain, score in item_domains(row, domain_terms).items()
        )
        return (round(static + turn, 4), 1 if row.get("pinned") else 0, -idx)

    return [row for _, row in sorted(enumerate(rows), key=key, reverse=True)]


def rank_initiatives(items: list[dict] | None, user_text: str | None = None) -> list[dict]:
    rows = list(items or [])
    if not config.ATTENTION_ENABLED or len(rows) < 2:
        return rows
    current = state()
    turn_signal = score_domains(user_text or "")
    domain_terms = _domain_terms()

    def key(pair):
        idx, row = pair
        priority = float(row.get("priority") or 0)
        static = attention_score(row, current, domain_terms)
        turn = max(
            turn_signal.get(domain, 0.0) * score
            for domain, score in item_domains(row, domain_terms).items()
        )
        return (priority, round(static + turn, 4), row.get("date") or "", -idx)

    return [row for _, row in sorted(enumerate(rows), key=key, reverse=True)]


def prompt_snapshot(current: dict | None = None) -> dict:
    if not config.ATTENTION_ENABLED:
        return {"enabled": False}
    current = current or state()
    domains = current.get("domains") or {}
    top = sorted(domains.items(), key=lambda item: item[1], reverse=True)[:4]
    return {
        "enabled": True,
        "focused_domain": current.get("focused_domain"),
        "top_domains": [{"domain": domain, "score": score} for domain, score in top],
        "last_signals": current.get("last_signals") or {},
        "guidance": (
            "Use attention only to prioritize relevant context and proactive timing. "
            "Do not treat high attention as proof that a fact is true."
        ),
    }
