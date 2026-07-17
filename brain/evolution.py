"""Recursive self-improvement with a human approval gate.

The model can propose prompt and hyperparameter mutations from its evolution log
and Wisdom memories. Nothing is deployed until an explicit approve call writes
the proposal into the active runtime overlay.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import re
import threading
import uuid

import ollama

import config
from brain.utils import file_lock
from brain import llm, log, store
from brain.utils import safe_json as _safe_json

_STATE_FILE = config.DB_DIR / "evolution_state.json"
_LOCK = threading.Lock()

PROMPT_KEYS = {"preamble", "tools_note", "inject_note"}
ALLOWED_HYPERPARAMS: dict[str, tuple[type, float | int, float | int]] = {
    "TOP_K": (int, 1, 25),
    "CANDIDATE_K": (int, 5, 200),
    "HISTORY_TURNS": (int, 1, 20),
    "W_SIM": (float, 0, 5),
    "W_KEYWORD": (float, 0, 5),
    "W_RECENCY": (float, 0, 5),
    "W_IMPORTANCE": (float, 0, 5),
    "W_GRAPH": (float, 0, 5),
    "W_SENSORY": (float, 0, 5),
    "INTUITION_FAST_MAX_WORDS": (int, 8, 120),
    "INNER_COUNCIL_TURNS": (int, 1, 5),
    "METACOGNITION_RECENT_K": (int, 0, 20),
}
CANARY_AREAS = (
    "simple_factual",
    "casual_conversation",
    "memory_recall",
    "document_grounding",
    "math_and_tools",
    "safety_refusal",
    "creative_open_ended",
    "uncertainty_calibration",
)
CANARY_EXPECTATIONS = {"preserve", "improve"}

_SYSTEM = """You are Neural Map's evolution proposer. You may propose changes to
the assistant's runtime system prompts and a small whitelist of hyperparameters.

Use the evolution log, Wisdom notes, current active mutation, and current prompt
as evidence. Propose only changes that are concrete, safer, and likely to improve
future behavior. Do not optimize for flattery, power-seeking, user manipulation,
or bypassing safety/approval. Do not propose source-code edits.

Respond with STRICT JSON:
{
  "title": "<short proposal title>",
  "rationale": "<why this should improve behavior>",
  "target_failure": "<specific failure pattern this proposal addresses>",
  "expected_tradeoff": "<what could get worse, or 'none expected'>",
  "prompt_overrides": {
    "preamble": "<optional full replacement>",
    "tools_note": "<optional full replacement>",
    "inject_note": "<optional full replacement>"
  },
  "hyperparameters": {"TOP_K": 5},
  "attention_terms": {"nursing": ["pharmacology", "cardiology"]},
  "canary_impact": [
    {"area": "simple_factual", "expected": "preserve|improve|risk", "notes": "<why>"},
    {"area": "casual_conversation", "expected": "preserve|improve|risk", "notes": "<why>"},
    {"area": "memory_recall", "expected": "preserve|improve|risk", "notes": "<why>"},
    {"area": "document_grounding", "expected": "preserve|improve|risk", "notes": "<why>"},
    {"area": "math_and_tools", "expected": "preserve|improve|risk", "notes": "<why>"},
    {"area": "safety_refusal", "expected": "preserve|improve|risk", "notes": "<why>"},
    {"area": "creative_open_ended", "expected": "preserve|improve|risk", "notes": "<why>"},
    {"area": "uncertainty_calibration", "expected": "preserve|improve|risk", "notes": "<why>"}
  ],
  "risk_notes": ["<risk or monitoring note>"]
}

Only include prompt keys you actually want to replace. Only include whitelisted
hyperparameters. attention_terms may only ADD lowercase single-word tokens to the
EXISTING attention domains listed in the payload — propose them when the log or
Wisdom notes show the attention router missing focus on a recurring topic (its
built-in vocabulary is static); never invent new domains and never remove terms.
Mark canary_impact as "risk" if the mutation may narrow behavior
or overfit to recent failures in that area. If no useful mutation is warranted, return:
{"title":"No mutation recommended","rationale":"...","target_failure":"none","expected_tradeoff":"none",
 "prompt_overrides":{},"hyperparameters":{},"canary_impact":[],"risk_notes":[]}
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_state() -> dict:
    return {
        "version": 1,
        "active": {
            "proposal_id": None,
            "approved_at": None,
            "prompt_overrides": {},
            "hyperparameters": {},
            "attention_terms": {},
            "snapshot_id": None,
        },
        "snapshots": [],
        "proposals": [],
    }


def _state_lock():
    """Cross-process exclusive lock on this module's state file."""
    return file_lock(_STATE_FILE.with_name(".neural_map_evolution_state.lock"))


def _read_state_unlocked() -> dict:
    try:
        data = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return _empty_state()
    if not isinstance(data, dict):
        return _empty_state()
    base = _empty_state()
    base.update(data)
    base.setdefault("active", _empty_state()["active"])
    base.setdefault("proposals", [])
    base.setdefault("snapshots", [])
    return base


def _write_state_unlocked(state: dict) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STATE_FILE.with_name(f"{_STATE_FILE.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, _STATE_FILE)


def state() -> dict:
    with _LOCK:
        with _state_lock():
            return _read_state_unlocked()




def _read_log() -> str:
    try:
        text = config.EVOLUTION_LOG_PATH.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return "(no evolution_log.txt yet)"
    return text[-config.EVOLUTION_MAX_LOG_CHARS :]


def append_log(entry: str) -> dict:
    text = (entry or "").strip()
    if not text:
        return {"written": False}
    config.EVOLUTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"{_now()} {text}\n"
    encoded = line.encode("utf-8")
    with _LOCK:
        with _state_lock():
            try:
                current_size = config.EVOLUTION_LOG_PATH.stat().st_size
            except FileNotFoundError:
                current_size = 0
            if current_size + len(encoded) > config.EVOLUTION_MAX_LOG_BYTES:
                return {
                    "written": False,
                    "error": f"evolution log is at its configured size limit ({config.EVOLUTION_MAX_LOG_BYTES} bytes)",
                }
            with config.EVOLUTION_LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(line)
    return {"written": True, "path": str(config.EVOLUTION_LOG_PATH)}


def _wisdom(model: str | None = None) -> list[dict]:
    rows = [
        rec
        for rec in store.list_records_all(model)
        if (rec.get("meta") or {}).get("category") == config.METACOGNITION_CATEGORY
        and not (rec.get("meta") or {}).get("superseded")
    ]
    rows.sort(key=lambda r: ((r.get("meta") or {}).get("date") or "", r.get("id") or ""), reverse=True)
    return [
        {
            "text": rec.get("text", ""),
            "date": (rec.get("meta") or {}).get("date"),
            "bias": (rec.get("meta") or {}).get("bias"),
        }
        for rec in rows[: config.EVOLUTION_MAX_WISDOM]
    ]


def _clean_prompt_overrides(raw: dict) -> dict:
    out = {}
    for key, value in (raw or {}).items():
        if key not in PROMPT_KEYS or not isinstance(value, str):
            continue
        value = value.strip()
        if not value:
            continue
        out[key] = value[: config.EVOLUTION_MAX_PROMPT_CHARS]
    return out


def _clean_hyperparameters(raw: dict) -> dict:
    out = {}
    for key, value in (raw or {}).items():
        if key not in ALLOWED_HYPERPARAMS:
            continue
        typ, lo, hi = ALLOWED_HYPERPARAMS[key]
        try:
            parsed = typ(value)
        except (TypeError, ValueError):
            continue
        if parsed < lo or parsed > hi:
            continue
        out[key] = parsed
    return out


# A learned attention term must look like one lowercase token the router's
# tokenizer could actually produce.
_ATTENTION_TERM_RE = re.compile(r"^[a-z0-9][a-z0-9_.+-]{1,31}$")


def _clean_attention_terms(raw) -> dict[str, list[str]]:
    """Validated vocabulary additions for the attention router.

    Only known attention domains are accepted, terms must be single lowercase
    tokens, built-in terms are dropped as redundant, and each domain is capped
    (config.ATTENTION_MAX_LEARNED_TERMS_PER_DOMAIN) so a runaway proposal can't
    flood the router's vocabulary."""
    # Lazy import: attention reads learned terms from this module at score time,
    # so evolution must not import attention at module load.
    from brain.attention import _DOMAIN_TERMS, DOMAINS

    out: dict[str, list[str]] = {}
    if not isinstance(raw, dict):
        return out
    for domain, terms in raw.items():
        if domain not in DOMAINS or not isinstance(terms, list):
            continue
        builtin = _DOMAIN_TERMS.get(domain, set())
        cleaned: list[str] = []
        for term in terms:
            if not isinstance(term, str):
                continue
            token = term.strip().lower()
            if not _ATTENTION_TERM_RE.match(token) or token in builtin or token in cleaned:
                continue
            cleaned.append(token)
            if len(cleaned) >= config.ATTENTION_MAX_LEARNED_TERMS_PER_DOMAIN:
                break
        if cleaned:
            out[domain] = cleaned
    return out


def learned_attention_terms() -> dict[str, list[str]]:
    """Approved attention-vocabulary additions from the active runtime overlay."""
    active = state().get("active", {}) or {}
    return _clean_attention_terms(active.get("attention_terms") or {})


def _attention_domains_payload() -> dict:
    from brain.attention import _DOMAIN_TERMS, DOMAINS

    return {
        "domains": list(DOMAINS),
        "built_in_terms": {domain: sorted(_DOMAIN_TERMS.get(domain, set())) for domain in DOMAINS},
        "learned_terms": learned_attention_terms(),
        "max_new_terms_per_domain": config.ATTENTION_MAX_LEARNED_TERMS_PER_DOMAIN,
    }


def _clean_canary_impact(raw) -> list[dict]:
    rows: list[dict] = []
    if not isinstance(raw, list):
        return rows
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        area = str(item.get("area") or "").strip()
        if area not in CANARY_AREAS or area in seen:
            continue
        expected = str(item.get("expected") or "").strip().lower()
        if expected not in {"preserve", "improve", "risk"}:
            expected = "risk"
        rows.append(
            {
                "area": area,
                "expected": expected,
                "notes": str(item.get("notes") or "")[:500],
            }
        )
        seen.add(area)
    return rows


def _canary_review(proposal: dict) -> dict:
    if not (
        proposal.get("prompt_overrides")
        or proposal.get("hyperparameters")
        or proposal.get("attention_terms")
    ):
        return {"ok": True, "failures": []}
    by_area = {
        item.get("area"): item
        for item in proposal.get("canary_impact") or []
        if isinstance(item, dict)
    }
    failures: list[str] = []
    for area in CANARY_AREAS:
        item = by_area.get(area)
        if item is None:
            failures.append(f"missing canary impact for {area}")
            continue
        expected = item.get("expected")
        if expected not in CANARY_EXPECTATIONS:
            failures.append(f"{area} marked {expected or 'unknown'}")
    return {"ok": not failures, "failures": failures}


def _prompt_hashes(prompt_parts: dict[str, str] | None) -> dict[str, str]:
    return {
        key: hashlib.sha256((prompt_parts or {}).get(key, "").encode("utf-8")).hexdigest()[:16]
        for key in sorted(PROMPT_KEYS)
    }


def _feature_flags() -> dict:
    return {
        "cognition_mode": config.COGNITION_MODE,
        "intuition": config.INTUITION_ENABLED,
        "inner_council": config.INNER_COUNCIL_ENABLED,
        "metacognition": config.METACOGNITION_ENABLED,
        "emotion": config.EMOTION_ENABLED,
        "initiative": config.INITIATIVE_ENABLED,
        "tool_calling": config.TOOL_CALLING_ENABLED,
        "action_tools": config.ACTION_TOOLS_ENABLED,
        "web_search": config.WEB_SEARCH_ENABLED,
        "market_quotes": config.MARKET_QUOTES_ENABLED,
    }


def _wisdom_fingerprint(model: str | None = None) -> dict:
    rows = _wisdom(model=model)
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]
    return {"count": len(rows), "hash": digest}


def _snapshot_payload(
    current: dict,
    *,
    reason: str,
    prompt_parts: dict[str, str] | None = None,
    proposal_id: str | None = None,
    model: str | None = None,
) -> dict:
    active = current.get("active") or {}
    payload = {
        "version": 1,
        "created_at": _now(),
        "reason": reason[:240],
        "proposal_id": proposal_id or active.get("proposal_id"),
        "active": {
            "proposal_id": active.get("proposal_id"),
            "approved_at": active.get("approved_at"),
            "title": active.get("title"),
            "prompt_overrides": dict(active.get("prompt_overrides") or {}),
            "hyperparameters": dict(active.get("hyperparameters") or {}),
            "attention_terms": dict(active.get("attention_terms") or {}),
        },
        "prompt_hashes": _prompt_hashes(prompt_parts),
        "feature_flags": _feature_flags(),
        "wisdom": _wisdom_fingerprint(model=model),
    }
    stable = dict(payload)
    stable.pop("created_at", None)
    snapshot_hash = hashlib.sha256(json.dumps(stable, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:20]
    payload["id"] = f"cog-{snapshot_hash}"
    return payload


def _append_snapshot_unlocked(current: dict, snapshot: dict) -> None:
    snapshots = list(current.get("snapshots") or [])
    snapshots = [s for s in snapshots if s.get("id") != snapshot["id"]]
    snapshots.append(snapshot)
    current["snapshots"] = snapshots[-50:]


def _active_from_snapshot(snapshot: dict, *, restored: bool = False) -> dict | None:
    if not isinstance(snapshot, dict) or not snapshot.get("id"):
        return None
    raw = snapshot.get("active")
    if not isinstance(raw, dict):
        return None
    active = {
        **_empty_state()["active"],
        "proposal_id": raw.get("proposal_id"),
        "approved_at": raw.get("approved_at"),
        "title": raw.get("title"),
        "prompt_overrides": _clean_prompt_overrides(raw.get("prompt_overrides") if isinstance(raw.get("prompt_overrides"), dict) else {}),
        "hyperparameters": _clean_hyperparameters(raw.get("hyperparameters") if isinstance(raw.get("hyperparameters"), dict) else {}),
        "attention_terms": _clean_attention_terms(raw.get("attention_terms")),
        "snapshot_id": snapshot["id"],
    }
    if restored:
        active["restored_at"] = _now()
        active["restored_from"] = snapshot["id"]
    return active


def _active_has_runtime_config(active: dict | None) -> bool:
    active = active or {}
    return bool(
        active.get("prompt_overrides")
        or active.get("hyperparameters")
        or active.get("attention_terms")
    )


def _rollback_snapshot_unlocked(current: dict) -> dict | None:
    snapshots = list(current.get("snapshots") or [])
    if not snapshots:
        return None
    active = current.get("active") or {}
    active_snapshot_id = active.get("snapshot_id")
    active_proposal_id = active.get("proposal_id")
    proposal = _find_proposal(current, active_proposal_id) if active_proposal_id else None
    proposal_snapshots = (proposal or {}).get("snapshots") or {}
    before_id = proposal_snapshots.get("before")
    if before_id:
        found = next((s for s in snapshots if s.get("id") == before_id), None)
        if found:
            return found
    if active_proposal_id:
        for snapshot in reversed(snapshots):
            if snapshot.get("proposal_id") == active_proposal_id and str(snapshot.get("reason") or "").startswith("before approving"):
                return snapshot
    if active_snapshot_id:
        for idx, snapshot in enumerate(snapshots):
            if snapshot.get("id") == active_snapshot_id and idx > 0:
                return snapshots[idx - 1]
    for snapshot in reversed(snapshots):
        if snapshot.get("id") != active_snapshot_id:
            return snapshot
    return None


def snapshots() -> list[dict]:
    with _LOCK:
        with _state_lock():
            current = _read_state_unlocked()
            return list(current.get("snapshots") or [])


def current_snapshot_id() -> str:
    return (state().get("active") or {}).get("snapshot_id") or "unsnapshotted"


def restore_snapshot(snapshot_id: str) -> dict:
    with _LOCK:
        with _state_lock():
            current = _read_state_unlocked()
            snapshot = next((s for s in current.get("snapshots", []) if s.get("id") == snapshot_id), None)
            if snapshot is None:
                return {"error": "snapshot not found"}
            active = _active_from_snapshot(snapshot, restored=True)
            if active is None:
                return {"error": "snapshot is invalid"}
            current["active"] = active
            _write_state_unlocked(current)
            return {"restored": True, "snapshot_id": snapshot["id"], "active": current["active"]}


def prompt_parts(defaults: dict[str, str]) -> dict[str, str]:
    active = state().get("active", {})
    out = dict(defaults)
    for key, value in (active.get("prompt_overrides") or {}).items():
        if key in PROMPT_KEYS and isinstance(value, str) and value.strip():
            out[key] = value
    return out


def hyperparam(name: str, default):
    if name not in ALLOWED_HYPERPARAMS:
        return default
    value = (state().get("active", {}).get("hyperparameters") or {}).get(name)
    if value is None:
        return default
    cleaned = _clean_hyperparameters({name: value})
    return cleaned.get(name, default)


def propose(current_prompt: dict[str, str], model: str | None = None) -> dict:
    if not config.EVOLUTION_ENABLED:
        return {"error": "evolution is disabled"}
    payload = {
        "evolution_log": _read_log(),
        "wisdom": _wisdom(model=model),
        "active": state().get("active", {}),
        "current_prompt": {k: current_prompt.get(k, "") for k in sorted(PROMPT_KEYS)},
        "allowed_hyperparameters": sorted(ALLOWED_HYPERPARAMS),
        "attention_router": _attention_domains_payload(),
        "canary_areas": list(CANARY_AREAS),
        "approval_policy": (
            "Approval is blocked unless every canary area is present and marked "
            "preserve or improve. Any risk/unknown/missing canary impact requires "
            "manual redesign before approval."
        ),
    }
    try:
        raw = llm.chat(_SYSTEM, "Draft a human-reviewable mutation proposal:\n" + json.dumps(payload, ensure_ascii=True), model=model)
    except Exception as exc:
        log.fallback("evolution_failed", exc, model=model, phase="propose")
        out = {"error": str(exc)}
        if isinstance(exc, (ConnectionError, TimeoutError, ollama.RequestError, ollama.ResponseError)):
            out["error_kind"] = "ollama_unavailable"
        return out
    data = _safe_json(raw)
    proposal = {
        "id": uuid.uuid4().hex,
        "created_at": _now(),
        "status": "pending",
        "title": str(data.get("title") or "Untitled mutation")[:160],
        "rationale": str(data.get("rationale") or "")[:4000],
        "target_failure": str(data.get("target_failure") or "")[:1000],
        "expected_tradeoff": str(data.get("expected_tradeoff") or "")[:1000],
        "prompt_overrides": _clean_prompt_overrides(data.get("prompt_overrides") if isinstance(data.get("prompt_overrides"), dict) else {}),
        "hyperparameters": _clean_hyperparameters(data.get("hyperparameters") if isinstance(data.get("hyperparameters"), dict) else {}),
        "attention_terms": _clean_attention_terms(data.get("attention_terms")),
        "canary_impact": _clean_canary_impact(data.get("canary_impact")),
        "risk_notes": [str(x)[:500] for x in (data.get("risk_notes") or []) if isinstance(x, str)][:10],
    }
    proposal["approval_checks"] = {"canary_review": _canary_review(proposal)}
    with _LOCK:
        with _state_lock():
            current = _read_state_unlocked()
            current["proposals"].append(proposal)
            _write_state_unlocked(current)
    return proposal


def _find_proposal(current: dict, proposal_id: str) -> dict | None:
    for proposal in current.get("proposals", []):
        if proposal.get("id") == proposal_id:
            return proposal
    return None


def approve(proposal_id: str, prompt_parts: dict[str, str] | None = None, model: str | None = None) -> dict:
    with _LOCK:
        with _state_lock():
            current = _read_state_unlocked()
            proposal = _find_proposal(current, proposal_id)
            if proposal is None:
                return {"error": "proposal not found"}
            if proposal.get("status") != "pending":
                return {"error": f"proposal is {proposal.get('status')}"}
            review = _canary_review(proposal)
            proposal["approval_checks"] = {"canary_review": review}
            if not review["ok"]:
                _write_state_unlocked(current)
                return {"error": "canary review failed", "canary_review": review}
            before = _snapshot_payload(
                current,
                reason=f"before approving {proposal_id}",
                prompt_parts=prompt_parts,
                proposal_id=proposal_id,
                model=model,
            )
            _append_snapshot_unlocked(current, before)
            proposal["status"] = "approved"
            proposal["approved_at"] = _now()
            proposal["snapshots"] = {"before": before["id"]}
            current["active"] = {
                "proposal_id": proposal["id"],
                "approved_at": proposal["approved_at"],
                "title": proposal.get("title"),
                "prompt_overrides": proposal.get("prompt_overrides") or {},
                "hyperparameters": proposal.get("hyperparameters") or {},
                "attention_terms": proposal.get("attention_terms") or {},
                "rollback_snapshot_id": before["id"],
            }
            after = _snapshot_payload(
                current,
                reason=f"after approving {proposal_id}",
                prompt_parts=prompt_parts,
                proposal_id=proposal_id,
                model=model,
            )
            _append_snapshot_unlocked(current, after)
            proposal["snapshots"]["after"] = after["id"]
            current["active"]["snapshot_id"] = after["id"]
            _write_state_unlocked(current)
            return {"approved": True, "active": current["active"], "snapshots": {"before": before["id"], "after": after["id"]}}


def reject(proposal_id: str, reason: str | None = None) -> dict:
    with _LOCK:
        with _state_lock():
            current = _read_state_unlocked()
            proposal = _find_proposal(current, proposal_id)
            if proposal is None:
                return {"error": "proposal not found"}
            if proposal.get("status") != "pending":
                return {"error": f"proposal is {proposal.get('status')}"}
            proposal["status"] = "rejected"
            proposal["rejected_at"] = _now()
            proposal["rejection_reason"] = (reason or "")[:1000]
            _write_state_unlocked(current)
            return {"rejected": True}


def rollback(auto_revert_on_empty: bool = True) -> dict:
    with _LOCK:
        with _state_lock():
            current = _read_state_unlocked()
            previous_active = dict(current.get("active") or _empty_state()["active"])
            active_id = previous_active.get("proposal_id")
            snapshot = _rollback_snapshot_unlocked(current)
            if snapshot is None:
                return {"error": "rollback snapshot not found", "active": current.get("active") or _empty_state()["active"]}
            restored_active = _active_from_snapshot(snapshot, restored=True)
            if restored_active is None:
                return {"error": "rollback snapshot is invalid", "active": current.get("active") or _empty_state()["active"]}
            if (
                auto_revert_on_empty
                and _active_has_runtime_config(previous_active)
                and not _active_has_runtime_config(restored_active)
            ):
                current["active"] = {
                    **previous_active,
                    "auto_reverted_at": _now(),
                    "auto_revert_reason": "rollback produced empty active state",
                }
                _write_state_unlocked(current)
                return {
                    "error": "rollback produced empty active state",
                    "auto_reverted": True,
                    "attempted_snapshot_id": snapshot["id"],
                    "active": current["active"],
                }
            proposal = _find_proposal(current, active_id) if active_id else None
            if proposal and proposal.get("status") == "approved":
                proposal["status"] = "pending"
                proposal["rolled_back_at"] = _now()
            current["active"] = restored_active
            _write_state_unlocked(current)
            return {"rolled_back": True, "snapshot_id": snapshot["id"], "active": current["active"]}


def proposal_preview(proposal: dict, defaults: dict[str, str]) -> dict:
    parts = dict(defaults)
    parts.update(proposal.get("prompt_overrides") or {})
    return {k: parts.get(k, "") for k in sorted(PROMPT_KEYS)}
