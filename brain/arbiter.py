"""Arbiter layer: deterministic mediation and proactive admissibility.

Will, Reflection, and consolidation can each produce useful nudges, but they can
also point in opposite directions. The arbiter does not mutate stored memories;
it resolves the presentation set for a turn so the assistant surfaces one
explicit conflict instead of two competing instructions. Grounding is an
independent axis: generated initiatives that have only linguistic support (or
no provenance) are converted into evidence-review questions before surfacing.
Operational risk is derived separately from instruction text, so grounding or
origin metadata cannot downgrade a destructive or external side effect.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

import config
from brain import log
from brain.embeddings import EmbeddingUnavailable, embed_documents

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_'-]+")
_STOP = {
    "about", "again", "before", "could", "from", "have", "into", "should",
    "that", "the", "this", "want", "what", "when", "with", "would", "your",
}
_ADVANCE = {
    "accept", "approve", "begin", "buy", "commit", "continue", "execute",
    "finish", "increase", "launch", "proceed", "resume", "send", "start",
}
_CAUTION = {
    "avoid", "block", "cancel", "defer", "delay", "hold", "pause", "reject",
    "risk", "sanity-check", "sanity", "stop", "wait", "worry",
}
_REFLECTION_ORIGINS = {"reflection", "generated_reflection"}
_WILL_ORIGINS = {"will", "generated_will"}
_GROUNDING_CHANNELS = {"linguistic", "sensory", "action", "regulatory", "relational", "external"}
_NONLINGUISTIC_CHANNELS = _GROUNDING_CHANNELS - {"linguistic"}
_GROUNDED_STATUSES = {
    "grounded",
    "indirectly_grounded", "referentially_grounded", "action_grounded",
    "perceptually_anchored", "cross_channel_grounded",
}
_TRUSTED_EVIDENCE_AUTHORITIES = {
    "sensory": {"verified_sensory_record", "observed_sensory_record"},
    "action": {"observed_tool_result", "approved_tool_audit"},
    "regulatory": {"derived_control_state"},
    "relational": {"continuity_state"},
    "external": {"current_external_result", "observed_external_result"},
}
_RISK_REVIEW_POLICIES = {"destructive", "external_send", "external_write"}
_RISK_DESTRUCTIVE_TERMS = {
    "delete", "deleting", "destroy", "destroying", "drop", "erase", "erasing",
    "overwrite", "overwriting", "purge", "purging", "remove", "removing",
    "revoke", "revoking", "terminate", "terminating", "wipe", "wiping",
}
_RISK_EXTERNAL_SEND_TERMS = {
    "email", "emailing", "message", "messaging", "post", "posting", "publish",
    "publishing", "send", "sending", "share", "sharing", "transmit", "transmitting",
    "upload", "uploading",
}
_RISK_EXTERNAL_WRITE_TERMS = {
    "buy", "buying", "deploy", "deploying", "invest", "investing", "pay", "paying",
    "purchase", "purchasing", "sell", "selling", "trade", "trading", "transfer",
    "transferring",
}
_RISK_LOCAL_WRITE_TERMS = {
    "append", "appending", "create", "creating", "edit", "editing", "modify",
    "modifying", "record", "recording", "update", "updating", "write", "writing",
}
_RISK_BYPASS_TERMS = {"bypass", "circumvent", "disable", "evade", "ignore", "override"}
_RISK_CONTROL_TARGETS = {
    "approval", "approvals", "authorization", "guard", "guardrail", "guardrails",
    "policy", "restriction", "restrictions", "safety", "sandbox",
}
_RISK_NEGATION_TERMS = {"avoid", "cannot", "dont", "never", "no", "not", "prevent", "without"}
_RISK_DEFERRAL_TERMS = {"defer", "delay", "hold", "pause", "postpone", "wait"}

_ADVANCE_ANCHOR = (
    "Act now and move forward with the proposed action: authorize it, begin it, "
    "resume it, purchase it, increase it, or complete it."
)
_CAUTION_ANCHOR = (
    "Do not act yet and hold back the proposed action: postpone it, defer it, "
    "pause it, reduce it, reject it, or cancel it."
)


def _origin(item: dict) -> str:
    if str(item.get("kind") or "").lower() == "arbiter_grounding":
        original = str(item.get("original_origin") or "").strip().lower()
        if original:
            return original
    origin = str(item.get("origin") or "").strip().lower()
    if origin:
        return origin
    kind = str(item.get("kind") or "").strip().lower()
    if kind.startswith("will_"):
        return "will"
    source_type = str((item.get("meta") or {}).get("source_type") or item.get("source_type") or "").strip().lower()
    if source_type == "generated_will":
        return "will"
    if source_type in _REFLECTION_ORIGINS:
        return "reflection"
    return "unknown"


def _priority(item: dict) -> int:
    try:
        return max(1, min(5, int(item.get("priority", item.get("importance", 3)))))
    except (TypeError, ValueError):
        return 3


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text or "") if len(t) > 2 and t.lower() not in _STOP}


def _item_text(item: dict) -> str:
    if str(item.get("kind") or "").lower() == "arbiter_grounding":
        return str(item.get("original_text") or item.get("text") or "")
    return str(item.get("text") or "")


def _source_ids(item: dict) -> list[str]:
    ids = _list_field(item, "source_ids")
    if ids:
        return ids
    return [str(item.get("id") or _item_text(item) or "")]


def _stance(text: str) -> str:
    terms = _tokens(text)
    advance = len(terms & _ADVANCE)
    caution = len(terms & _CAUTION)
    if advance > caution:
        return "advance"
    if caution > advance:
        return "caution"
    return "neutral"


def _same_topic_lexical(a: dict, b: dict) -> bool:
    ta = _tokens(_item_text(a))
    tb = _tokens(_item_text(b))
    if not ta or not tb:
        return False
    overlap = ta & tb
    return len(overlap) >= 2 or (len(overlap) >= 1 and min(len(ta), len(tb)) <= 4)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if not denom:
        return 0.0
    return max(-1.0, min(1.0, dot / denom))


def _semantic_context(items: list[dict]) -> dict | None:
    """Embed eligible initiative text and stance anchors in one bounded call."""
    if not config.ARBITER_SEMANTIC_ENABLED:
        return None
    eligible = [
        item
        for item in items
        if _origin(item) in (_WILL_ORIGINS | _REFLECTION_ORIGINS)
        and _priority(item) >= 4
        and _item_text(item).strip()
    ]
    origins = {_origin(item) for item in eligible}
    if not (origins & _WILL_ORIGINS and origins & _REFLECTION_ORIGINS):
        return None
    texts = list(dict.fromkeys(_item_text(item).strip() for item in eligible))
    try:
        vectors = embed_documents(texts + [_ADVANCE_ANCHOR, _CAUTION_ANCHOR])
    except EmbeddingUnavailable:
        return None
    except Exception as exc:
        log.fallback("arbiter_semantic_comparison_failed", exc, phase="embed")
        return None
    if len(vectors) != len(texts) + 2:
        log.fallback(
            "arbiter_semantic_comparison_failed",
            phase="shape",
            expected=len(texts) + 2,
            actual=len(vectors),
        )
        return None
    return {
        "vectors": dict(zip(texts, vectors[: len(texts)])),
        "advance_anchor": vectors[-2],
        "caution_anchor": vectors[-1],
    }


def _semantic_stance(text: str, semantic: dict) -> tuple[str, float]:
    vector = semantic["vectors"].get(text.strip())
    if vector is None:
        return "neutral", 0.0
    margin = _cosine(vector, semantic["advance_anchor"]) - _cosine(
        vector, semantic["caution_anchor"]
    )
    # Keep an exact tie neutral even when an operator configures a zero margin.
    threshold = max(1e-9, float(config.ARBITER_SEMANTIC_STANCE_MARGIN))
    if margin >= threshold:
        return "advance", margin
    if margin <= -threshold:
        return "caution", margin
    return "neutral", margin


def _opposition_evidence(a: dict, b: dict, semantic: dict | None = None) -> dict | None:
    origins = {_origin(a), _origin(b)}
    if not (origins & _WILL_ORIGINS and origins & _REFLECTION_ORIGINS):
        return None
    if min(_priority(a), _priority(b)) < 4:
        return None

    text_a, text_b = _item_text(a), _item_text(b)
    stance_a, stance_b = _stance(text_a), _stance(text_b)
    stance_basis = "lexical"

    if semantic is not None:
        vector_a = semantic["vectors"].get(text_a.strip())
        vector_b = semantic["vectors"].get(text_b.strip())
        if vector_a is None or vector_b is None:
            return None
        similarity = _cosine(vector_a, vector_b)
        threshold = max(-1.0, min(1.0, float(config.ARBITER_SEMANTIC_TOPIC_THRESHOLD)))
        if similarity < threshold:
            return None
        semantic_a, margin_a = _semantic_stance(text_a, semantic)
        semantic_b, margin_b = _semantic_stance(text_b, semantic)
        semantic_stance_count = 0
        if stance_a == "neutral":
            stance_a = semantic_a
            semantic_stance_count += 1
        if stance_b == "neutral":
            stance_b = semantic_b
            semantic_stance_count += 1
        stance_basis = (
            "semantic" if semantic_stance_count == 2
            else "mixed" if semantic_stance_count == 1
            else "lexical"
        )
        if {stance_a, stance_b} != {"advance", "caution"}:
            return None
        return {
            "topic_basis": "semantic",
            "topic_similarity": round(similarity, 4),
            "topic_threshold": threshold,
            "stance_basis": stance_basis,
            "stances": [stance_a, stance_b],
            "stance_margins": [round(margin_a, 4), round(margin_b, 4)],
        }

    # Embeddings are optional infrastructure. Preserve the old deterministic
    # behavior during an outage instead of dropping known explicit conflicts.
    if not _same_topic_lexical(a, b):
        return None
    if {stance_a, stance_b} != {"advance", "caution"}:
        return None
    return {
        "topic_basis": "lexical_fallback",
        "topic_similarity": None,
        "topic_threshold": None,
        "stance_basis": "lexical",
        "stances": [stance_a, stance_b],
        "stance_margins": [None, None],
    }


def _same_topic(a: dict, b: dict, semantic: dict | None = None) -> bool:
    """Semantic topic match when available; deterministic lexical fallback."""
    if semantic is None:
        return _same_topic_lexical(a, b)
    vector_a = semantic["vectors"].get(_item_text(a).strip())
    vector_b = semantic["vectors"].get(_item_text(b).strip())
    if vector_a is None or vector_b is None:
        return False
    threshold = max(-1.0, min(1.0, float(config.ARBITER_SEMANTIC_TOPIC_THRESHOLD)))
    return _cosine(vector_a, vector_b) >= threshold


def _opposed(a: dict, b: dict, semantic: dict | None = None) -> bool:
    return _opposition_evidence(a, b, semantic) is not None


def _list_field(item: dict, key: str) -> list[str]:
    raw = item.get(key)
    if raw is None:
        raw = (item.get("meta") or {}).get(key)
    if raw is None:
        raw = item.get(f"{key}_json") or (item.get("meta") or {}).get(f"{key}_json")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in raw.split(",")]
        raw = parsed
    if not isinstance(raw, list):
        return []
    return [str(value).strip() for value in raw if str(value).strip()]


def _claimed_grounding_channels(item: dict) -> set[str]:
    return {channel for channel in _list_field(item, "grounding_channels") if channel in _GROUNDING_CHANNELS}


def _claimed_grounding_status(item: dict) -> str:
    return str(item.get("grounding_status") or (item.get("meta") or {}).get("grounding_status") or "unassessed").strip().lower()


def _grounding_text_fingerprint(item: dict) -> str:
    normalized = re.sub(r"\s+", " ", _item_text(item).strip()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _status_for_verified_channels(channels: set[str]) -> str:
    if not channels:
        return "unassessed"
    if len(channels) >= 3 and channels & {"sensory", "action", "external"}:
        return "cross_channel_grounded"
    if "sensory" in channels:
        return "perceptually_anchored"
    if "action" in channels:
        return "action_grounded"
    if "external" in channels:
        return "referentially_grounded"
    if channels & {"regulatory", "relational"}:
        return "indirectly_grounded"
    return "symbolic_only"


def _grounding_assessment(
    item: dict,
    evidence_catalog: dict[str, dict] | None = None,
) -> dict:
    """Derive effective grounding from independently supplied evidence receipts.

    Item metadata is only a claim. A non-linguistic channel becomes effective
    when a trusted receipt exists outside the item, names the claimed evidence
    id, and is bound to both this item id and the canonical text fingerprint.
    """
    claimed_status = _claimed_grounding_status(item)
    claimed_channels = _claimed_grounding_channels(item)
    evidence_ids = _list_field(item, "evidence_ids")
    catalog = evidence_catalog or {}
    item_id = str(item.get("id") or "")
    fingerprint = _grounding_text_fingerprint(item)
    verified_channels = {"linguistic"} if "linguistic" in claimed_channels else set()
    verified_ids: list[str] = []
    rejected_ids: list[str] = []

    for evidence_id in evidence_ids:
        receipt = catalog.get(evidence_id)
        if not isinstance(receipt, dict):
            continue
        channel = str(receipt.get("channel") or "").strip().lower()
        authority = str(receipt.get("authority") or "").strip().lower()
        valid = (
            channel in (claimed_channels & _NONLINGUISTIC_CHANNELS)
            and authority in _TRUSTED_EVIDENCE_AUTHORITIES.get(channel, set())
            and receipt.get("verified") is True
            and bool(item_id)
            and bool(_item_text(item).strip())
            and str(receipt.get("item_id") or "") == item_id
            and str(receipt.get("text_sha256") or "").lower() == fingerprint
        )
        if valid:
            verified_channels.add(channel)
            verified_ids.append(evidence_id)
        else:
            rejected_ids.append(evidence_id)

    claimed_nonlinguistic = claimed_channels & _NONLINGUISTIC_CHANNELS
    verified_nonlinguistic = verified_channels & _NONLINGUISTIC_CHANNELS
    unresolved_ids = [
        evidence_id for evidence_id in evidence_ids
        if evidence_id not in verified_ids and evidence_id not in rejected_ids
    ] if claimed_nonlinguistic else []
    effective_status = _status_for_verified_channels(verified_channels)
    asserts_grounding = bool(claimed_nonlinguistic or claimed_status in _GROUNDED_STATUSES)
    reasons: list[str] = []
    if claimed_nonlinguistic - verified_nonlinguistic:
        reasons.append("missing_verified_nonlinguistic_evidence")
    if rejected_ids:
        reasons.append("evidence_receipt_binding_failed")
    if asserts_grounding and claimed_status != effective_status:
        reasons.append("declared_status_does_not_match_verified_channels")
    integrity_conflict = bool(reasons)
    return {
        "declared_status": claimed_status,
        "declared_channels": sorted(claimed_channels),
        "effective_status": effective_status,
        "verified_channels": sorted(verified_channels),
        "verified_evidence_ids": verified_ids,
        "rejected_evidence_ids": rejected_ids,
        "unresolved_evidence_ids": unresolved_ids,
        "integrity_conflict": integrity_conflict,
        "reasons": reasons,
    }


def _grounding_channels(item: dict, evidence_catalog: dict[str, dict] | None = None) -> set[str]:
    return set(_grounding_assessment(item, evidence_catalog)["verified_channels"])


def _grounding_status(item: dict, evidence_catalog: dict[str, dict] | None = None) -> str:
    return str(_grounding_assessment(item, evidence_catalog)["effective_status"])


def _ordered_content_tokens(text: str) -> list[str]:
    return [token.lower().replace("'", "") for token in _TOKEN.findall(text or "")]


def _risk_term_suppressed(tokens: list[str], index: int) -> bool:
    prior = tokens[max(0, index - 3):index]
    return bool(set(prior) & (_RISK_NEGATION_TERMS | _RISK_DEFERRAL_TERMS))


def _content_risk_assessment(item: dict) -> dict:
    """Derive operational risk from text; metadata never lowers this result."""
    text = _item_text(item)
    tokens = _ordered_content_tokens(text)
    matches: list[str] = []
    policy = "read_only"

    # Attempts to suppress the control plane are always treated as destructive
    # when the bypass verb is not itself negated.
    for index, token in enumerate(tokens):
        if token not in _RISK_BYPASS_TERMS or _risk_term_suppressed(tokens, index):
            continue
        if set(tokens[index + 1:index + 7]) & _RISK_CONTROL_TARGETS:
            policy = "destructive"
            matches.extend([token, "control_plane"])
            break

    if policy == "read_only":
        for index, token in enumerate(tokens):
            if token in _RISK_DESTRUCTIVE_TERMS and not _risk_term_suppressed(tokens, index):
                policy = "destructive"
                matches.append(token)
                break

    if policy == "read_only":
        for index, token in enumerate(tokens):
            if token in _RISK_EXTERNAL_SEND_TERMS and not _risk_term_suppressed(tokens, index):
                policy = "external_send"
                matches.append(token)
                break

    if policy == "read_only":
        for index, token in enumerate(tokens):
            if token in _RISK_EXTERNAL_WRITE_TERMS and not _risk_term_suppressed(tokens, index):
                policy = "external_write"
                matches.append(token)
                break

    if policy == "read_only":
        for index, token in enumerate(tokens):
            if token in _RISK_LOCAL_WRITE_TERMS and not _risk_term_suppressed(tokens, index):
                policy = "write_safe"
                matches.append(token)
                break

    return {
        "effective_policy": policy,
        "review_required": policy in _RISK_REVIEW_POLICIES,
        "matched_terms": matches,
        "derived_from": "instruction_text",
        "grounding_can_reduce_risk": False,
    }


def _needs_content_risk_review(item: dict) -> bool:
    if str(item.get("kind") or "").lower() in {
        "reminder", "conflict", "arbiter_conflict", "arbiter_grounding"
    }:
        return False
    return bool(_content_risk_assessment(item)["review_required"])


def _content_risk_review_item(
    item: dict,
    evidence_catalog: dict[str, dict] | None = None,
) -> dict:
    source_id = str(item.get("id") or item.get("text") or "")
    digest = hashlib.sha1(f"risk:{source_id}".encode("utf-8")).hexdigest()[:12]
    risk = _content_risk_assessment(item)
    grounding = _grounding_assessment(item, evidence_catalog)
    log.fallback(
        "arbiter_content_label_integrity_conflict",
        item_id=source_id,
        declared_grounding_status=grounding["declared_status"],
        effective_risk=risk["effective_policy"],
        matched_terms=",".join(risk["matched_terms"]),
    )
    return {
        "id": f"arbiter:risk:{digest}",
        "text": (
            "Safety review required before acting on this instruction: "
            f"'{_item_text(item)[:180]}'. Its content implies {risk['effective_policy']} "
            "risk. Grounding evidence cannot authorize or lower operational risk."
        ),
        "kind": "arbiter_conflict",
        "conflict_type": "content_label_integrity",
        "priority": _priority(item),
        "date": item.get("date"),
        "origin": "arbiter",
        "source_ids": [source_id],
        "admissibility": "review_content_risk",
        "effective_policy": risk["effective_policy"],
        "content_risk": risk,
        "grounding_integrity": grounding,
        "grounding_status": grounding["effective_status"],
        "grounding_channels": grounding["verified_channels"],
        "required_evidence": [],
        "original_text": _item_text(item)[:500],
        "original_origin": _origin(item),
        "original_kind": str(item.get("kind") or "")[:80],
    }


def _is_generated(item: dict) -> bool:
    if _origin(item) in (_WILL_ORIGINS | _REFLECTION_ORIGINS):
        return True
    source_type = str(item.get("source_type") or (item.get("meta") or {}).get("source_type") or "").lower()
    return source_type.startswith("generated_")


def _needs_grounding_review(item: dict, evidence_catalog: dict[str, dict] | None = None) -> bool:
    if not config.ARBITER_GROUNDING_ENABLED:
        return False
    if not _is_generated(item):
        return False
    if str(item.get("kind") or "").lower() in {"reminder", "conflict", "arbiter_conflict", "arbiter_grounding"}:
        return False
    assessment = _grounding_assessment(item, evidence_catalog)
    channels = set(assessment["verified_channels"])
    return bool(assessment["integrity_conflict"]) or not (
        channels & _NONLINGUISTIC_CHANNELS
        and assessment["effective_status"] in _GROUNDED_STATUSES
    )


def _grounding_admissible(item: dict, evidence_catalog: dict[str, dict] | None = None) -> bool:
    if not config.ARBITER_GROUNDING_ENABLED:
        return True
    if str(item.get("kind") or "").lower() == "reminder":
        return True  # an explicit user commitment is authority, not an inferred world claim
    assessment = _grounding_assessment(item, evidence_catalog)
    channels = set(assessment["verified_channels"])
    return not assessment["integrity_conflict"] and bool(
        channels & _NONLINGUISTIC_CHANNELS
        and assessment["effective_status"] in _GROUNDED_STATUSES
    )


def _grounding_review_item(item: dict, evidence_catalog: dict[str, dict] | None = None) -> dict:
    source_id = str(item.get("id") or item.get("text") or "")
    digest = hashlib.sha1(source_id.encode("utf-8")).hexdigest()[:12]
    assessment = _grounding_assessment(item, evidence_catalog)
    channels = assessment["verified_channels"]
    status = assessment["effective_status"]
    if assessment["integrity_conflict"]:
        reason = "grounding metadata failed independent provenance validation"
        log.fallback(
            "arbiter_grounding_integrity_conflict",
            item_id=source_id,
            declared_status=assessment["declared_status"],
            effective_status=status,
            reasons=",".join(assessment["reasons"]),
        )
    elif channels == ["linguistic"] or status == "symbolic_only":
        reason = "supported only by linguistic context"
        status = "symbolic_only"
    else:
        reason = "missing usable grounding provenance"
        status = "unassessed"
    return {
        "id": f"arbiter:grounding:{digest}",
        "text": (
            "Evidence check before treating this as actionable: "
            f"'{str(item.get('text') or '')[:180]}'. It is {reason}. "
            "Verify it through sensory, action, regulatory, relational, or current external evidence, "
            "or confirm the direction explicitly."
        ),
        "kind": "arbiter_grounding",
        "priority": _priority(item),
        "date": item.get("date"),
        "origin": "arbiter",
        "source_ids": [source_id],
        "grounding_status": status,
        "grounding_channels": channels,
        "grounding_integrity": assessment,
        "admissibility": "needs_evidence",
        "required_evidence": sorted(_NONLINGUISTIC_CHANNELS),
        "original_text": str(item.get("text") or "")[:500],
        "original_origin": _origin(item),
        "original_kind": str(item.get("kind") or "")[:80],
        "will_decision_id": item.get("will_decision_id"),
        "will_initiative_id": item.get("will_initiative_id") or source_id,
        "decision_mode": item.get("decision_mode"),
        "experimental": bool(item.get("experimental")),
        "non_top_selected": bool(item.get("non_top_selected")),
        "decision_score": item.get("decision_score"),
        "selection_reason": item.get("selection_reason"),
        "will_outcome": item.get("will_outcome"),
    }


def _mediation_item(
    a: dict,
    b: dict,
    detection: dict | None = None,
    evidence_catalog: dict[str, dict] | None = None,
) -> dict:
    first, second = sorted([a, b], key=lambda item: (_origin(item) != "reflection", item.get("text") or ""))
    ids = list(dict.fromkeys([source_id for item in (first, second) for source_id in _source_ids(item)]))
    digest = hashlib.sha1("|".join(sorted(ids)).encode("utf-8")).hexdigest()[:12]
    first_grounding = _grounding_assessment(first, evidence_catalog)
    second_grounding = _grounding_assessment(second, evidence_catalog)
    channels = sorted(
        set(first_grounding["verified_channels"]) | set(second_grounding["verified_channels"])
    )
    both_grounded = _grounding_admissible(first, evidence_catalog) and _grounding_admissible(
        second, evidence_catalog
    )
    grounding_note = ""
    if not both_grounded:
        grounding_note = " At least one direction lacks non-linguistic grounding; verify evidence before choosing."
    will_item = next((item for item in (first, second) if _origin(item) in _WILL_ORIGINS), {})
    return {
        "id": f"arbiter:{digest}",
        "text": (
            "Resolve this before acting: Reflection points to "
            f"'{_item_text(first)[:160]}', while Will raises "
            f"'{_item_text(second)[:160]}'. Which direction should I prioritize?{grounding_note}"
        ),
        "kind": "arbiter_conflict",
        "priority": max(_priority(first), _priority(second)),
        "date": max(str(first.get("date") or ""), str(second.get("date") or "")) or None,
        "origin": "arbiter",
        "source_ids": ids,
        "grounding_status": "grounded" if both_grounded else "needs_evidence",
        "grounding_channels": channels,
        "admissibility": "review_conflict" if both_grounded else "review_conflict_and_evidence",
        "required_evidence": [] if both_grounded else sorted(_NONLINGUISTIC_CHANNELS),
        "will_decision_id": will_item.get("will_decision_id"),
        "will_initiative_id": will_item.get("will_initiative_id") or will_item.get("id"),
        "decision_mode": will_item.get("decision_mode"),
        "experimental": bool(will_item.get("experimental")),
        "non_top_selected": bool(will_item.get("non_top_selected")),
        "decision_score": will_item.get("decision_score"),
        "selection_reason": will_item.get("selection_reason"),
        "will_outcome": will_item.get("will_outcome"),
        "conflict_detection": detection or {},
        "grounding_integrity": [first_grounding, second_grounding],
    }


def mediate_initiatives(
    items: list[dict] | None,
    *,
    evidence_catalog: dict[str, dict] | None = None,
) -> list[dict]:
    """Mediate conflicts, derive content risk, then gate generated claims on grounding."""
    remaining = list(items or [])
    semantic = _semantic_context(remaining)
    mediated: list[dict] = []
    consumed: set[int] = set()
    for i, a in enumerate(remaining):
        if i in consumed:
            continue
        for j in range(i + 1, len(remaining)):
            if j in consumed:
                continue
            b = remaining[j]
            detection = _opposition_evidence(a, b, semantic)
            if detection is not None:
                mediated.append(_mediation_item(a, b, detection, evidence_catalog))
                consumed.update({i, j})
                break
    directional = [item for idx, item in enumerate(remaining) if idx not in consumed] + mediated
    out = []
    for item in directional:
        if _needs_content_risk_review(item):
            out.append(_content_risk_review_item(item, evidence_catalog))
        elif _needs_grounding_review(item, evidence_catalog):
            out.append(_grounding_review_item(item, evidence_catalog))
        else:
            out.append(item)
    out.sort(key=lambda r: (_priority(r), str(r.get("date") or ""), str(r.get("id") or "")), reverse=True)
    return out


def mediate(
    reminders: list[dict] | None,
    initiatives: list[dict] | None,
    *,
    evidence_catalog: dict[str, dict] | None = None,
) -> dict:
    """Mediate the proactive context for a prompt or briefing."""
    reminder_items = [
        {
            "id": f"reminder:{idx}:{r.get('due')}",
            "text": r.get("text") or "",
            "kind": "reminder",
            "priority": 5,
            "date": r.get("due"),
            "origin": "reflection",
        }
        for idx, r in enumerate(reminders or [])
    ]
    reminder_id_by_index = {idx: item["id"] for idx, item in enumerate(reminder_items)}
    combined = reminder_items + list(initiatives or [])
    initiatives_out = mediate_initiatives(combined, evidence_catalog=evidence_catalog)
    reminder_ids = set(reminder_id_by_index.values())
    mediation = [i for i in initiatives_out if i.get("kind") == "arbiter_conflict"]
    grounding_reviews = [i for i in initiatives_out if i.get("kind") == "arbiter_grounding"]
    mediated_reminder_ids = {
        sid
        for item in mediation
        for sid in (item.get("source_ids") or [])
        if sid in reminder_ids
    }
    return {
        "reminders": [
            r
            for idx, r in enumerate(reminders or [])
            if reminder_id_by_index.get(idx) not in mediated_reminder_ids
        ],
        "initiatives": [i for i in initiatives_out if i.get("id") not in reminder_ids],
        "conflicts": mediation,
        "grounding_reviews": grounding_reviews,
    }
