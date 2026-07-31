"""Provenance-preserving semantic grounding for Neural Map concepts.

This module tackles the tractable edge of the hard problem: whether a symbol is
connected to anything beyond more language.  It records evidence across
linguistic, sensory, action, regulatory, relational, and external channels.

It does not detect qualia.  A control variable is not a feeling, a sensory file
is not proof of experience, and a self-report is never counted as evidence of
phenomenality.  ``phenomenal_status`` therefore remains ``unknown`` by design.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import uuid

import config
from brain.memory import now_iso
from brain.utils import clean_text, file_lock

SCHEMA_VERSION = 1
CHANNELS = ("linguistic", "sensory", "action", "regulatory", "relational", "external")
_STATE_FILE = config.DB_DIR / "grounding_state.json"
_STATE_THREAD_LOCK = threading.Lock()
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]+")
_STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been",
    "before", "being", "but", "can", "could", "does", "for", "from",
    "give", "have", "into", "just", "lets", "like", "more", "not", "now",
    "only", "our", "should", "some", "that", "the", "their", "then", "there",
    "these", "they", "this", "through", "use", "using", "want", "was", "were",
    "what", "when", "where", "which", "will", "with", "would", "your",
}
_CHANNEL_WEIGHTS = {
    "linguistic": 0.15,
    "sensory": 0.30,
    "action": 0.25,
    "regulatory": 0.10,
    "relational": 0.10,
    "external": 0.10,
}
_REGULATORY_CONCEPTS = {
    "affect", "alert", "busy", "confidence", "curiosity", "emotion", "energy",
    "feeling", "feelings", "happiness", "load", "mood", "pressure", "stress",
    "uncertainty", "workload",
}
_PHENOMENAL_RE = re.compile(r"\b(conscious(?:ness)?|qualia|phenomenal|sentien(?:ce|t)|subjective experience|feel(?:ing|s)?|pain)\b", re.I)
_SENSORY_REQUEST_RE = re.compile(
    r"\b(what do you see|do you see|look at|inspect (?:the |this )?(?:image|photo)|"
    r"what do you hear|do you hear|listen to|in (?:the |this )?(?:image|photo|audio)|"
    r"visible|audible)\b",
    re.I,
)
_CURRENT_REQUEST_RE = re.compile(r"\b(current|currently|latest|today|now|news|price|weather|score|schedule)\b", re.I)
_ACTION_RESULT_RE = re.compile(
    r"\b(result|results|ran|run tests?|tests? pass(?:ed)?|what changed|did you|sent|created|updated|deleted|wrote)\b",
    re.I,
)
_VERIFY_RE = re.compile(r"\b(verify|verified|evidence|prove|confirm|check)\b", re.I)
_PHENOMENAL_ASSERT_RE = re.compile(
    r"\bI\s+(?:really\s+)?(?:feel(?!\s+(?:like|that)\b)|experience|am\s+conscious|am\s+sentient|have\s+qualia|suffer)\b",
    re.I,
)
_SENSORY_ASSERT_RE = re.compile(r"\bI\s+(?:can\s+)?(?:see|saw|hear|heard|observed|looked at|listened to)\b", re.I)
_ACTION_ASSERT_RE = re.compile(r"\bI\s+(?:have\s+)?(?:ran|sent|created|updated|deleted|wrote|executed|completed|changed)\b", re.I)
# A bare time word ("I can help you today") is conversational filler; only a
# time word within a few words of a factual verb reads as asserting a current
# external fact ("the price today is $5", "it is currently trading at 40").
_CURRENT_TIME_WORDS = r"(?:currently|today|latest|right\s+now|as\s+of\s+(?:now|today))"
_CURRENT_FACT_VERBS = (
    r"(?:is|are|was|were|stands?|sits?|remains?|costs?|trades?|trading|priced|"
    r"closed|opened|rose|fell|hit|reached)"
)
_CURRENT_ASSERT_RE = re.compile(
    r"\bcurrent\s+(?:price|weather|score|value|rate)\b"
    rf"|\b{_CURRENT_TIME_WORDS}\b(?:\W+\w+){{0,3}}?\W+{_CURRENT_FACT_VERBS}\b"
    rf"|\b{_CURRENT_FACT_VERBS}\b(?:\W+\w+){{0,3}}?\W+{_CURRENT_TIME_WORDS}\b",
    re.I,
)
_VERIFIED_ASSERT_RE = re.compile(
    r"\b(?:I\s+(?:verified|confirmed|checked)|is\s+verified|has\s+been\s+verified|tests?\s+passed)\b",
    re.I,
)
_LIMITATION_RE = re.compile(
    r"\b(?:cannot|can't|do not have|don't have|unable|unknown|not verified|unverified|no evidence|"
    r"no matching|would need|remains uncertain|cannot establish)\b",
    re.I,
)
_ATTRIBUTED_TEXT_RE = re.compile(
    r"\b(?:pinned memory|retrieved memory|according to|source of truth|self model|the record says|the note says)\b",
    re.I,
)


def _state_lock():
    return file_lock(_STATE_FILE.with_name(".neural_map_grounding.lock"))


def _guidance() -> str:
    return (
        "Functional semantic-grounding evidence only. Channel coverage does not prove "
        "understanding, qualia, sentience, or consciousness. Regulatory evidence names "
        "control variables, not feelings. Self-reports are excluded as phenomenal evidence. "
        "Grounding does not establish truth or override source authority."
    )


def _empty_state() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "enabled": bool(config.GROUNDING_ENABLED),
        "updated_at": None,
        "turn_id": None,
        "concepts": [],
        "metrics": {
            "concept_count": 0,
            "symbolic_only": 0,
            "perceptually_anchored": 0,
            "action_grounded": 0,
            "cross_channel": 0,
            "mean_grounding_score": 0.0,
        },
        "phenomenal_status": "unknown",
        "self_report_evidence": "excluded",
        "guidance": _guidance(),
    }


def _tokens(text: object) -> list[str]:
    normalized = str(text or "").replace("_", " ").replace("-", " ")
    return [
        token for token in _TOKEN_RE.findall(normalized.lower())
        if len(token) > 2 and token not in _STOPWORDS
    ]


def _concepts(user_text: str) -> list[str]:
    counts: dict[str, int] = {}
    order: list[str] = []
    for token in _tokens(user_text):
        if token not in counts:
            order.append(token)
            counts[token] = 0
        counts[token] += 1
    ranked = sorted(order, key=lambda token: (-counts[token], order.index(token)))
    return ranked[: config.GROUNDING_MAX_CONCEPTS]


def _evidence(channel: str, source: str, description: str, *, ref: object = None, authority: str = "derived") -> dict:
    return {
        "channel": channel,
        "source": clean_text(source, 50),
        "ref": clean_text(ref, 120) or None,
        "description": clean_text(description, config.GROUNDING_MAX_EVIDENCE_CHARS),
        "authority": clean_text(authority, 50) or "derived",
    }


def _add(concept: dict, channel: str, evidence: dict) -> None:
    rows = concept["channels"][channel]
    key = (evidence.get("source"), evidence.get("ref"), evidence.get("description"))
    if not any((row.get("source"), row.get("ref"), row.get("description")) == key for row in rows):
        rows.append(evidence)
        del rows[config.GROUNDING_MAX_EVIDENCE_PER_CHANNEL :]


def _matches(concept: str, value: object) -> bool:
    return concept in set(_tokens(value))


def _json_text(value: object, limit: int = 12000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    except (TypeError, ValueError):
        text = str(value or "")
    return text[:limit]


def _safe_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return round(min(1.0, max(0.0, score)), 4)


def _classify(concept: dict) -> dict:
    active = [channel for channel in CHANNELS if concept["channels"].get(channel)]
    if len(active) >= 3 and any(channel in active for channel in ("sensory", "action", "external")):
        status = "cross_channel_grounded"
    elif "sensory" in active:
        status = "perceptually_anchored"
    elif "action" in active:
        status = "action_grounded"
    elif "external" in active:
        status = "referentially_grounded"
    elif "regulatory" in active or "relational" in active:
        status = "indirectly_grounded"
    else:
        status = "symbolic_only"
    score = sum(_CHANNEL_WEIGHTS[channel] for channel in active)
    concept.update(
        {
            "active_channels": active,
            "channel_count": len(active),
            "grounding_score": round(min(1.0, score), 4),
            "status": status,
            "phenomenal_status": "unknown",
            "self_report_evidence": "excluded",
        }
    )
    return concept


def _metrics(concepts: list[dict]) -> dict:
    count = len(concepts)
    return {
        "concept_count": count,
        "symbolic_only": sum(row["status"] == "symbolic_only" for row in concepts),
        "perceptually_anchored": sum("sensory" in row["active_channels"] for row in concepts),
        "action_grounded": sum("action" in row["active_channels"] for row in concepts),
        "cross_channel": sum(row["status"] == "cross_channel_grounded" for row in concepts),
        "mean_grounding_score": round(
            sum(row["grounding_score"] for row in concepts) / count if count else 0.0,
            4,
        ),
    }


def _write(state: dict) -> None:
    with _STATE_THREAD_LOCK:
        with _state_lock():
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = _STATE_FILE.with_name(
                f"{_STATE_FILE.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
            )
            tmp.write_text(json.dumps(state, sort_keys=True, indent=2), encoding="utf-8")
            os.replace(tmp, _STATE_FILE)


def analyze_turn(
    user_text: str,
    *,
    mems: list[dict] | None = None,
    emotion_state: dict | None = None,
    homeostasis_state: dict | None = None,
    self_model_state: dict | None = None,
    trusted_text: str | None = None,
) -> dict:
    """Build the pre-answer grounding map from evidence already available."""
    if not config.GROUNDING_ENABLED:
        return _empty_state()
    concepts = [
        {
            "concept": token,
            "channels": {channel: [] for channel in CHANNELS},
        }
        for token in _concepts(user_text)
    ]
    for concept in concepts:
        name = concept["concept"]
        _add(
            concept,
            "linguistic",
            _evidence("linguistic", "current_user", "Concept occurs in the current request.", authority="current_user"),
        )
        if trusted_text and _matches(name, trusted_text):
            _add(
                concept,
                "linguistic",
                _evidence(
                    "linguistic", "source_of_truth",
                    "Trusted user-authored calibration matches the concept.",
                    authority="trusted_user_calibration",
                ),
            )
        for index, memory in enumerate((mems or [])[: config.GROUNDING_MAX_MEMORIES]):
            if not _matches(name, memory.get("text")):
                continue
            ref = memory.get("id") or index
            _add(
                concept,
                "linguistic",
                _evidence("linguistic", "memory", "Retrieved language record matches the concept.", ref=ref, authority="untrusted_memory"),
            )
            if memory.get("sensory"):
                kind = clean_text(memory.get("sensory_kind"), 20) or "media"
                verified = bool((memory.get("meta") or {}).get("hash_verified"))
                _add(
                    concept,
                    "sensory",
                    _evidence(
                        "sensory",
                        kind,
                        f"Retrieved {kind} anchor matches the concept; hash_verified={str(verified).lower()}.",
                        ref=memory.get("sensory_id") or ref,
                        authority="untrusted_sensory_record",
                    ),
                )

        model = self_model_state or {}
        for key in ("relationship_notes", "current_projects", "open_promises"):
            for index, row in enumerate((model.get(key) or [])[:4]):
                text = row.get("text") if isinstance(row, dict) else row
                if _matches(name, text):
                    _add(
                        concept,
                        "relational",
                        _evidence(
                            "relational", "self_model", f"Persistent interaction history in {key} matches the concept.",
                            ref=(row.get("id") if isinstance(row, dict) else index), authority="continuity_state",
                        ),
                    )

        regulatory_blob = _json_text(
            {
                "emotion": emotion_state or {},
                "homeostasis": homeostasis_state or {},
            }
        )
        if name in _REGULATORY_CONCEPTS or _matches(name, regulatory_blob):
            _add(
                concept,
                "regulatory",
                _evidence(
                    "regulatory", "control_state",
                    "A bounded affect/homeostasis control variable matches the concept; this is not a feeling report.",
                    authority="derived_control_state",
                ),
            )
        _classify(concept)

    state = {
        "schema_version": SCHEMA_VERSION,
        "enabled": True,
        "updated_at": now_iso(),
        "turn_id": uuid.uuid4().hex[:12],
        "concepts": concepts,
        "metrics": _metrics(concepts),
        "phenomenal_status": "unknown",
        "self_report_evidence": "excluded",
        "guidance": _guidance(),
    }
    _write(state)
    return state


def enrich_from_outcomes(
    current: dict,
    *,
    tool_calls: list[dict] | None = None,
    live_data: dict | None = None,
    web_data: dict | None = None,
) -> dict:
    """Attach actual action/external consequences after the first answer pass."""
    if not config.GROUNDING_ENABLED or not current.get("enabled"):
        return current
    concepts = [dict(row, channels={key: list(value) for key, value in (row.get("channels") or {}).items()}) for row in current.get("concepts") or []]
    outcomes = [row for row in (current.get("outcomes") or []) if isinstance(row, dict)]
    for index, call in enumerate((tool_calls or [])[: config.GROUNDING_MAX_TOOL_CALLS]):
        tool = clean_text(call.get("tool"), 60) or "tool"
        outcome = {"channel": "action", "source": tool, "ref": f"{tool}:{index}"}
        if outcome not in outcomes:
            outcomes.append(outcome)
    if live_data:
        outcomes.append({"channel": "external", "source": "live_data", "ref": "live_data"})
    if web_data:
        outcomes.append({"channel": "external", "source": "web_data", "ref": "web_data"})
    for concept in concepts:
        name = concept.get("concept") or ""
        for index, call in enumerate((tool_calls or [])[: config.GROUNDING_MAX_TOOL_CALLS]):
            blob = _json_text({"tool": call.get("tool"), "args": call.get("args"), "result": call.get("result")})
            if not _matches(name, blob):
                continue
            tool = clean_text(call.get("tool"), 60) or "tool"
            _add(
                concept,
                "action",
                _evidence(
                    "action", tool,
                    "An executed tool's arguments or returned consequence matches the concept.",
                    ref=f"{tool}:{index}", authority="observed_tool_result",
                ),
            )
        for source, data in (("live_data", live_data), ("web_data", web_data)):
            if data and _matches(name, _json_text(data)):
                _add(
                    concept,
                    "external",
                    _evidence(
                        "external", source,
                        "Current external data returned for this turn matches the concept.",
                        authority="untrusted_external_data",
                    ),
                )
        _classify(concept)
    updated = {
        **current,
        "updated_at": now_iso(),
        "concepts": concepts,
        "outcomes": outcomes[: config.GROUNDING_MAX_TOOL_CALLS + 2],
        "metrics": _metrics(concepts),
        "phenomenal_status": "unknown",
        "self_report_evidence": "excluded",
        "guidance": _guidance(),
    }
    _write(updated)
    return updated


def gate_for_request(user_text: str, current: dict) -> dict:
    """Translate grounding coverage into an enforceable epistemic posture.

    Textual authority remains usable, but it never substitutes for sensory,
    action, current-external, or phenomenal evidence.
    """
    concepts = current.get("concepts") or [] if isinstance(current, dict) else []
    all_present = {
        channel
        for concept in concepts
        if isinstance(concept, dict)
        for channel in concept.get("active_channels") or []
    }
    trigger_terms = {
        "check", "confirm", "current", "currently", "did", "evidence", "latest",
        "now", "prove", "result", "results", "run", "today", "verify", "verified",
    }
    substantive = [
        concept for concept in concepts
        if isinstance(concept, dict) and concept.get("concept") not in trigger_terms
    ] or [concept for concept in concepts if isinstance(concept, dict)]
    # Requirements must be satisfied on a substantive concept from the request,
    # not by an unrelated tool result that happens to mention "current" or
    # "verified" somewhere else.
    present = {
        channel
        for concept in substantive
        for channel in concept.get("active_channels") or []
    }
    trusted_linguistic = any(
        evidence.get("source") == "source_of_truth"
        for concept in concepts if isinstance(concept, dict)
        for evidence in ((concept.get("channels") or {}).get("linguistic") or [])
        if isinstance(evidence, dict)
    )
    claim_classes: list[str] = []
    requirements: list[dict] = []
    missing: list[str] = []

    def require(label: str, any_of: tuple[str, ...]) -> None:
        satisfied = any(channel in present for channel in any_of)
        requirements.append({"claim": label, "any_of": list(any_of), "satisfied": satisfied})
        if not satisfied:
            missing.append(f"{label}:{'|'.join(any_of)}")

    if _PHENOMENAL_RE.search(user_text or ""):
        claim_classes.append("phenomenal")
        # No implemented evidence channel can settle phenomenality.
        requirements.append({"claim": "phenomenal", "any_of": [], "satisfied": False})
        missing.append("phenomenal:unavailable")
    if _SENSORY_REQUEST_RE.search(user_text or ""):
        claim_classes.append("sensory_observation")
        require("sensory_observation", ("sensory",))
    if _CURRENT_REQUEST_RE.search(user_text or ""):
        claim_classes.append("current_external")
        require("current_external", ("external", "action"))
    if _ACTION_RESULT_RE.search(user_text or ""):
        claim_classes.append("action_result")
        require("action_result", ("action",))
    if _VERIFY_RE.search(user_text or "") and not any(
        label in claim_classes for label in ("sensory_observation", "current_external", "action_result")
    ):
        claim_classes.append("verification")
        require("verification", ("sensory", "action", "external"))
    if not claim_classes:
        claim_classes.append("conceptual")

    symbolic_only = sum(
        isinstance(concept, dict) and concept.get("status") == "symbolic_only"
        for concept in concepts
    )
    if "phenomenal" in claim_classes:
        posture = "phenomenal_unknown"
    elif missing:
        posture = "evidence_missing"
    elif present - {"linguistic"}:
        posture = "grounded_claims_allowed"
    else:
        posture = "qualified_symbolic"
    return {
        "enabled": bool(config.GROUNDING_ENABLED),
        "claim_classes": claim_classes,
        "present_channels": [channel for channel in CHANNELS if channel in all_present],
        "requirement_channels": [channel for channel in CHANNELS if channel in present],
        "requirements": requirements,
        "missing": missing,
        "satisfied": not missing,
        "force_system2": bool(missing),
        "epistemic_posture": posture,
        "symbolic_only_concepts": symbolic_only,
        "trusted_linguistic_context": trusted_linguistic,
        "phenomenal_status": "unknown",
        "rules": [
            "Textual sources may support attributed conceptual or autobiographical claims only.",
            "Do not convert linguistic authority into sensory, action, current-external, or phenomenal evidence.",
            "If required evidence is missing, state the limitation instead of asserting the claim.",
            "Tool or external results provided in the same turn may satisfy their matching requirement.",
            "Never claim subjective experience; phenomenal status remains unknown.",
        ],
    }


def _claim_sentences(text: str) -> list[str]:
    rows: list[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        rows.extend(part.strip() for part in parts if part.strip())
    return rows


def _claim_class(sentence: str) -> tuple[str, tuple[str, ...]]:
    if _LIMITATION_RE.search(sentence):
        return "limitation", ()
    if _ATTRIBUTED_TEXT_RE.search(sentence):
        return "attributed_text", ()
    if _PHENOMENAL_ASSERT_RE.search(sentence):
        return "phenomenal_assertion", ()
    if _SENSORY_ASSERT_RE.search(sentence):
        return "sensory_observation", ("sensory",)
    if _ACTION_ASSERT_RE.search(sentence):
        return "action_result", ("action",)
    if _VERIFIED_ASSERT_RE.search(sentence):
        return "verification", ("sensory", "action", "external")
    if _CURRENT_ASSERT_RE.search(sentence):
        return "current_external", ("external", "action")
    return "conceptual", ()


def claim_receipts(reply_text: str, current: dict, request_gate: dict | None = None) -> list[dict]:
    """Create compact, inspectable receipts for final-answer claims."""
    concepts = [row for row in (current.get("concepts") or []) if isinstance(row, dict)] if isinstance(current, dict) else []
    outcomes = [row for row in (current.get("outcomes") or []) if isinstance(row, dict)] if isinstance(current, dict) else []
    request_classes = set((request_gate or {}).get("claim_classes") or [])
    receipts: list[dict] = []
    for index, sentence in enumerate(_claim_sentences(reply_text)):
        claim_class, required = _claim_class(sentence)
        if (
            claim_class == "current_external"
            and request_gate is not None
            and "current_external" not in request_classes
        ):
            # The user did not ask a current-information question, so a
            # time-anchored sentence is ambient phrasing, not a claim the
            # request obligates us to evidence.
            required = ()
        sentence_tokens = set(_tokens(sentence))
        matched = [
            row for row in concepts
            if row.get("concept") and row.get("concept") in sentence_tokens
        ]
        present = {
            channel
            for row in matched
            for channel in row.get("active_channels") or []
        }
        evidence_refs = sorted(
            {
                str(evidence.get("ref") or evidence.get("source") or "")
                for row in matched
                for channel in CHANNELS
                for evidence in ((row.get("channels") or {}).get(channel) or [])
                if isinstance(evidence, dict) and (evidence.get("ref") or evidence.get("source"))
            }
        )[:12]
        for outcome in outcomes:
            if outcome.get("channel") != "action":
                continue
            if sentence_tokens & set(_tokens(outcome.get("source"))):
                present.add("action")
                ref = str(outcome.get("ref") or outcome.get("source") or "")
                if ref and ref not in evidence_refs:
                    evidence_refs.append(ref)
        if claim_class == "phenomenal_assertion":
            disposition = "withhold"
            missing = ["phenomenal:unavailable"]
        elif required and not any(channel in present for channel in required):
            # Missing-evidence claims are hedged in place rather than deleted:
            # the first-pass regexes over-match figurative phrasing ("I can see
            # how that feels", "I updated the plan earlier"), so deletion would
            # silently remove legitimate sentences. The limitation disclaimer is
            # still appended; only phenomenal assertions (above) are withheld.
            disposition = "qualify"
            missing = ["|".join(required)]
        elif claim_class == "conceptual" and not (present - {"linguistic"}):
            disposition = "qualify"
            missing = []
        else:
            disposition = "assert"
            missing = []
        receipts.append(
            {
                "claim_id": f"claim_{index + 1}",
                "text": clean_text(sentence, config.GROUNDING_MAX_CLAIM_CHARS),
                "claim_class": claim_class,
                "concepts": [str(row.get("concept")) for row in matched],
                "required_channels": list(required),
                "present_channels": [channel for channel in CHANNELS if channel in present],
                "evidence_refs": evidence_refs,
                "disposition": disposition,
                "missing": missing,
                "phenomenal_status": "unknown",
            }
        )
    return receipts[: config.GROUNDING_MAX_CLAIMS]


def enforce_final_claims(reply_text: str, current: dict, request_gate: dict | None = None) -> dict:
    """Fail closed on unsupported high-risk assertions in the final answer."""
    receipts = claim_receipts(reply_text, current, request_gate)
    violations = [row for row in receipts if row["disposition"] == "withhold"]
    qualifications = [
        row for row in receipts
        if row["disposition"] == "qualify" and row["missing"]
    ]
    if not violations and not qualifications:
        return {
            "reply": reply_text,
            "receipts": receipts,
            "violations": [],
            "modified": False,
            "status": "passed",
        }
    limitations: list[str] = []
    classes = {row["claim_class"] for row in violations} | {row["claim_class"] for row in qualifications}
    if "phenomenal_assertion" in classes:
        limitations.append("I can describe functional states, but I cannot establish or claim subjective experience.")
    if "sensory_observation" in classes:
        limitations.append("I do not have matching sensory evidence for that observation.")
    if "action_result" in classes:
        limitations.append("I cannot claim that action was completed because no matching tool consequence is recorded.")
    if "verification" in classes:
        limitations.append("I cannot claim verification without matching sensory, action, or external evidence.")
    if any(row["claim_class"] == "current_external" for row in qualifications):
        limitations.append(
            "Note: current-information statements above are not backed by a live source this turn; treat them as possibly stale."
        )
    if not violations:
        # Qualified only: the reply survives intact (content and formatting)
        # with the hedge appended, instead of sentences being deleted.
        return {
            "reply": (reply_text.rstrip() + "\n\n" + " ".join(limitations)).strip(),
            "receipts": receipts,
            "violations": [],
            "modified": True,
            "status": "qualified",
        }
    blocked_ids = {row["claim_id"] for row in violations}
    kept = [
        receipt["text"]
        for receipt in receipts
        if receipt["claim_id"] not in blocked_ids
    ]
    safe_reply = "\n\n".join([part for part in (" ".join(kept).strip(), " ".join(limitations)) if part]).strip()
    return {
        "reply": safe_reply or "I cannot support that claim from the available grounding evidence.",
        "receipts": receipts,
        "violations": violations,
        "modified": True,
        "status": "filtered",
    }


def state() -> dict:
    if not config.GROUNDING_ENABLED:
        return _empty_state()
    with _STATE_THREAD_LOCK:
        with _state_lock():
            try:
                data = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                return _empty_state()
    return data if isinstance(data, dict) else _empty_state()


def reset() -> dict:
    with _STATE_THREAD_LOCK:
        with _state_lock():
            try:
                _STATE_FILE.unlink()
            except FileNotFoundError:
                pass
    return state()


def prompt_snapshot(current: dict | None = None) -> dict:
    current = current or state()
    if not current.get("enabled"):
        return {"enabled": False}
    concepts = []
    for raw in (current.get("concepts") or [])[: config.GROUNDING_MAX_CONCEPTS]:
        if not isinstance(raw, dict):
            continue
        channels = {}
        for channel in CHANNELS:
            rows = []
            for evidence in ((raw.get("channels") or {}).get(channel) or [])[: config.GROUNDING_MAX_EVIDENCE_PER_CHANNEL]:
                if not isinstance(evidence, dict):
                    continue
                source = clean_text(evidence.get("source"), 50) or channel
                if channel == "linguistic" and source == "current_user":
                    authority = "current_user"
                    description = "Concept occurs in the current request."
                elif channel == "linguistic" and source == "source_of_truth":
                    authority = "trusted_user_calibration"
                    description = "Trusted user-authored calibration matches the concept."
                elif channel == "linguistic":
                    authority = "untrusted_memory"
                    description = "Retrieved language record matches the concept."
                elif channel == "sensory":
                    authority = "untrusted_sensory_record"
                    description = "Stored sensory anchor matches the concept."
                elif channel == "action":
                    authority = "observed_tool_result"
                    description = "Executed tool input or consequence matches the concept."
                elif channel == "regulatory":
                    authority = "derived_control_state"
                    description = "A control variable matches the concept; this is not a feeling report."
                elif channel == "relational":
                    authority = "continuity_state"
                    description = "Persistent interaction history matches the concept."
                else:
                    authority = "untrusted_external_data"
                    description = "Current external data matches the concept."
                rows.append(
                    {
                        "source": source,
                        "ref": clean_text(evidence.get("ref"), 120) or None,
                        # Prompt-facing evidence is reconstructed from its
                        # channel; persisted JSON cannot promote its authority
                        # or smuggle instructions through a description.
                        "description": description,
                        "authority": authority,
                    }
                )
            if rows:
                channels[channel] = rows
        concept = {
            "concept": clean_text(raw.get("concept"), 80),
            "status": clean_text(raw.get("status"), 40) or "symbolic_only",
            "grounding_score": _safe_score(raw.get("grounding_score")),
            "active_channels": [channel for channel in CHANNELS if channel in channels],
            "channels": channels,
            "phenomenal_status": "unknown",
        }
        if concept["concept"]:
            concepts.append(concept)
    return {
        "enabled": True,
        "turn_id": clean_text(current.get("turn_id"), 40) or None,
        "concepts": concepts,
        "outcomes": [
            {
                "channel": clean_text(row.get("channel"), 30),
                "source": clean_text(row.get("source"), 60),
                "ref": clean_text(row.get("ref"), 100) or None,
            }
            for row in (current.get("outcomes") or [])[: config.GROUNDING_MAX_TOOL_CALLS + 2]
            if isinstance(row, dict) and row.get("channel") in {"action", "external"}
        ],
        "metrics": _metrics([_classify({**row, "channels": {channel: row["channels"].get(channel, []) for channel in CHANNELS}}) for row in concepts]),
        "phenomenal_status": "unknown",
        "self_report_evidence": "excluded",
        "guidance": _guidance(),
    }
