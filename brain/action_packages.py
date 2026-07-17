"""Build reviewable Action Packages from proactive initiative candidates."""
from __future__ import annotations

from datetime import datetime, timezone

import config
from brain import store, tools
from brain.memory import clamp_importance

# Taint label for a package whose claimed source initiative could not be matched
# to a stored, still-open initiative record with the same text. The chat
# one-word approval shortcut refuses these (panel review only), and their
# priority is capped below "urgent".
UNVERIFIED_TAINT = "unverified_initiative"
UNVERIFIED_MAX_PRIORITY = 3


def _clean(value: object, limit: int = 1000) -> str:
    return str(value or "").strip()[:limit]


def _title_from_text(text: str) -> str:
    words = text.replace("\n", " ").split()
    title = " ".join(words[:10]).strip(" .")
    return title or "Review proactive initiative"


def _package_body(item: dict) -> str:
    text = _clean(item.get("text"), 4000)
    kind = _clean(item.get("kind") or "initiative", 80)
    priority = item.get("priority") or 4
    date_prepared = datetime.now(timezone.utc).date().isoformat()
    source_id = _clean(item.get("id"), 160) or "unspecified initiative"
    return f"""## Package Header

- Package: {_title_from_text(text)}
- Date prepared: {date_prepared}
- Trigger: Proactive initiative surfaced by Neural Map.
- Priority: {priority}
- Approval required: Yes
- Reversibility: Reversible

## 1. Decision Needed

Decide whether this initiative should become an active next step: {text}

## 2. Confirmed Facts

| Fact | Source | Date |
|---|---|---|
| Neural Map surfaced this initiative: "{text}" | {source_id} | {date_prepared} |
| Initiative kind is "{kind}" with priority {priority}. | local initiative metadata | {date_prepared} |

## 3. Assumptions

| Assumption | Confidence | How to verify |
|---|---:|---|
| This initiative is still relevant to the current situation. | Medium | Review current facts before acting. |
| The next useful step is planning, not execution. | High | Approve only this package; create a separate action for any real-world execution. |

## 4. Recommendation

Direct verdict: Review this package, then either approve it as a planning item, revise it, or reject it.

Rationale: The initiative appears consequential enough to deserve a structured decision instead of a casual nudge.

Confidence: Medium until current facts are verified.

## 5. Numbers

| Item | Amount / value | Notes |
|---|---:|---|
| Priority | {priority} | 1 low, 5 high |

## 6. Consequences

Direct effects: Converts a loose initiative into a reviewed plan.

Liquidity effects: None unless a later approved action involves money.

Tax effects: None unless a later approved action involves taxes.

Behavioral effects: Reduces prompting burden while preserving your approval authority.

Timing effects: Keeps the loop visible before it becomes urgent.

Strategic effects: Moves Neural Map from reactive answers toward bounded agency.

## 7. Risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| The initiative is stale or based on incomplete context. | A proactive package can overfocus the wrong issue. | Verify current facts before any follow-on action. |
| Approval is mistaken for execution approval. | The package may mention trades, messages, or checklists. | Approval records the plan only; external action requires a separate approval. |

## 8. Alternatives

| Option | When it is better | Tradeoff |
|---|---|---|
| Approve as planning item | The initiative is relevant and worth tracking. | Still requires separate execution later. |
| Revise package | The idea is useful but missing facts or framing. | Takes one more pass. |
| Reject and close loop | The initiative is stale or noisy. | Neural Map backs off this item. |

## 9. Draft Action

No external action is authorized by this package. If approved, record this as a reviewed planning package and prepare any real-world next step separately.

## 10. Approval Gate

Choose one:

- Approve as drafted.
- Revise with changes.
- Reject and close the loop.
"""


def _stored_open_initiative(source_id: str, model: str | None = None) -> dict | None:
    """The stored, still-open initiative record behind a claimed source id."""
    if not source_id:
        return None
    try:
        for rec in store.list_records_all(model):
            meta = rec.get("meta") or {}
            if (
                rec.get("id") == source_id
                and meta.get("category") == config.INITIATIVE_CATEGORY
                and meta.get("status") in (None, "open")
            ):
                return rec
    except Exception:
        return None
    return None


def queue_from_initiative(item: dict, model: str | None = None) -> dict:
    if not config.ACTION_PACKAGES_ENABLED:
        return {"error": "action packages are disabled"}
    text = _clean(item.get("text"), 4000)
    if not text:
        return {"error": "initiative text is required"}
    source_id = _clean(item.get("id"), 160)
    for action in tools.list_pending():
        args = action.get("args") or {}
        if (
            action.get("tool") == "review_action_package"
            and source_id
            and args.get("source_initiative_id") == source_id
        ):
            return {"pending": True, "action": action, "existing": True}
    # Provenance check: the request body (id, text, priority) is client-supplied,
    # so verify the claimed initiative against the store instead of trusting it.
    # Verified means the id resolves to a stored, still-open initiative whose
    # text matches what the caller wants packaged. Anything else — unknown id,
    # dismissed/expired initiative, edited text, no id at all — is queued but
    # tainted as unverified and capped below "urgent" priority, so fabricated
    # content can't arrive looking like a trusted high-priority package.
    stored = _stored_open_initiative(source_id, model)
    verified = bool(stored and _clean(stored.get("text"), 4000) == text)
    priority = clamp_importance(item.get("priority", 4))
    taint = ["initiative"]
    if not verified:
        taint.append(UNVERIFIED_TAINT)
        priority = min(priority, UNVERIFIED_MAX_PRIORITY)
    args = {
        "title": _title_from_text(text),
        "body": _package_body({**item, "priority": priority}),
        "source_initiative_id": source_id,
        "priority": priority,
        "__model": model,
    }
    result = tools.dispatch("review_action_package", args, taint=taint)
    if isinstance(result, dict):
        result["verified_initiative"] = verified
    return result

