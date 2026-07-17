"""Review-only Action Package support."""
from __future__ import annotations

from brain import store
from brain.embeddings import EmbeddingUnavailable, embed_document, text_only_embedding
from brain.memory import MemoryUnit, clamp_importance, now_iso

import config


def _clean_text(value: object, limit: int) -> str:
    text = str(value or "").strip()
    return text[:limit]


def review_action_package(args: dict) -> dict:
    """Record an approved Action Package locally.

    Approving this tool means "save the reviewed plan"; it never performs the
    draft action described inside the package.
    """
    title = _clean_text(args.get("title") or "Action Package", 160)
    body = _clean_text(args.get("body"), config.MAX_MEMORY_CHARS)
    if not body:
        return {"error": "action package body is required"}
    priority = clamp_importance(args.get("priority", 4))
    source_id = _clean_text(args.get("source_initiative_id"), 160) or None
    package = f"# {title}\n\n{body}" if not body.startswith("#") else body
    unit = MemoryUnit(
        text=package,
        category=config.ACTION_PACKAGE_CATEGORY,
        importance=priority,
        extra={
            "source_type": "approved_action_package",
            "package_status": "approved",
            "approved_at": now_iso(),
            "source_initiative_id": source_id,
            "external_action_authorized": False,
        },
    )
    try:
        store.add(unit, embed_document(unit.text, model=args.get("__model")), model=args.get("__model"))
    except (EmbeddingUnavailable, Exception):
        store.add(unit, text_only_embedding(), model=args.get("__model"), text_only=True)
    return {
        "ok": True,
        "id": unit.id,
        "category": config.ACTION_PACKAGE_CATEGORY,
        "message": "Action Package recorded locally. No external action was executed.",
    }

