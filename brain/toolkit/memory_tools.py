"""Memory maintenance tools."""
from __future__ import annotations

import config
from brain import store
from brain.embeddings import EmbeddingUnavailable, embed_document, text_only_embedding
from brain.memory import MemoryUnit, now_iso

BASE_META_KEYS = {
    "schema_version",
    "category",
    "importance",
    "date",
    "access_count",
    "source",
    "source_type",
    "pinned",
}


def _tool_model(args: dict | None = None) -> str | None:
    model = (args or {}).get("__model")
    if model is None:
        model = (args or {}).get("model")
    return str(model).strip() or None


def memory_record(memory_id: str, model: str | None = None) -> dict | None:
    for rec in store.list_records_all(model):
        if rec.get("id") == memory_id:
            return rec
    return None


def _store_memory_unit(unit: MemoryUnit, model: str | None = None) -> tuple[str, str]:
    try:
        embedding = embed_document(unit.text, model=model)
    except EmbeddingUnavailable:
        store.add(unit, text_only_embedding(), model=model, text_only=True)
        return unit.id, "text_only"
    store.add(unit, embedding, model=model)
    return unit.id, "vector"


def update_memory(args: dict) -> dict:
    model = _tool_model(args)
    memory_id = str((args or {}).get("id", "")).strip()
    text = str((args or {}).get("text", "")).strip()
    if not memory_id or not text:
        return {"error": "id and text are required"}
    if len(text) > config.MAX_MEMORY_CHARS:
        return {"id": memory_id, "error": "text too long"}
    rec = memory_record(memory_id, model=model)
    if rec is None:
        return {"id": memory_id, "error": "memory not found"}
    meta = rec.get("meta") or {}
    extra = {k: v for k, v in meta.items() if k not in BASE_META_KEYS}
    extra["source_type"] = meta.get("source_type", "tool_updated")
    extra["pinned"] = bool(meta.get("pinned", False))
    unit = MemoryUnit(
        id=memory_id,
        text=text,
        category=str((args or {}).get("category") or meta.get("category") or "note")[: config.MAX_CATEGORY_CHARS],
        importance=(args or {}).get("importance") if (args or {}).get("importance") is not None else meta.get("importance", 3),
        date=meta.get("date") or now_iso(),
        source=meta.get("source"),
        access_count=int(meta.get("access_count", 0) or 0),
        extra=extra,
    )
    old_text_only = rec.get("mode") == "text_only"
    try:
        try:
            embedding = embed_document(unit.text, model=model)
            new_text_only = False
        except EmbeddingUnavailable:
            embedding = text_only_embedding()
            new_text_only = True
        # Write the replacement FIRST via upsert (overwrites in place for the
        # same collection). Only once it has landed do we remove the old copy,
        # and only when the storage mode changed so the stale record lives in a
        # DIFFERENT collection. If the write fails, the original is untouched —
        # a failed update must never destroy the memory it was replacing.
        store.upsert(unit, embedding, model=model, text_only=new_text_only)
        if old_text_only != new_text_only:
            store.delete([memory_id], model=model, text_only=old_text_only)
        mode = "text_only" if new_text_only else "vector"
    except Exception as exc:
        return {"id": memory_id, "error": str(exc)}
    return {"id": memory_id, "mode": mode, "category": unit.category, "importance": unit.importance}


def pin_memory(args: dict) -> dict:
    model = _tool_model(args)
    memory_id = str((args or {}).get("id", "")).strip()
    if not memory_id:
        return {"error": "id is required"}
    pinned = bool((args or {}).get("pinned", True))
    n = store.update_metadata_all([memory_id], {"pinned": pinned}, model=model)
    if n <= 0:
        return {"id": memory_id, "error": "memory not found"}
    return {"id": memory_id, "pinned": pinned}


def mark_memory_wrong(args: dict) -> dict:
    model = _tool_model(args)
    memory_id = str((args or {}).get("id", "")).strip()
    if not memory_id:
        return {"error": "id is required"}
    patch = {"superseded": True, "status": "wrong", "wrong_reason": str((args or {}).get("reason", ""))[:500]}
    n = store.update_metadata_all([memory_id], patch, model=model)
    if n <= 0:
        return {"id": memory_id, "error": "memory not found"}
    return {"id": memory_id, "superseded": True}


def merge_memories(args: dict) -> dict:
    model = _tool_model(args)
    raw_ids = (args or {}).get("ids") or []
    ids = raw_ids if isinstance(raw_ids, list) else [raw_ids]
    ids = [str(i).strip() for i in ids if str(i).strip()]
    text = str((args or {}).get("text", "")).strip()
    if len(ids) < 2:
        return {"error": "at least two ids are required"}
    if not text:
        return {"error": "text is required"}
    records = [memory_record(i, model=model) for i in ids]
    missing = [mid for mid, rec in zip(ids, records) if rec is None]
    if missing:
        return {"error": "some memories were not found", "missing": missing}
    unit = MemoryUnit(
        text=text,
        category=str((args or {}).get("category") or (records[0].get("meta") or {}).get("category") or "note")[: config.MAX_CATEGORY_CHARS],
        importance=max(int((r.get("meta") or {}).get("importance", 3) or 3) for r in records if r),
        extra={"source_type": "tool_merged", "merged_from": ",".join(ids[:20])},
    )
    try:
        mid, mode = _store_memory_unit(unit, model=model)
        store.update_metadata_all(ids, {"superseded": True, "status": "merged", "merged_into": mid}, model=model)
    except Exception as exc:
        return {"error": str(exc)}
    return {"id": mid, "mode": mode, "merged_from": ids}


def cite_sources(args: dict | None = None) -> dict:
    args = args or {}
    model = _tool_model(args)
    raw_ids = args.get("memory_ids") or []
    ids = raw_ids if isinstance(raw_ids, list) else [raw_ids]
    rows = []
    for memory_id in ids:
        rec = memory_record(str(memory_id), model=model)
        if rec is None:
            rows.append({"id": memory_id, "error": "memory not found"})
            continue
        meta = rec.get("meta") or {}
        rows.append(
            {
                "id": rec["id"],
                "category": meta.get("category"),
                "source": meta.get("source"),
                "source_type": meta.get("source_type"),
                "date": meta.get("date"),
                "text": (rec.get("text") or "")[:1000],
            }
        )
    query = args.get("query")
    if query and not rows:
        try:
            from brain.retriever import retrieve_weighted

            for hit in retrieve_weighted(str(query), top_k=5, model=model):
                rows.append(
                    {
                        "id": hit.get("id"),
                        "category": hit.get("category"),
                        "source": hit.get("source"),
                        "date": hit.get("date"),
                        "score": hit.get("score"),
                        "text": (hit.get("text") or "")[:1000],
                    }
                )
        except Exception as exc:
            return {"error": str(exc), "sources": rows}
    return {"sources": rows}
