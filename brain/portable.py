"""Portable backup: export memories to a JSON/Markdown shape and restore them.

ChromaDB and the graph SQLite file are implementation details. A memory is really
just text + metadata (category, importance, date, source, ...) plus the entity
edges that link facts together. This module serializes exactly that, so a backup
survives a Chroma version bump, a different machine, or even a different embedding
model.

Embeddings are deliberately NOT exported: they're large and specific to one model.
Import re-embeds each memory's text with the active model (falling back to the
text-only store when embeddings are unavailable, same as normal ingest), so a
backup is portable across embedding models — the text is the source of truth.

JSON is the canonical round-trip format (ids and metadata preserved); Markdown is
a human-readable archive of the same data.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import config
from brain import graph, store
from brain.embeddings import EmbeddingUnavailable, embed_documents, text_only_embedding
from brain.memory import SCHEMA_VERSION, MemoryUnit, now_iso

EXPORT_VERSION = 1
BACKUP_DIR = config.DB_DIR / "backups"

# Metadata keys that map to MemoryUnit's own fields; everything else in a record's
# metadata (superseded, due, status, last_accessed, ...) round-trips via `extra`.
_CORE_META = {"category", "importance", "date", "source", "access_count", "schema_version"}
_NON_META = {"id", "text", "mode"}


def export_data(model: str | None = None) -> dict:
    """The portable snapshot: every memory for `model` (both stores) flattened to
    text+metadata, plus the graph edges that link them. Self-describing envelope.

    The export is scoped to one model's store, so the graph is filtered to edges
    whose fact_id is among the exported memories. That keeps the snapshot internally
    consistent: no dangling edges pointing at facts that aren't in the payload
    (whether they belong to another collection or were deleted out from under a
    stale edge). Fact-less edges carry no recall signal and are omitted."""
    memories = []
    ids: set[str] = set()
    for rec in store.list_records_all(model):
        meta = rec.get("meta") or {}
        memories.append({"id": rec["id"], "text": rec["text"], "mode": rec["mode"], **meta})
        ids.add(rec["id"])
    collections = {
        store.collection_name(model),
        store.collection_name(model, text_only=True),
        None,  # legacy/unscoped edges are best-effort portable context
    }
    edges = [
        e
        for e in graph.export_edges()
        if e.get("fact_id") in ids and e.get("collection") in collections
    ]
    return {
        "neural_map_export": {
            "version": EXPORT_VERSION,
            "exported_at": now_iso(),
            "schema_version": SCHEMA_VERSION,
            "embedding_model": config.embedding_model(model),
            "count": len(memories),
            "memories": memories,
            "graph": edges,
        }
    }


def _safe_reason(reason: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", reason.strip().lower()).strip("-")
    return slug[:60] or "backup"


def save_backup(reason: str, model: str | None = None) -> Path:
    """Write a timestamped portable JSON backup and return its path."""
    data = export_data(model=model)
    stamp = now_iso().replace(":", "").replace("+", "Z")
    path = BACKUP_DIR / f"{stamp}_{_safe_reason(reason)}.json"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(json.dumps(data, ensure_ascii=False, indent=2))
    finally:
        if fd >= 0:
            os.close(fd)
    return path


def _payload(data: dict) -> dict:
    payload = (data or {}).get("neural_map_export")
    if not isinstance(payload, dict) or not isinstance(payload.get("memories"), list):
        raise ValueError("not a Neural Map export (missing 'neural_map_export.memories')")
    return payload


def _unit_from_record(rec: dict) -> MemoryUnit:
    """Rebuild a MemoryUnit from an exported record, preserving id and carrying any
    non-core metadata through `extra` so it survives the round trip."""
    extra = {k: v for k, v in rec.items() if k not in _CORE_META and k not in _NON_META}
    return MemoryUnit(
        id=rec["id"],
        text=rec.get("text", ""),
        category=rec.get("category") or "conversation",
        importance=rec.get("importance", 3),
        date=rec.get("date") or now_iso(),
        source=rec.get("source"),
        access_count=rec.get("access_count", 0),
        extra=extra or None,
    )


def _snapshot_existing(ids: set[str], model: str | None = None) -> dict[bool, tuple[list[MemoryUnit], list[list[float]]]]:
    """Existing rows about to be overwritten, including their current embeddings."""
    snapshot: dict[bool, tuple[list[MemoryUnit], list[list[float]]]] = {False: ([], []), True: ([], [])}
    if not ids:
        return snapshot
    for text_only in (False, True):
        data = store.get_all(model, text_only=text_only)
        units, embeddings = snapshot[text_only]
        stored_embeddings = data.get("embeddings")
        for cid, text, meta, embedding in zip(
            data.get("ids") or [],
            data.get("documents") or [],
            data.get("metadatas") or [],
            stored_embeddings if stored_embeddings is not None else [],
        ):
            if cid not in ids:
                continue
            units.append(_unit_from_record({"id": cid, "text": text, **(meta or {})}))
            embeddings.append(list(embedding))
    return snapshot


def _snapshot_all(model: str | None = None) -> dict[bool, tuple[list[MemoryUnit], list[list[float]]]]:
    """Every existing row for the target model, with embeddings, for replace rollback."""
    snapshot: dict[bool, tuple[list[MemoryUnit], list[list[float]]]] = {False: ([], []), True: ([], [])}
    for text_only in (False, True):
        data = store.get_all(model, text_only=text_only)
        units, embeddings = snapshot[text_only]
        stored_embeddings = data.get("embeddings")
        for cid, text, meta, embedding in zip(
            data.get("ids") or [],
            data.get("documents") or [],
            data.get("metadatas") or [],
            stored_embeddings if stored_embeddings is not None else [],
        ):
            units.append(_unit_from_record({"id": cid, "text": text, **(meta or {})}))
            embeddings.append(list(embedding))
    return snapshot


def _restore_snapshot(snapshot: dict[bool, tuple[list[MemoryUnit], list[list[float]]]], model: str | None = None) -> None:
    for text_only, (units, embeddings) in snapshot.items():
        if units:
            store.add_many(units, embeddings, model=model, text_only=text_only)


def _replace_graph_collections(model: str | None) -> list[str]:
    return [
        store.collection_name(model),
        store.collection_name(model, text_only=True),
    ]


def _snapshot_replace_edges(model: str | None) -> list[dict]:
    collections = set(_replace_graph_collections(model))
    collections.add(None)  # legacy/unscoped edges are cleared by replace too
    return [e for e in graph.export_edges() if e.get("collection") in collections]


def _snapshot_merge_edges(imported_ids: set[str], model: str | None) -> list[dict]:
    """Existing edges of the memories a merge import is about to overwrite."""
    collections = set(store.graph_collection_names(model))
    return [
        e
        for e in graph.export_edges()
        if e.get("fact_id") in imported_ids and e.get("collection") in collections
    ]


def import_data(data: dict, model: str | None = None, replace: bool = False) -> dict:
    """Restore memories + graph from an export.

    replace=True wipes both stores for the target model and that model's scoped
    graph edges first (authoritative restore for this model); otherwise records
    are merged, overwriting any with the same id. Re-embeds text with the active
    model, and rewrites restored edges' collection to the destination store, so
    it works regardless of which model produced the backup. Returns
    {"memories", "edges", "mode"}.
    """
    payload = _payload(data)
    units = [_unit_from_record(r) for r in payload["memories"] if isinstance(r, dict) and r.get("id")]
    imported_ids = {u.id for u in units}

    text_only = False
    embeddings: list[list[float]] = []
    if units:
        texts = [u.text for u in units]
        try:
            embeddings = embed_documents(texts, model=model)
        except EmbeddingUnavailable:
            text_only = True
            embeddings = [text_only_embedding() for _ in units]

    rollback: dict[bool, tuple[list[MemoryUnit], list[list[float]]]] | None = None
    rollback_edges: list[dict] = []
    if replace:
        rollback = _snapshot_all(model)
        rollback_edges = _snapshot_replace_edges(model)
        store.clear_all(model)
        graph.clear_collections(_replace_graph_collections(model), include_legacy=True)
    elif units:
        rollback = _snapshot_existing(imported_ids, model=model)
        rollback_edges = _snapshot_merge_edges(imported_ids, model)
        # Merge: drop any existing copy (in either store) so ids don't collide and
        # a memory never ends up duplicated across the vector/text-only stores.
        # Embeddings are prepared BEFORE this destructive step, so a hard embed
        # failure cannot delete the user's existing records. Existing rows are
        # snapshotted too, so a Chroma write failure after this point can roll back
        # to the pre-import state.
        store.delete_all([u.id for u in units], model=model)
        # An overwritten memory's old edges describe its old text; keeping them
        # alongside the imported ones would leave both associations pointing at
        # the updated record. Replace them wholesale (the snapshot above brings
        # them back on rollback).
        graph.delete_fact_ids(list(imported_ids), collections=store.graph_collection_names(model))

    try:
        if units:
            store.add_many(units, embeddings, model=model, text_only=text_only)
        # The memories were just re-embedded into THIS store, so their facts now live in
        # the destination collection. Rewrite each restored edge's collection to the
        # actual landing collection (vector or text-only) so graph recall can resolve
        # it even when the backup came from a different embedding model or embeddings
        # are currently unavailable. Edges whose fact wasn't imported keep their
        # original collection (we can't know where that fact lives).
        destination = store.collection_name(model, text_only=text_only)
        edge_rows = [
            {**row, "collection": destination} if isinstance(row, dict) and row.get("fact_id") in imported_ids else row
            for row in (payload.get("graph") or [])
        ]
        edges = graph.import_edges(edge_rows)
    except Exception:
        if replace:
            store.clear_all(model)
            graph.clear_collections(_replace_graph_collections(model), include_legacy=True)
            if rollback is not None:
                _restore_snapshot(rollback, model=model)
            graph.import_edges(rollback_edges)
        else:
            store.delete_all(list(imported_ids), model=model)
            # Clear whatever landed for the imported ids (including any edges
            # import_edges wrote before failing), then put back the snapshots.
            graph.delete_fact_ids(list(imported_ids), collections=store.graph_collection_names(model))
            if rollback is not None:
                _restore_snapshot(rollback, model=model)
            graph.import_edges(rollback_edges)
        raise
    return {"memories": len(units), "edges": edges, "mode": "text_only" if text_only else "vector"}


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def to_markdown(data: dict) -> str:
    """Render an export as a human-readable Markdown archive, grouped by category."""
    payload = _payload(data)
    lines = [
        "# Neural Map Memory Export",
        "",
        f"_Exported {payload.get('exported_at', '')} · {payload.get('count', 0)} memories · "
        f"embedding model `{payload.get('embedding_model', '')}`_",
        "",
    ]
    by_cat: dict[str, list[dict]] = {}
    for m in payload["memories"]:
        by_cat.setdefault(m.get("category") or "uncategorized", []).append(m)
    for cat in sorted(by_cat):
        lines.append(f"## {cat}")
        lines.append("")
        for m in by_cat[cat]:
            facets = [f"importance {m.get('importance', 3)}"]
            if m.get("date"):
                facets.append(str(m["date"])[:10])
            if m.get("source"):
                facets.append(str(m["source"]))
            lines.append(f"- **{' · '.join(facets)}**  ")
            lines.append(f"  {_one_line(m.get('text', ''))}  ")
            lines.append(f"  `id: {m.get('id', '')} · {m.get('mode', '')}`")
        lines.append("")
    graph_rows = payload.get("graph") or []
    if graph_rows:
        lines.append(f"## graph ({len(graph_rows)} edges)")
        lines.append("")
        for e in graph_rows:
            lines.append(f"- {e.get('subject')} — *{e.get('relation')}* → {e.get('object')}")
        lines.append("")
    return "\n".join(lines)
