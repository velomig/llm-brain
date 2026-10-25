"""Vector store wrapper (ChromaDB, persisted to disk).

We always pass embeddings in explicitly (computed by the selected Ollama model),
so Chroma never tries to download its own default embedding model. Each embedding
model gets its own cosine-space collection because vector dimensions can differ.
"""
from __future__ import annotations

from functools import lru_cache, wraps
import hashlib
import json
import os
import re
import threading
import uuid

import chromadb

import config
from brain import graph
from brain.utils import file_lock
from brain.memory import MemoryUnit, age_days, clamp_importance, now_iso, repaired_metadata

_client = chromadb.PersistentClient(path=str(config.DB_DIR))
_VERSIONS: dict[str, int] = {}
_REVISION_FILE = config.DB_DIR / ".neural_map_revisions.json"
_REVISION_THREAD_LOCK = threading.Lock()


def mutation_lock():
    """Serialize memory mutations with approval snapshot checks and undo.

    Reentrant across nested store/tool operations and exclusive across processes.
    A store-wide lock also covers merges and changes between storage modes.
    """
    return file_lock(config.DB_DIR / ".neural_map_memory_mutations.lock")


def memory_transaction(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        with mutation_lock():
            return func(*args, **kwargs)
    return wrapped


def _collection_name(model: str | None = None, text_only: bool = False) -> str:
    embed_model = config.embedding_model(model)
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", embed_model).strip("_")[:32] or "model"
    digest = hashlib.sha1(embed_model.encode("utf-8")).hexdigest()[:12]
    mode = "text" if text_only else "vec"
    return f"{config.COLLECTION_NAME}_{mode}_{slug}_{digest}"


def _bump_version(model: str | None = None, text_only: bool = False) -> None:
    name = _collection_name(model, text_only=text_only)
    _bump_version_name(name)


def _bump_version_name(name: str) -> None:
    with _REVISION_THREAD_LOCK:
        with _revision_lock():
            versions = _read_versions()
            next_version = max(_VERSIONS.get(name, 0), int(versions.get(name, 0))) + 1
            _VERSIONS[name] = next_version
            versions[name] = next_version
            _write_versions(versions)


def _revision_lock():
    """Cross-process exclusive lock on this module's state file."""
    return file_lock(_REVISION_FILE.with_name(".neural_map_revisions.lock"))


_METADATA_THREAD_LOCK = threading.Lock()


def _metadata_lock():
    """Cross-process lock serializing metadata read-modify-write cycles.

    reinforce/mark_superseded/update_metadata each rebuild the WHOLE metadata
    object from a fresh read. Without serialization, a concurrent recall's
    reinforce can write back stale pinned/status/importance fields it read
    before another process's update landed — silently undoing a user choice
    (and pinned participates in outbound portfolio-symbol authorization)."""
    return file_lock(config.DB_DIR / ".neural_map_metadata.lock")


def _read_versions() -> dict[str, int]:
    try:
        with _REVISION_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): int(v) for k, v in data.items() if isinstance(v, int | float | str) and str(v).isdigit()}


def _write_versions(versions: dict[str, int]) -> None:
    _REVISION_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _REVISION_FILE.with_name(f"{_REVISION_FILE.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(versions, fh, sort_keys=True, separators=(",", ":"))
    os.replace(tmp, _REVISION_FILE)


@lru_cache(maxsize=32)
def _collection(model: str | None = None, text_only: bool = False):
    embed_model = config.embedding_model(model)
    return _client.get_or_create_collection(
        name=_collection_name(model, text_only=text_only),
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": embed_model,
            "mode": "text_only" if text_only else "vector",
        },
    )


@memory_transaction
def add(unit: MemoryUnit, embedding: list[float], model: str | None = None, text_only: bool = False) -> None:
    _collection(model, text_only=text_only).add(
        ids=[unit.id],
        embeddings=[embedding],
        documents=[unit.text],
        metadatas=[unit.metadata()],
    )
    _bump_version(model, text_only=text_only)


@memory_transaction
def upsert(unit: MemoryUnit, embedding: list[float], model: str | None = None, text_only: bool = False) -> None:
    """Insert or replace a record by id in one collection. Unlike add(), an
    existing id is overwritten in place rather than erroring, so a caller can
    write a replacement without first deleting the original — the update stays
    non-destructive if the write fails."""
    _collection(model, text_only=text_only).upsert(
        ids=[unit.id],
        embeddings=[embedding],
        documents=[unit.text],
        metadatas=[unit.metadata()],
    )
    _bump_version(model, text_only=text_only)


@memory_transaction
def restore_record(unit: MemoryUnit, embedding: list[float], model: str | None = None, text_only: bool = False) -> None:
    """Restore exact metadata without deleting the existing record first."""
    collection = _collection(model, text_only=text_only)
    current = collection.get(ids=[unit.id], include=["metadatas"])
    if not current["ids"]:
        upsert(unit, embedding, model=model, text_only=text_only)
        return
    metadata = unit.metadata()
    # Chroma update merges metadata; null removes fields introduced by the action.
    patch = {key: None for key in (current["metadatas"][0] or {}) if key not in metadata}
    collection.update(ids=[unit.id], documents=[unit.text], embeddings=[embedding],
                      metadatas=[{**patch, **metadata}])
    _bump_version(model, text_only=text_only)


@memory_transaction
def add_many(
    units: list[MemoryUnit],
    embeddings: list[list[float]],
    model: str | None = None,
    text_only: bool = False,
    batch_size: int = 1000,
) -> None:
    """Bulk insert, chunked to stay under Chroma's per-call limit (fast ingest)."""
    units = list(units)
    embeddings = list(embeddings)
    col = _collection(model, text_only=text_only)
    wrote = False
    try:
        for i in range(0, len(units), batch_size):
            batch = units[i:i + batch_size]
            col.add(
                ids=[u.id for u in batch],
                embeddings=embeddings[i:i + batch_size],
                documents=[u.text for u in batch],
                metadatas=[u.metadata() for u in batch],
            )
            wrote = True
    finally:
        if wrote:
            _bump_version(model, text_only=text_only)


def get_all(model: str | None = None, text_only: bool = False) -> dict:
    """Everything in the index (ids, documents, metadatas, embeddings).

    Fine for a personal-scale "second brain"; the hybrid re-ranker loads this
    once per query. For a truly massive corpus you'd push BM25 into a real
    keyword index instead of loading all docs into RAM.
    """
    return _collection(model, text_only=text_only).get(include=["documents", "metadatas", "embeddings"])


def list_records(model: str | None = None, text_only: bool = False) -> dict:
    """Everything needed for memory management, without pulling embeddings."""
    return _collection(model, text_only=text_only).get(include=["documents", "metadatas"])


def get_by_category(category: str, model: str | None = None, text_only: bool = False) -> dict:
    return _collection(model, text_only=text_only).get(where={"category": category}, include=["documents", "metadatas"])


def get_by_ids(ids: list[str], model: str | None = None) -> dict[str, dict]:
    """Fetch documents+metadata for specific ids (across both collections).

    Returns {id: {"text", "meta"}}. Used by graph-based associative recall to
    turn fact ids back into candidate memories.
    """
    out: dict[str, dict] = {}
    if not ids:
        return out
    for text_only in (False, True):
        got = _collection(model, text_only=text_only).get(ids=ids, include=["documents", "metadatas"])
        for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
            out[cid] = {"text": doc, "meta": meta or {}}
    return out


def count(model: str | None = None, text_only: bool = False) -> int:
    return _collection(model, text_only=text_only).count()


def collection_name(model: str | None = None, text_only: bool = False) -> str:
    """Public name of the collection for a model (used as a cache key)."""
    return _collection_name(model, text_only=text_only)


def collection_version(model: str | None = None, text_only: bool = False) -> int:
    """Persisted mutation counter for retrieval caches, shared across processes."""
    name = _collection_name(model, text_only=text_only)
    version = max(_VERSIONS.get(name, 0), int(_read_versions().get(name, 0)))
    _VERSIONS[name] = version
    return version


def query(
    query_embedding: list[float],
    n_results: int,
    model: str | None = None,
    text_only: bool = False,
    where: dict | None = None,
) -> dict:
    """ChromaDB ANN search → nearest neighbors by cosine distance.

    Uses the HNSW index, so it does NOT load the whole collection into RAM.
    An optional `where` metadata filter restricts the search to a subset
    (e.g. one category, or importance >= N) — used for balanced candidate recall.
    """
    col = _collection(model, text_only=text_only)
    cnt = col.count()
    if cnt == 0:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": min(n_results, cnt),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    return col.query(**kwargs)


def categories(model: str | None = None, text_only: bool = False) -> list[str]:
    """Distinct category labels present in the collection (sorted).

    Pulls metadata only (no embeddings). Callers should cache this — it scans
    every record's metadata, so it's cheap per record but not free at scale.
    """
    metas = _collection(model, text_only=text_only).get(include=["metadatas"])["metadatas"] or []
    return sorted({(m or {}).get("category") for m in metas if (m or {}).get("category")})


@memory_transaction
def reinforce(ids: list[str], model: str | None = None, text_only: bool = False) -> int:
    """Strengthen recalled memories: bump access_count and last_accessed.

    Deliberately does NOT bump the collection version — text and ids are
    unchanged, so the BM25/keyword cache stays warm across turns. (Importance
    isn't touched here, so cached metadata only goes mildly stale on the
    access_count field, which no ranking signal reads.)
    """
    if not ids:
        return 0
    col = _collection(model, text_only=text_only)
    with _METADATA_THREAD_LOCK, _metadata_lock():
        got = col.get(ids=ids, include=["metadatas"])
        found = got["ids"]
        if not found:
            return 0
        stamp = now_iso()
        metas = [
            {**(m or {}), "access_count": int((m or {}).get("access_count", 0)) + 1, "last_accessed": stamp}
            for m in got["metadatas"]
        ]
        col.update(ids=found, metadatas=metas)
    return len(found)


@memory_transaction
def mark_superseded(ids: list[str], model: str | None = None, text_only: bool = False) -> int:
    """Retire memories that newer information has made obsolete (belief update).

    They stay on disk (audit trail) but retrieval filters them out, so the model
    answers from current truth.
    """
    if not ids:
        return 0
    col = _collection(model, text_only=text_only)
    with _METADATA_THREAD_LOCK, _metadata_lock():
        got = col.get(ids=ids, include=["metadatas"])
        found = got["ids"]
        if not found:
            return 0
        metas = [{**(m or {}), "superseded": True} for m in got["metadatas"]]
        col.update(ids=found, metadatas=metas)
    _bump_version(model, text_only=text_only)
    return len(found)


@memory_transaction
def update_metadata(ids: list[str], patch: dict, model: str | None = None, text_only: bool = False) -> int:
    """Patch metadata for existing records without touching text or embeddings.

    INITIATIVE LAYER: this is used to dismiss an initiative by setting
    status="dismissed". Keeping it as a generic, tiny helper makes rollback easy:
    remove the initiative endpoints and this remains a harmless management API.
    """
    if not ids or not patch:
        return 0
    col = _collection(model, text_only=text_only)
    with _METADATA_THREAD_LOCK, _metadata_lock():
        got = col.get(ids=ids, include=["metadatas"])
        found = got["ids"]
        if not found:
            return 0
        metas = [{**(m or {}), **patch} for m in got["metadatas"]]
        col.update(ids=found, metadatas=metas)
    _bump_version(model, text_only=text_only)
    return len(found)


@memory_transaction
def prune_episodes(
    model: str | None = None,
    older_than_days: float | None = None,
    max_importance: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Forget raw episodes that are old, low-importance, and never recalled.

    Mimics human forgetting: the gist (facts) is consolidated and kept, while the
    raw transcript fades. A memory that was ever recalled (access_count > 0) is
    spared. Facts and ingested documents are never touched. Superseded episodes
    are always eligible. Returns {"pruned": n, "ids": [...]}.
    """
    older = config.PRUNE_EPISODE_AFTER_DAYS if older_than_days is None else older_than_days
    max_imp = config.PRUNE_MAX_IMPORTANCE if max_importance is None else max_importance
    fade_cats = {config.EPISODE_CATEGORY, "conversation"}
    victims: list[tuple[str, bool]] = []
    for text_only in (False, True):
        col = _collection(model, text_only=text_only)
        got = col.get(include=["metadatas"])
        for cid, meta in zip(got["ids"], got["metadatas"]):
            meta = meta or {}
            if meta.get("category") not in fade_cats:
                continue
            if meta.get("pinned"):
                continue
            if int(meta.get("access_count", 0)) > 0:
                continue
            if meta.get("superseded"):
                victims.append((cid, text_only))
                continue
            if clamp_importance(meta.get("importance", 3)) > max_imp:
                continue
            age = age_days(meta.get("date"))
            if age is None or age < older:
                continue
            victims.append((cid, text_only))
    if not dry_run:
        for text_only in (False, True):
            ids = [cid for cid, t in victims if t == text_only]
            if ids:
                _collection(model, text_only=text_only).delete(ids=ids)
                _bump_version(model, text_only=text_only)
        # A pruned episode's graph edges must go with it, or every prune turns
        # graph-linked episodes into dangling_graph_fact integrity errors that
        # repair has to mop up. Done here so every caller (API endpoint,
        # reflect job, chat CLI) gets the cleanup, not just one of them.
        if victims:
            graph.delete_fact_ids(
                [cid for cid, _ in victims], collections=graph_collection_names(model)
            )
    return {"pruned": len(victims), "ids": [cid for cid, _ in victims]}


@memory_transaction
def delete(ids: list[str], model: str | None = None, text_only: bool = False) -> None:
    if ids:
        _collection(model, text_only=text_only).delete(ids=ids)
        _bump_version(model, text_only=text_only)


@memory_transaction
def delete_by_source(source: str, model: str | None = None, text_only: bool = False) -> None:
    """Remove every chunk that came from a given source file (for clean re-ingest)."""
    _collection(model, text_only=text_only).delete(where={"source": source})
    _bump_version(model, text_only=text_only)


# --- Both-store helpers (text-only as a first-class degraded mode) -------------
# A memory lives in the VECTOR collection normally, or in the TEXT-ONLY one when
# embeddings were unavailable at write time. Anything that searches, counts,
# manages, or deletes memories must span BOTH so the degraded-mode records are
# never invisible. These helpers centralize that iteration so no caller has to
# remember to do it twice (and can't get it half-right).
ALL_MODES = (False, True)   # vector store, then the text-only fallback store


def count_all(model: str | None = None) -> int:
    """Total memories across both collections."""
    return sum(count(model, text_only=t) for t in ALL_MODES)


def graph_collection_names(model: str | None = None) -> list[str | None]:
    """The graph-edge scopes a memory deletion must clean: vector, text-only,
    and the legacy/untagged scope — the same set integrity repair targets for
    dangling edges. Anything that calls delete_all on graph-backed memories
    passes this to graph.delete_fact_ids so the edges go with them."""
    return [collection_name(model), collection_name(model, text_only=True), None]


@memory_transaction
def delete_all(ids: list[str], model: str | None = None) -> None:
    """Delete the given ids from whichever collection holds them (missing ids are
    a harmless no-op in Chroma), so callers needn't know which store a row is in."""
    for t in ALL_MODES:
        delete(ids, model=model, text_only=t)


@memory_transaction
def delete_source_all(source: str, model: str | None = None) -> None:
    """Remove a source file's chunks from both collections (clean re-ingest)."""
    for t in ALL_MODES:
        delete_by_source(source, model=model, text_only=t)


@memory_transaction
def mark_superseded_all(ids: list[str], model: str | None = None) -> int:
    """Retire memories by id across both collections. Returns total marked."""
    return sum(mark_superseded(ids, model=model, text_only=t) for t in ALL_MODES)


@memory_transaction
def reinforce_all(ids: list[str], model: str | None = None) -> int:
    """Strengthen recalled memories across both collections. Returns total bumped."""
    return sum(reinforce(ids, model=model, text_only=t) for t in ALL_MODES)


@memory_transaction
def update_metadata_all(ids: list[str], patch: dict, model: str | None = None) -> int:
    """Patch metadata by id across vector and text-only collections."""
    return sum(update_metadata(ids, patch, model=model, text_only=t) for t in ALL_MODES)


def categories_all(model: str | None = None) -> list[str]:
    """Distinct category labels present in EITHER collection (sorted)."""
    cats: set[str] = set()
    for t in ALL_MODES:
        cats.update(categories(model, text_only=t))
    return sorted(cats)


def get_by_category_all(category: str, model: str | None = None) -> dict:
    """Merged ids/documents/metadatas for a category across both collections, each
    row tagged with its store `mode` ("vector" | "text_only")."""
    merged: dict[str, list] = {"ids": [], "documents": [], "metadatas": [], "modes": []}
    for t in ALL_MODES:
        got = get_by_category(category, model=model, text_only=t)
        cids = got.get("ids") or []
        merged["ids"].extend(cids)
        merged["documents"].extend(got.get("documents") or [])
        merged["metadatas"].extend(got.get("metadatas") or [])
        merged["modes"].extend(["text_only" if t else "vector"] * len(cids))
    return merged


def list_records_all(model: str | None = None) -> list[dict]:
    """Every stored memory across both collections as flat rows: each is
    {"id", "text", "meta", "mode"} with mode "vector" | "text_only"."""
    rows: list[dict] = []
    for t in ALL_MODES:
        data = list_records(model, text_only=t)
        mode = "text_only" if t else "vector"
        for cid, text, meta in zip(data.get("ids") or [], data.get("documents") or [], data.get("metadatas") or []):
            rows.append({"id": cid, "text": text, "meta": meta or {}, "mode": mode})
    return rows


@memory_transaction
def clear_all(model: str | None = None) -> int:
    """Delete every memory from both collections. Used by a full (--replace)
    restore so the imported set is authoritative. Returns how many were removed."""
    removed = 0
    for t in ALL_MODES:
        col = _collection(model, text_only=t)
        ids = col.get().get("ids") or []
        if ids:
            col.delete(ids=ids)
            removed += len(ids)
            _bump_version(model, text_only=t)
    return removed


@memory_transaction
def repair_schema(model: str | None = None, dry_run: bool = False) -> dict:
    """Upgrade pre-versioned records to the current metadata schema, in both stores.

    Metadata-only (no re-embedding), and idempotent: records already at the
    current SCHEMA_VERSION are left untouched, so it's safe to run repeatedly.
    Only collections that actually change get their version bumped (keeping
    retrieval caches warm). Returns {"scanned", "migrated", "by_mode": {...}}.
    See migrate.py for the CLI wrapper.
    """
    summary: dict = {"scanned": 0, "migrated": 0, "by_mode": {}}
    for t in ALL_MODES:
        col = _collection(model, text_only=t)
        got = col.get(include=["metadatas"])
        ids = got.get("ids") or []
        metas = got.get("metadatas") or []
        summary["scanned"] += len(ids)
        fix_ids: list[str] = []
        fix_metas: list[dict] = []
        for cid, meta in zip(ids, metas):
            upgraded = repaired_metadata(meta)
            if upgraded is not None:
                fix_ids.append(cid)
                fix_metas.append(upgraded)
        mode = "text_only" if t else "vector"
        summary["by_mode"][mode] = len(fix_ids)
        summary["migrated"] += len(fix_ids)
        if fix_ids and not dry_run:
            col.update(ids=fix_ids, metadatas=fix_metas)
            _bump_version(model, text_only=t)
    return summary
