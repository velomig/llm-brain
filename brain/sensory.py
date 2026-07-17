"""Sensory memory: image/audio vectors kept beside text memories.

The text memory path stores descriptions and OCR. This module stores the raw
sensory signature itself in separate Chroma collections, plus a small text anchor
so normal text questions can retrieve "I remember seeing/hearing that source."

The vector backend is deliberately dependency-free for now. It is not CLIP; it is
a stable perceptual/statistical signature with the same storage/retrieval shape a
CLIP backend would use later.
"""
from __future__ import annotations

from datetime import timezone
from io import BytesIO
import hashlib
import math
import re
import wave

import chromadb

import config
from brain.memory import now_iso

_client = chromadb.PersistentClient(path=str(config.DB_DIR))
_BACKEND = "sensory-stat-v1"
_MEDIA_VECTOR_MAX_SAMPLES = 8192
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".aiff", ".aif"}


def is_supported_audio(path: str) -> bool:
    from pathlib import Path

    return Path(path).suffix.lower() in AUDIO_SUFFIXES


def _norm(vec: list[float]) -> list[float]:
    mag = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / mag for v in vec]


def _hash_bucket(token: str, dim: int) -> int:
    return int(hashlib.blake2b(token.encode("utf-8"), digest_size=4).hexdigest(), 16) % dim


def text_anchor_vector(text: str) -> list[float]:
    dim = config.SENSORY_VECTOR_DIM
    vec = [0.0] * dim
    for token in re.findall(r"\w+", (text or "").lower()):
        if len(token) < 2:
            continue
        vec[_hash_bucket(token, dim)] += 1.0
    if not any(vec):
        vec[0] = 1.0
    return _norm(vec)


def media_vector(data: bytes, kind: str) -> list[float]:
    """A fixed-size raw sensory signature from bytes.

    It combines byte histograms, local transitions, and file-kind markers. This
    gives exact-source and near-source recall now, while keeping the Chroma
    collection ready for a future CLIP/audio embedding backend.
    """
    dim = config.SENSORY_VECTOR_DIM
    vec = [0.0] * dim
    if not data:
        vec[0] = 1.0
        return vec
    step = max(1, len(data) // _MEDIA_VECTOR_MAX_SAMPLES)
    weight = float(step)
    for pos in range(0, len(data), step):
        b = data[pos]
        vec[b % dim] += weight
        if pos:
            vec[(b ^ data[pos - 1]) % dim] += 0.35 * weight
    digest = hashlib.sha256(data).digest()
    for i, b in enumerate(digest):
        vec[(i * 7 + b) % dim] += 2.0
    vec[_hash_bucket(f"kind:{kind}", dim)] += 8.0
    return _norm(vec)


def audio_summary(data: bytes, filename: str) -> str:
    """Small deterministic audio description when no transcript is available."""
    details = [f"Audio sensation stored from {filename}."]
    try:
        with wave.open(BytesIO(data), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            duration = frames / rate if rate else 0.0
            details.append(
                f"WAV metadata: {duration:.2f}s, {wav.getnchannels()} channel(s), {rate} Hz."
            )
    except Exception:
        details.append("No transcript is available; retrieval uses the audio sensory vector and filename anchor.")
    return " ".join(details)


def _collection(kind: str, anchor: bool = False):
    name = f"{config.COLLECTION_NAME}_sense_{'anchor' if anchor else 'vec'}_{kind}_{config.SENSORY_VECTOR_DIM}"
    return _client.get_or_create_collection(
        name=name[:63],
        metadata={
            "hnsw:space": "cosine",
            "mode": "sensory_anchor" if anchor else "sensory_vector",
            "backend": _BACKEND,
            "kind": kind,
        },
    )


def _memory_id(kind: str, source: str, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()[:24]
    source_digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    return f"sense_{kind}_{source_digest}_{digest}"


def _media_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _verified_pair_meta(kind: str, cid: str, anchor_meta: dict) -> dict | None:
    """Return vector metadata only when the anchor/vector records match.

    Sensory retrieval starts from lightweight text anchors, but the durable
    identity of a media item lives in its full content hash. Verifying the paired
    vector record prevents a hash-bucket or truncated-id collision from returning
    an anchor that no longer matches the stored sensory signature.
    """
    media_hash = anchor_meta.get("media_sha256")
    if not media_hash:
        return None
    got = _collection(kind).get(ids=[cid], include=["metadatas"])
    metas = got.get("metadatas") or []
    if not metas:
        return None
    vector_meta = metas[0] or {}
    if vector_meta.get("media_sha256") != media_hash:
        return None
    if vector_meta.get("source_sha256") != anchor_meta.get("source_sha256"):
        return None
    if vector_meta.get("kind") != kind or anchor_meta.get("kind") != kind:
        return None
    if vector_meta.get("backend") != anchor_meta.get("backend"):
        return None
    return vector_meta


def _record_snapshot(col, cid: str) -> dict | None:
    """The stored (embedding, document, metadata) for one id, or None."""
    got = col.get(ids=[cid], include=["embeddings", "documents", "metadatas"])
    if not (got.get("ids") or []):
        return None
    embeddings = got.get("embeddings")
    embedding = embeddings[0] if embeddings is not None and len(embeddings) else None
    return {
        "embedding": [float(v) for v in embedding] if embedding is not None else None,
        "document": (got.get("documents") or [None])[0],
        "metadata": (got.get("metadatas") or [None])[0],
    }


def add_media(
    kind: str,
    data: bytes,
    source: str,
    filename: str,
    description: str,
    category: str,
    importance: int = 3,
) -> str | None:
    if not config.SENSORY_EMBEDDINGS_ENABLED or not data or kind not in {"image", "audio"}:
        return None
    sid = _memory_id(kind, source, data)
    media_sha256 = _media_hash(data)
    meta = {
        "schema_version": 2,
        "kind": kind,
        "source": source,
        "source_sha256": _source_hash(source),
        "media_sha256": media_sha256,
        "id_sha256_prefix": media_sha256[:24],
        "filename": filename,
        "category": category,
        "importance": int(max(1, min(5, importance))),
        "date": now_iso(),
        "source_type": f"sensory_{kind}",
        "backend": _BACKEND,
        "description": (description or "")[:3000],
    }
    doc = f"I remember {'seeing' if kind == 'image' else 'hearing'} {filename}. {description}".strip()
    # Write the replacement FIRST, delete the superseded records LAST. The old
    # delete-first order meant a failure between the two writes destroyed the
    # existing record and could strand a vector without its anchor. upsert
    # overwrites in place when the id already exists (identical bytes
    # re-uploaded for the same source), so the prior pair is snapshotted and
    # restored if the second write fails.
    vec_col = _collection(kind)
    anchor_col = _collection(kind, anchor=True)
    written: list[tuple] = []
    try:
        prior_vec = _record_snapshot(vec_col, sid)
        prior_anchor = _record_snapshot(anchor_col, sid)
        vec_col.upsert(
            ids=[sid],
            embeddings=[media_vector(data, kind)],
            documents=[doc],
            metadatas=[meta],
        )
        written.append((vec_col, prior_vec))
        anchor_col.upsert(
            ids=[sid],
            embeddings=[text_anchor_vector(f"{filename} {description}")],
            documents=[doc],
            metadatas=[meta],
        )
        written.append((anchor_col, prior_anchor))
    except Exception:
        # Best-effort rollback of the half-written replacement so the failure
        # leaves neither an orphan vector nor a clobbered prior record.
        for col, prior in written:
            try:
                if prior is None:
                    col.delete(ids=[sid])
                else:
                    col.upsert(
                        ids=[sid],
                        embeddings=[prior["embedding"]],
                        documents=[prior["document"]],
                        metadatas=[prior["metadata"]],
                    )
            except Exception:
                pass
        raise
    delete_source(source, kind=kind, keep_ids={sid})
    return sid


def delete_source(source: str, kind: str | None = None, keep_ids: set[str] | None = None) -> None:
    keep = keep_ids or set()
    kinds = [kind] if kind else ["image", "audio"]
    for k in kinds:
        for anchor in (False, True):
            col = _collection(k, anchor=anchor)
            got = col.get(where={"source": source})
            ids = [cid for cid in (got.get("ids") or []) if cid not in keep]
            if ids:
                col.delete(ids=ids)


def list_records(source: str | None = None) -> list[dict]:
    rows: list[dict] = []
    for kind in ("image", "audio"):
        kwargs = {"include": ["documents", "metadatas"]}
        if source:
            kwargs["where"] = {"source": source}
        data = _collection(kind, anchor=True).get(**kwargs)
        for cid, doc, meta in zip(data.get("ids") or [], data.get("documents") or [], data.get("metadatas") or []):
            rows.append({"id": cid, "text": doc, "meta": meta or {}, "kind": kind})
    return rows


def retrieve_text(question: str, top_k: int | None = None, category: str | None = None) -> list[dict]:
    if not config.SENSORY_EMBEDDINGS_ENABLED:
        return []
    k = top_k or config.SENSORY_RECALL_K
    qvec = text_anchor_vector(question)
    rows: list[dict] = []
    for kind in ("image", "audio"):
        col = _collection(kind, anchor=True)
        if col.count() <= 0:
            continue
        res = col.query(
            query_embeddings=[qvec],
            n_results=min(k, col.count()),
            include=["documents", "metadatas", "distances"],
        )
        for rank, cid in enumerate(res.get("ids", [[]])[0]):
            meta = (res.get("metadatas", [[]])[0][rank] or {})
            if category and meta.get("category") != category:
                continue
            vector_meta = _verified_pair_meta(kind, cid, meta)
            if vector_meta is None:
                continue
            distance = float(res.get("distances", [[]])[0][rank])
            meta = {**meta, "hash_verified": True, "vector_backend": vector_meta.get("backend")}
            rows.append(
                {
                    "id": f"sense:{cid}",
                    "text": res.get("documents", [[]])[0][rank],
                    "meta": meta,
                    "kind": kind,
                    "sensory_id": cid,
                    "sensory_similarity": round(1.0 - distance, 4),
                    "_rank": rank,
                }
            )
    rows.sort(key=lambda r: (-r["sensory_similarity"], r["kind"], r["sensory_id"]))
    return rows[:k]


def clear_all() -> None:
    for kind in ("image", "audio"):
        for anchor in (False, True):
            col = _collection(kind, anchor=anchor)
            ids = col.get().get("ids") or []
            if ids:
                col.delete(ids=ids)
