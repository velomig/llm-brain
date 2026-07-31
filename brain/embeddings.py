"""Embedding model wrapper (Ollama).

Documents stored in the index use `search_document:` and live queries use
`search_query:`. Models trained with task prefixes use that signal for better
retrieval, and models that do not recognize it still receive a consistent prompt.
"""
from __future__ import annotations

import ollama

import config
from brain import log

_client = ollama.Client(host=config.OLLAMA_HOST, timeout=config.OLLAMA_TIMEOUT_SECONDS)


class EmbeddingUnavailable(RuntimeError):
    """The selected model/server cannot produce embeddings."""


def _is_embedding_unavailable(exc: Exception) -> bool:
    message = str(exc).lower()
    return "does not support embeddings" in message or ("embedding" in message and "not support" in message)


def _embed(text: str, model: str | None = None) -> list[float]:
    name = config.embedding_model(model)
    try:
        resp = _client.embeddings(model=name, prompt=text)
    except Exception as exc:
        if _is_embedding_unavailable(exc):
            log.fallback("embedding_unavailable", exc, model=name, mode="single")
            raise EmbeddingUnavailable(str(exc)) from exc
        raise
    return list(resp["embedding"])


def embed_document(text: str, model: str | None = None) -> list[float]:
    """Embed text that will be STORED in the vector index."""
    return _embed(f"search_document: {text}", model=model)


def embed_query(text: str, model: str | None = None) -> list[float]:
    """Embed a live QUESTION used to search the index."""
    return _embed(f"search_query: {text}", model=model)


def _embed_batch(prompts: list[str], model: str | None = None) -> list[list[float]]:
    name = config.embedding_model(model)
    try:
        resp = _client.embed(model=name, input=prompts)
    except Exception as exc:
        if _is_embedding_unavailable(exc):
            log.fallback("embedding_unavailable", exc, model=name, mode="batch", count=len(prompts))
            raise EmbeddingUnavailable(str(exc)) from exc
        raise
    embs = getattr(resp, "embeddings", None)
    if embs is None and isinstance(resp, dict):
        embs = resp["embeddings"]
    return [list(e) for e in embs]


def embed_documents(texts: list[str], model: str | None = None, batch_size: int | None = None) -> list[list[float]]:
    """Embed many documents for STORAGE, batched (one Ollama call per batch).

    ~3-4x faster than one-at-a-time for bulk ingestion.
    """
    size = batch_size or config.EMBED_BATCH_SIZE
    out: list[list[float]] = []
    for i in range(0, len(texts), size):
        out.extend(_embed_batch([f"search_document: {t}" for t in texts[i:i + size]], model=model))
    return out


def text_only_embedding() -> list[float]:
    """Placeholder vector for Chroma records that are retrieved without semantics."""
    return [0.0]
