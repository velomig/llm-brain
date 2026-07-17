"""Tiny helpers for building memories in tests."""
from __future__ import annotations

from brain import store
from brain.embeddings import embed_document, text_only_embedding
from brain.memory import MemoryUnit


def store_memory(
    text: str,
    category: str = "fact",
    importance: int = 3,
    source: str | None = None,
    extra: dict | None = None,
    date: str | None = None,
    text_only: bool = False,
    model: str | None = None,
) -> MemoryUnit:
    """Persist a memory and return its MemoryUnit (so tests can use its id).

    text_only=True simulates the embeddings-unavailable path (placeholder vector
    in the text-only collection); otherwise it embeds via the patched fake client.
    """
    fields: dict = {"text": text, "category": category, "importance": importance}
    if source is not None:
        fields["source"] = source
    if extra is not None:
        fields["extra"] = extra
    if date is not None:
        fields["date"] = date
    unit = MemoryUnit(**fields)
    if text_only:
        store.add(unit, text_only_embedding(), model=model, text_only=True)
    else:
        store.add(unit, embed_document(text, model=model), model=model, text_only=False)
    return unit


def ids(results: list[dict]) -> list[str]:
    return [r["id"] for r in results]
