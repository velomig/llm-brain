"""When embeddings are unavailable, memories must land in the text-only
collection and still be retrievable (via the keyword path)."""
from __future__ import annotations

from brain import store
from brain.documents import ingest_document_text
from brain.retriever import retrieve_weighted
from util import store_memory

DOC = "# Portfolio\nThe FNDY position covers growth sectors that FNDX screens out.\n"


def test_ingest_falls_back_to_text_only_when_embeddings_down(fake_embed):
    fake_embed.available = False
    n, _cat = ingest_document_text(DOC, filename="portfolio.md", source="src:portfolio")
    assert n >= 1
    assert store.count(text_only=True) == n   # everything went to the text-only store
    assert store.count(text_only=False) == 0  # nothing reached the vector store


def test_text_only_memory_still_retrievable_by_keyword(fake_embed):
    # Seed filler while embeddings work so "FNDY" stays rare (positive BM25 idf);
    # the combined keyword index spans both the vector and text-only collections.
    for i in range(6):
        store_memory(f"unrelated filler note {i} about gardening and weather")
    fake_embed.available = False
    ingest_document_text(DOC, filename="portfolio.md", source="src:portfolio")
    # Embeddings stay down, so semantic is dead; keyword must still surface it.
    res = retrieve_weighted("FNDY", w_sim=0, w_keyword=1, w_recency=0, w_importance=0, w_graph=0)
    assert res, "text-only memory was not retrievable by keyword"
    assert any("FNDY" in r["text"] for r in res)


def test_recovery_to_vector_store_when_embeddings_return(fake_embed):
    # Down: lands in text-only. Back up + re-ingest: replaces into the vector store.
    fake_embed.available = False
    ingest_document_text(DOC, filename="portfolio.md", source="src:portfolio")
    assert store.count(text_only=True) >= 1

    fake_embed.available = True
    n, _ = ingest_document_text(DOC, filename="portfolio.md", source="src:portfolio")
    assert store.count(text_only=False) == n
    # The replace step clears the stale text-only chunks for this source.
    assert store.count(text_only=True) == 0
