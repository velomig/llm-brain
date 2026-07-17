"""Re-ingesting a document replaces its chunks instead of duplicating them, and
leaves other sources alone."""
from __future__ import annotations

import pytest

from brain import graph, store
from brain.documents import ingest_document_text

DOC = (
    "# Plan\n"
    "Contribute monthly to SCHD and FXAIX at a sixty forty split.\n\n"
    "# Risk\n"
    "Keep at most two speculative positions.\n"
)


def test_reingest_same_source_does_not_duplicate():
    n1, _ = ingest_document_text(DOC, filename="plan.md", source="src:plan")
    assert store.count() == n1

    n2, _ = ingest_document_text(DOC, filename="plan.md", source="src:plan")
    assert n2 == n1
    assert store.count() == n1   # replaced, not doubled


def test_reingest_updates_content_for_same_source():
    ingest_document_text(DOC, filename="plan.md", source="src:plan")
    new_doc = "# Plan\nSwitch entirely to a treasury ladder strategy.\n"
    n, _ = ingest_document_text(new_doc, filename="plan.md", source="src:plan")

    assert store.count() == n
    texts = " ".join(store.list_records().get("documents") or [])
    assert "treasury ladder" in texts
    assert "sixty forty" not in texts   # old content gone


def test_other_sources_survive_reingest():
    ingest_document_text(DOC, filename="plan.md", source="src:plan")
    other_n, _ = ingest_document_text(
        "# Notes\nUnrelated content here.\n", filename="notes.md", source="src:notes"
    )
    plan_n, _ = ingest_document_text(DOC, filename="plan.md", source="src:plan")

    assert store.count() == plan_n + other_n   # re-ingest of plan left notes intact


def test_replace_false_appends():
    n1, _ = ingest_document_text(DOC, filename="plan.md", source="src:plan")
    n2, _ = ingest_document_text(DOC, filename="plan.md", source="src:plan", replace=False)
    assert store.count() == n1 + n2   # without replace, chunks accumulate


def test_reingest_removes_replaced_chunks_graph_edges():
    ingest_document_text(DOC, filename="plan.md", source="src:plan")
    old_ids = store.list_records().get("ids") or []
    graph.add_relations(
        [{"subject": "SCHD", "relation": "part_of", "object": "plan"}],
        fact_id=old_ids[0],
        collection=store.collection_name(None),
    )

    ingest_document_text(
        "# Plan\nSwitch entirely to a treasury ladder strategy.\n",
        filename="plan.md",
        source="src:plan",
    )

    # The replaced chunk is gone from the store, so its edge must go too —
    # otherwise it lingers as a dangling_graph_fact integrity error.
    dangling = [row for row in graph.export_edges() if row.get("fact_id") in set(old_ids)]
    assert dangling == []


def test_reingest_write_failure_preserves_old_chunks(monkeypatch):
    ingest_document_text(DOC, filename="plan.md", source="src:plan")
    old_ids = set(store.list_records().get("ids") or [])
    original_add_many = store.add_many

    def _partial_write_then_fail(*args, **kwargs):
        original_add_many(*args, **kwargs)
        raise RuntimeError("store write failed")

    monkeypatch.setattr("brain.documents.store.add_many", _partial_write_then_fail)

    with pytest.raises(RuntimeError):
        ingest_document_text(
            "# Plan\nSwitch entirely to a treasury ladder strategy.\n",
            filename="plan.md",
            source="src:plan",
        )

    records = store.list_records()
    assert set(records.get("ids") or []) == old_ids
    texts = " ".join(records.get("documents") or [])
    assert "sixty forty" in texts
    assert "treasury ladder" not in texts


def test_replace_false_partial_write_failure_removes_new_units(monkeypatch):
    """The media endpoint stages replacement chunks with replace=False. A
    later-batch failure must remove the staged UUIDs while preserving the old
    source records."""
    ingest_document_text(DOC, filename="plan.md", source="src:plan")
    old_ids = set(store.list_records().get("ids") or [])

    def _write_one_then_fail(units, embeddings, model=None, text_only=False, **kwargs):
        store.add(units[0], embeddings[0], model=model, text_only=text_only)
        raise RuntimeError("later batch failed")

    monkeypatch.setattr("brain.documents.store.add_many", _write_one_then_fail)

    with pytest.raises(RuntimeError):
        ingest_document_text(
            "# Plan\nA newly staged portfolio replacement.\n",
            filename="plan.md",
            source="src:plan",
            replace=False,
        )

    assert set(store.list_records().get("ids") or []) == old_ids
