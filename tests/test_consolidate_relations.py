"""Consolidation must link each extracted relation to the fact it came from.

Regression: previously every relation was attached to the FIRST stored fact, so
a triple belonging to a later fact resolved to the wrong memory in graph recall.
The model call is monkeypatched so the test is hermetic (no Ollama)."""
from __future__ import annotations

import json

import pytest

import config
from brain import consolidate, graph, llm, store


@pytest.fixture
def consolidation_on(monkeypatch):
    # conftest sets NEURAL_MAP_CONSOLIDATION=0, so re-enable it for these tests.
    monkeypatch.setattr(config, "MEMORY_CONSOLIDATION", True)


def _patch_model(monkeypatch, payload: dict):
    monkeypatch.setattr(llm, "chat", lambda *a, **k: json.dumps(payload))


def _text_by_id() -> dict[str, str]:
    rows = store.list_records()
    return dict(zip(rows.get("ids") or [], rows.get("documents") or []))


def test_each_relation_links_to_its_own_fact(consolidation_on, monkeypatch):
    _patch_model(
        monkeypatch,
        {
            "facts": [
                {"text": "Alex holds SCHD.", "importance": 4,
                 "relations": [{"subject": "Alex", "relation": "holds", "object": "SCHD"}]},
                {"text": "Alex sold JEPQ.", "importance": 4,
                 "relations": [{"subject": "Alex", "relation": "sold", "object": "JEPQ"}]},
            ],
            "superseded": [],
        },
    )

    result = consolidate.consolidate("u", "a")
    assert result["stored"] == 2
    assert result["relations"] == 2

    schd_ids = graph.fact_ids_for({"schd"})
    jepq_ids = graph.fact_ids_for({"jepq"})
    assert len(schd_ids) == 1 and len(jepq_ids) == 1
    # The bug attached both edges to the first fact; they must now differ.
    assert schd_ids != jepq_ids

    text_by_id = _text_by_id()
    assert "SCHD" in text_by_id[schd_ids[0]]
    assert "JEPQ" in text_by_id[jepq_ids[0]]   # the JEPQ edge resolves to the JEPQ fact


def test_flat_relations_fallback_links_only_for_single_fact(consolidation_on, monkeypatch):
    # A model that ignores the nested schema and emits a single fact + top-level
    # relations should still get linked (unambiguous mapping).
    _patch_model(
        monkeypatch,
        {
            "facts": [{"text": "Alice knows Bob.", "importance": 3}],
            "superseded": [],
            "relations": [{"subject": "Alice", "relation": "knows", "object": "Bob"}],
        },
    )
    result = consolidate.consolidate("u", "a")
    assert result["stored"] == 1
    assert result["relations"] == 1
    assert len(graph.fact_ids_for({"bob"})) == 1


def test_flat_relations_skipped_when_ambiguous(consolidation_on, monkeypatch):
    # Multiple facts + only top-level relations = no fact mapping. Skip rather than
    # mislink every edge to the first fact.
    _patch_model(
        monkeypatch,
        {
            "facts": [
                {"text": "Alex holds SCHD."},
                {"text": "Alex sold JEPQ."},
            ],
            "superseded": [],
            "relations": [
                {"subject": "Alex", "relation": "holds", "object": "SCHD"},
                {"subject": "Alex", "relation": "sold", "object": "JEPQ"},
            ],
        },
    )
    result = consolidate.consolidate("u", "a")
    assert result["stored"] == 2
    assert result["relations"] == 0          # ambiguous edges skipped
    assert graph.stats()["edges"] == 0
