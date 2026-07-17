"""RetrievalSettings validation must treat every producer signal as valid.

Regression: the validator rejected requests where semantic+keyword were both 0,
even though graph recall (on by default via config.W_GRAPH) can produce candidates
on its own. A UI user zeroing Semantic+BM25 while leaving Recency/Importance set
got a validation error for a query the retriever could actually answer."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

import config
from server import RetrievalSettings


def test_graph_default_makes_semantic_keyword_zero_valid():
    # The reported path: Semantic=0, BM25=0, Recency/Importance nonzero. Graph is
    # left at its default (config.W_GRAPH > 0), so this must be accepted.
    assert config.W_GRAPH > 0
    s = RetrievalSettings(semantic=0, keyword=0, recency=0.3, importance=0.1)
    assert s.graph == config.W_GRAPH


def test_explicit_graph_only_is_valid():
    s = RetrievalSettings(semantic=0, keyword=0, graph=1.0)
    assert s.graph == 1.0


def test_recency_or_importance_only_is_now_valid():
    # Recency and importance are real producers now (most-recent / most-important
    # memories), so they're valid even with semantic/keyword/graph/sensory all zero.
    assert RetrievalSettings(semantic=0, keyword=0, graph=0, sensory=0, recency=1, importance=0).recency == 1
    assert RetrievalSettings(semantic=0, keyword=0, graph=0, sensory=0, recency=0, importance=1).importance == 1


def test_explicit_sensory_only_is_valid():
    s = RetrievalSettings(semantic=0, keyword=0, graph=0, recency=0, importance=0, sensory=1.0)
    assert s.sensory == 1.0


def test_all_weights_zero_is_rejected():
    # The only useless request: every signal off, so nothing can produce.
    with pytest.raises(ValidationError):
        RetrievalSettings(semantic=0, keyword=0, graph=0, sensory=0, recency=0, importance=0)


def test_graph_is_exposed_in_defaults():
    # Drives /api/retrieval_defaults and the UI slider, so it must be present.
    assert RetrievalSettings().model_dump()["graph"] == config.W_GRAPH


def test_sensory_is_exposed_in_defaults():
    assert RetrievalSettings().model_dump()["sensory"] == config.W_SENSORY
