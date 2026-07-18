"""/api/health is a never-raising diagnostics snapshot.

It answers "why is my brain quiet today?" — so it must report Ollama being down
as a *state* (200 with reachable=False), keep the memory/graph counts it can still
compute, and split vector vs. text-only counts so a degraded-mode brain is visible.
Ollama isn't mocked by conftest, so each test patches llm.list_models directly."""
from __future__ import annotations

import config
import server
from util import store_memory


def _health(monkeypatch, available, model=None):
    monkeypatch.setattr(server.llm, "list_models", lambda: list(available))
    return server.health(model=model)


def test_health_reports_models_counts_and_features(monkeypatch):
    store_memory("a vector memory", text_only=False)
    store_memory("a fallback memory", text_only=True)
    store_memory("another fallback", text_only=True)

    available = [
        config.CHAT_MODEL,
        config.EMBED_MODEL,
        config.COUNCIL_MODEL,
        config.CONSOLIDATION_MODEL,
        config.METACOGNITION_MODEL,
        config.VISION_MODEL,
    ]
    out = _health(monkeypatch, available)

    assert out["ok"] is True
    assert out["ollama"] == {"reachable": True, "error": None}
    assert out["chat_model"] == {"name": config.CHAT_MODEL, "available": True}
    assert out["embed_model"] == {"name": config.EMBED_MODEL, "available": True}
    assert out["role_models"]["council"] == {"name": config.COUNCIL_MODEL, "available": True}
    assert out["role_models"]["consolidation"]["available"] is True
    assert out["role_models"]["metacognition"]["available"] is True
    assert out["role_models"]["vision"]["available"] is True
    assert out["memories"]["vector"] == 1
    assert out["memories"]["text_only"] == 2
    assert out["memories"]["total"] == 3
    assert out["memories"]["pinned"] == 0
    assert out["memories"]["source_types"] == {"user_entered": 3}
    assert out["features"]["web_search"] == config.WEB_SEARCH_ENABLED
    assert out["features"]["web_search_strict_context_guard"] == config.WEB_SEARCH_STRICT_CONTEXT_GUARD
    assert out["features"]["web_search_monitor_unsourced_context"] == config.WEB_SEARCH_MONITOR_UNSOURCED_CONTEXT
    assert out["features"]["web_search_unsourced_context_guard"] == config.WEB_SEARCH_BLOCK_WITH_UNSOURCED_CONTEXT
    assert out["features"]["web_search_user_derived_with_context"] == config.WEB_SEARCH_ALLOW_USER_DERIVED_WITH_CONTEXT
    assert out["features"]["market_quotes"] == config.MARKET_QUOTES_ENABLED
    assert out["features"]["initiative"] == config.INITIATIVE_ENABLED
    assert out["features"]["attention"] == config.ATTENTION_ENABLED
    assert out["features"]["arbiter"] is True
    assert out["features"]["tool_calling"] == config.TOOL_CALLING_ENABLED
    assert out["features"]["action_tools"] == config.ACTION_TOOLS_ENABLED
    assert out["features"]["metacognition"] == config.METACOGNITION_ENABLED
    assert out["features"]["inner_council"] == config.INNER_COUNCIL_ENABLED
    assert out["features"]["evolution"] == config.EVOLUTION_ENABLED
    assert out["features"]["observability"] is True
    assert out["features"]["sensory"] == config.SENSORY_EMBEDDINGS_ENABLED


def test_health_flags_missing_models(monkeypatch):
    # Ollama is up but neither configured model is pulled — the common "quiet
    # brain" cause. Reachable, but not ok, and each model marked unavailable.
    out = _health(monkeypatch, ["some-unrelated-model:latest"])

    assert out["ollama"]["reachable"] is True
    assert out["chat_model"]["available"] is False
    assert out["embed_model"]["available"] is False
    assert out["ok"] is False


def test_health_survives_ollama_down(monkeypatch):
    # Ollama unreachable: a reported state, not a 500. Model availability is
    # unknown (None, not False) and the counts it can still compute are present.
    store_memory("still countable", text_only=False)

    def _boom():
        raise ConnectionError("connection refused")

    monkeypatch.setattr(server.llm, "list_models", _boom)
    out = server.health()

    assert out["ok"] is False
    assert out["ollama"]["reachable"] is False
    assert "connection refused" in out["ollama"]["error"]
    assert out["chat_model"]["available"] is None
    assert out["embed_model"]["available"] is None
    assert out["memories"]["vector"] == 1


def test_health_reports_graph_edges(monkeypatch):
    from brain import graph

    graph.add_relations(
        [{"subject": "Ada", "relation": "knows", "object": "Lovelace"}],
        fact_id="f1",
    )
    out = _health(monkeypatch, [config.CHAT_MODEL, config.EMBED_MODEL])

    assert out["graph"]["edges"] == 1
    assert out["graph"]["entities"] == 2
