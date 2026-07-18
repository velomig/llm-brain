"""Regression tests for the audit fixes.

Covers: capability-probe caching (a transient Ollama outage must not disable
tool-calling until restart), web-upload replacement by filename (an edited
re-upload must not leave stale chunks), the optional web_search doc-context
guard (prompt-injection exfiltration defense), deterministic graph fact
ordering, and the num_ctx cap for models whose max context is undiscoverable.
"""
from __future__ import annotations

import config
import server
from brain import graph, llm, pipeline, store, websearch


# --- supports_tools: failed probes are not cached ------------------------------

class _FlakyShow:
    """Fake ollama client whose show() fails until told otherwise."""

    def __init__(self) -> None:
        self.calls = 0
        self.fail = True

    def show(self, name):
        self.calls += 1
        if self.fail:
            raise ConnectionError("ollama down")
        return {"capabilities": ["tools"]}


def test_supports_tools_failed_probe_is_not_cached(monkeypatch):
    fake = _FlakyShow()
    monkeypatch.setattr(llm, "_client", fake)
    llm._TOOLS_CAPABILITY.clear()

    assert llm.supports_tools("m1") is False   # outage -> injection for THIS turn only
    fake.fail = False
    assert llm.supports_tools("m1") is True    # next turn re-probes and recovers
    assert llm.supports_tools("m1") is True    # a successful probe IS cached
    assert fake.calls == 2

    llm._TOOLS_CAPABILITY.clear()


# --- web upload: edited re-upload replaces stale chunks ------------------------

def _upload(filename: str, text: str):
    return server.ingest_markdown(server.IngestReq(filename=filename, text=text))


def test_edited_reupload_replaces_stale_chunks():
    _upload("plan.md", "# Plan\nContribute at a sixty forty split.\n")
    _upload("plan.md", "# Plan\nSwitch entirely to a treasury ladder.\n")

    records = store.list_records()
    texts = " ".join(records.get("documents") or [])
    assert "treasury ladder" in texts
    assert "sixty forty" not in texts   # previous upload's chunks were swept

    sources = {(m or {}).get("source") for m in records.get("metadatas") or []}
    assert len(sources) == 1            # only the latest upload's source remains


def test_reupload_leaves_other_filenames_alone():
    _upload("plan.md", "# Plan\nContribute at a sixty forty split.\n")
    _upload("notes.md", "# Notes\nUnrelated content here.\n")
    _upload("plan.md", "# Plan\nSwitch entirely to a treasury ladder.\n")

    texts = " ".join(store.list_records().get("documents") or [])
    assert "Unrelated content" in texts
    assert "treasury ladder" in texts
    assert "sixty forty" not in texts


def test_edited_reupload_sweeps_stale_chunks_graph_edges():
    _upload("plan.md", "# Plan\nContribute at a sixty forty split.\n")
    old_ids = store.list_records().get("ids") or []
    graph.add_relations(
        [{"subject": "split", "relation": "part_of", "object": "plan"}],
        fact_id=old_ids[0],
        collection=store.collection_name(None),
    )

    _upload("plan.md", "# Plan\nSwitch entirely to a treasury ladder.\n")

    # The stale upload's chunks are swept via the shared deletion path, so
    # their edges must go too — otherwise they linger as dangling_graph_fact
    # integrity errors.
    dangling = [row for row in graph.export_edges() if row.get("fact_id") in set(old_ids)]
    assert dangling == []


def test_identical_reupload_still_replaces_by_content_key():
    doc = "# Plan\nContribute at a sixty forty split.\n"
    first = _upload("plan.md", doc)
    second = _upload("plan.md", doc)
    assert second["chunks"] == first["chunks"]
    assert store.count() == first["chunks"]   # replaced, not doubled


# --- web_search context guard --------------------------------------------------

def test_web_search_tool_guard_can_block_document_context(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["system"] = system
        captured["specs"] = specs
        captured["dispatch"] = dispatch
        return "ok", []

    assert config.WEB_SEARCH_BLOCK_WITH_DOC_CONTEXT is True
    assert config.WEB_SEARCH_STRICT_CONTEXT_GUARD is True
    monkeypatch.setattr(config, "WEB_SEARCH_ALLOW_USER_DERIVED_WITH_CONTEXT", False)
    # "plan" is a complexity term: the intuition gate would route to System 2,
    # whose llm.chat calls aren't mocked here. The gate has its own tests.
    monkeypatch.setattr(config, "INTUITION_ENABLED", False)
    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)

    doc_mems = [{"id": "m1", "text": "chunk", "source": "upload:abc123:plan.pdf"}]
    result = pipeline.answer("what does my plan say?", doc_mems)
    assert result["reply"] == "ok"
    names = {s["function"]["name"] for s in captured["specs"]}
    assert "web_search" not in names
    assert "web_search" not in captured["system"]
    assert "disabled" in captured["dispatch"]("web_search", {"query": "secrets"})["error"]


def test_web_search_strict_guard_blocks_memory_without_source(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["system"] = system
        captured["specs"] = specs
        captured["dispatch"] = dispatch
        return "ok", []

    monkeypatch.setattr(config, "INTUITION_ENABLED", False)
    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(config, "WEB_SEARCH_STRICT_CONTEXT_GUARD", True)
    monkeypatch.setattr(config, "WEB_SEARCH_ALLOW_USER_DERIVED_WITH_CONTEXT", False)

    mems = [{"id": "m1", "text": "private stored fact", "source": None}]
    result = pipeline.answer("hello", mems)

    assert result["reply"] == "ok"
    names = {s["function"]["name"] for s in captured["specs"]}
    assert "web_search" not in names
    assert "web_search" not in captured["system"]
    refused = captured["dispatch"]("web_search", {"query": "private stored fact"})
    assert "retrieved memory context" in refused["error"]


def test_web_search_unsourced_context_guard_blocks_when_strict_guard_disabled(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["system"] = system
        captured["specs"] = specs
        captured["dispatch"] = dispatch
        return "ok", []

    monkeypatch.setattr(config, "INTUITION_ENABLED", False)
    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(config, "WEB_SEARCH_STRICT_CONTEXT_GUARD", False)
    monkeypatch.setattr(config, "WEB_SEARCH_BLOCK_WITH_UNSOURCED_CONTEXT", True)
    monkeypatch.setattr(config, "WEB_SEARCH_ALLOW_USER_DERIVED_WITH_CONTEXT", False)

    mems = [{"id": "m1", "text": "private stored fact", "source": None}]
    pipeline.answer("hello", mems)

    names = {s["function"]["name"] for s in captured["specs"]}
    assert "web_search" not in names
    assert "web_search" not in captured["system"]
    refused = captured["dispatch"]("web_search", {"query": "private stored fact"})
    assert "unsourced retrieved memory context" in refused["error"]


def test_web_search_monitor_logs_unsourced_context(monkeypatch):
    events = []

    monkeypatch.setattr(config, "INTUITION_ENABLED", False)
    monkeypatch.setattr(config, "WEB_SEARCH_MONITOR_UNSOURCED_CONTEXT", True)
    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: False)
    monkeypatch.setattr(pipeline.llm, "chat", lambda *a, **k: "ok")
    monkeypatch.setattr(pipeline, "market_context_for", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "web_context_for", lambda *a, **k: None)
    monkeypatch.setattr(pipeline.log, "fallback", lambda event, *a, **k: events.append((event, k)))

    pipeline.answer("hello", [{"id": "m1", "text": "private stored fact", "source": None}])

    assert any(event == "unsourced_context_detected" and fields["count"] == 1 for event, fields in events)


def test_web_search_tool_guarded_with_document_context(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["system"] = system
        captured["specs"] = specs
        captured["dispatch"] = dispatch
        return "ok", []

    monkeypatch.setattr(config, "INTUITION_ENABLED", False)   # keep System 2 out of this test
    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(config, "WEB_SEARCH_BLOCK_WITH_DOC_CONTEXT", True)
    monkeypatch.setattr(config, "WEB_SEARCH_ALLOW_USER_DERIVED_WITH_CONTEXT", False)

    doc_mems = [{"id": "m1", "text": "chunk", "source": "upload:abc123:plan.pdf"}]
    result = pipeline.answer("what does my plan say?", doc_mems)
    assert result["reply"] == "ok"
    names = {s["function"]["name"] for s in captured["specs"]}
    assert "web_search" not in names                       # unadvertised...
    assert "web_search" not in captured["system"]           # ...and unprompted
    refused = captured["dispatch"]("web_search", {"query": "secrets"})
    assert "disabled" in refused["error"]                  # ...AND refused at dispatch
    assert captured["dispatch"]("get_current_datetime", {})["date"]  # others still work

    no_doc_mems = [{"id": "m2", "text": "a stored fact", "source": None}]
    pipeline.answer("hello", no_doc_mems)
    names = {s["function"]["name"] for s in captured["specs"]}
    assert "web_search" not in names                       # strict guard bites on any memory
    assert "web_search" not in captured["system"]
    refused = captured["dispatch"]("web_search", {"query": "a stored fact"})
    assert "retrieved memory context" in refused["error"]


def test_web_search_tool_untouched_when_guard_disabled(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["system"] = system
        captured["specs"] = specs
        return "ok", []

    monkeypatch.setattr(config, "INTUITION_ENABLED", False)   # keep System 2 out of this test
    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(config, "WEB_SEARCH_STRICT_CONTEXT_GUARD", False)
    monkeypatch.setattr(config, "WEB_SEARCH_BLOCK_WITH_DOC_CONTEXT", False)
    monkeypatch.setattr(config, "WEB_SEARCH_BLOCK_WITH_UNSOURCED_CONTEXT", False)

    doc_mems = [{"id": "m1", "text": "chunk", "source": "upload:abc123:plan.pdf"}]
    pipeline.answer("what does my plan say?", doc_mems)
    names = {s["function"]["name"] for s in captured["specs"]}
    assert "web_search" in names
    assert "web_search" in captured["system"]


def test_web_search_tool_allows_user_derived_query(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["result"] = dispatch("web_search", {"query": "Phoenix weather"})
        return "ok", [{"tool": "web_search", "args": {"query": "Phoenix weather"}, "result": captured["result"]}]

    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(
        pipeline.tools,
        "dispatch",
        lambda name, args: {"query": args["query"], "results": [{"title": "ok"}], "errors": []},
    )

    result = pipeline.answer(
        "what is the weather in Phoenix today?",
        [{"id": "m1", "text": "private stored fact", "source": None}],
    )
    assert result["reply"] == "ok"
    assert captured["result"]["query"] == "Phoenix weather"
    assert "error" not in captured["result"]


def test_web_search_tool_rejects_filler_word_as_query(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["result"] = dispatch("web_search", {"query": "like"})
        return "ok", [{"tool": "web_search", "args": {"query": "like"}, "result": captured["result"]}]

    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    # Simulate a provider prefetch that yielded nothing, leaving the tool
    # available for a corrected model-generated query.
    monkeypatch.setattr(pipeline, "web_context_for", lambda query: None)

    pipeline.answer("I would like news from today in politics in the US", [])

    assert "too vague" in captured["result"]["error"]


def test_tool_path_prefetches_general_nonpolitical_search(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["system"] = system
        captured["specs"] = specs
        return "ok", []

    prefetched = {
        "query": "Research beginner sourdough hydration ratios",
        "provider": "DuckDuckGo",
        "results": [{"title": "Sourdough guide", "url": "https://example.com/sourdough"}],
        "errors": [],
    }

    monkeypatch.setattr(config, "INTUITION_ENABLED", False)
    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(pipeline, "web_context_for", lambda query: prefetched)

    result = pipeline.answer("Research beginner sourdough hydration ratios", [])
    names = {s["function"]["name"] for s in captured["specs"]}

    assert result["web_data"] == prefetched
    assert "web_search" not in names
    assert "Sourdough guide" in captured["system"]


def test_generic_web_followup_prefetches_active_ticker_not_filler(monkeypatch):
    captured = {}

    def fake_web_context(query):
        captured["query"] = query
        return {
            "query": query,
            "provider": "test",
            "results": [{"title": "EXCO result", "url": "https://example.com/exco"}],
            "errors": [],
        }

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["system"] = system
        captured["specs"] = specs
        return "grounded answer", []

    monkeypatch.setattr(config, "INTUITION_ENABLED", False)
    monkeypatch.setattr(pipeline, "web_context_for", fake_web_context)
    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)

    result = pipeline.answer(
        "search the web for information",
        [],
        history=[
            {"role": "user", "content": "I would like an independent analysis of EXCO."},
            {"role": "assistant", "content": "I can analyze it against your portfolio."},
            {"role": "user", "content": "Don't look at it against my portfolio."},
            {"role": "assistant", "content": "No specific headlines were retrieved."},
        ],
    )

    names = {s["function"]["name"] for s in captured["specs"]}
    # The prior request names its topic beyond the bare symbol, so it is
    # retained whole (reducing it to just the ticker loses the topic); the
    # pronoun clarification and the filler continuation are never searched
    # on their own.
    expected = "I would like an independent analysis of EXCO.\nsearch the web for information"
    assert captured["query"] == expected
    assert result["web_data"]["query"] == expected
    assert "EXCO result" in captured["system"]
    assert "web_search" not in names


def test_generic_web_followup_recovers_after_empty_prefetch(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["result"] = dispatch(
            "web_search",
            {"query": "Cerebras technology market competition"},
        )
        return "grounded answer", [
            {"tool": "web_search", "args": {}, "result": captured["result"]},
        ]

    monkeypatch.setattr(config, "INTUITION_ENABLED", False)
    monkeypatch.setattr(
        pipeline,
        "web_context_for",
        lambda query: {"query": query, "provider": "test", "results": [], "errors": []},
    )
    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(
        pipeline.tools,
        "dispatch",
        lambda name, args: {"query": args["query"], "results": [{"title": "EXCO result"}], "errors": []},
    )

    pipeline.answer(
        "search the web for information",
        [],
        history=[
            {"role": "user", "content": "I would like an independent analysis of EXCO."},
            {"role": "assistant", "content": "I can analyze it against your portfolio."},
            {"role": "user", "content": "Don't look at it against my portfolio."},
            {"role": "assistant", "content": "No specific headlines were retrieved."},
        ],
    )

    assert captured["result"]["query"] == "EXCO analysis"
    assert "error" not in captured["result"]


def test_web_search_tool_allows_compacted_user_phrase(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["result"] = dispatch("web_search", {"query": "SpaceX news"})
        return "ok", [{"tool": "web_search", "args": {"query": "SpaceX news"}, "result": captured["result"]}]

    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(
        pipeline.tools,
        "dispatch",
        lambda name, args: {"query": args["query"], "results": [{"title": "ok"}], "errors": []},
    )

    pipeline.answer("can you access the web for news on Space X?", [])
    assert captured["result"]["query"] == "SpaceX news"
    assert "error" not in captured["result"]


def test_web_search_tool_allows_retry_from_previous_user_request(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["result"] = dispatch("web_search", {"query": "SpaceX news"})
        return "ok", [{"tool": "web_search", "args": {"query": "SpaceX news"}, "result": captured["result"]}]

    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(
        pipeline.tools,
        "dispatch",
        lambda name, args: {"query": args["query"], "results": [{"title": "ok"}], "errors": []},
    )

    pipeline.answer(
        "can you try again?",
        [],
        history=[{"role": "user", "content": "can you access the web for news on Space X?"}],
    )
    assert captured["result"]["query"] == "SpaceX news"
    assert "error" not in captured["result"]


def test_web_search_tool_allows_safe_news_refiners(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["result"] = dispatch("web_search", {"query": "SpaceX official updates July 8 2026"})
        return "ok", [{"tool": "web_search", "args": {"query": "SpaceX official updates July 8 2026"}, "result": captured["result"]}]

    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(
        pipeline.tools,
        "dispatch",
        lambda name, args: {"query": args["query"], "results": [{"title": "ok"}], "errors": []},
    )

    pipeline.answer("can you access the web for news on Space X?", [])
    assert captured["result"]["query"] == "SpaceX official updates July 8 2026"
    assert "error" not in captured["result"]


def test_web_search_tool_allows_contextual_followup_from_previous_user_request(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["result"] = dispatch("web_search", {"query": "SpaceX news July 8 2026 ACME price drop"})
        return "ok", [{"tool": "web_search", "args": {"query": "SpaceX news July 8 2026 ACME price drop"}, "result": captured["result"]}]

    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(
        pipeline.tools,
        "dispatch",
        lambda name, args: {"query": args["query"], "results": [{"title": "ok"}], "errors": []},
    )

    pipeline.answer(
        "can you correlate between the news and the price drop today?",
        [],
        history=[{"role": "user", "content": "can you access the web for news on Space X and ACME?"}],
    )
    assert captured["result"]["query"] == "SpaceX news July 8 2026 ACME price drop"
    assert "error" not in captured["result"]


def test_web_search_tool_allows_public_political_pronoun_followup(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["result"] = dispatch(
            "web_search",
            {"query": "Trump election commission ousted members news"},
        )
        return "ok", [{"tool": "web_search", "args": {}, "result": captured["result"]}]

    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(
        pipeline.tools,
        "dispatch",
        lambda name, args: {"query": args["query"], "results": [{"title": "ok"}], "errors": []},
    )

    pipeline.answer(
        "Did he oust members of his political election commission?",
        [],
        history=[{"role": "user", "content": "I would like to know any news regarding President Trump today."}],
    )
    assert "error" not in captured["result"]


def test_web_search_tool_allows_public_model_refinements(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["result"] = dispatch(
            "web_search",
            {"query": "SpaceX Starship launch test results FAA regulatory news last 48 hours"},
        )
        return "ok", [{"tool": "web_search", "args": {}, "result": captured["result"]}]

    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(
        pipeline.tools,
        "dispatch",
        lambda name, args: {"query": args["query"], "results": [{"title": "ok"}], "errors": []},
    )

    pipeline.answer("can you access the web for news on Space X?", [])
    assert "error" not in captured["result"]


def test_web_search_tool_allows_acme_stock_news_terms(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["result"] = dispatch(
            "web_search",
            {"query": "ACME stock decline today earnings reports analyst downgrades SEC filings news"},
        )
        return "ok", [{"tool": "web_search", "args": {}, "result": captured["result"]}]

    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    # The two-question prompt routes answer() into System 2, whose critique and
    # rewrite go through llm.chat — stub it so the test never talks to Ollama.
    monkeypatch.setattr(pipeline.llm, "chat", lambda *args, **kwargs: "ok")
    monkeypatch.setattr(
        pipeline.tools,
        "dispatch",
        lambda name, args: {"query": args["query"], "results": [{"title": "ok"}], "errors": []},
    )

    pipeline.answer("Why did the ACME stock decline today? What is the news that caused that?", [])
    assert "error" not in captured["result"]


def test_tool_path_prefetches_web_results_and_skips_web_tool(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["system"] = system
        captured["specs"] = specs
        return "ok", []

    prefetched = {
        "query": "Why did the ACME stock decline today? What is the news that caused that?",
        "results": [{"title": "ACME news", "url": "https://example.com/news", "snippet": "headline"}],
        "errors": [],
    }

    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    # The two-question prompt routes answer() into System 2, whose critique and
    # rewrite go through llm.chat and would overwrite the mocked "ok" draft with
    # a live generation — stub it so the reply assertion is deterministic.
    monkeypatch.setattr(pipeline.llm, "chat", lambda *args, **kwargs: "ok")
    monkeypatch.setattr(pipeline, "web_context_for", lambda query: prefetched)

    result = pipeline.answer("Why did the ACME stock decline today? What is the news that caused that?", [])
    names = {s["function"]["name"] for s in captured["specs"]}

    assert result["reply"] == "ok"
    assert result["web_data"] == prefetched
    assert "web_search" not in names
    assert "ACME news" in captured["system"]


def test_web_results_suppress_unpinned_stale_memory_in_prompt(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["system"] = system
        captured["specs"] = specs
        return "ok", []

    prefetched = {
        "query": "Trump election news today",
        "provider": "Bing News",
        "results": [
            {
                "title": "Election headline",
                "url": "https://example.com/election",
                "snippet": "Current election news from the search result.",
            }
        ],
        "errors": [],
    }

    stale_memory = {
        "id": "m1",
        "text": "Assistant: no specific named news items were found; informational waywardness.",
        "category": "episode",
        "source": "chat",
        "pinned": False,
    }

    monkeypatch.setattr(config, "INTUITION_ENABLED", False)
    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(pipeline, "web_context_for", lambda query: prefetched)

    result = pipeline.answer("Isn't there Trump news regarding the election?", [stale_memory])

    names = {s["function"]["name"] for s in captured["specs"]}
    assert result["reply"] == "ok"
    assert "web_search" not in names
    assert "Election headline" in captured["system"]
    assert "Current Web Grounding Rule" in captured["system"]
    assert "informational waywardness" not in captured["system"]


def test_empty_tool_loop_reply_preserves_executed_tools(monkeypatch):
    used = [
        {
            "tool": "send_email",
            "args": {"recipient": "person@example.com", "subject": "Hi", "body": "Done"},
            "result": {"sent": True, "draft": False},
        }
    ]

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        return "", used

    monkeypatch.setattr(config, "TOOL_CALLING_ENABLED", True)
    monkeypatch.setattr(config, "INTUITION_ENABLED", False)
    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(
        pipeline,
        "market_context_for",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("eager fallback must not run")),
    )
    monkeypatch.setattr(
        pipeline,
        "web_context_for",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("eager fallback must not run")),
    )

    result = pipeline.answer("Please send that email.", [])

    assert result["tools_used"] == used
    assert "send_email" in result["reply"]


def test_empty_tool_loop_reply_surfaces_retrieved_memory(monkeypatch):
    used = [
        {
            "tool": "search_local_files",
            "args": {"query": "FNDX"},
            "result": {"results": []},
        }
    ]

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        return "", used

    monkeypatch.setattr(config, "TOOL_CALLING_ENABLED", True)
    monkeypatch.setattr(config, "INTUITION_ENABLED", False)
    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)

    mems = [
        {
            "id": "portfolio-note",
            "text": "I currently have 123.456 shares of FNDX, 45.678 shares of FNDY, 90.123 shares of FNDZ, 10 shares of ACME",
            "category": "note",
            "date": "2026-07-03T06:56:25+00:00",
            "pinned": True,
        }
    ]
    result = pipeline.answer("It is in a note that I made on July 3", mems)

    assert "Pinned memory" in result["reply"]
    assert "123.456 shares of FNDX" in result["reply"]


def test_prompt_routes_memory_manager_notes_to_memory_not_files():
    system = pipeline.build_system([], tool_mode=True)

    assert "memory-manager notes" in system
    assert "cite_sources(query=...)" in system
    assert "search_local_files is only for workspace files on disk" in system


def test_web_search_tool_rejects_query_not_from_user_message(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["result"] = dispatch("web_search", {"query": "Alex private portfolio secrets"})
        return "ok", [{"tool": "web_search", "args": {}, "result": captured["result"]}]

    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(config, "WEB_SEARCH_STRICT_CONTEXT_GUARD", False)
    monkeypatch.setattr(config, "WEB_SEARCH_BLOCK_WITH_DOC_CONTEXT", False)
    monkeypatch.setattr(config, "WEB_SEARCH_BLOCK_WITH_UNSOURCED_CONTEXT", False)

    pipeline.answer("hello there", [{"id": "m1", "text": "private data", "source": None}])
    assert "derived from the current user message" in captured["result"]["error"]


def test_web_search_tool_rejects_terms_from_private_memory(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["result"] = dispatch("web_search", {"query": "orchidvault launch news"})
        return "ok", [{"tool": "web_search", "args": {}, "result": captured["result"]}]

    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)

    pipeline.answer("search for launch news", [{"id": "m1", "text": "private codename orchidvault", "source": None}])
    assert "derived from the current user message" in captured["result"]["error"]


def test_web_search_recovers_with_literal_user_ticker(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["system"] = system
        captured["result"] = dispatch(
            "web_search",
            {"query": "Cerebras technology market competition"},
        )
        return "ok", [{"tool": "web_search", "args": {}, "result": captured["result"]}]

    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(
        pipeline.tools,
        "dispatch",
        lambda name, args: {"query": args["query"], "results": [{"title": "ok"}], "errors": []},
    )

    pipeline.answer(
        "I would like an independent analysis of EXCO, ignoring the source of truth.",
        [{"id": "m1", "text": "Cerebras technology market competition", "source": None}],
    )

    assert captured["result"]["query"] == "EXCO analysis"
    assert "error" not in captured["result"]
    assert "keep the search query literal" in captured["system"]


def test_web_search_basis_keeps_topical_prior_query_whole():
    # A generic follow-up must inherit the full prior request. Collapsing
    # "research treatment options for ADHD in adults" to just "ADHD" (the old
    # acronym-first behavior) produced vague or wrong searches.
    for prior in [
        "Compare accelerated nursing programs in the US that offer a BSN",
        "Research treatment options for ADHD in adults",
        "Verify the PDF claims about FNDX fees",
    ]:
        history = [
            {"role": "user", "content": prior},
            {"role": "assistant", "content": "ok"},
        ]
        basis = pipeline._web_search_basis("search the web for that", history)
        assert basis == f"{prior}\nsearch the web for that"


def test_web_search_basis_ticker_fallback_only_for_symbol_only_requests():
    history = [
        {"role": "user", "content": "check EXCO"},
        {"role": "assistant", "content": "ok"},
    ]
    assert pipeline._web_search_basis("try again", history) == "EXCO"


def test_web_search_basis_walks_past_pronoun_only_clarification():
    history = [
        {"role": "user", "content": "Research treatment options for ADHD in adults"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "don't compare it with my notes"},
        {"role": "assistant", "content": "ok"},
    ]
    basis = pipeline._web_search_basis("search the web for that", history)
    assert basis == "Research treatment options for ADHD in adults\nsearch the web for that"


def test_web_search_basis_understands_current_sources_followup():
    history = [
        {"role": "user", "content": "Explain the evidence for simulation theory"},
        {"role": "assistant", "content": "Here is the non-live evidence I know."},
    ]
    followup = "Can you search the web and get more current sources?"

    basis = pipeline._web_search_basis(followup, history)

    assert basis == f"Explain the evidence for simulation theory\n{followup}"
    assert pipeline._needs_web_prefetch(followup, history) is True


def test_web_search_current_sources_followup_recovers_from_model_expansion(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["result"] = dispatch(
            "web_search",
            {"query": "simulation theory quantum gravity peer reviewed sources"},
        )
        return "grounded answer", [{"tool": "web_search", "args": {}, "result": captured["result"]}]

    monkeypatch.setattr(config, "INTUITION_ENABLED", False)
    monkeypatch.setattr(pipeline, "web_context_for", lambda query: None)
    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(
        pipeline.tools,
        "dispatch",
        lambda name, args: {"query": args["query"], "results": [{"title": "ok"}], "errors": []},
    )

    pipeline.answer(
        "Can you search the web and get more current sources?",
        [],
        history=[
            {"role": "user", "content": "Explain the evidence for simulation theory"},
            {"role": "assistant", "content": "Here is the non-live evidence I know."},
        ],
    )

    # "quantum gravity" and "peer reviewed" came from the model, not the
    # user. They remain blocked; the dispatcher safely retries the literal
    # prior user-authored topic instead of returning an error loop.
    assert captured["result"]["query"] == "Explain the evidence for simulation theory"
    assert "error" not in captured["result"]


def test_web_search_tool_rejects_terms_from_uploaded_document(monkeypatch):
    # A codename that lives only in a sourced upload is exactly as private as an
    # unsourced note: user topic + upload codename must not reach the provider.
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["result"] = dispatch("web_search", {"query": "launch news orchidvault"})
        return "ok", [{"tool": "web_search", "args": {}, "result": captured["result"]}]

    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(pipeline, "web_context_for", lambda *a, **k: None)

    pipeline.answer(
        "search for launch news",
        [{"id": "m1", "text": "codename orchidvault roadmap", "source": "upload:abc123:roadmap.pdf"}],
    )
    assert "derived from the current user message" in captured["result"]["error"]


def test_web_search_tool_rejects_underived_terms_off_every_blocklist(monkeypatch):
    # The derivation check is an allowlist: a term of unknown provenance is
    # refused even when no private/sensitive blocklist knows about it.
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["result"] = dispatch("web_search", {"query": "spacex starship zanzibar"})
        return "ok", [{"tool": "web_search", "args": {}, "result": captured["result"]}]

    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(pipeline, "web_context_for", lambda *a, **k: None)

    pipeline.answer("any spacex starship updates?", [])
    assert "derived from the current user message" in captured["result"]["error"]


def test_web_search_tool_allows_safe_refinement_with_doc_context(monkeypatch):
    # The safe refinement vocabulary stays usable alongside sourced context, as
    # long as the refinement term does not itself occur in the retrieved text.
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["result"] = dispatch("web_search", {"query": "spacex starship launch"})
        return "ok", []

    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(pipeline, "web_context_for", lambda *a, **k: None)
    monkeypatch.setattr(
        pipeline.tools,
        "dispatch",
        lambda name, args: {"query": args["query"], "results": [], "errors": []},
    )

    pipeline.answer(
        "any spacex starship updates?",
        [{"id": "m1", "text": "codename orchidvault roadmap", "source": "upload:abc123:roadmap.pdf"}],
    )
    assert "error" not in captured["result"]


def test_web_search_tool_vetoes_safe_term_found_in_private_context(monkeypatch):
    # Belt-and-braces: a safe refinement term loses its escape hatch when the
    # same word occurs in retrieved memory text, so context cannot ride out on
    # the generic vocabulary either.
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["result"] = dispatch("web_search", {"query": "spacex starship launch"})
        return "ok", []

    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(pipeline, "web_context_for", lambda *a, **k: None)

    pipeline.answer(
        "any spacex starship updates?",
        [{"id": "m1", "text": "internal launch plan for orchidvault", "source": "upload:abc123:plan.pdf"}],
    )
    assert "derived from the current user message" in captured["result"]["error"]


def test_web_search_tool_rejects_second_search_same_turn(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["first"] = dispatch("web_search", {"query": "Phoenix weather"})
        captured["second"] = dispatch("web_search", {"query": "Phoenix forecast"})
        return "ok", []

    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(
        pipeline.tools,
        "dispatch",
        lambda name, args: {"query": args["query"], "results": [], "errors": []},
    )

    pipeline.answer("Phoenix weather forecast", [])
    assert "error" not in captured["first"]
    assert "one call per turn" in captured["second"]["error"]


def test_web_search_refusal_does_not_block_corrected_retry(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["bad"] = dispatch("web_search", {"query": "Alex private portfolio secrets"})
        captured["good"] = dispatch("web_search", {"query": "Phoenix weather"})
        return "ok", []

    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(
        pipeline.tools,
        "dispatch",
        lambda name, args: {"query": args["query"], "results": [], "errors": []},
    )

    pipeline.answer("Phoenix weather", [])
    assert "derived from the current user message" in captured["bad"]["error"]
    assert "error" not in captured["good"]


def test_web_search_tool_rejects_after_prior_tool_result(monkeypatch):
    captured = {}

    def fake_chat_with_tools(system, user, specs, dispatch, **kwargs):
        captured["date"] = dispatch("get_current_datetime", {})
        captured["search"] = dispatch("web_search", {"query": "Phoenix weather"})
        return "ok", []

    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)

    pipeline.answer("Phoenix weather", [])
    assert captured["date"]["date"]
    assert "after another tool result" in captured["search"]["error"]


# --- eager web search: the raw prompt must not leave the box wholesale ----------

def _forbid_search(monkeypatch):
    def boom(query, max_results=None):
        raise AssertionError(f"search_web must not be called, got query: {query!r}")
    monkeypatch.setattr(websearch, "search_web", boom)


def test_eager_search_skipped_for_long_prompt(monkeypatch):
    _forbid_search(monkeypatch)
    private = "latest thoughts: " + "my private notes about finances and health " * 20
    assert len(private) > config.MAX_EAGER_WEB_QUERY_CHARS
    assert websearch.web_context_for(private) is None


def test_eager_search_skipped_for_pasted_block(monkeypatch):
    _forbid_search(monkeypatch)
    pasted = "check the latest on this:\nline one of my notes\nline two\nline three"
    assert len(pasted) < config.MAX_EAGER_WEB_QUERY_CHARS   # short, but pasted-shaped
    assert websearch.web_context_for(pasted) is None


def test_eager_search_sends_short_question_collapsed(monkeypatch):
    captured = {}

    def fake_search(query, max_results=None):
        captured["query"] = query
        return {"query": query, "results": [], "errors": []}

    monkeypatch.setattr(websearch, "search_web", fake_search)
    result = websearch.web_context_for("what is the latest\n  news in Phoenix?")
    assert result is not None
    assert captured["query"] == "what is the latest news in Phoenix?"


def test_eager_search_guard_applies_in_always_mode(monkeypatch):
    _forbid_search(monkeypatch)
    monkeypatch.setattr(config, "WEB_SEARCH_MODE", "always")
    private = "word " * 200   # no search terms needed in always mode
    assert websearch.web_context_for(private) is None


# --- graph: deterministic, newest-first fact ordering ---------------------------

def test_graph_fact_ids_are_newest_first_and_deterministic():
    graph.add_relations(
        [{"subject": "Alex", "relation": "holds", "object": "FNDX"}],
        fact_id="fact-old", date="2024-01-01T00:00:00+00:00",
    )
    graph.add_relations(
        [{"subject": "Alex", "relation": "holds", "object": "FNDZ"}],
        fact_id="fact-new", date="2026-01-01T00:00:00+00:00",
    )
    graph.add_relations(
        [{"subject": "Alex", "relation": "holds", "object": "FNDW"}],
        fact_id="fact-mid", date="2025-01-01T00:00:00+00:00",
    )

    assert graph.fact_ids_for({"alex"}) == ["fact-new", "fact-mid", "fact-old"]


# --- num_ctx: conservative cap when the model max is undiscoverable -------------

def test_unknown_model_num_ctx_is_capped(monkeypatch):
    monkeypatch.setattr(server.llm, "model_max_context", lambda name: None)
    assert server.llm.normalize_num_ctx(262_144, None) == config.UNKNOWN_MODEL_NUM_CTX

    monkeypatch.setattr(server.llm, "model_max_context", lambda name: 8192)
    assert server.llm.normalize_num_ctx(262_144, None) == 8192   # known max still wins
