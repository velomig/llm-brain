"""Swallowed fallback events must be logged (structured), even though the chat
keeps working. Each test drives a real code path into its failure branch and
asserts a "neural_map" log record with the expected event= identifier."""
from __future__ import annotations

import logging

import pytest

import config
from brain import log, observability
from util import store_memory


def _raise(exc: Exception):
    def _fn(*a, **k):
        raise exc
    return _fn


@pytest.fixture
def log_records():
    """Capture records emitted on the neural_map logger (independent of its own
    handler / propagate setting), at DEBUG so even low-level events are seen."""
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture()
    log.logger.addHandler(handler)
    old_level = log.logger.level
    log.logger.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        log.logger.removeHandler(handler)
        log.logger.setLevel(old_level)


def _has_event(records, name: str) -> bool:
    return any(f"event={name}" in r.getMessage() for r in records)


def test_embedding_unavailable_is_logged(fake_embed, log_records):
    from brain import embeddings

    fake_embed.available = False
    with pytest.raises(embeddings.EmbeddingUnavailable):
        embeddings.embed_document("hello world")
    assert _has_event(log_records, "embedding_unavailable")


def test_graph_recall_failure_is_logged(monkeypatch, log_records):
    from brain import retriever

    store_memory("a memory about quarterly taxes", category="fact")
    monkeypatch.setattr(retriever.graph, "expand", _raise(RuntimeError("graph boom")))
    # Semantic still produces candidates, so the chat path survives the graph error.
    retriever.retrieve_weighted("taxes", w_sim=1, w_keyword=0, w_graph=1)
    assert _has_event(log_records, "graph_failed")


def test_consolidation_failure_is_logged(monkeypatch, log_records):
    from brain import consolidate

    monkeypatch.setattr(config, "MEMORY_CONSOLIDATION", True)
    monkeypatch.setattr(consolidate.llm, "chat", _raise(RuntimeError("llm down")))
    result = consolidate.consolidate("user said something", "assistant replied")
    assert result["stored"] == 0
    assert _has_event(log_records, "consolidation_failed")


def test_reflection_failure_is_logged(monkeypatch, log_records):
    from brain import reflect

    store_memory("a durable fact to reflect on", category="fact", importance=4)
    monkeypatch.setattr(reflect.llm, "chat", _raise(RuntimeError("llm down")))
    assert reflect.reflect() == {"insights": 0, "reminders": 0, "initiatives": 0}
    assert _has_event(log_records, "reflection_failed")


def test_tool_calling_failure_is_logged(monkeypatch, log_records):
    from brain import pipeline

    monkeypatch.setattr(config, "TOOL_CALLING_ENABLED", True)
    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)
    monkeypatch.setattr(pipeline.llm, "chat_with_tools", _raise(RuntimeError("tool channel error")))
    monkeypatch.setattr(pipeline.llm, "chat", lambda *a, **k: "fallback reply")
    monkeypatch.setattr(pipeline, "market_context_for", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "web_context_for", lambda *a, **k: None)

    out = pipeline.answer("what is my portfolio worth", [])
    assert out["reply"] == "fallback reply"          # the turn still succeeds
    assert _has_event(log_records, "tool_calling_failed")  # ...but the failure is visible


def test_metacognition_failure_is_logged(monkeypatch, log_records):
    from brain import metacognition

    monkeypatch.setattr(config, "METACOGNITION_ENABLED", True)
    monkeypatch.setattr(metacognition.llm, "chat", _raise(RuntimeError("supervisor down")))
    result = metacognition.supervise("user", "reply")

    assert result["stored"] == 0
    assert _has_event(log_records, "metacognition_failed")


def test_inner_council_failure_is_logged(monkeypatch, log_records):
    from brain import council

    monkeypatch.setattr(config, "INNER_COUNCIL_ENABLED", True)
    monkeypatch.setattr(council.llm, "chat", _raise(RuntimeError("council down")))
    result = council.deliberate("Should I make a major career decision?", "draft")

    assert result is None
    assert _has_event(log_records, "inner_council_failed")


def test_evolution_failure_is_logged(monkeypatch, log_records):
    from brain import evolution

    monkeypatch.setattr(config, "EVOLUTION_ENABLED", True)
    monkeypatch.setattr(evolution.llm, "chat", _raise(RuntimeError("mutation model down")))
    result = evolution.propose({"preamble": "p", "tools_note": "t", "inject_note": "i"})

    assert "mutation model down" in result["error"]
    assert _has_event(log_records, "evolution_failed")


def test_structured_fields_are_rendered(log_records):
    log.fallback("graph_failed", ValueError("bad entity"), phase="recall", model="m")
    msg = next(r.getMessage() for r in log_records if "event=graph_failed" in r.getMessage())
    assert "phase=recall" in msg
    assert "model=m" in msg
    assert 'error="ValueError: bad entity"' in msg


def test_fallback_logs_include_active_trace_context(log_records):
    trace = observability.start("unit", user_text="private text")

    log.fallback("unit_failed", RuntimeError("boom"), phase="unit")

    msg = next(r.getMessage() for r in log_records if "event=unit_failed" in r.getMessage())
    assert f"trace_id={trace['id']}" in msg
    assert "trace_kind=unit" in msg
    recent = log.recent(1)[0]
    assert recent["fields"]["trace_id"] == trace["id"]
    assert recent["fields"]["trace_kind"] == "unit"
    assert "private text" not in str(recent)
    observability.finish(trace)


def test_finished_trace_context_does_not_leak_to_later_logs(log_records):
    trace = observability.start("unit")
    observability.finish(trace)

    log.fallback("after_trace", RuntimeError("boom"), phase="unit")

    msg = next(r.getMessage() for r in log_records if "event=after_trace" in r.getMessage())
    assert "trace_id=" not in msg
    assert "trace_kind=" not in msg
