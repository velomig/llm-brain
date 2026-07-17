"""System 1 / System 2 decision gate."""
from __future__ import annotations

import math

import config
from brain import emotion, pipeline


def _quiet_context(monkeypatch):
    monkeypatch.setattr(config, "TOOL_CALLING_ENABLED", False)
    monkeypatch.setattr(pipeline, "market_context_for", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "web_context_for", lambda *a, **k: None)


def test_simple_high_confidence_query_uses_system1(monkeypatch):
    _quiet_context(monkeypatch)
    calls = []
    monkeypatch.setattr(pipeline.llm, "chat", lambda *a, **k: calls.append(a[1]) or "quick reply")
    emotion.set_state({"confidence": 0.4})

    out = pipeline.answer("coffee?", [{"id": "m1", "text": "Alex likes black coffee.", "similarity": 0.8}])

    assert out["reply"] == "quick reply"
    assert out["thinking"]["path"] == "system1"
    assert out["thinking"]["cognitive_state_id"]
    assert out["thinking"]["resonance"] == 0.8
    assert out["trace"]["kind"] == "chat_turn"
    assert any(e["layer"] == "intuition" and e["status"] == "system1" for e in out["trace"]["events"])
    assert len(calls) == 1


def test_low_confidence_query_uses_system2_draft_critique_rewrite(monkeypatch):
    _quiet_context(monkeypatch)
    replies = iter(["draft answer", "critique the draft", "final answer"])
    prompts = []

    def fake_chat(system, user, **kwargs):
        prompts.append(user)
        return next(replies)

    monkeypatch.setattr(pipeline.llm, "chat", fake_chat)
    emotion.set_state({"confidence": -0.5})

    out = pipeline.answer("coffee?", [])

    assert out["reply"] == "final answer"
    assert out["thinking"]["path"] == "system2"
    assert "low affective confidence" in out["thinking"]["reason"]
    assert "Critique this draft" in prompts[1]
    assert "Rewrite the answer" in prompts[2]


def test_fast_cognition_mode_suppresses_system2(monkeypatch):
    _quiet_context(monkeypatch)
    calls = []
    monkeypatch.setattr(config, "COGNITION_MODE", "fast")
    monkeypatch.setattr(pipeline.llm, "chat", lambda *a, **k: calls.append(a[1]) or "fast reply")
    emotion.set_state({"confidence": -0.5})

    out = pipeline.answer("Explain the tradeoff and plan the fix", [])

    assert out["reply"] == "fast reply"
    assert out["thinking"]["path"] == "standard"
    assert out["thinking"]["cognition_mode"] == "fast"
    assert out["thinking"]["foreground_llm_calls_estimate"] == 1
    assert len(calls) == 1


def test_complex_query_uses_system2_even_with_neutral_confidence(monkeypatch):
    _quiet_context(monkeypatch)
    replies = iter(["draft", "critique", "final"])
    monkeypatch.setattr(pipeline.llm, "chat", lambda *a, **k: next(replies))
    emotion.reset()

    out = pipeline.answer("Explain the tradeoff and plan the fix", [])

    assert out["reply"] == "final"
    assert out["thinking"]["path"] == "system2"
    assert "deliberative wording" in out["thinking"]["reason"]


def test_malformed_emotion_state_falls_back_to_system2(monkeypatch):
    _quiet_context(monkeypatch)
    replies = iter(["draft", "critique", "final"])
    monkeypatch.setattr(pipeline.llm, "chat", lambda *a, **k: next(replies))
    monkeypatch.setattr(pipeline.emotion, "state", lambda: {"confidence": "not-a-float"})

    out = pipeline.answer("coffee?", [])

    assert out["reply"] == "final"
    assert out["thinking"]["path"] == "system2"
    assert "malformed emotion state" in out["thinking"]["reason"]
    assert out["thinking"]["confidence"] is None


def test_non_finite_emotion_state_falls_back_to_system2(monkeypatch):
    _quiet_context(monkeypatch)
    replies = iter(["draft", "critique", "final"])
    monkeypatch.setattr(pipeline.llm, "chat", lambda *a, **k: next(replies))
    monkeypatch.setattr(pipeline.emotion, "state", lambda: {"confidence": math.nan})

    out = pipeline.answer("coffee?", [])

    assert out["reply"] == "final"
    assert out["thinking"]["path"] == "system2"
    assert "malformed emotion state" in out["thinking"]["reason"]
