"""Metacognitive supervisor writes compact Wisdom notes, not noisy transcripts."""
from __future__ import annotations

import os
import subprocess
import sys

import config
from brain import consolidate
from brain import metacognition, pipeline, store


def test_supervisor_stores_actionable_wisdom_note(monkeypatch):
    monkeypatch.setattr(config, "METACOGNITION_ENABLED", True)
    monkeypatch.setattr(
        metacognition.llm,
        "chat",
        lambda *a, **k: (
            '{"note":{"text":"I should verify current facts before answering time-sensitive questions.",'
            '"bias":"stale certainty","confidence":0.82}}'
        ),
    )

    result = metacognition.supervise(
        "what is the latest price?",
        "It is probably unchanged.",
        mems=[{"text": "Old price memory", "category": "fact", "score": 0.4}],
        thinking={"path": "system1", "reason": "short request"},
        emotion_state={"confidence": 0.7},
    )

    assert result["stored"] == 1
    assert result["status"] == "ok"
    rows = store.list_records_all()
    assert len(rows) == 1
    assert rows[0]["text"] == "I should verify current facts before answering time-sensitive questions."
    assert rows[0]["meta"]["category"] == config.METACOGNITION_CATEGORY
    assert rows[0]["meta"]["source_type"] == "generated_metacognition"
    assert rows[0]["meta"]["bias"] == "stale certainty"
    assert rows[0]["meta"]["confidence"] == 0.82
    assert rows[0]["meta"]["importance"] == 3


def test_supervisor_skips_null_note(monkeypatch):
    monkeypatch.setattr(config, "METACOGNITION_ENABLED", True)
    monkeypatch.setattr(metacognition.llm, "chat", lambda *a, **k: '{"note": null}')

    result = metacognition.supervise("hello", "hi")
    assert result["status"] == "skipped"
    assert result["reason"] == "no_note"
    assert result["stored"] == 0
    assert result["id"] is None
    assert store.count_all() == 0


def test_supervisor_stops_nested_reflection_at_max_depth(monkeypatch):
    nested = {}
    monkeypatch.setattr(config, "METACOGNITION_ENABLED", True)
    monkeypatch.setattr(config, "METACOGNITION_MAX_REFLECTION_DEPTH", 1)

    def fake_chat(*args, **kwargs):
        nested["result"] = metacognition.supervise("inner audit", "inner reply")
        return '{"note": null}'

    monkeypatch.setattr(metacognition.llm, "chat", fake_chat)

    result = metacognition.supervise("outer audit", "outer reply")
    assert result["status"] == "skipped"
    assert nested["result"]["status"] == "exhausted"
    assert nested["result"]["reason"] == "max_reflection_depth"
    assert nested["result"]["stored"] == 0
    assert nested["result"]["id"] is None
    assert store.count_all() == 0


def test_supervisor_tolerates_malformed_max_reflection_depth(monkeypatch):
    monkeypatch.setattr(config, "METACOGNITION_ENABLED", True)
    monkeypatch.setattr(config, "METACOGNITION_MAX_REFLECTION_DEPTH", "not-an-int")
    monkeypatch.setattr(metacognition.llm, "chat", lambda *a, **k: '{"note": null}')

    result = metacognition.supervise("hello", "hi")
    assert result["status"] == "skipped"
    assert result["stored"] == 0


def test_config_import_tolerates_malformed_max_reflection_depth():
    env = {**os.environ, "NEURAL_MAP_METACOGNITION_MAX_REFLECTION_DEPTH": "not-an-int"}
    result = subprocess.run(
        [sys.executable, "-c", "import config; print(config.METACOGNITION_MAX_REFLECTION_DEPTH)"],
        cwd=str(config.BASE_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "1"


def test_capture_passes_process_context_to_supervisor(monkeypatch):
    captured = {}
    monkeypatch.setattr(config, "MEMORY_CONSOLIDATION", False)
    monkeypatch.setattr(config, "COGNITION_MODE", "deep")

    def fake_supervise(user_text, reply_text, **kwargs):
        captured["user"] = user_text
        captured["reply"] = reply_text
        captured.update(kwargs)
        return {"stored": 0, "id": None}

    monkeypatch.setattr(pipeline.metacognition, "supervise", fake_supervise)
    monkeypatch.setattr(pipeline.emotion, "update_from_turn", lambda *a, **k: {})
    mems = [{"id": "m1", "text": "poetic memory", "category": "dream"}]
    thinking = {"path": "system2", "reason": "deliberative wording"}

    pipeline.capture("plan the fix", "final answer", mems=mems, thinking=thinking)

    assert captured["user"] == "plan the fix"
    assert captured["reply"] == "final answer"
    assert captured["mems"] == mems
    assert captured["thinking"] == thinking
    assert isinstance(captured["emotion_state"], dict)


def test_balanced_capture_skips_metacognition_for_simple_turn(monkeypatch):
    monkeypatch.setattr(config, "MEMORY_CONSOLIDATION", False)
    monkeypatch.setattr(config, "COGNITION_MODE", "balanced")
    monkeypatch.setattr(pipeline.emotion, "update_from_turn", lambda *a, **k: {})

    def boom(*args, **kwargs):
        raise AssertionError("metacognition should be skipped for cheap balanced turns")

    monkeypatch.setattr(pipeline.metacognition, "supervise", boom)

    pipeline.capture("hello", "hi", thinking={"path": "system1", "confidence": 0.4})


def test_balanced_capture_runs_metacognition_for_system2(monkeypatch):
    captured = {}
    monkeypatch.setattr(config, "MEMORY_CONSOLIDATION", False)
    monkeypatch.setattr(config, "COGNITION_MODE", "balanced")
    monkeypatch.setattr(pipeline.emotion, "update_from_turn", lambda *a, **k: {})

    def fake_supervise(user_text, reply_text, **kwargs):
        captured["thinking"] = kwargs.get("thinking")
        return {"stored": 0, "id": None}

    monkeypatch.setattr(pipeline.metacognition, "supervise", fake_supervise)

    pipeline.capture("plan", "answer", thinking={"path": "system2", "reason": "hard"})

    assert captured["thinking"]["path"] == "system2"


def test_capture_survives_malformed_metacognition_depth(monkeypatch):
    monkeypatch.setattr(config, "MEMORY_CONSOLIDATION", False)
    monkeypatch.setattr(config, "COGNITION_MODE", "balanced")
    monkeypatch.setattr(config, "METACOGNITION_ENABLED", True)
    monkeypatch.setattr(config, "METACOGNITION_MAX_REFLECTION_DEPTH", "not-an-int")
    monkeypatch.setattr(pipeline.metacognition.llm, "chat", lambda *a, **k: '{"note": null}')
    monkeypatch.setattr(pipeline.emotion, "update_from_turn", lambda *a, **k: {})

    pipeline.capture("plan", "answer", thinking={"path": "system2", "reason": "hard"})


def test_capture_treats_max_depth_as_exhausted_decision(monkeypatch):
    monkeypatch.setattr(config, "MEMORY_CONSOLIDATION", False)
    monkeypatch.setattr(config, "COGNITION_MODE", "balanced")
    monkeypatch.setattr(pipeline.emotion, "state", lambda: {})
    monkeypatch.setattr(pipeline.emotion, "update_from_turn", lambda *a, **k: {})
    monkeypatch.setattr(
        pipeline.metacognition,
        "supervise",
        lambda *a, **k: {"stored": 0, "id": None, "skipped": "max_reflection_depth"},
    )

    pipeline.capture("plan", "final answer", thinking={"path": "system2", "reason": "hard"})

    rows = store.list_records_all()
    assert rows[0]["text"] == "User: plan\nAssistant: final answer"


def test_capture_treats_legacy_max_depth_status_as_exhausted_decision(monkeypatch):
    monkeypatch.setattr(config, "MEMORY_CONSOLIDATION", False)
    monkeypatch.setattr(config, "COGNITION_MODE", "balanced")
    monkeypatch.setattr(pipeline.emotion, "state", lambda: {})
    monkeypatch.setattr(pipeline.emotion, "update_from_turn", lambda *a, **k: {})
    monkeypatch.setattr(pipeline.metacognition, "supervise", lambda *a, **k: {"status": "MAX_DEPTH_REACHED"})

    pipeline.capture("plan", "final answer", thinking={"path": "system2", "reason": "hard"})

    rows = store.list_records_all()
    assert rows[0]["text"] == "User: plan\nAssistant: final answer"


def test_capture_treats_bare_max_depth_string_as_exhausted_decision(monkeypatch):
    monkeypatch.setattr(config, "MEMORY_CONSOLIDATION", False)
    monkeypatch.setattr(config, "COGNITION_MODE", "balanced")
    monkeypatch.setattr(pipeline.emotion, "state", lambda: {})
    monkeypatch.setattr(pipeline.emotion, "update_from_turn", lambda *a, **k: {})
    monkeypatch.setattr(pipeline.metacognition, "supervise", lambda *a, **k: "MAX_DEPTH_REACHED")

    pipeline.capture("plan", "final answer", thinking={"path": "system2", "reason": "hard"})

    rows = store.list_records_all()
    assert rows[0]["text"] == "User: plan\nAssistant: final answer"


def test_fast_capture_skips_background_llm_passes(monkeypatch):
    calls = {}
    monkeypatch.setattr(config, "MEMORY_CONSOLIDATION", True)
    monkeypatch.setattr(config, "COGNITION_MODE", "fast")
    monkeypatch.setattr(
        consolidate,
        "consolidate",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("consolidation should be skipped in fast mode")),
    )
    monkeypatch.setattr(
        pipeline.metacognition,
        "supervise",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("metacognition should be skipped in fast mode")),
    )

    def fake_emotion_update(*args, **kwargs):
        calls["llm_analysis"] = kwargs.get("llm_analysis")
        return {}

    monkeypatch.setattr(pipeline.emotion, "update_from_turn", fake_emotion_update)

    pipeline.capture("hello", "hi", thinking={"path": "system2"})

    assert calls["llm_analysis"] is False


def test_balanced_capture_uses_heuristic_emotion(monkeypatch):
    calls = {}
    monkeypatch.setattr(config, "MEMORY_CONSOLIDATION", False)
    monkeypatch.setattr(config, "COGNITION_MODE", "balanced")
    monkeypatch.setattr(pipeline.metacognition, "supervise", lambda *a, **k: {"stored": 0, "id": None})

    def fake_emotion_update(*args, **kwargs):
        calls["llm_analysis"] = kwargs.get("llm_analysis")
        return {}

    monkeypatch.setattr(pipeline.emotion, "update_from_turn", fake_emotion_update)

    pipeline.capture("plan", "answer", thinking={"path": "system2"})

    assert calls["llm_analysis"] is False
