"""Emotional State / affective stance layer."""
from __future__ import annotations

import config
import server
from brain import emotion, pipeline


def test_emotion_state_clamps_persists_and_resets():
    out = emotion.set_state({"happiness": 2, "stress": -2, "confidence": 0.5, "curiosity": 0.25, "reason": "manual"})

    assert out["happiness"] == 1.0
    assert out["stress"] == -1.0
    assert out["confidence"] == 0.5
    assert out["curiosity"] == 0.25
    assert emotion.state()["reason"] == "manual"

    reset = emotion.reset()
    assert {k: reset[k] for k in emotion.KEYS} == {k: 0.0 for k in emotion.KEYS}


def test_emotion_update_uses_small_llm_delta_and_decay(monkeypatch):
    monkeypatch.setattr(config, "EMOTION_LLM_ANALYSIS", True)
    monkeypatch.setattr(
        emotion,
        "_llm_delta",
        lambda user, reply, model=None: {
            "happiness": -0.2,
            "stress": 0.2,
            "confidence": -0.1,
            "curiosity": 0.05,
            "reason": "test",
        },
    )
    emotion.set_state({"happiness": 0.5, "stress": 0, "confidence": 0, "curiosity": 0})

    out = emotion.update_from_turn("bad day", "I can help")

    assert out["happiness"] == 0.31  # 0.5 * 0.92 + clamped -0.15
    assert out["stress"] == 0.15
    assert out["confidence"] == -0.1
    assert out["curiosity"] == 0.05
    assert out["reason"] == "test"


def test_emotion_heuristic_fallback_when_llm_disabled(monkeypatch):
    monkeypatch.setattr(config, "EMOTION_LLM_ANALYSIS", False)

    out = emotion.update_from_turn("I am frustrated and this is wrong", "Maybe we can fix it")

    assert out["stress"] > 0
    assert out["happiness"] < 0
    assert out["confidence"] < 0


def test_build_system_injects_emotional_state():
    system = pipeline.build_system([], emotion_state=emotion.set_state({"stress": 0.6, "confidence": -0.4, "curiosity": 0.3}))

    assert "Emotional State JSON:" in system
    assert '"stress": 0.6' in system
    assert "subtle tone/style guidance" in system


def test_emotion_api_patch_and_reset():
    patched = server.set_emotion(server.EmotionReq(stress=0.4, curiosity=0.2, reason="api"))
    assert patched["stress"] == 0.4
    assert patched["curiosity"] == 0.2
    assert patched["reason"] == "api"

    reset = server.reset_emotion()
    assert reset["stress"] == 0.0
