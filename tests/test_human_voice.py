"""Human-likeness prompt layer: the output contract scales to the question,
voice direction is positive and personality-driven, retrieved memories carry
hedged-recall bands, and the emotional stance shapes pacing — not just tone."""
from __future__ import annotations

import json

import config
from brain import emotion
from brain.pipeline import _memory_match_bands, _voice_note, build_system


def _memories_json(system: str) -> list[dict]:
    block = system.split("Retrieved Memories JSON:", 1)[1].split("Proactive Reminders JSON:", 1)[0].strip()
    return json.loads(block)


# --- output contract: structure is earned, not mandatory ---------------------

def test_output_contract_scales_to_the_question():
    system = build_system([])
    assert "Readable Output Contract" in system
    assert "Simple questions get a plain sentence or two" in system
    # The old always-on mandate is gone.
    assert "persistent presentation requirement" not in system


# --- voice: positive direction that follows the personality profile -----------

def test_voice_note_licenses_opinions_and_mirrors_personality():
    note = _voice_note()
    assert "commit to an opinion" in note
    assert "instead of presenting a neutral menu" in note
    # Trait words track the configured Will profile (defaults: curious .70,
    # cautious .65, playful .35).
    if config.WILL_PERSONALITY_CURIOUS >= 0.55:
        assert "curious" in note
    if config.WILL_PERSONALITY_PLAYFUL >= 0.30:
        assert "playful" in note


def test_voice_note_reaches_the_system_prompt():
    assert "Voice: speak like a person, not a system." in build_system([])


# --- match bands: hazy recall hedges, strong recall speaks plainly ------------

def test_match_bands_split_strong_from_possible():
    mems = [
        {"id": "1", "text": "close hit", "similarity": 0.90},
        {"id": "2", "text": "distant hit", "similarity": 0.10},
    ]
    assert _memory_match_bands(mems) == ["strong", "possible"]


def test_match_bands_pinned_is_always_strong():
    mems = [{"id": "1", "text": "pinned", "pinned": True, "similarity": 0.05}]
    assert _memory_match_bands(mems) == ["strong"]


def test_match_bands_none_without_scores():
    assert _memory_match_bands([{"id": "1", "text": "x"}]) is None


def test_prompt_marks_weak_recall_and_offers_the_hedge():
    mems = [
        {"id": "1", "text": "strong", "category": "fact", "source": None, "date": "2026-06-12", "similarity": 0.9},
        {"id": "2", "text": "weak", "category": "fact", "source": None, "date": "2026-06-12", "similarity": 0.1},
    ]
    system = build_system(mems)
    rows = _memories_json(system)
    assert rows[0]["match"] == "strong"
    assert rows[1]["match"] == "possible"
    assert "if I'm remembering right" in system


def test_prompt_omits_match_field_when_no_scores():
    rows = _memories_json(build_system([{"id": "1", "text": "x", "category": "note", "source": None, "date": "2026-06-12"}]))
    assert "match" not in rows[0]


# --- emotion: pacing and verbosity, not just tone -----------------------------

def test_emotion_guidance_shapes_pacing_and_verbosity():
    block = emotion.prompt_block({
        "happiness": 0.0, "stress": 0.6, "confidence": -0.4, "curiosity": 0.0,
        "updated_at": None, "reason": "test",
    })
    data = json.loads(block)
    assert "cautious" in data["labels"]
    assert "subtle tone/style guidance" in data["guidance"]
    assert "pacing" in data["guidance"]
    assert "keep replies tighter" in data["guidance"]


def test_emotion_guidance_tracks_the_current_stance():
    block = emotion.prompt_block({
        "happiness": 0.0, "stress": -0.5, "confidence": 0.5, "curiosity": 0.3,
        "updated_at": None, "reason": "test",
    })
    data = json.loads(block)
    assert "more expansive" in data["guidance"]      # settled
    assert "commit to recommendations plainly" in data["guidance"]  # decisive
