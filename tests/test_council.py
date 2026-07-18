"""Inner Council multi-agent deliberation."""
from __future__ import annotations

import config
from brain import council, pipeline


def test_council_gate_requires_high_stakes_decision(monkeypatch):
    monkeypatch.setattr(config, "INNER_COUNCIL_ENABLED", True)

    assert council.should_convene("Should I quit my job and change careers?")
    assert council.should_convene("Should I accept this job offer?")
    assert council.should_convene("Should I make a major financial decision?")
    assert council.should_convene("Use the inner council to debate this")
    assert not council.should_convene("What did the city council decide about business permits?")
    assert not council.should_convene("How can I invest more time in my health?")
    assert not council.should_convene("Can you help with job interview practice?")
    assert not council.should_convene("What is a good coffee ratio?")


def test_council_gate_ignores_study_questions(monkeypatch):
    # Nursing coursework is full of decision + medical phrasing; none of it is a
    # personal high-stakes decision, so no 5-call debate.
    monkeypatch.setattr(config, "INNER_COUNCIL_ENABLED", True)

    assert not council.should_convene(
        "NCLEX practice: should I give the beta blocker medication before checking heart rate?"
    )
    assert not council.should_convene("Quiz me: which symptom should I assess first?")
    assert not council.should_convene("For the exam, should I memorize the surgery checklist?")
    assert not council.should_convene("Explain the rationale: should I clamp before the doctor arrives?")
    # An explicit request overrides the study suppressor...
    assert council.convene_reason("Convene the council: should I drop pharmacology this exam cycle?") == "explicit_request"
    # ...and real personal decisions still convene with an attributable reason.
    assert council.convene_reason("Should I quit my job to focus on my portfolio?") == "personal_decision"
    # Loose "recommend a resource" phrasing no longer triggers on topic words.
    assert not council.should_convene("Can you recommend a pharmacology textbook about medication safety?")


def test_council_defaults_to_one_round(monkeypatch):
    monkeypatch.setattr(config, "INNER_COUNCIL_ENABLED", True)
    monkeypatch.setattr(config, "INNER_COUNCIL_TURNS", 1)
    calls = []

    def fake_chat(system, user, **kwargs):
        calls.append((system, user))
        if "Synthesize this inner-council debate" in user:
            return "moderated final"
        return f"statement {len(calls)}"

    monkeypatch.setattr(council.llm, "chat", fake_chat)

    out = council.deliberate("Should I make a major career decision?", "draft answer")

    assert out["turns"] == 1
    assert len(out["transcript"]) == 4
    assert len(calls) == 5


def test_council_routes_all_five_calls_to_auxiliary_model(monkeypatch):
    monkeypatch.setattr(config, "INNER_COUNCIL_ENABLED", True)
    monkeypatch.setattr(config, "INNER_COUNCIL_TURNS", 1)
    monkeypatch.setattr(config, "COUNCIL_MODEL", "gemma4:e4b")
    calls = []

    def fake_chat(system, user, **kwargs):
        calls.append(kwargs)
        return "moderated final" if "Synthesize this inner-council debate" in user else "statement"

    monkeypatch.setattr(council.llm, "chat", fake_chat)

    out = council.deliberate(
        "Should I make a major career decision?",
        "draft answer",
        model="gemma4:26b",
        num_ctx=262144,
    )

    assert out["model"] == "gemma4:e4b"
    assert len(calls) == 5
    assert {call["model"] for call in calls} == {"gemma4:e4b"}
    assert {call["num_ctx"] for call in calls} == {config.AUXILIARY_NUM_CTX}
    assert {call["think"] for call in calls} == {False}


def test_council_runs_four_personas_three_rounds_then_moderator(monkeypatch):
    monkeypatch.setattr(config, "INNER_COUNCIL_ENABLED", True)
    monkeypatch.setattr(config, "INNER_COUNCIL_TURNS", 3)
    calls = []

    def fake_chat(system, user, **kwargs):
        calls.append((system, user))
        if "Synthesize this inner-council debate" in user:
            return "moderated final"
        return f"statement {len(calls)}"

    monkeypatch.setattr(council.llm, "chat", fake_chat)

    out = council.deliberate("Should I make a major career decision?", "draft answer")

    assert out["final"] == "moderated final"
    assert out["turns"] == 3
    assert out["personas"] == ["Optimist", "Cynic", "Strategist", "Ethicist"]
    assert len(out["transcript"]) == 12
    assert len(calls) == 13
    assert out["transcript"][0]["persona"] == "Optimist"
    assert out["transcript"][-1]["round"] == 3


def test_pipeline_replaces_draft_with_council_final(monkeypatch):
    monkeypatch.setattr(config, "TOOL_CALLING_ENABLED", False)
    monkeypatch.setattr(config, "INTUITION_ENABLED", False)
    monkeypatch.setattr(pipeline, "market_context_for", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "web_context_for", lambda *a, **k: None)
    monkeypatch.setattr(pipeline.llm, "chat", lambda *a, **k: "draft")

    def fake_deliberate(user_text, draft_reply, **kwargs):
        return {
            "used": True,
            "turns": 3,
            "personas": ["Optimist", "Cynic", "Strategist", "Ethicist"],
            "transcript": [{"round": 1, "persona": "Optimist", "statement": "go"}],
            "final": "council final",
        }

    monkeypatch.setattr(pipeline.council, "deliberate", fake_deliberate)

    out = pipeline.answer("Should I make a major financial decision?", [])

    assert out["reply"] == "council final"
    assert out["council"]["used"] is True
    assert out["thinking"]["council"] == "inner council convened"


def test_pipeline_skips_deliberation_without_a_trigger(monkeypatch):
    monkeypatch.setattr(config, "TOOL_CALLING_ENABLED", False)
    monkeypatch.setattr(config, "INTUITION_ENABLED", False)
    monkeypatch.setattr(pipeline, "market_context_for", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "web_context_for", lambda *a, **k: None)
    monkeypatch.setattr(pipeline.llm, "chat", lambda *a, **k: "answer")
    convened = []
    monkeypatch.setattr(pipeline.council, "deliberate", lambda *a, **k: convened.append(1))

    out = pipeline.answer("Quiz me: which medication should I give first?", [])

    assert convened == []          # study question — no 5-call debate
    assert out["council"]["used"] is False


def test_pipeline_emits_one_cognition_gates_summary(monkeypatch):
    monkeypatch.setattr(config, "TOOL_CALLING_ENABLED", False)
    monkeypatch.setattr(config, "INTUITION_ENABLED", False)
    monkeypatch.setattr(pipeline, "market_context_for", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "web_context_for", lambda *a, **k: None)
    monkeypatch.setattr(pipeline.llm, "chat", lambda *a, **k: "answer")

    out = pipeline.answer("What is a good coffee ratio?", [])

    summaries = [e for e in out["trace"]["events"] if e["layer"] == "cognition_gates"]
    assert len(summaries) == 1
    fields = summaries[0]["fields"]
    assert fields["council"] == "not_used"
    assert fields["council_reason"] == "no_trigger"
    assert fields["path"] in ("standard", "system1", "system2")
    assert fields["foreground_llm_calls"] >= 1


def test_pipeline_feeds_system2_result_to_council(monkeypatch):
    monkeypatch.setattr(config, "TOOL_CALLING_ENABLED", False)
    monkeypatch.setattr(config, "INTUITION_ENABLED", True)
    monkeypatch.setattr(pipeline, "market_context_for", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "web_context_for", lambda *a, **k: None)
    monkeypatch.setattr(
        pipeline,
        "_intuition_gate",
        lambda *a, **k: {"path": "system2", "reason": "test", "resonance": 0.0},
    )
    replies = iter(["system1 draft", "critique", "system2 final"])
    monkeypatch.setattr(pipeline.llm, "chat", lambda *a, **k: next(replies))
    seen = {}

    def fake_deliberate(user_text, draft_reply, **kwargs):
        seen["draft_reply"] = draft_reply
        return None

    monkeypatch.setattr(pipeline.council, "deliberate", fake_deliberate)

    out = pipeline.answer("Should I make a major financial decision?", [])

    assert seen["draft_reply"] == "system2 final"
    assert out["reply"] == "system2 final"
