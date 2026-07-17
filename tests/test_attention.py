from __future__ import annotations

from fastapi.testclient import TestClient

import config
import server
from brain import attention, pipeline


def test_attention_updates_domain_focus_from_turn():
    state = attention.update_from_turn(
        "SPCX stock price dropped; review the portfolio allocation.",
        mems=[{"text": "SPCX position note", "category": "note"}],
        tools=["get_stock_quotes"],
    )

    assert state["enabled"] is True
    assert state["focused_domain"] == "finance"
    assert state["domains"]["finance"] > state["domains"]["nursing"]


def test_attention_ranks_memories_without_dropping_anything():
    attention.update_from_turn("Review the nursing exam and clinical schedule.")
    rows = [
        {"id": "finance", "text": "SPCX portfolio allocation note", "category": "fact"},
        {"id": "nursing", "text": "Nursing clinical exam schedule", "category": "fact"},
    ]

    ranked = attention.rank_memories("What should I study for clinical?", rows)

    assert [row["id"] for row in ranked] == ["nursing", "finance"]
    assert {row["id"] for row in ranked} == {"nursing", "finance"}


def test_attention_ranks_same_priority_initiatives_by_focus():
    attention.update_from_turn("Let's fix the server tests and code.")
    initiatives = [
        {"id": "finance", "text": "Review the portfolio allocation", "kind": "finance", "priority": 4},
        {"id": "code", "text": "Run tests for the server code", "kind": "system", "priority": 4},
    ]

    ranked = attention.rank_initiatives(initiatives, user_text="continue the code work")

    assert ranked[0]["id"] == "code"


def test_build_system_includes_attention_state():
    attention.update_from_turn("How can dynamic attention route memory?")

    system = pipeline.build_system([])

    assert "Attention State JSON:" in system
    assert "Use attention only to prioritize relevant context" in system


def test_attention_api_endpoint():
    attention.update_from_turn("Can you fix the web app?")
    client = TestClient(server.app, base_url="http://127.0.0.1")

    res = client.get("/api/attention")

    assert res.status_code == 200
    assert res.json()["focused_domain"] == "system_maintenance"


def test_web_search_does_not_create_finance_focus():
    attention.reset()

    state = attention.update_from_turn("President election news today.", tools=["web_search"])

    assert state["focused_domain"] == "general"
    assert state["domains"]["finance"] == 0.0


def test_protected_memory_domains_do_not_create_focus_without_user_cue():
    attention.reset()

    state = attention.update_from_turn(
        "President election news today.",
        mems=[
            {"text": "Portfolio shares stock ticker allocation note", "category": "episode"},
            {"text": "Nursing clinical NCLEX medication patient note", "category": "episode"},
        ],
    )

    assert state["focused_domain"] == "general"
    assert state["domains"]["finance"] == 0.0
    assert state["domains"]["nursing"] == 0.0


def test_health_reports_attention(monkeypatch):
    monkeypatch.setattr(server.llm, "list_models", lambda: [config.CHAT_MODEL, config.EMBED_MODEL])

    out = server.health()

    assert out["features"]["attention"] == config.ATTENTION_ENABLED
    assert out["attention"]["enabled"] is True



# --- momentum: focus shifts are gradual, not single-message jumps ---------------

_FINANCE_TURN = "Review the portfolio allocation: SPCX, SCHD, SGOV shares and dividend income."
_NURSING_TURN = "Nursing clinical exam prep: NCLEX medication review with a patient study lab."


def _establish_finance_focus(turns: int = 3) -> None:
    for _ in range(turns):
        attention.update_from_turn(_FINANCE_TURN)


def test_single_offtopic_burst_does_not_displace_established_focus():
    _establish_finance_focus()

    state = attention.update_from_turn(_NURSING_TURN)

    assert state["focused_domain"] == "finance"
    assert state["domains"]["finance"] > state["domains"]["nursing"]
    # The burst still registers — it accumulates instead of being ignored.
    assert state["domains"]["nursing"] > 0


def test_sustained_topic_shift_takes_over_within_a_few_turns():
    _establish_finance_focus()

    state = attention.state()
    for _ in range(4):
        state = attention.update_from_turn(_NURSING_TURN)

    assert state["focused_domain"] == "nursing"
    assert state["domains"]["nursing"] > state["domains"]["finance"]


def test_cold_start_still_focuses_on_first_clear_message():
    attention.reset()

    state = attention.update_from_turn(_FINANCE_TURN)

    assert state["focused_domain"] == "finance"


def test_unsignaled_domains_still_decay():
    _establish_finance_focus()
    before = attention.state()["domains"]["finance"]

    after = attention.update_from_turn("Completely unrelated pleasantries, nothing topical.")

    assert after["domains"]["finance"] < before


# --- learned vocabulary: evolution-approved terms close the static blind spot ---

def test_learned_terms_extend_domain_vocabulary_end_to_end(monkeypatch, tmp_path):
    from brain import evolution
    import json as _json

    message = "Pharmacology flashcards for the cardiology rotation."

    attention.reset()
    baseline = attention.update_from_turn(message)
    assert baseline["focused_domain"] == "general"   # the blind spot

    canary = _json.dumps(
        [{"area": a, "expected": "preserve", "notes": "covered"} for a in evolution.CANARY_AREAS]
    )
    monkeypatch.setattr(config, "EVOLUTION_LOG_PATH", tmp_path / "evolution_log.txt")
    monkeypatch.setattr(
        evolution.llm,
        "chat",
        lambda *a, **k: (
            '{"title":"Teach nursing vocabulary","rationale":"r","target_failure":"t",'
            '"expected_tradeoff":"none","prompt_overrides":{},"hyperparameters":{},'
            '"attention_terms":{"nursing":["pharmacology","cardiology","rotation"]},'
            f'"canary_impact":{canary},"risk_notes":[]}}'
        ),
    )
    proposal = evolution.propose({"preamble": "P", "tools_note": "T", "inject_note": "I"})
    assert evolution.approve(proposal["id"])["approved"] is True

    attention.reset()
    state = attention.update_from_turn(message)

    assert state["focused_domain"] == "nursing"
    assert state["domains"]["nursing"] > 0
