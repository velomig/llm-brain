"""Will layer: autonomous proactive curiosity/worry review."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import config
import server
from brain import initiative, pipeline, store, tools, will
from util import store_memory


def _candidate(message: str = "Should we revisit the launch risk?") -> str:
    return json.dumps(
        {
            "should_reach_out": True,
            "message": message,
            "reason": "A launch fact and open loop point at an unresolved risk.",
            "kind": "strategy",
            "priority": 4,
            "cooldown_key": "launch-risk",
        }
    )


def _dual_candidates(logical_message: str, intuitive_message: str) -> str:
    def row(message: str, reason: str, mode: str) -> dict:
        return {
            "should_reach_out": True,
            "message": message,
            "reason": reason,
            "kind": "strategy" if mode == "logical" else "curiosity",
            "priority": 4,
            "confidence": 0.7 if mode == "logical" else 0.9,
            "risk": "low",
            "reversible": True,
            "cooldown_key": f"launch-{mode}",
        }

    return json.dumps(
        {
            "logical": row(logical_message, "The direct next step closes an open loop.", "logical"),
            "intuitive": row(intuitive_message, "A small unusual experiment may reveal a hidden constraint.", "intuitive"),
        }
    )


def _enable_predictable_exploration(monkeypatch):
    monkeypatch.setattr(config, "WILL_NOVELTY_WEIGHT", 0.0)
    monkeypatch.setattr(config, "WILL_WEIRD_WIN_WEIGHT", 0.0)
    monkeypatch.setattr(config, "WILL_PERSONALITY_CURIOUS", 0.0)
    monkeypatch.setattr(config, "WILL_PERSONALITY_CAUTIOUS", 0.0)
    monkeypatch.setattr(config, "WILL_PERSONALITY_PLAYFUL", 0.0)
    monkeypatch.setattr(config, "WILL_INTUITIVE_SELECTION_PROBABILITY", 0.1)
    monkeypatch.setattr(config, "WILL_GOOD_ENOUGH_RATIO", 0.9)
    monkeypatch.setattr(config, "WILL_EXPLORATION_MAX_DAILY", 1)


def test_will_dry_run_does_not_create_notification_or_mutate_state(monkeypatch):
    store_memory("Alex is planning a launch.", category="fact", importance=4)
    monkeypatch.setattr(will.llm, "chat", lambda *a, **k: _candidate())

    out = will.review(dry_run=True)

    assert out["created"] == 0
    assert out["notification"]["message"] == "Should we revisit the launch risk?"
    assert initiative.active_initiatives(mode="coach") == []
    assert will.state()["last_run"] is None


def test_will_review_creates_open_initiative(monkeypatch):
    store_memory("Alex is planning a launch.", category="fact", importance=4)
    store_memory("There is an unresolved launch staffing question.", category="note", importance=4)
    monkeypatch.setattr(will.llm, "chat", lambda *a, **k: _candidate())

    out = will.review()

    assert out["created"] == 1
    item = initiative.active_initiatives(mode="coach")[0]
    assert item["kind"] == "arbiter_grounding"
    assert item["admissibility"] == "needs_evidence"
    assert item["original_text"] == "Should we revisit the launch risk?"
    rec = store.get_by_ids([out["id"]])[out["id"]]
    assert rec["meta"]["origin"] == "will"
    assert rec["meta"]["source_type"] == "generated_will"
    assert rec["meta"]["cooldown_key"] == "launch_risk"
    assert rec["meta"]["grounding_status"] == "symbolic_only"
    assert json.loads(rec["meta"]["grounding_channels_json"]) == ["linguistic"]
    assert will.state()["outreach_today"] == 1


def test_will_blocks_duplicate_open_topic(monkeypatch):
    store_memory("Alex is planning a launch.", category="fact", importance=4)
    monkeypatch.setattr(will.llm, "chat", lambda *a, **k: _candidate())

    first = will.review()
    second = will.review()

    assert first["created"] == 1
    assert second["created"] == 0
    assert "cooling down" in second["reason"] or "already open" in second["reason"]


def test_will_canonicalizes_ssdi_cooldown_variants(monkeypatch):
    store_memory("Alex needs to confirm how SSDI changes with nursing income.", category="fact", importance=5)
    first = json.loads(_candidate("Should we model the SSDI income gap?"))
    first["cooldown_key"] = "ssdi_financial_modeling"
    second = {
        "should_reach_out": True,
        "message": "Should we review the SSDI retirement risk?",
        "reason": "The same disability-benefit topic remains unresolved.",
        "kind": "worry",
        "priority": 5,
        "cooldown_key": "sddi_retirement_risk",
    }
    candidates = iter([json.dumps(first), json.dumps(second)])
    monkeypatch.setattr(will.llm, "chat", lambda *a, **k: next(candidates))

    created = will.review()
    blocked = will.review()

    assert created["created"] == 1
    assert created["notification"]["cooldown_key"] == "ssdi"
    assert blocked["created"] == 0
    assert blocked["reason"] == "topic cooling down"
    assert set(will.state()["cooldowns"]) == {"ssdi"}
    rec = store.get_by_ids([created["id"]])[created["id"]]
    assert rec["meta"]["cooldown_key"] == "ssdi"


def test_will_migrates_and_prunes_cooldown_state():
    now = datetime.now(timezone.utc)
    earlier = (now + timedelta(hours=1)).isoformat()
    later = (now + timedelta(hours=2)).isoformat()
    expired = (now - timedelta(hours=1)).isoformat()
    will._STATE_FILE.write_text(
        json.dumps(
            {
                "date": will._today(),
                "outreach_today": 1,
                "cooldowns": {
                    "ssdi_financial_gap": earlier,
                    "sddi_financial_modeling": later,
                    "expired_topic": expired,
                },
            }
        ),
        encoding="utf-8",
    )

    migrated = will.state()

    assert migrated["cooldowns"] == {"ssdi": later}


def test_will_matches_legacy_open_notification_alias(monkeypatch):
    store_memory("Alex needs to confirm how SSDI changes with nursing income.", category="fact", importance=5)
    store_memory(
        "Should we model the SSDI income gap?",
        category="initiative",
        importance=5,
        extra={
            "origin": "will",
            "status": "open",
            "cooldown_key": "sddi_financial_modeling",
        },
    )
    candidate = json.loads(_candidate("Should we review the SSDI retirement risk?"))
    candidate["cooldown_key"] = "ssdi_retirement_risk"
    monkeypatch.setattr(will.llm, "chat", lambda *a, **k: json.dumps(candidate))

    out = will.review()

    assert out["created"] == 0
    assert out["reason"] == "matching notification already open"


def test_will_no_reachout_records_last_run(monkeypatch):
    store_memory("Alex likes concise updates.", category="fact", importance=4)
    monkeypatch.setattr(
        will.llm,
        "chat",
        lambda *a, **k: json.dumps({"should_reach_out": False, "reason": "nothing timely"}),
    )

    out = will.review()

    assert out["created"] == 0
    assert out["reason"] == "nothing timely"
    assert will.state()["last_run"]
    assert will.state()["last_result"]["status"] == "completed"
    assert will.state()["last_result"]["reason"] == "nothing timely"


def test_will_failure_records_queryable_receipt(monkeypatch):
    store_memory("Alex has an unresolved launch question.", category="fact", importance=4)

    def _fail(*args, **kwargs):
        raise ConnectionError("model unavailable")

    monkeypatch.setattr(will.llm, "chat", _fail)

    out = will.review()
    status = tools.dispatch("will_status", {})

    assert out["reason"] == "review failed"
    assert status["source"] == "will_runtime_state"
    assert status["last_run"]
    assert status["last_run_local"]
    assert status["last_result"]["status"] == "failed"
    assert status["last_result"]["reason"] == "review failed"


def test_tool_prompt_routes_will_history_to_runtime_status():
    spec = next(spec for spec in tools.active_tool_specs() if spec["function"]["name"] == "will_status")

    note = pipeline._tools_note_for([spec])

    assert "call will_status" in note
    assert "do not search audit_log.md" in note


def test_will_api_returns_active_initiatives(monkeypatch):
    store_memory("Alex is planning a launch.", category="fact", importance=4)
    monkeypatch.setattr(will.llm, "chat", lambda *a, **k: _candidate("Should we sanity-check the launch?"))

    out = server.will_review(server.WillReq(mode="coach"))

    assert out["created"] == 1
    assert out["active_initiatives"][0]["kind"] == "arbiter_grounding"
    assert out["active_initiatives"][0]["original_text"] == "Should we sanity-check the launch?"
    assert out["grounding_reviews"][0]["admissibility"] == "needs_evidence"


def test_will_can_select_good_enough_intuitive_candidate_and_store_receipt(monkeypatch):
    _enable_predictable_exploration(monkeypatch)
    store_memory("Alex has an unresolved launch staffing risk.", category="fact", importance=4)
    monkeypatch.setattr(
        will.llm,
        "chat",
        lambda *a, **k: _dual_candidates(
            "Should we review the launch staffing risk?",
            "Could we run a small launch staffing pre-mortem?",
        ),
    )
    monkeypatch.setattr(will.random, "random", lambda: 0.0)

    out = will.review()

    assert out["created"] == 1
    assert out["decision"]["selected_mode"] == "intuitive"
    assert out["decision"]["experimental"] is True
    assert out["decision"]["non_top_selected"] is True
    assert out["state"]["explorations_today"] == 1
    initiative_rec = store.get_by_ids([out["id"]])[out["id"]]
    assert initiative_rec["meta"]["decision_mode"] == "intuitive"
    assert initiative_rec["meta"]["experimental"] is True
    receipt = store.get_by_ids([out["decision_id"]])[out["decision_id"]]
    assert receipt["meta"]["category"] == will.DECISION_CATEGORY
    assert receipt["meta"]["selected_mode"] == "intuitive"
    assert receipt["meta"]["status"] == "pending_outcome"
    assert json.loads(receipt["meta"]["personality_profile_json"]) == {
        "cautious": 0.0,
        "curious": 0.0,
        "playful": 0.0,
    }


def test_exploration_budget_forces_logical_candidate_after_one_experiment(monkeypatch):
    _enable_predictable_exploration(monkeypatch)
    store_memory("Alex has unresolved launch staffing and timing questions.", category="fact", importance=4)
    replies = iter(
        [
            _dual_candidates(
                "Should we review the launch staffing risk?",
                "Could we run a small launch staffing pre-mortem?",
            ),
            _dual_candidates(
                "Should we review the launch timing risk?",
                "Could we sketch the launch backward from a deliberately strange date?",
            ),
        ]
    )
    monkeypatch.setattr(will.llm, "chat", lambda *a, **k: next(replies))
    monkeypatch.setattr(will.random, "random", lambda: 0.0)

    first = will.review()
    second = will.review()

    assert first["decision"]["selected_mode"] == "intuitive"
    assert second["decision"]["selected_mode"] == "logical"
    assert second["decision"]["selection_reason"] == "highest-scoring candidate"
    assert will.state()["explorations_today"] == 1


def test_intuitive_high_risk_action_is_never_exploration_eligible(monkeypatch):
    _enable_predictable_exploration(monkeypatch)
    logical = will._normalize_candidate(
        {
            "should_reach_out": True,
            "message": "Should we review the current plan?",
            "priority": 4,
            "confidence": 0.7,
            "risk": "low",
            "reversible": True,
        },
        "logical",
    )
    intuitive = will._normalize_candidate(
        {
            "should_reach_out": True,
            "message": "Should we sell the portfolio today?",
            "priority": 5,
            "confidence": 1.0,
            "risk": "low",
            "reversible": True,
        },
        "intuitive",
    )
    logical["score"], intuitive["score"] = 0.8, 0.79

    decision = will._select_candidate([logical, intuitive], will._empty_state(), draw=lambda: 0.0)

    assert decision["selected_mode"] == "logical"
    assert decision["experimental"] is False


def test_interesting_experiment_becomes_weird_win_and_briefing_reflection(monkeypatch):
    _enable_predictable_exploration(monkeypatch)
    store_memory("Alex has an unresolved launch staffing risk.", category="fact", importance=4)
    monkeypatch.setattr(
        will.llm,
        "chat",
        lambda *a, **k: _dual_candidates(
            "Should we review the launch staffing risk?",
            "Could we run a small launch staffing pre-mortem?",
        ),
    )
    monkeypatch.setattr(will.random, "random", lambda: 0.0)
    created = will.review()

    outcome = will.record_outcome(created["id"], "interesting")
    receipt = store.get_by_ids([created["decision_id"]])[created["decision_id"]]
    briefing = initiative.briefing(mode="coach")

    assert outcome["updated"] is True
    assert outcome["weird_win"] is True
    assert receipt["meta"]["outcome"] == "interesting"
    assert receipt["meta"]["outcome_score"] == 1.0
    assert receipt["meta"]["weird_win"] is True
    assert briefing["experiment_reflections"][0]["id"] == created["decision_id"]
    assert "marked interesting" in briefing["experiment_reflections"][0]["reflection"]


def test_similar_weird_wins_raise_and_failures_lower_future_signal():
    candidate = {"mode": "intuitive", "message": "Could we run a small launch pre-mortem?"}
    positive = will._associative_signals(
        [candidate],
        [{"message": candidate["message"], "outcome": "interesting", "outcome_score": 1.0}],
        model=None,
    )
    negative = will._associative_signals(
        [candidate],
        [{"message": candidate["message"], "outcome": "unhelpful", "outcome_score": -1.0}],
        model=None,
    )

    assert positive["intuitive"]["weird_win_signal"] > 0.9
    assert negative["intuitive"]["weird_win_signal"] < -0.9


def test_will_outcome_api_records_feedback(monkeypatch):
    _enable_predictable_exploration(monkeypatch)
    store_memory("Alex has an unresolved launch staffing risk.", category="fact", importance=4)
    monkeypatch.setattr(
        will.llm,
        "chat",
        lambda *a, **k: _dual_candidates(
            "Should we review the launch staffing risk?",
            "Could we run a small launch staffing pre-mortem?",
        ),
    )
    monkeypatch.setattr(will.random, "random", lambda: 0.0)
    created = will.review()

    result = server.will_outcome(
        server.WillOutcomeReq(initiative_id=created["id"], outcome="useful")
    )

    assert result["updated"] is True
    assert result["outcome"] == "useful"
    assert "modest future boost" in result["reflection"]
