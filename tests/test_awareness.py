from __future__ import annotations

from fastapi.testclient import TestClient

import server
from brain import awareness, pipeline


def test_awareness_updates_focus_and_open_loops():
    result = awareness.update_from_turn(
        "lets do it, analyze the portfolio again",
        "The current portfolio uses the July 3 share counts.",
        mems=[{"text": "portfolio note", "category": "note"}],
        thinking={"path": "standard", "tools_used": ["get_stock_quotes", "calculate"]},
        reminders=[{"text": "Review COIN exit framework"}],
        initiatives=[{"text": "Update Source of Truth with current holdings"}],
        emotion_state={"labels": ["curious"], "confidence": 0.2, "curiosity": 0.6},
    )

    assert result["enabled"] is True
    assert result["current_focus"] == "portfolio accuracy"
    assert "focused" in result["stance"]
    assert result["last_user_intent"] == "make a code change"
    assert result["last_tools"] == ["get_stock_quotes", "calculate"]
    assert result["open_loops"][0] == "Review COIN exit framework"
    assert "tool results" in result["next_best_action"]


def test_build_system_includes_awareness_state():
    awareness.update_from_turn(
        "How can this feel more aware?",
        "Add an awareness layer.",
        thinking={"path": "standard"},
    )

    system = pipeline.build_system([])

    assert "Awareness State JSON:" in system
    assert "awareness design" in system
    assert "Do not claim literal consciousness" in system


def test_awareness_api_endpoint():
    awareness.update_from_turn("Can you fix the code?", "Done.", thinking={"path": "standard"})
    client = TestClient(server.app, base_url="http://127.0.0.1")

    res = client.get("/api/awareness")

    assert res.status_code == 200
    assert res.json()["current_focus"] == "code repair"


# --- open-loop TTL: orphaned loops expire instead of recycling forever ----------

def test_orphaned_open_loop_expires_after_ttl_turns(monkeypatch):
    import config

    monkeypatch.setattr(config, "AWARENESS_LOOP_TTL_TURNS", 2)
    awareness.reset()

    state = awareness.update_from_turn(
        "anything", "ok", reminders=[{"text": "Call the CPA about tax-home rules"}]
    )
    assert "Call the CPA about tax-home rules" in state["open_loops"]
    assert state["open_loop_meta"]["Call the CPA about tax-home rules"] == 0

    # Reminder resolved (no live source): the loop is carried, aging each turn.
    state = awareness.update_from_turn("next", "ok")
    assert state["open_loop_meta"]["Call the CPA about tax-home rules"] == 1
    state = awareness.update_from_turn("next", "ok")
    assert state["open_loop_meta"]["Call the CPA about tax-home rules"] == 2

    # Past the TTL it retires instead of recycling at the top of the list.
    state = awareness.update_from_turn("next", "ok")
    assert "Call the CPA about tax-home rules" not in state["open_loops"]
    assert "Call the CPA about tax-home rules" not in state["open_loop_meta"]


def test_live_source_keeps_loop_confirmed_indefinitely(monkeypatch):
    import config

    monkeypatch.setattr(config, "AWARENESS_LOOP_TTL_TURNS", 1)
    awareness.reset()

    for _ in range(4):
        state = awareness.update_from_turn(
            "anything", "ok", reminders=[{"text": "Review COIN exit framework"}]
        )

    assert state["open_loops"][0] == "Review COIN exit framework"
    assert state["open_loop_meta"]["Review COIN exit framework"] == 0


def test_fresh_loops_lead_carried_orphans():
    awareness.reset()
    awareness.update_from_turn("anything", "ok", reminders=[{"text": "Old orphaned task"}])

    state = awareness.update_from_turn(
        "next", "ok", reminders=[{"text": "New live reminder"}]
    )

    assert state["open_loops"].index("New live reminder") < state["open_loops"].index("Old orphaned task")


# --- energy momentum: activity nudges the persona instead of snapping it --------

def test_energy_eases_toward_activity_instead_of_jumping():
    awareness.reset()

    state = awareness.update_from_turn(
        "busy turn",
        "ok",
        mems=[{"text": "note", "category": "note"}],
        thinking={"tools_used": ["calculate"]},
        reminders=[{"text": "Review COIN exit framework"}],
    )

    # Target for loops+tools+mems is 0.75; from the 0.5 baseline one turn moves
    # only a fraction of the way there (0.3 smoothing -> 0.575), not the full jump.
    assert 0.5 < state["energy"] < 0.75
    assert state["energy"] == 0.575


def test_energy_converges_to_target_over_sustained_turns():
    awareness.reset()

    for _ in range(20):
        state = awareness.update_from_turn(
            "busy turn",
            "ok",
            mems=[{"text": "note", "category": "note"}],
            thinking={"tools_used": ["calculate"]},
            reminders=[{"text": "Review COIN exit framework"}],
        )

    assert abs(state["energy"] - 0.75) < 0.01


def test_quiet_turns_hold_homeostatic_baseline():
    awareness.reset()

    for _ in range(30):
        state = awareness.update_from_turn("quiet turn", "ok")

    assert state["energy"] == 0.5


def test_energy_recovers_to_baseline_after_activity():
    awareness.reset()

    for _ in range(12):
        state = awareness.update_from_turn(
            "busy turn",
            "ok",
            mems=[{"text": "note", "category": "note"}],
            thinking={"tools_used": ["calculate"]},
        )
    assert state["energy"] > 0.62

    for _ in range(30):
        state = awareness.update_from_turn("quiet turn", "ok")

    assert abs(state["energy"] - 0.5) < 0.001


def test_energy_baseline_is_configurable_and_used_by_reset(monkeypatch):
    import config

    monkeypatch.setattr(config, "AWARENESS_ENERGY_BASELINE", 0.6)

    state = awareness.reset()

    assert state["energy"] == 0.6
    assert awareness._smoothed_energy(0.3, loops=[], tools=[], mems=[]) > 0.3


def test_energy_smoothing_one_restores_instant_behavior(monkeypatch):
    import config

    monkeypatch.setattr(config, "AWARENESS_ENERGY_SMOOTHING", 1.0)
    awareness.reset()

    state = awareness.update_from_turn(
        "busy turn",
        "ok",
        mems=[{"text": "note", "category": "note"}],
        thinking={"tools_used": ["calculate"]},
        reminders=[{"text": "Review COIN exit framework"}],
    )

    assert state["energy"] == 0.75
