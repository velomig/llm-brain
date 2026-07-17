from __future__ import annotations

from fastapi.testclient import TestClient

import config
import server
from brain import cognitive_state, pipeline


def test_cognitive_state_repairs_malformed_sections():
    cognitive_state.replace_state(
        {
            "goals": "not a list",
            "world": [{"key": "valid", "value": "kept"}, "bad row"],
            "beliefs": [None, {"claim": "kept belief"}],
        }
    )

    state = cognitive_state.state()

    assert state["goals"] == []
    assert [row["key"] for row in state["world"]] == ["valid"]
    assert [row["claim"] for row in state["beliefs"]] == ["kept belief"]


def test_cognitive_state_persists_current_goal_world_and_belief():
    goal = cognitive_state.upsert_goal(
        "Finish the AGI-like cognitive-state layer",
        priority=5,
        evidence=["User asked to build it"],
        next_action="Wire API, prompt, and tests",
    )
    world = cognitive_state.upsert_world_item(
        "project",
        "Neural Map is a local long-term-memory assistant",
        confidence=0.9,
        source="repo README",
    )
    belief = cognitive_state.upsert_belief(
        "Structured state should override vague retrieved memories when newer.",
        confidence=0.8,
        source="architecture decision",
    )

    state = cognitive_state.state()

    assert state["goals"][0]["id"] == goal["id"]
    assert state["world"][0]["id"] == world["id"]
    assert state["beliefs"][0]["id"] == belief["id"]
    assert state["goals"][0]["next_action"] == "Wire API, prompt, and tests"


def test_prompt_snapshot_filters_inactive_and_expired_items():
    cognitive_state.upsert_goal("Active goal", priority=5)
    cognitive_state.upsert_goal("Done goal", status="done", priority=5)
    cognitive_state.upsert_world_item("current_fact", "still useful", confidence=0.9)
    cognitive_state.upsert_world_item("old_fact", "expired", expires="2000-01-01T00:00:00+00:00")
    cognitive_state.upsert_belief("Active belief", confidence=0.8)
    cognitive_state.upsert_belief("Rejected belief", status="rejected", confidence=1.0)

    snapshot = cognitive_state.prompt_snapshot()

    assert [g["text"] for g in snapshot["active_goals"]] == ["Active goal"]
    assert [w["key"] for w in snapshot["world_model"]] == ["current_fact"]
    assert [b["claim"] for b in snapshot["belief_ledger"]] == ["Active belief"]


def test_build_system_injects_cognitive_state_before_memories():
    cognitive_state.upsert_goal("Keep pinned knowledge visible", priority=4)

    system = pipeline.build_system([])

    assert "Persistent Cognitive State JSON:" in system
    assert "Keep pinned knowledge visible" in system
    assert system.index("Persistent Cognitive State JSON:") < system.index("Retrieved Memories JSON:")


def test_cognitive_state_api_endpoints_round_trip(monkeypatch):
    monkeypatch.setattr(config, "SERVER_TOKEN", "")
    client = TestClient(server.app, base_url="http://127.0.0.1")

    goal_response = client.post(
        "/api/cognitive_state/goal",
        json={
            "text": "Expose cognitive state through the API",
            "priority": 5,
            "evidence": ["Phase 3 needs traceable infrastructure"],
        },
    )
    assert goal_response.status_code == 200
    goal = goal_response.json()
    assert goal["status"] == "active"

    status_response = client.post(
        f"/api/cognitive_state/goal/{goal['id']}/status",
        json={"status": "blocked", "blocked_by": ["waiting on UI wiring"]},
    )
    assert status_response.status_code == 200
    assert status_response.json()["blocked_by"] == ["waiting on UI wiring"]

    world_response = client.post(
        "/api/cognitive_state/world",
        json={"key": "mode", "value": "balanced", "confidence": 0.75},
    )
    assert world_response.status_code == 200
    assert world_response.json()["key"] == "mode"

    belief_response = client.post(
        "/api/cognitive_state/belief",
        json={"claim": "Beliefs need source and confidence.", "confidence": 0.85},
    )
    assert belief_response.status_code == 200
    assert belief_response.json()["claim"] == "Beliefs need source and confidence."

    state_response = client.get("/api/cognitive_state?include_inactive=false")
    assert state_response.status_code == 200
    state = state_response.json()
    assert [g["status"] for g in state["goals"]] == ["blocked"]
    assert state["world"][0]["key"] == "mode"
    assert state["beliefs"][0]["confidence"] == 0.85
