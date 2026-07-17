from __future__ import annotations

from fastapi.testclient import TestClient

import config
import server
from brain import cognitive_state, homeostasis, pipeline

_HIGH_LOAD_ATTENTION = {"enabled": True, "domains": {"nursing": 0.9, "benefits": 0.4}}


def _baseline(name: str) -> float:
    return homeostasis.VARIABLES[name]["baseline"]


def _drive_load(turns: int = 8, hours: str = "40 hours/week") -> dict:
    cognitive_state.upsert_world_item(
        key="clinical_workload",
        value=f"clinical rotation, {hours}",
        confidence=0.9,
        source="test",
    )
    state = {}
    for _ in range(turns):
        state = homeostasis.update_from_turn(attention_state=_HIGH_LOAD_ATTENTION)
    return state


# --- coupling: a workload change re-weights the behavior dials ------------------

def test_sustained_clinical_load_caps_portfolio_intensity():
    state = _drive_load()

    assert state["enabled"] is True
    assert state["variables"]["user_load"] > 0.5
    assert state["variables"]["portfolio_intensity"] < _baseline("portfolio_intensity")
    assert "load_caps_portfolio_intensity" in state["active_rules"]
    fired = [a for a in state["adjustments"] if a.get("event") == "fired"]
    assert any(a["rule"] == "load_caps_portfolio_intensity" for a in fired)
    assert state["posture_levels"]["portfolio_intensity"] in {"balanced", "conservative"}


def test_single_mention_does_not_fire_a_rule():
    state = homeostasis.update_from_turn(attention_state=_HIGH_LOAD_ATTENTION)

    assert state["active_rules"] == []
    assert state["variables"]["portfolio_intensity"] == _baseline("portfolio_intensity")


def test_attention_salience_alone_cannot_fire_rules():
    # No world-model evidence: attention is scaled below every rule threshold.
    state = {}
    for _ in range(12):
        state = homeostasis.update_from_turn(attention_state=_HIGH_LOAD_ATTENTION, world_items=[])

    assert state["active_rules"] == []


def test_rules_release_with_hysteresis_when_load_drops():
    _drive_load()
    cognitive_state.upsert_world_item(
        key="clinical_workload",
        value="on break, 4 hours/week",
        confidence=0.9,
        source="test",
    )
    state = {}
    for _ in range(10):
        state = homeostasis.update_from_turn(
            attention_state={"enabled": True, "domains": {"nursing": 0.05}}
        )

    assert state["active_rules"] == []
    released = [a for a in state["adjustments"] if a.get("event") == "released"]
    assert any(a["rule"] == "load_caps_portfolio_intensity" for a in released)
    assert state["variables"]["portfolio_intensity"] > 0.5  # recovering toward baseline


def test_adjustments_are_attributable():
    state = _drive_load()

    fired = next(a for a in state["adjustments"] if a.get("event") == "fired")
    assert fired["source"] == "user_load"
    assert fired["target"] in homeostasis.DERIVED
    assert 0.0 <= fired["source_value"] <= 1.0
    assert fired["target_after"] <= fired["target_before"]


# --- governance: rules only change through an explicit call ---------------------

def test_set_rules_validates_and_replaces():
    updated = homeostasis.set_rules(
        [
            {
                "id": "custom_rule",
                "source": "user_load",
                "target": "proactivity",
                "weight": -1.0,
                "threshold": 0.3,
                "release": 0.2,
                "note": "custom",
            }
        ]
    )

    assert [r["id"] for r in homeostasis.effective_rules()] == ["custom_rule"]
    assert any(a.get("event") == "rules_replaced" for a in updated["adjustments"])

    restored = homeostasis.set_rules(None)
    assert [r["id"] for r in homeostasis.effective_rules()] == [r["id"] for r in homeostasis.DEFAULT_RULES]
    assert sum(1 for a in restored["adjustments"] if a.get("event") == "rules_replaced") == 2


def test_set_rules_rejects_unknown_variables_and_bad_hysteresis():
    for bad in (
        {"id": "x", "source": "portfolio_intensity", "target": "proactivity", "weight": -1.0},
        {"id": "x", "source": "user_load", "target": "user_load", "weight": -1.0},
        {"id": "x", "source": "user_load", "target": "proactivity", "weight": -1.0, "threshold": 0.3, "release": 0.5},
    ):
        try:
            homeostasis.set_rules([bad])
        except ValueError:
            continue
        raise AssertionError(f"rule should have been rejected: {bad}")


# --- gating: conservative proactivity trims low-priority suggestions ------------

def test_gate_initiatives_only_when_conservative():
    rows = [
        {"id": "low", "text": "nice-to-have", "priority": 3},
        {"id": "high", "text": "urgent", "priority": 5},
        {"id": "arbiter:abc", "text": "resolve conflict", "priority": 3},
    ]

    kept, dropped = homeostasis.gate_initiatives(rows)
    assert dropped == 0 and len(kept) == 3  # baseline posture is not conservative

    _drive_load(turns=12)
    kept, dropped = homeostasis.gate_initiatives(rows)
    assert {r["id"] for r in kept} == {"high", "arbiter:abc"}
    assert dropped == 1


# --- prompt + API surface --------------------------------------------------------

def test_build_system_includes_homeostasis_posture():
    _drive_load()

    system = pipeline.build_system([])

    assert "Homeostasis Posture JSON:" in system
    assert "Derived operating posture" in system


def test_homeostasis_api_endpoints():
    client = TestClient(server.app, base_url="http://127.0.0.1")

    res = client.get("/api/homeostasis")
    assert res.status_code == 200
    assert res.json()["enabled"] is True

    res = client.get("/api/homeostasis/rules")
    assert res.status_code == 200
    assert [r["id"] for r in res.json()["rules"]] == [r["id"] for r in homeostasis.DEFAULT_RULES]

    res = client.post(
        "/api/homeostasis/rules",
        json={"rules": [{"id": "x", "source": "nope", "target": "proactivity", "weight": -1.0}]},
    )
    assert res.status_code == 422

    res = client.post("/api/homeostasis/reset")
    assert res.status_code == 200
    assert res.json()["active_rules"] == []


def test_world_upsert_propagates_through_coupling():
    client = TestClient(server.app, base_url="http://127.0.0.1")

    for _ in range(8):
        res = client.post(
            "/api/cognitive_state/world",
            json={"key": "clinical_workload", "value": "40 hours/week rotation", "confidence": 0.9},
        )
        assert res.status_code == 200

    state = homeostasis.state()
    assert state["variables"]["user_load"] > 0.5
    assert "load_caps_portfolio_intensity" in state["active_rules"]


def test_health_reports_homeostasis(monkeypatch):
    monkeypatch.setattr(server.llm, "list_models", lambda: [config.CHAT_MODEL, config.EMBED_MODEL])

    out = server.health()

    assert out["features"]["homeostasis"] == config.HOMEOSTASIS_ENABLED
    assert out["homeostasis"]["enabled"] is True
