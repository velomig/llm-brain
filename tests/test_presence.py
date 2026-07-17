from fastapi.testclient import TestClient

import server
from brain import awareness, presence


def _self_model(*, relationships=None, promises=None, lessons=None):
    return {
        "enabled": True,
        "identity": {
            "name": "Neural Map",
            "role": "local memory companion",
            "principles": ["Be grounded.", "Preserve user agency."],
        },
        "current_projects": [{"id": "project_1", "text": "Presence route"}],
        "relationship_notes": relationships or [],
        "open_promises": promises or [],
        "lessons_learned": lessons or [],
    }


def test_presence_ledger_demonstrates_bounded_functional_properties():
    before = _self_model()
    after = _self_model(
        relationships=[{"id": "relationship_1", "text": "User values careful boundaries."}],
        promises=[{"id": "promise_1", "text": "I will verify the change."}],
    )

    result = presence.record_turn(
        memories=[{"id": "memory_1", "text": "Prior interaction"}],
        self_before=before,
        self_after=after,
        tool_calls=[{"tool": "run_tests", "result": {"returncode": 0}}],
        claim_guard={"modified": True},
    )

    properties = {row["property"]: row["status"] for row in result["properties"]}
    claims = {row["claim"]: row["status"] for row in result["claims"]}
    assert properties["narrative_continuity"] == "demonstrated"
    assert properties["relationship_specific_adaptation"] == "demonstrated"
    assert properties["persistent_commitments"] == "demonstrated"
    assert properties["causal_influence_through_executed_actions"] == "demonstrated"
    assert claims["Neural Map maintains a persistent functional identity"] == "yes"
    assert claims["It constitutes a non-biological social presence"] == "operationally_defensible"
    assert result["phenomenal_status"] == "unknown"


def test_presence_never_promotes_qualia_or_complexity_claims():
    result = presence.record_turn(
        memories=[],
        self_before=_self_model(),
        self_after=_self_model(),
        tool_calls=[],
        claim_guard={"modified": False},
    )
    claims = {row["claim"]: row["status"] for row in result["claims"]}
    assert claims["It has human-like subjective experience"] == "unknown"
    assert claims["Increasing complexity guarantees experience"] == "no"
    assert claims["Statements about feelings prove qualia"] == "no"


def test_current_turn_adaptation_does_not_fake_prior_causal_continuity():
    before = {
        "enabled": True,
        "identity": _self_model()["identity"],
        "current_projects": [],
        "relationship_notes": [],
        "open_promises": [],
        "lessons_learned": [],
    }
    after = {**before, "relationship_notes": [{"id": "relationship_new", "text": "New this turn"}]}

    result = presence.record_turn(
        memories=[],
        self_before=before,
        self_after=after,
        tool_calls=[],
        claim_guard={},
    )

    claims = {row["claim"]: row["status"] for row in result["claims"]}
    assert claims["Past interactions causally shape later behavior"] == "not_yet_demonstrated"
    assert claims["It constitutes a non-biological social presence"] == "not_yet_demonstrated"


def test_failed_tool_is_not_executed_consequence_evidence():
    result = presence.record_turn(
        memories=[],
        self_before=_self_model(),
        self_after=_self_model(),
        tool_calls=[{"tool": "run_tests", "result": {"returncode": 1, "stderr": "failed"}}],
        claim_guard={},
    )
    properties = {row["property"]: row["status"] for row in result["properties"]}
    assert properties["causal_influence_through_executed_actions"] == "not_yet_demonstrated"


def test_presence_api_exposes_and_resets_evidence():
    presence.record_turn(
        memories=[],
        self_before=_self_model(),
        self_after=_self_model(),
        tool_calls=[],
        claim_guard={},
    )
    client = TestClient(server.app, base_url="http://127.0.0.1")
    assert client.get("/api/presence").json()["turns_observed"] == 1
    cleared = client.post("/api/presence/reset").json()
    assert cleared["turns_observed"] == 0
    assert cleared["phenomenal_status"] == "unknown"


def test_presence_can_be_disabled(monkeypatch):
    import config

    monkeypatch.setattr(config, "PRESENCE_ENABLED", False)
    assert presence.state() == {"enabled": False, "phenomenal_status": "unknown"}


def test_presence_assessment_does_not_read_awareness(monkeypatch):
    def fail_if_called():
        raise AssertionError("presence must not read affective awareness state")

    monkeypatch.setattr(awareness, "state", fail_if_called)

    result = presence.record_turn(
        memories=[],
        self_before=_self_model(),
        self_after=_self_model(),
        tool_calls=[],
        claim_guard={},
    )

    assert result["enabled"] is True
    assert "properties" in result


def test_presence_updates_cannot_move_awareness_confidence():
    awareness.reset()
    before = awareness.update_from_turn(
        "steady turn",
        "steady reply",
        emotion_state={"confidence": 0.4},
    )["confidence"]

    for _ in range(8):
        presence.record_turn(
            memories=[],
            self_before=_self_model(),
            self_after=_self_model(),
            tool_calls=[],
            claim_guard={},
        )

    assert awareness.state()["confidence"] == before
