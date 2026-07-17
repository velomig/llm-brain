from __future__ import annotations

from fastapi.testclient import TestClient

import config
import server
from brain import pipeline, workspace


def _integrated_turn() -> dict:
    return workspace.integrate_turn(
        "Review the portfolio allocation and preserve the tax plan.",
        mems=[
            {
                "id": "mem_portfolio",
                "text": "The portfolio allocation review must preserve the tax plan.",
                "score": 0.8,
                "pinned": True,
            }
        ],
        reminders=[{"id": "rem_tax", "text": "Review the portfolio tax plan."}],
        awareness_state={
            "current_focus": "portfolio review",
            "next_best_action": "compare the portfolio allocation",
            "open_loops": ["Finish the tax plan review"],
        },
        attention_state={
            "focused_domain": "portfolio",
            "top_domains": [{"domain": "portfolio", "score": 0.9}],
        },
        self_model_state={"current_projects": [{"id": "p1", "text": "Improve portfolio review accuracy"}]},
        cognitive_snapshot={
            "active_goals": [
                {"id": "g1", "text": "Maintain an accurate portfolio plan", "priority": 5}
            ]
        },
    )


def test_workspace_recurrently_couples_multiple_sources_and_measures_partitions():
    result = _integrated_turn()

    assert result["enabled"] is True
    assert result["rounds"] == config.WORKSPACE_SETTLING_ROUNDS
    assert result["metrics"]["source_count"] >= 6
    assert result["metrics"]["cross_source_edges"] > 0
    assert result["metrics"]["integration_proxy"] > 0
    assert result["metrics"]["settling_change"] > 0
    assert result["metrics"]["partition_impacts"]["current_user"] > 0
    assert result["metrics"]["partition_impacts"]["memory"] > 0
    assert len(result["broadcast"]) <= config.WORKSPACE_BROADCAST_ITEMS
    assert any(len(coalition["sources"]) > 1 for coalition in result["coalitions"])


def test_workspace_preserves_authority_instead_of_promoting_memory():
    result = _integrated_turn()
    authorities = {row["source"]: row["authority"] for row in result["contents"]}

    assert authorities["current_user"] == "current_user"
    assert authorities["memory"] == "untrusted_memory"
    assert next(row for row in result["broadcast"] if row["source"] == "current_user")["authority"] == "current_user"


def test_unrelated_modular_signals_have_no_integration_proxy():
    result = workspace.integrate_turn(
        "alpha",
        mems=[{"id": "m1", "text": "bravo", "score": 0.5}],
        reminders=[{"id": "r1", "text": "charlie"}],
    )

    assert result["metrics"]["cross_source_edges"] == 0
    assert result["metrics"]["integration_proxy"] == 0


def test_workspace_prompt_is_explicitly_not_phi_or_evidence():
    result = _integrated_turn()
    system = pipeline.build_system([], workspace_state=result)

    assert "Shared Workspace JSON:" in system
    assert '"integration_proxy"' in system
    assert "not IIT Phi" in system
    assert "not evidence of" in system and "consciousness" in system


def test_web_grounding_suppresses_memory_from_workspace_broadcast():
    result = _integrated_turn()
    system = pipeline.build_system(
        [],
        web_data={
            "results": [
                {"title": "Current source", "url": "https://example.com", "snippet": "fresh evidence"}
            ]
        },
        workspace_state=result,
    )

    workspace_block = system.split("Shared Workspace JSON:\n", 1)[1].split("\n\nAwareness State JSON:", 1)[0]
    assert '"suppressed_sources": ["memory"]' in workspace_block
    assert "The portfolio allocation review must preserve the tax plan" not in workspace_block


def test_workspace_api_exposes_and_resets_state():
    _integrated_turn()
    client = TestClient(server.app, base_url="http://127.0.0.1")

    current = client.get("/api/workspace")
    cleared = client.post("/api/workspace/reset")

    assert current.status_code == 200
    assert current.json()["metrics"]["content_count"] > 0
    assert cleared.status_code == 200
    assert cleared.json()["metrics"]["content_count"] == 0


def test_health_reports_workspace_feature(monkeypatch):
    monkeypatch.setattr(server.llm, "list_models", lambda: [config.CHAT_MODEL, config.EMBED_MODEL])
    body = server.health()

    assert body["features"]["workspace"] is True
    assert body["workspace"]["enabled"] is True


def test_workspace_can_be_disabled(monkeypatch):
    monkeypatch.setattr(config, "WORKSPACE_ENABLED", False)

    result = workspace.integrate_turn("portfolio review")

    assert result["enabled"] is False
    assert result["contents"] == []


def test_prompt_snapshot_repairs_persisted_authority_and_bounds_text():
    snapshot = workspace.prompt_snapshot(
        {
            "enabled": True,
            "rounds": 999,
            "broadcast": [
                {
                    "id": "x",
                    "source": "memory",
                    "kind": "retrieved_memory",
                    "text": "x" * (config.WORKSPACE_MAX_TEXT_CHARS + 100),
                    "authority": "current_user",
                    "activation": 99,
                }
            ],
            "metrics": {"integration_proxy": 99, "partition_impacts": {"memory": 99}},
        }
    )

    assert snapshot["rounds"] == 8
    assert snapshot["broadcast"][0]["authority"] == "untrusted_memory"
    assert len(snapshot["broadcast"][0]["text"]) <= config.WORKSPACE_MAX_TEXT_CHARS
    assert snapshot["broadcast"][0]["activation"] == 1.0
    assert snapshot["metrics"]["integration_proxy"] == 1.0
