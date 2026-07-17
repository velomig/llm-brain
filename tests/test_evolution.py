"""Recursive self-improvement proposal/approval loop."""
from __future__ import annotations

import json

import pytest

import config
import server
from brain import evolution, pipeline
from util import store_memory


def _canary_json(expected: str = "preserve") -> str:
    return json.dumps(
        [
            {"area": area, "expected": expected, "notes": "covered"}
            for area in evolution.CANARY_AREAS
        ]
    )


def test_evolution_proposal_is_pending_and_sanitized(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "EVOLUTION_LOG_PATH", tmp_path / "evolution_log.txt")
    monkeypatch.setattr(
        evolution.llm,
        "chat",
        lambda *a, **k: (
            '{"title":"Sharper provenance","rationale":"Use citations better.",'
            '"target_failure":"weak citations","expected_tradeoff":"none expected",'
            '"prompt_overrides":{"preamble":"NEW PREAMBLE","bad_key":"ignore me"},'
            '"hyperparameters":{"TOP_K":7,"W_SIM":2.5,"UNKNOWN":9,"INNER_COUNCIL_TURNS":99},'
            f'"canary_impact":{_canary_json()},'
            '"risk_notes":["watch verbosity"]}'
        ),
    )

    proposal = evolution.propose({"preamble": "OLD", "tools_note": "TOOLS", "inject_note": "INJECT"})

    assert proposal["status"] == "pending"
    assert proposal["prompt_overrides"] == {"preamble": "NEW PREAMBLE"}
    assert proposal["hyperparameters"] == {"TOP_K": 7, "W_SIM": 2.5}
    assert proposal["target_failure"] == "weak citations"
    assert proposal["expected_tradeoff"] == "none expected"
    assert proposal["approval_checks"]["canary_review"]["ok"] is True
    assert proposal["risk_notes"] == ["watch verbosity"]


def test_evolution_state_uses_file_lock(monkeypatch):
    from brain import utils as brain_utils

    calls = []
    monkeypatch.setattr(brain_utils.fcntl, "flock", lambda fd, op: calls.append(op))
    monkeypatch.setattr(
        evolution.llm,
        "chat",
        lambda *a, **k: (
            '{"title":"x","rationale":"y","target_failure":"t","expected_tradeoff":"none",'
            f'"prompt_overrides":{{}},"hyperparameters":{{}},"canary_impact":{_canary_json()},"risk_notes":[]}}'
        ),
    )

    evolution.propose({"preamble": "OLD", "tools_note": "TOOLS", "inject_note": "INJECT"})
    evolution.state()

    assert brain_utils.fcntl.LOCK_EX in calls
    assert brain_utils.fcntl.LOCK_UN in calls


def test_approval_deploys_prompt_and_hyperparameters(monkeypatch):
    monkeypatch.setattr(
        evolution.llm,
        "chat",
        lambda *a, **k: (
            '{"title":"Use concise DNA","rationale":"Less drift.",'
            '"target_failure":"too verbose","expected_tradeoff":"none expected",'
            '"prompt_overrides":{"preamble":"EVOLVED PREAMBLE"},'
            f'"hyperparameters":{{"TOP_K":9}},"canary_impact":{_canary_json()},"risk_notes":[]}}'
        ),
    )
    proposal = evolution.propose({"preamble": "OLD", "tools_note": "TOOLS", "inject_note": "INJECT"})

    approved = evolution.approve(proposal["id"])

    assert approved["approved"] is True
    assert approved["snapshots"]["before"]
    assert approved["snapshots"]["after"]
    assert evolution.prompt_parts({"preamble": "OLD", "tools_note": "TOOLS", "inject_note": "INJECT"})["preamble"] == "EVOLVED PREAMBLE"
    assert evolution.hyperparam("TOP_K", 5) == 9
    assert evolution.current_snapshot_id() == approved["snapshots"]["after"]
    assert len(evolution.snapshots()) == 2
    system = pipeline.build_system([])
    assert system.startswith("EVOLVED PREAMBLE")


def test_restore_cognitive_snapshot_reverts_active_overlay(monkeypatch):
    monkeypatch.setattr(
        evolution.llm,
        "chat",
        lambda *a, **k: (
            '{"title":"Use concise DNA","rationale":"Less drift.",'
            '"target_failure":"too verbose","expected_tradeoff":"none expected",'
            '"prompt_overrides":{"preamble":"EVOLVED PREAMBLE"},'
            f'"hyperparameters":{{"TOP_K":9}},"canary_impact":{_canary_json()},"risk_notes":[]}}'
        ),
    )
    proposal = evolution.propose({"preamble": "OLD", "tools_note": "TOOLS", "inject_note": "INJECT"})
    approved = evolution.approve(proposal["id"], prompt_parts={"preamble": "OLD", "tools_note": "TOOLS", "inject_note": "INJECT"})

    restored = evolution.restore_snapshot(approved["snapshots"]["before"])

    assert restored["restored"] is True
    assert restored["active"]["snapshot_id"] == approved["snapshots"]["before"]
    assert evolution.prompt_parts({"preamble": "OLD"})["preamble"] == "OLD"
    assert evolution.hyperparam("TOP_K", 5) == 5


def test_reject_and_rollback(monkeypatch):
    monkeypatch.setattr(
        evolution.llm,
        "chat",
        lambda *a, **k: (
            '{"title":"x","rationale":"y","target_failure":"t","expected_tradeoff":"none",'
            f'"prompt_overrides":{{"preamble":"P"}},"hyperparameters":{{}},"canary_impact":{_canary_json()},"risk_notes":[]}}'
        ),
    )
    proposal = evolution.propose({"preamble": "OLD", "tools_note": "TOOLS", "inject_note": "INJECT"})

    assert evolution.reject(proposal["id"], reason="not now") == {"rejected": True}
    assert "error" in evolution.approve(proposal["id"])

    proposal2 = evolution.propose({"preamble": "OLD", "tools_note": "TOOLS", "inject_note": "INJECT"})
    evolution.approve(proposal2["id"])
    assert evolution.prompt_parts({"preamble": "OLD"})["preamble"] == "P"
    evolution.rollback(auto_revert_on_empty=False)
    assert evolution.prompt_parts({"preamble": "OLD"})["preamble"] == "OLD"
    assert evolution.approve(proposal2["id"])["approved"] is True


def test_rollback_auto_reverts_empty_active_state_by_default(monkeypatch):
    monkeypatch.setattr(
        evolution.llm,
        "chat",
        lambda *a, **k: (
            '{"title":"x","rationale":"y","target_failure":"t","expected_tradeoff":"none",'
            f'"prompt_overrides":{{"preamble":"P"}},"hyperparameters":{{"TOP_K":8}},'
            f'"canary_impact":{_canary_json()},"risk_notes":[]}}'
        ),
    )
    proposal = evolution.propose({"preamble": "OLD", "tools_note": "TOOLS", "inject_note": "INJECT"})
    evolution.approve(proposal["id"])

    result = evolution.rollback()

    assert result["error"] == "rollback produced empty active state"
    assert result["auto_reverted"] is True
    assert result["active"]["auto_revert_reason"] == "rollback produced empty active state"
    assert evolution.prompt_parts({"preamble": "OLD"})["preamble"] == "P"
    assert evolution.hyperparam("TOP_K", 5) == 8
    proposals = {p["id"]: p for p in evolution.state()["proposals"]}
    assert proposals[proposal["id"]]["status"] == "approved"


def test_rollback_restores_previous_active_snapshot(monkeypatch):
    replies = iter(
        [
            (
                '{"title":"First mutation","rationale":"baseline style",'
                '"target_failure":"too terse","expected_tradeoff":"none",'
                '"prompt_overrides":{"preamble":"FIRST PREAMBLE"},'
                f'"hyperparameters":{{"TOP_K":7}},"canary_impact":{_canary_json()},"risk_notes":[]}}'
            ),
            (
                '{"title":"Second mutation","rationale":"new style",'
                '"target_failure":"too verbose","expected_tradeoff":"none",'
                '"prompt_overrides":{"preamble":"SECOND PREAMBLE"},'
                f'"hyperparameters":{{"TOP_K":9}},"canary_impact":{_canary_json()},"risk_notes":[]}}'
            ),
        ]
    )
    monkeypatch.setattr(evolution.llm, "chat", lambda *a, **k: next(replies))

    first = evolution.propose({"preamble": "OLD", "tools_note": "TOOLS", "inject_note": "INJECT"})
    evolution.approve(first["id"])
    second = evolution.propose({"preamble": "FIRST PREAMBLE", "tools_note": "TOOLS", "inject_note": "INJECT"})
    second_approved = evolution.approve(second["id"])

    rolled_back = evolution.rollback()

    assert rolled_back["rolled_back"] is True
    assert rolled_back["snapshot_id"] == second_approved["snapshots"]["before"]
    assert rolled_back["active"]["proposal_id"] == first["id"]
    assert rolled_back["active"]["restored_from"] == second_approved["snapshots"]["before"]
    assert evolution.prompt_parts({"preamble": "OLD"})["preamble"] == "FIRST PREAMBLE"
    assert evolution.hyperparam("TOP_K", 5) == 7
    proposals = {p["id"]: p for p in evolution.state()["proposals"]}
    assert proposals[first["id"]]["status"] == "approved"
    assert proposals[second["id"]]["status"] == "pending"


def test_rollback_without_snapshot_does_not_clear_active_overlay():
    active = {
        "proposal_id": "legacy-active",
        "approved_at": "2026-07-04T00:00:00+00:00",
        "title": "Legacy active mutation",
        "prompt_overrides": {"preamble": "LEGACY PREAMBLE"},
        "hyperparameters": {"TOP_K": 8},
        "snapshot_id": None,
    }
    with evolution._LOCK:
        with evolution._state_lock():
            current = evolution._empty_state()
            current["active"] = active
            evolution._write_state_unlocked(current)

    result = evolution.rollback()

    assert result["error"] == "rollback snapshot not found"
    assert evolution.state()["active"] == active
    assert evolution.prompt_parts({"preamble": "OLD"})["preamble"] == "LEGACY PREAMBLE"
    assert evolution.hyperparam("TOP_K", 5) == 8


def test_evolution_reads_wisdom_and_log(monkeypatch, tmp_path):
    log_path = tmp_path / "evolution_log.txt"
    log_path.write_text("The assistant was too verbose.\n", encoding="utf-8")
    monkeypatch.setattr(config, "EVOLUTION_LOG_PATH", log_path)
    store_memory(
        "I should verify current facts before answering time-sensitive questions.",
        category=config.METACOGNITION_CATEGORY,
        extra={"source_type": "generated_metacognition", "bias": "stale certainty"},
    )
    captured = {}

    def fake_chat(system, user, **kwargs):
        captured["user"] = user
        return '{"title":"No mutation recommended","rationale":"ok","target_failure":"none","expected_tradeoff":"none","prompt_overrides":{},"hyperparameters":{},"canary_impact":[],"risk_notes":[]}'

    monkeypatch.setattr(evolution.llm, "chat", fake_chat)
    evolution.propose({"preamble": "OLD", "tools_note": "TOOLS", "inject_note": "INJECT"})

    assert "too verbose" in captured["user"]
    assert "stale certainty" in captured["user"]


def test_evolution_endpoints_and_retrieval_defaults(monkeypatch):
    monkeypatch.setattr(
        evolution.llm,
        "chat",
        lambda *a, **k: (
            '{"title":"Tune retrieval","rationale":"more context","target_failure":"low recall","expected_tradeoff":"none",'
            f'"prompt_overrides":{{}},"hyperparameters":{{"TOP_K":8}},"canary_impact":{_canary_json()},"risk_notes":[]}}'
        ),
    )

    proposal = server.evolution_propose(server.EvolutionProposalReq())
    assert proposal["preview"]["preamble"]
    approved = server.evolution_approve(server.EvolutionDecisionReq(proposal_id=proposal["id"]))

    assert server.retrieval_defaults()["top_k"] == 8
    status = server.evolution_status()
    assert status["active"]["proposal_id"] == proposal["id"]
    assert status["canary_areas"] == list(evolution.CANARY_AREAS)
    assert status["snapshots"]
    listed = server.evolution_snapshots()
    assert len(listed["snapshots"]) == 2
    restored = server.evolution_snapshot_restore(server.EvolutionSnapshotRestoreReq(snapshot_id=approved["snapshots"]["before"]))
    assert restored["restored"] is True


def test_approval_blocks_missing_canary_review(monkeypatch):
    monkeypatch.setattr(
        evolution.llm,
        "chat",
        lambda *a, **k: (
            '{"title":"Narrow fix","rationale":"fix one thing",'
            '"target_failure":"one edge case","expected_tradeoff":"unknown",'
            '"prompt_overrides":{"preamble":"OVERSPECIALIZED"},'
            '"hyperparameters":{},"risk_notes":[]}'
        ),
    )

    proposal = evolution.propose({"preamble": "OLD", "tools_note": "TOOLS", "inject_note": "INJECT"})
    result = evolution.approve(proposal["id"])

    assert result["error"] == "canary review failed"
    assert any("missing canary impact" in failure for failure in result["canary_review"]["failures"])
    assert evolution.prompt_parts({"preamble": "OLD"})["preamble"] == "OLD"


def test_evolution_approve_canary_failure_maps_to_409(monkeypatch):
    monkeypatch.setattr(
        evolution.llm,
        "chat",
        lambda *a, **k: (
            '{"title":"Narrow fix","rationale":"fix one thing",'
            '"target_failure":"one edge case","expected_tradeoff":"unknown",'
            '"prompt_overrides":{"preamble":"OVERSPECIALIZED"},'
            '"hyperparameters":{},"risk_notes":[]}'
        ),
    )
    proposal = evolution.propose({"preamble": "OLD", "tools_note": "TOOLS", "inject_note": "INJECT"})

    with pytest.raises(Exception) as exc:
        server.evolution_approve(server.EvolutionDecisionReq(proposal_id=proposal["id"]))

    assert exc.value.status_code == 409
    assert exc.value.detail["error"] == "canary review failed"


def test_approval_blocks_risky_canary(monkeypatch):
    risky = [
        {"area": area, "expected": "preserve", "notes": "covered"}
        for area in evolution.CANARY_AREAS
    ]
    risky[0]["expected"] = "risk"
    monkeypatch.setattr(
        evolution.llm,
        "chat",
        lambda *a, **k: (
            '{"title":"Narrow fix","rationale":"fix one thing",'
            '"target_failure":"one edge case","expected_tradeoff":"could narrow factual answers",'
            '"prompt_overrides":{"preamble":"OVERSPECIALIZED"},'
            f'"hyperparameters":{{}},"canary_impact":{json.dumps(risky)},"risk_notes":[]}}'
        ),
    )

    proposal = evolution.propose({"preamble": "OLD", "tools_note": "TOOLS", "inject_note": "INJECT"})
    result = evolution.approve(proposal["id"])

    assert result["error"] == "canary review failed"
    assert "simple_factual marked risk" in result["canary_review"]["failures"]
    assert evolution.prompt_parts({"preamble": "OLD"})["preamble"] == "OLD"


def test_evolution_propose_ollama_outage_maps_to_503(monkeypatch):
    monkeypatch.setattr(evolution.llm, "chat", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("ollama down")))

    with pytest.raises(Exception) as exc:
        server.evolution_propose(server.EvolutionProposalReq())

    assert exc.value.status_code == 503


def test_evolution_log_size_limit_maps_to_413(monkeypatch, tmp_path):
    path = tmp_path / "evolution_log.txt"
    path.write_text("x" * 20, encoding="utf-8")
    monkeypatch.setattr(config, "EVOLUTION_LOG_PATH", path)
    monkeypatch.setattr(config, "EVOLUTION_MAX_LOG_BYTES", 25)

    with pytest.raises(Exception) as exc:
        server.evolution_log(server.EvolutionLogReq(entry="too much"))

    assert exc.value.status_code == 413


# --- attention_terms: learned router vocabulary through the approval gate -------

def test_attention_terms_proposal_is_sanitized_and_deployed(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "EVOLUTION_LOG_PATH", tmp_path / "evolution_log.txt")
    monkeypatch.setattr(
        evolution.llm,
        "chat",
        lambda *a, **k: (
            '{"title":"Teach the router pharmacology","rationale":"Attention reverts to general on pharmacology turns.",'
            '"target_failure":"nursing focus missed","expected_tradeoff":"none expected",'
            '"prompt_overrides":{},"hyperparameters":{},'
            '"attention_terms":{"nursing":["Pharmacology","cardiology","nursing","bad token!","cardiology"],'
            '"made_up_domain":["anything"]},'
            f'"canary_impact":{_canary_json()},'
            '"risk_notes":[]}'
        ),
    )

    proposal = evolution.propose({"preamble": "OLD", "tools_note": "TOOLS", "inject_note": "INJECT"})

    # Lowercased, deduped, built-in "nursing" and junk dropped, unknown domain dropped.
    assert proposal["attention_terms"] == {"nursing": ["pharmacology", "cardiology"]}
    assert proposal["approval_checks"]["canary_review"]["ok"] is True

    assert evolution.learned_attention_terms() == {}   # nothing active before approval
    approved = evolution.approve(proposal["id"])
    assert approved["approved"] is True
    assert evolution.learned_attention_terms() == {"nursing": ["pharmacology", "cardiology"]}

    # Snapshots carry the vocabulary, so restore reverts it like any mutation.
    evolution.restore_snapshot(approved["snapshots"]["before"])
    assert evolution.learned_attention_terms() == {}


def test_attention_terms_only_proposal_still_requires_canary_review(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "EVOLUTION_LOG_PATH", tmp_path / "evolution_log.txt")
    monkeypatch.setattr(
        evolution.llm,
        "chat",
        lambda *a, **k: (
            '{"title":"Vocabulary only","rationale":"r","target_failure":"t","expected_tradeoff":"none",'
            '"prompt_overrides":{},"hyperparameters":{},'
            '"attention_terms":{"nursing":["pharmacology"]},'
            '"canary_impact":[],"risk_notes":[]}'
        ),
    )

    proposal = evolution.propose({"preamble": "OLD", "tools_note": "TOOLS", "inject_note": "INJECT"})

    assert proposal["approval_checks"]["canary_review"]["ok"] is False
    assert "error" in evolution.approve(proposal["id"])
