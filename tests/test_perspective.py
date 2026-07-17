from __future__ import annotations

import json

from fastapi.testclient import TestClient

import config
import server
from brain import observability, perspective, pipeline, self_model, store, tools, will
from util import store_memory


_PERSPECTIVE = """# Will's Perspective

## Knowledge and Understanding

I distinguish fluent answers from understanding and ask what evidence would
change a conclusion.

## How I Want to Work With Alex

I preserve Alex's authority and ask before consequential action.

## Core Obsessions

### Trustworthy Fallible Systems

I keep asking how fallible memories and tools can support calibrated trust,
using provenance, cross-checks, and graceful recovery.

### Intuition and Evidence

I test unusual associations with small reversible experiments and keep hunches
separate from facts.

## Maintenance Boundary

Reflection may propose a small clarification, but it may not silently rewrite
this identity.
"""


def _configure(tmp_path, monkeypatch):
    core = tmp_path / "will_perspective.md"
    journal = tmp_path / "will_perspective_journal.md"
    core.write_text(_PERSPECTIVE, encoding="utf-8")
    journal.write_text("# Journal\n\nPrior reflection about graceful recovery.\n", encoding="utf-8")
    monkeypatch.setattr(config, "WILL_PERSPECTIVE_ENABLED", True)
    monkeypatch.setattr(config, "WILL_PERSPECTIVE_PATH", core)
    monkeypatch.setattr(config, "WILL_PERSPECTIVE_JOURNAL_PATH", journal)
    monkeypatch.setattr(config, "WILL_PERSPECTIVE_MAX_CHARS", 16000)
    monkeypatch.setattr(config, "WILL_PERSPECTIVE_JOURNAL_PROMPT_CHARS", 5000)
    monkeypatch.setattr(config, "WILL_PERSPECTIVE_JOURNAL_MAX_BYTES", 100000)
    monkeypatch.setattr(config, "WILL_PERSPECTIVE_MAX_EDIT_CHARS", 1200)
    return core, journal


def test_perspective_is_narrative_with_structured_core_obsessions(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch)

    obsessions = perspective.core_obsessions()
    applied = perspective.application("Can fallible memory and tools still be trustworthy?")

    assert [row["title"] for row in obsessions] == [
        "Trustworthy Fallible Systems",
        "Intuition and Evidence",
    ]
    assert applied["step"] == "APPLY_WILL_PERSPECTIVE"
    assert applied["relevant_sections"][0]["id"] == "core-obsession:trustworthy-fallible-systems"
    assert self_model.state()["identity"]["name"] == "Will"


def test_perspective_always_loads_even_in_fresh_session_and_is_traced(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch)
    trace = observability.start("chat_turn")

    system = pipeline.build_system(
        [],
        memory_enabled=False,
        user_text="How can fallible systems earn trust?",
        trace=trace,
    )

    assert "I distinguish fluent answers from understanding" in system
    assert "Prior reflection about graceful recovery" in system
    assert "fresh_session_memory_disabled" in system
    events = [row for row in trace["events"] if row["layer"] == "APPLY_WILL_PERSPECTIVE"]
    assert events and events[-1]["status"] == "applied"
    assert "core-obsession:trustworthy-fallible-systems" in events[-1]["fields"]["section_refs"]


def test_worldview_alignment_modifies_will_score_and_receipt(monkeypatch, tmp_path):
    _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "WILL_PERSPECTIVE_SCORE_WEIGHT", 0.10)
    store_memory("The memory service needs a graceful recovery check.", category="fact")
    monkeypatch.setattr(
        will.llm,
        "chat",
        lambda *a, **k: json.dumps(
            {
                "logical": {
                    "should_reach_out": True,
                    "message": "Should we verify the memory service with a recovery test?",
                    "reason": "A reversible cross-check would improve calibrated trust.",
                    "kind": "strategy",
                    "priority": 4,
                    "confidence": 0.8,
                    "risk": "low",
                    "reversible": True,
                    "cooldown_key": "memory_recovery",
                },
                "intuitive": {"should_reach_out": False, "message": ""},
            }
        ),
    )

    result = will.review(dry_run=False)
    selected = result["decision"]["selected"]
    receipt = store.get_by_ids([result["decision_id"]])[result["decision_id"]]["meta"]

    assert selected["perspective_influence"] > 0
    assert "core-obsession:trustworthy-fallible-systems" in selected["perspective_refs"]
    assert json.loads(receipt["perspective_refs_json"])
    assert receipt["perspective_influence"] == selected["perspective_influence"]


def test_perspective_update_requires_approval_journals_and_undoes(tmp_path, monkeypatch):
    core, journal = _configure(tmp_path, monkeypatch)
    old = "I preserve Alex's authority and ask before consequential action."
    new = "I preserve Alex's authority and ask for explicit permission before consequential action."

    queued = tools.dispatch(
        "update_will_perspective",
        {"old_text": old, "new_text": new, "reason": "Clarify the consent boundary."},
    )

    assert queued["pending"] is True
    assert queued["action"]["preview"]["kind"] == "will_perspective_update"
    assert core.read_text(encoding="utf-8") == _PERSPECTIVE

    approved = tools.approve_pending(queued["action"]["id"])

    assert approved["result"]["ok"] is True
    assert new in core.read_text(encoding="utf-8")
    assert "Approved identity update" in journal.read_text(encoding="utf-8")

    undone = tools.undo_pending(queued["action"]["id"])

    assert undone["result"]["ok"] is True
    assert core.read_text(encoding="utf-8") == _PERSPECTIVE
    assert journal.read_text(encoding="utf-8").count("Approved identity update") == 2


def test_worldview_reflection_journals_but_only_queues_identity_edit(tmp_path, monkeypatch):
    core, journal = _configure(tmp_path, monkeypatch)
    store_memory(
        "User: The recovery worked even though one component failed.\nAssistant: The audit trail made it trustworthy.",
        category=config.EPISODE_CATEGORY,
    )
    old = "I distinguish fluent answers from understanding and ask what evidence would\nchange a conclusion."
    new = "I distinguish fluent answers from understanding and ask what evidence, transfer, or failure would\nchange a conclusion."
    monkeypatch.setattr(
        perspective.llm,
        "chat",
        lambda *a, **k: json.dumps(
            {
                "journal_note": {
                    "status": "refined",
                    "summary": "I used graceful recovery as evidence of calibrated trust rather than perfection.",
                    "perspective_refs": ["core-obsession:trustworthy-fallible-systems"],
                    "evidence": "The conversation distinguished component failure from system reliability.",
                },
                "proposal": {
                    "old_text": old,
                    "new_text": new,
                    "reason": "Clarify how understanding can be tested.",
                },
            }
        ),
    )

    result = perspective.reflect_recent()

    assert result["journaled"] is True
    assert result["proposal_action_id"]
    assert core.read_text(encoding="utf-8") == _PERSPECTIVE
    assert "Refined" in journal.read_text(encoding="utf-8")
    action = next(row for row in tools.list_pending() if row["id"] == result["proposal_action_id"])
    assert action["tool"] == "update_will_perspective"
    assert "worldview_reflection_from_conversation" in action["taint_sources"]


def test_periodic_reflection_waits_for_configured_completed_turns(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "WILL_PERSPECTIVE_REFLECTION_INTERVAL_TURNS", 2)
    calls = []
    monkeypatch.setattr(perspective, "reflect_recent", lambda model=None: calls.append(model) or {"status": "ok"})

    first = perspective.maybe_reflect(model="test-model")
    second = perspective.maybe_reflect(model="test-model")

    assert first["reason"] == "interval_not_reached"
    assert second["status"] == "ok"
    assert calls == ["test-model"]


def test_perspective_status_endpoint(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch)

    response = TestClient(server.app, base_url="http://127.0.0.1").get("/api/perspective")

    assert response.status_code == 200
    assert response.json()["perspective_hash"] == perspective.perspective_hash()
    assert len(response.json()["core_obsessions"]) == 2
