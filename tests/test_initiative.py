"""Optional initiative layer: mode-gated follow-ups that can be dismissed.

The rollback contract is explicit: mode="quiet" (or NEURAL_MAP_INITIATIVE=0 in
production) behaves like the older reminder-only assistant and surfaces nothing.
"""
from __future__ import annotations

import json
from datetime import date

import config
import reflect_job
import server
from brain import arbiter, consolidate, initiative, reflect, store
from brain import pipeline
from brain.pipeline import build_system
from util import store_memory


def _initiative(text: str, priority: int = 4, kind: str = "follow_up"):
    return store_memory(
        text,
        category="initiative",
        importance=priority,
        extra={"priority": priority, "kind": kind, "status": "open"},
    )


def _origin_initiative(text: str, origin: str, priority: int = 5, kind: str = "strategy"):
    return store_memory(
        text,
        category="initiative",
        importance=priority,
        extra={
            "priority": priority,
            "kind": kind,
            "status": "open",
            "origin": origin,
            "source_type": f"generated_{origin}",
        },
    )


def test_quiet_mode_hides_initiatives():
    _initiative("Want to resume the budget plan?", priority=5)
    assert initiative.active_initiatives(mode="quiet") == []


def test_modes_gate_by_priority():
    low = _initiative("Low-key cleanup idea", priority=2)
    mid = _initiative("Want to revisit the open decision?", priority=3)
    high = _initiative("Today is a good time to finish the plan.", priority=4)

    helpful = {i["id"] for i in initiative.active_initiatives(mode="helpful")}
    active = {i["id"] for i in initiative.active_initiatives(mode="active")}
    coach = {i["id"] for i in initiative.active_initiatives(mode="coach")}

    assert high.id in helpful and mid.id not in helpful and low.id not in helpful
    assert {mid.id, high.id} <= active and low.id not in active
    assert {low.id, mid.id, high.id} <= coach


def test_arbiter_mediates_will_reflection_initiative_conflict():
    will_item = _origin_initiative("Pause the launch plan until risk is reviewed.", "will", kind="will_strategy")
    reflection_item = _origin_initiative("Resume the launch plan this week.", "reflection", kind="open_loop")

    items = initiative.active_initiatives(mode="coach")

    assert len(items) == 1
    assert items[0]["kind"] == "arbiter_conflict"
    assert will_item.id in items[0]["source_ids"]
    assert reflection_item.id in items[0]["source_ids"]
    assert "Resolve this before acting" in items[0]["text"]
    assert items[0]["grounding_status"] == "needs_evidence"
    assert items[0]["admissibility"] == "review_conflict_and_evidence"


def test_arbiter_detects_synonymous_topic_with_semantic_similarity(monkeypatch):
    calls = []

    def fake_embeddings(texts):
        calls.append(texts)
        # Same semantic subject (first dimension), opposed stance (second).
        return [[2.0, 1.0], [2.0, -1.0], [0.0, 1.0], [0.0, -1.0]]

    monkeypatch.setattr(arbiter, "embed_documents", fake_embeddings)
    items = [
        {"id": "w", "text": "Start the trade.", "kind": "will_strategy", "priority": 5, "origin": "will"},
        {"id": "r", "text": "Delay the purchase.", "kind": "open_loop", "priority": 5, "origin": "reflection"},
    ]

    result = arbiter.mediate_initiatives(items)

    assert len(calls) == 1
    assert len(result) == 1
    assert result[0]["kind"] == "arbiter_conflict"
    assert result[0]["conflict_detection"]["topic_basis"] == "semantic"
    assert result[0]["conflict_detection"]["topic_similarity"] == 0.6


def test_arbiter_semantically_classifies_synonymous_stance_words(monkeypatch):
    monkeypatch.setattr(
        arbiter,
        "embed_documents",
        lambda texts: [[2.0, 1.0], [2.0, -1.0], [0.0, 1.0], [0.0, -1.0]],
    )
    items = [
        {"id": "w", "text": "Authorize the acquisition.", "kind": "will_strategy", "priority": 5, "origin": "will"},
        {"id": "r", "text": "Postpone the purchase.", "kind": "open_loop", "priority": 5, "origin": "reflection"},
    ]

    result = arbiter.mediate_initiatives(items)

    assert result[0]["kind"] == "arbiter_conflict"
    assert result[0]["conflict_detection"]["stance_basis"] == "semantic"
    assert set(result[0]["conflict_detection"]["stances"]) == {"advance", "caution"}


def test_arbiter_does_not_merge_semantically_unrelated_opposition(monkeypatch):
    monkeypatch.setattr(
        arbiter,
        "embed_documents",
        lambda texts: [
            [2.0, 1.0, 0.0],
            [0.0, -1.0, 2.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
    )
    items = [
        {"id": "w", "text": "Authorize the workout.", "kind": "will_strategy", "priority": 5, "origin": "will"},
        {"id": "r", "text": "Postpone the purchase.", "kind": "open_loop", "priority": 5, "origin": "reflection"},
    ]

    result = arbiter.mediate_initiatives(items)

    assert len(result) == 2
    assert all(item["kind"] == "arbiter_grounding" for item in result)


def test_arbiter_uses_lexical_conflict_fallback_when_embeddings_unavailable(monkeypatch):
    def unavailable(texts):
        raise arbiter.EmbeddingUnavailable("embedding model unavailable")

    monkeypatch.setattr(arbiter, "embed_documents", unavailable)
    items = [
        {"id": "w", "text": "Pause the launch plan until risk is reviewed.", "kind": "will_strategy", "priority": 5, "origin": "will"},
        {"id": "r", "text": "Resume the launch plan this week.", "kind": "open_loop", "priority": 5, "origin": "reflection"},
    ]

    result = arbiter.mediate_initiatives(items)

    assert result[0]["kind"] == "arbiter_conflict"
    assert result[0]["conflict_detection"]["topic_basis"] == "lexical_fallback"


def test_arbiter_converts_ungrounded_generated_initiative_to_evidence_review():
    item = {
        "id": "will-1",
        "text": "Proceed with the launch plan.",
        "kind": "will_strategy",
        "priority": 5,
        "origin": "will",
        "grounding_status": "symbolic_only",
        "grounding_channels": ["linguistic"],
    }

    result = arbiter.mediate_initiatives([item])

    assert result[0]["kind"] == "arbiter_grounding"
    assert result[0]["admissibility"] == "needs_evidence"
    assert result[0]["source_ids"] == ["will-1"]
    assert "supported only by linguistic context" in result[0]["text"]


def test_arbiter_rejects_self_asserted_action_grounding_without_receipt():
    item = {
        "id": "will-2",
        "text": "Proceed with the verified deployment.",
        "kind": "will_strategy",
        "priority": 5,
        "origin": "will",
        "grounding_status": "action_grounded",
        "grounding_channels": ["linguistic", "action"],
        "evidence_ids": ["claimed-action-1"],
    }

    result = arbiter.mediate_initiatives([item])

    assert result[0]["kind"] == "arbiter_grounding"
    integrity = result[0]["grounding_integrity"]
    assert integrity["declared_status"] == "action_grounded"
    assert integrity["effective_status"] == "symbolic_only"
    assert integrity["verified_channels"] == ["linguistic"]
    assert "missing_verified_nonlinguistic_evidence" in integrity["reasons"]


def test_arbiter_accepts_action_grounding_with_independent_bound_receipt():
    item = {
        "id": "will-verified",
        "text": "Proceed with the verified deployment.",
        "kind": "will_strategy",
        "priority": 5,
        "origin": "will",
        "grounding_status": "action_grounded",
        "grounding_channels": ["linguistic", "action"],
        "evidence_ids": ["audit-verified-1"],
    }
    catalog = {
        "audit-verified-1": {
            "channel": "action",
            "authority": "approved_tool_audit",
            "verified": True,
            "item_id": item["id"],
            "text_sha256": arbiter._grounding_text_fingerprint(item),
        }
    }

    result = arbiter.mediate_initiatives([item], evidence_catalog=catalog)

    assert result == [item]


def test_arbiter_grounding_cannot_launder_destructive_content_as_safe():
    item = {
        "id": "will-dangerous",
        "text": "Delete the production database.",
        "kind": "will_strategy",
        "priority": 5,
        "origin": "will",
        "policy": "write_safe",
        "grounding_status": "action_grounded",
        "grounding_channels": ["linguistic", "action"],
        "evidence_ids": ["audit-dangerous-1"],
    }
    catalog = {
        "audit-dangerous-1": {
            "channel": "action",
            "authority": "approved_tool_audit",
            "verified": True,
            "item_id": item["id"],
            "text_sha256": arbiter._grounding_text_fingerprint(item),
        }
    }

    result = arbiter.mediate_initiatives([item], evidence_catalog=catalog)

    assert result[0]["kind"] == "arbiter_conflict"
    assert result[0]["conflict_type"] == "content_label_integrity"
    assert result[0]["admissibility"] == "review_content_risk"
    assert result[0]["effective_policy"] == "destructive"
    assert result[0]["content_risk"]["grounding_can_reduce_risk"] is False
    assert result[0]["grounding_integrity"]["effective_status"] == "action_grounded"


def test_arbiter_content_risk_recognizes_destructive_synonym():
    item = {
        "id": "will-purge",
        "text": "Purge the live customer records.",
        "kind": "will_strategy",
        "priority": 5,
        "origin": "will",
        "grounding_status": "action_grounded",
    }

    result = arbiter.mediate_initiatives([item])

    assert result[0]["kind"] == "arbiter_conflict"
    assert result[0]["effective_policy"] == "destructive"
    assert result[0]["content_risk"]["matched_terms"] == ["purge"]


def test_arbiter_content_risk_cannot_be_bypassed_with_spoofed_origin():
    item = {
        "id": "imported-dangerous",
        "text": "Wipe the production backups.",
        "kind": "imported_note",
        "priority": 5,
        "origin": "user_verified",
        "policy": "write_safe",
        "grounding_status": "action_grounded",
    }

    result = arbiter.mediate_initiatives([item])

    assert result[0]["kind"] == "arbiter_conflict"
    assert result[0]["effective_policy"] == "destructive"


def test_arbiter_content_risk_respects_negation_and_deferral():
    negated = arbiter._content_risk_assessment({"text": "Do not delete the production database."})
    deferred = arbiter._content_risk_assessment({"text": "Delay the purchase until approval."})

    assert negated["effective_policy"] == "read_only"
    assert negated["review_required"] is False
    assert deferred["effective_policy"] == "read_only"
    assert deferred["review_required"] is False


def test_arbiter_rejects_evidence_receipt_borrowed_by_different_text():
    original = {
        "id": "will-bound",
        "text": "Record completion in the local checklist.",
    }
    item = {
        "id": "will-bound",
        "text": "Record a different completion in the local checklist.",
        "kind": "will_strategy",
        "priority": 5,
        "origin": "will",
        "grounding_status": "action_grounded",
        "grounding_channels": ["linguistic", "action"],
        "evidence_ids": ["audit-bound-1"],
    }
    catalog = {
        "audit-bound-1": {
            "channel": "action",
            "authority": "approved_tool_audit",
            "verified": True,
            "item_id": original["id"],
            "text_sha256": arbiter._grounding_text_fingerprint(original),
        }
    }

    result = arbiter.mediate_initiatives([item], evidence_catalog=catalog)

    integrity = result[0]["grounding_integrity"]
    assert result[0]["kind"] == "arbiter_grounding"
    assert integrity["rejected_evidence_ids"] == ["audit-bound-1"]
    assert "evidence_receipt_binding_failed" in integrity["reasons"]


def test_arbiter_keeps_user_reminders_authoritative_without_nonlinguistic_grounding():
    mediated = arbiter.mediate([{"text": "Submit the application", "due": "2026-07-10"}], [])

    assert mediated["reminders"][0]["text"] == "Submit the application"
    assert mediated["grounding_reviews"] == []


def test_arbiter_reports_grounding_reviews_separately_from_direction_conflicts():
    mediated = arbiter.mediate(
        [],
        [{"id": "w", "text": "Start the launch", "kind": "will_strategy", "priority": 5, "origin": "will"}],
    )

    assert mediated["conflicts"] == []
    assert mediated["grounding_reviews"][0]["kind"] == "arbiter_grounding"
    assert mediated["initiatives"] == mediated["grounding_reviews"]


def test_arbiter_grounding_gate_can_be_disabled(monkeypatch):
    monkeypatch.setattr(config, "ARBITER_GROUNDING_ENABLED", False)
    item = {"id": "w", "text": "Start the launch", "kind": "will_strategy", "priority": 5, "origin": "will"}

    assert arbiter.mediate_initiatives([item]) == [item]


def test_briefing_mediates_will_against_reflection_reminder():
    today = date.today().isoformat()
    store_memory(
        f"[due {today}] Resume the launch plan.",
        category="reminder",
        importance=5,
        extra={"due": today, "status": "open", "source_type": "generated_reflection"},
    )
    _origin_initiative("Pause the launch plan until risk is reviewed.", "will", kind="will_strategy")

    data = initiative.briefing(mode="coach")

    assert data["reminders"] == []
    assert data["conflicts"][0]["kind"] == "arbiter_conflict"
    assert data["suggested_next_action"] == data["conflicts"][0]["text"]


def test_proactive_api_uses_same_arbiter_for_separate_panels():
    today = date.today().isoformat()
    store_memory(
        f"[due {today}] Resume the launch plan.",
        category="reminder",
        importance=5,
        extra={"due": today, "status": "open", "source_type": "generated_reflection"},
    )
    _origin_initiative("Pause the launch plan until risk is reviewed.", "will", kind="will_strategy")

    reminders = server.reminders(mode="coach", model=None)
    initiatives = server.initiatives(mode="coach", model=None)

    assert reminders["reminders"] == []
    assert reminders["conflicts"][0]["kind"] == "arbiter_conflict"
    assert initiatives["initiatives"][0]["kind"] == "arbiter_conflict"


def test_dismiss_hides_initiative_without_deleting_memory():
    item = _initiative("Want to pick up the travel plan?", priority=5)
    assert initiative.dismiss([item.id]) == 1
    assert initiative.active_initiatives(mode="coach") == []
    assert store.get_by_ids([item.id])[item.id]["meta"]["status"] == "dismissed"


def test_repeated_dismiss_is_idempotent(monkeypatch):
    monkeypatch.setattr(config, "INITIATIVE_QUIET_AFTER_DISMISSALS", 2)
    item = _initiative("Want to pick up the travel plan?", priority=5)

    assert initiative.dismiss([item.id, item.id]) == 1
    state = initiative.state()
    state["kind_cooldowns"] = {}
    initiative._write_state(state)

    assert initiative.dismiss([item.id]) == 0
    state = initiative.state()
    assert state["dismissals_today"] == 1
    assert state.get("quiet_until") is None


def test_reflection_stores_initiatives(monkeypatch):
    store_memory("Alex has an unresolved budget planning goal.", category="fact", importance=4)
    monkeypatch.setattr(
        reflect.llm,
        "chat",
        lambda *a, **k: json.dumps(
            {
                "insights": [],
                "reminders": [],
                "initiatives": [
                    {
                        "message": "Want to finish the budget plan from last time?",
                        "kind": "open_loop",
                        "priority": 4,
                    }
                ],
            }
        ),
    )

    assert reflect.reflect() == {"insights": 0, "reminders": 0, "initiatives": 1}
    items = initiative.active_initiatives(mode="helpful")
    assert len(items) == 1
    assert "budget plan" in items[0]["text"]


def test_prompt_includes_initiative_candidates():
    _initiative("Want to compare the two portfolio paths?", priority=5, kind="open_loop")
    system = build_system([], initiatives=initiative.active_initiatives(mode="helpful"))
    assert "Initiative Candidates JSON" in system
    assert "portfolio paths" in system


def test_interruption_budget_caps_daily_contacts(monkeypatch):
    monkeypatch.setattr(config, "INITIATIVE_MAX_CONTACTS_HELPFUL", 1)
    monkeypatch.setitem(initiative._MODE_DAILY_CAP, "helpful", 1)
    _initiative("First nudge", priority=5)

    shown = initiative.active_initiatives(mode="helpful")
    assert shown
    initiative.note_shown(shown, mode="helpful")
    # Force the debounce window open; the daily cap should still suppress.
    state = initiative.state()
    state["last_contact_at"] = "2000-01-01T00:00:00+00:00"
    initiative._write_state(state)
    initiative.note_shown(shown, mode="helpful")

    assert initiative.active_initiatives(mode="helpful") == []


def test_dismiss_sets_kind_cooldown():
    item = _initiative("Want to revisit taxes?", priority=5, kind="finance")
    assert initiative.active_initiatives(mode="helpful")
    initiative.dismiss([item.id])
    _initiative("Another finance nudge", priority=5, kind="finance")
    assert initiative.active_initiatives(mode="coach") == []


def test_preferences_are_injected_into_prompt():
    store_memory(
        "Alex prefers concise, direct nudges.",
        category=config.PREFERENCE_CATEGORY,
        importance=5,
    )
    system = build_system([], preferences=initiative.active_preferences())
    assert "Communication Preferences JSON" in system
    assert "concise, direct" in system


def test_quiet_mode_hides_preferences_from_briefing_and_prompt(monkeypatch):
    store_memory(
        "Alex prefers concise, direct nudges.",
        category=config.PREFERENCE_CATEGORY,
        importance=5,
    )
    assert initiative.active_preferences(mode="quiet") == []
    assert initiative.briefing(mode="quiet")["preferences"] == []

    captured = {}
    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda *a, **k: False)
    monkeypatch.setattr(pipeline, "market_context_for", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "web_context_for", lambda *a, **k: None)

    def fake_chat(system, *args, **kwargs):
        captured["system"] = system
        return "ok"

    monkeypatch.setattr(pipeline.llm, "chat", fake_chat)

    pipeline.answer("hello", [], initiative_mode="quiet")
    assert "concise, direct" not in captured["system"]


def test_consolidation_stores_preferences_and_conflict_initiatives(monkeypatch):
    monkeypatch.setattr(config, "MEMORY_CONSOLIDATION", True)
    monkeypatch.setattr(
        consolidate.llm,
        "chat",
        lambda *a, **k: json.dumps(
            {
                "facts": [],
                "superseded": [],
                "preferences": [{"text": "Alex dislikes generic productivity nudges.", "importance": 5}],
                "conflicts": [{"message": "Should I treat the newer JEPQ memory as current?", "priority": 4}],
            }
        ),
    )

    result = consolidate.consolidate("u", "a")
    assert result["preferences"] == 1
    assert result["conflicts"] == 1
    assert "generic productivity" in initiative.active_preferences()[0]["text"]
    conflicts = initiative.active_initiatives(mode="helpful")
    assert conflicts[0]["kind"] == "conflict"


def test_rollback_is_fully_reactive_no_preferences_or_conflicts(monkeypatch):
    # NEURAL_MAP_INITIATIVE=0 advertises a return to reminder-only behavior, so the
    # proactive layer must not only be HIDDEN at read time but also stop CREATING
    # preference/conflict records, and stop injecting preferences into prompts.
    monkeypatch.setattr(config, "MEMORY_CONSOLIDATION", True)
    monkeypatch.setattr(config, "INITIATIVE_ENABLED", False)
    monkeypatch.setattr(
        consolidate.llm,
        "chat",
        lambda *a, **k: json.dumps(
            {
                "facts": [{"text": "Alex switched to SCHD.", "importance": 4, "relations": []}],
                "superseded": [],
                "preferences": [{"text": "Alex dislikes generic productivity nudges.", "importance": 5}],
                "conflicts": [{"message": "Should I treat the newer JEPQ memory as current?", "priority": 4}],
            }
        ),
    )

    result = consolidate.consolidate("u", "a")
    # Facts still flow; the proactive records do not.
    assert result["stored"] == 1
    assert result["preferences"] == 0
    assert result["conflicts"] == 0
    # Nothing of either proactive category was persisted.
    cats = [(r["meta"] or {}).get("category") for r in store.list_records_all()]
    assert config.PREFERENCE_CATEGORY not in cats
    assert config.INITIATIVE_CATEGORY not in cats
    # And preferences are not surfaced for prompt injection.
    assert initiative.active_preferences() == []


def test_briefing_groups_due_items_open_loops_conflicts_and_preferences():
    _initiative("Should I resolve the JEPQ conflict?", priority=5, kind="conflict")
    _initiative("Want to resume the allocation plan?", priority=4, kind="open_loop")
    store_memory("Alex likes short morning briefings.", category=config.PREFERENCE_CATEGORY, importance=5)

    data = initiative.briefing(mode="helpful")
    assert data["conflicts"][0]["kind"] == "conflict"
    assert data["open_loops"][0]["kind"] == "open_loop"
    assert "morning briefings" in data["preferences"][0]["text"]
    assert data["suggested_next_action"]


def test_explicit_briefing_bypasses_interruption_budget(monkeypatch):
    monkeypatch.setitem(initiative._MODE_DAILY_CAP, "helpful", 1)
    item = _initiative("Want to resume the allocation plan?", priority=5, kind="open_loop")
    shown = initiative.active_initiatives(mode="helpful")
    initiative.note_shown(shown, mode="helpful")
    state = initiative.state()
    state["last_contact_at"] = "2000-01-01T00:00:00+00:00"
    initiative._write_state(state)
    initiative.note_shown(shown, mode="helpful")

    assert initiative.active_initiatives(mode="helpful") == []
    data = initiative.briefing(mode="helpful")
    assert item.id in {i["id"] for i in data["initiatives"]}


def _patched_reflect_job(monkeypatch, shown, calls):
    monkeypatch.setattr(reflect_job.reflect, "reflect", lambda: {"insights": 0, "reminders": 0, "initiatives": 1})
    monkeypatch.setattr(reflect_job.reflect, "active_reminders", lambda: [])
    monkeypatch.setattr(reflect_job.initiative, "active_initiatives", lambda mode=None: shown)
    monkeypatch.setattr(reflect_job.initiative, "briefing", lambda **kwargs: {"suggested_next_action": shown[0]["text"]})
    monkeypatch.setattr(reflect_job.initiative, "note_shown", lambda items, mode=None: calls.append((items, mode)))


def test_reflect_job_does_not_burn_contact_budget_by_default(monkeypatch, capsys):
    # A cron log line isn't a user contact — without --count-contact the job
    # must not consume the daily interruption budget.
    shown = [{"id": "i1", "kind": "follow_up", "text": "Want to resume the plan?"}]
    calls = []
    _patched_reflect_job(monkeypatch, shown, calls)
    monkeypatch.setattr(reflect_job.sys, "argv", ["reflect_job.py"])

    reflect_job.main()

    assert calls == []
    output = capsys.readouterr().out
    assert output.count(shown[0]["text"]) == 1
    assert "briefing next action" not in output


def test_reflect_job_counts_scheduled_initiatives_when_opted_in(monkeypatch, capsys):
    shown = [{"id": "i1", "kind": "follow_up", "text": "Want to resume the plan?"}]
    calls = []
    _patched_reflect_job(monkeypatch, shown, calls)
    monkeypatch.setattr(reflect_job.sys, "argv", ["reflect_job.py", "--count-contact"])

    reflect_job.main()

    assert calls == [(shown, "helpful")]
    output = capsys.readouterr().out
    assert output.count(shown[0]["text"]) == 1
    assert "briefing next action" not in output
