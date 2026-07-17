"""Reflection must not wipe existing insights/reminders when generating new ones.

Regression: reflect() used to delete the old insight/reminder categories BEFORE
storing replacements, so any failure after the delete (Ollama/embed/store error)
left the user with no reminders. The fix stores replacements first and deletes the
prior generation only once that succeeds. The model call is monkeypatched so these
tests are hermetic (no Ollama)."""
from __future__ import annotations

import json
from datetime import date

import pytest

from brain import reflect, store
from brain.reflect import active_reminders
from util import store_memory


def _seed_old_generation():
    """A prior reflection's output: one reminder (due today) and one insight.

    Stamped source_type=generated_reflection like reflect() itself writes —
    regeneration only owns (and deletes) records carrying that marker."""
    today = date.today().isoformat()
    store_memory(
        text=f"[due {today}] OLD reminder action",
        category="reminder",
        importance=5,
        extra={"due": today, "status": "open", "source_type": "generated_reflection"},
    )
    store_memory(
        "OLD insight pattern",
        category="insight",
        importance=4,
        extra={"source_type": "generated_reflection"},
    )


def _patch_model(monkeypatch, payload: dict):
    monkeypatch.setattr(reflect.llm, "chat", lambda *a, **k: json.dumps(payload))


def _insight_docs() -> list[str]:
    return store.get_by_category("insight").get("documents") or []


def test_failed_replacement_storage_keeps_old_reminders(monkeypatch):
    _seed_old_generation()
    # The model returns a valid new generation...
    _patch_model(
        monkeypatch,
        {
            "insights": ["NEW insight"],
            "reminders": [{"due": date.today().isoformat(), "action": "NEW action", "importance": 5}],
            "initiatives": [],
        },
    )
    # ...but embedding the replacements hard-fails (e.g. Ollama drops mid-run).
    def _boom(*a, **k):
        raise ConnectionError("ollama is down")

    monkeypatch.setattr(reflect, "embed_document", _boom)

    with pytest.raises(ConnectionError):
        reflect.reflect()

    # The old generation must survive untouched — this is the whole bug.
    rems = active_reminders()
    assert len(rems) == 1
    assert "OLD reminder action" in rems[0]["text"]
    assert _insight_docs() == ["OLD insight pattern"]


def test_successful_reflection_replaces_without_duplicating(monkeypatch):
    _seed_old_generation()
    _patch_model(
        monkeypatch,
        {
            "insights": ["NEW insight"],
            "reminders": [{"due": date.today().isoformat(), "action": "NEW action", "importance": 5}],
            "initiatives": [],
        },
    )

    result = reflect.reflect()
    assert result == {"insights": 1, "reminders": 1, "initiatives": 0}

    # Old gone, new present, no pile-up.
    rems = active_reminders()
    assert len(rems) == 1
    assert "NEW action" in rems[0]["text"]
    assert _insight_docs() == ["NEW insight"]


def test_successful_empty_reflection_clears_stale(monkeypatch):
    # A real, successful run that finds nothing current should still clear stale
    # items (regenerate-from-scratch semantics) — distinct from a FAILED run.
    _seed_old_generation()
    store_memory("a durable fact to reflect on", category="fact", importance=4)
    _patch_model(monkeypatch, {"insights": [], "reminders": [], "initiatives": []})

    result = reflect.reflect()
    assert result == {"insights": 0, "reminders": 0, "initiatives": 0}
    assert active_reminders() == []
    assert _insight_docs() == []


def test_unusable_due_dates_do_not_replace_valid_reminders(monkeypatch):
    # Regression: the model answered due="next Tuesday"; the reminder was stored
    # as-is (invisible to active_reminders forever), the run reported success,
    # and the valid previous generation was deleted — active reminders: empty.
    _seed_old_generation()
    _patch_model(
        monkeypatch,
        {
            "insights": [],
            "reminders": [{"due": "next Tuesday", "action": "NEW action", "importance": 5}],
            "initiatives": [],
        },
    )

    result = reflect.reflect()

    assert result["reminders"] == 0          # the unusable reminder was rejected
    rems = active_reminders()
    assert len(rems) == 1                     # ...and the valid old one survived
    assert "OLD reminder action" in rems[0]["text"]


def test_partially_valid_reminders_replace_and_drop_only_the_bad_one(monkeypatch):
    # One good date and one bad one: regeneration proceeds (old gone, good new
    # one live) and only the unusable item is dropped.
    _seed_old_generation()
    _patch_model(
        monkeypatch,
        {
            "insights": [],
            "reminders": [
                {"due": "next Tuesday", "action": "BAD action", "importance": 5},
                {"due": date.today().isoformat(), "action": "GOOD action", "importance": 5},
            ],
            "initiatives": [],
        },
    )

    result = reflect.reflect()

    assert result["reminders"] == 1
    rems = active_reminders()
    assert len(rems) == 1
    assert "GOOD action" in rems[0]["text"]


def test_datetime_due_is_normalized_to_date(monkeypatch):
    # An ISO datetime due is fine — normalized to its date so the stored record
    # and the [due ...] text agree with what active_reminders compares.
    _seed_old_generation()
    _patch_model(
        monkeypatch,
        {
            "insights": [],
            "reminders": [{"due": f"{date.today().isoformat()}T09:30:00", "action": "timed action", "importance": 4}],
            "initiatives": [],
        },
    )

    result = reflect.reflect()

    assert result["reminders"] == 1
    rems = active_reminders()
    assert len(rems) == 1
    assert rems[0]["due"] == date.today().isoformat()
    assert "timed action" in rems[0]["text"]


def test_reflection_never_deletes_user_created_insights_and_reminders(monkeypatch):
    # Regression: _generated_ids_for_category used to return EVERY id in the
    # insight/reminder categories, so a manually saved memory with one of those
    # categories was wiped by the next reflection pass.
    _seed_old_generation()
    today = date.today().isoformat()
    user_insight = store_memory("MY hand-written insight", category="insight", importance=5)
    user_reminder = store_memory(
        text=f"[due {today}] MY hand-written reminder",
        category="reminder",
        importance=5,
        extra={"due": today, "status": "open"},
    )
    legacy_insight = store_memory(
        "Pre-provenance insight of unknown origin",
        category="insight",
        importance=4,
        extra={"source_type": "legacy"},
    )
    pinned_generated = store_memory(
        "Generated but pinned insight",
        category="insight",
        importance=4,
        extra={"source_type": "generated_reflection", "pinned": True},
    )
    store_memory("a durable fact to reflect on", category="fact", importance=4)
    _patch_model(monkeypatch, {"insights": ["NEW insight"], "reminders": [], "initiatives": []})

    result = reflect.reflect()

    assert result == {"insights": 1, "reminders": 0, "initiatives": 0}
    survivors = store.get_by_ids(
        [user_insight.id, user_reminder.id, legacy_insight.id, pinned_generated.id]
    )
    assert user_insight.id in survivors
    assert user_reminder.id in survivors
    assert legacy_insight.id in survivors
    assert pinned_generated.id in survivors
    # The OLD generated ones were replaced; the user's reminder still fires.
    docs = _insight_docs()
    assert "OLD insight pattern" not in docs
    assert "NEW insight" in docs
    assert any("MY hand-written reminder" in r["text"] for r in active_reminders())


def test_reflection_replaces_only_reflection_owned_initiatives(monkeypatch):
    store_memory("a durable fact to reflect on", category="fact", importance=4)
    conflict = store_memory(
        "Should I treat the newer JEPQ memory as current?",
        category="initiative",
        importance=4,
        extra={"kind": "conflict", "priority": 4, "status": "open"},
    )
    stale_reflection = store_memory(
        "Want to resume the stale allocation plan?",
        category="initiative",
        importance=4,
        extra={"kind": "open_loop", "priority": 4, "status": "open", "origin": "reflection"},
    )
    legacy_reflection = store_memory(
        "Want to resume the legacy allocation plan?",
        category="initiative",
        importance=4,
        extra={"kind": "open_loop", "priority": 4, "status": "open", "expires_at": "2099-01-01T00:00:00+00:00"},
    )
    manual_initiative = store_memory(
        "Manual open-loop initiative should survive reflection.",
        category="initiative",
        importance=4,
        extra={"kind": "open_loop", "priority": 4, "status": "open"},
    )
    _patch_model(monkeypatch, {"insights": [], "reminders": [], "initiatives": []})

    result = reflect.reflect()

    assert result == {"insights": 0, "reminders": 0, "initiatives": 0}
    records = store.get_by_ids([conflict.id, stale_reflection.id, legacy_reflection.id, manual_initiative.id])
    assert conflict.id in records
    assert manual_initiative.id in records
    assert stale_reflection.id not in records
    assert legacy_reflection.id not in records
