"""active_reminders surfacing — including the P1 fix: reminders stored in the
text-only collection (embeddings-down fallback) must still be surfaced."""
from __future__ import annotations

from datetime import date, timedelta

from brain.reflect import active_reminders
from util import store_memory


def _reminder(due: str, status: str = "open", text_only: bool = False):
    return store_memory(
        text=f"[due {due}] do the thing",
        category="reminder",
        importance=5,
        extra={"due": due, "status": status},
        text_only=text_only,
    )


def _texts(rems):
    return " | ".join(r["text"] for r in rems)


def test_text_only_reminder_is_surfaced():
    # P1 regression: a reminder written to the text-only store must appear.
    _reminder(date.today().isoformat(), text_only=True)
    rems = active_reminders()
    assert any("do the thing" in r["text"] for r in rems), _texts(rems)


def test_vector_and_text_only_reminders_both_surface():
    _reminder(date.today().isoformat(), text_only=False)
    _reminder(date.today().isoformat(), text_only=True)
    assert len(active_reminders()) == 2


def test_reminder_beyond_horizon_is_hidden():
    far = (date.today() + timedelta(days=365)).isoformat()
    _reminder(far)
    assert active_reminders() == []


def test_due_within_horizon_is_shown():
    soon = (date.today() + timedelta(days=2)).isoformat()
    _reminder(soon)
    assert len(active_reminders()) == 1


def test_overdue_reminder_flagged():
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    _reminder(yesterday)
    rems = active_reminders()
    assert len(rems) == 1
    assert rems[0]["overdue"] is True


def test_non_open_status_is_skipped():
    _reminder(date.today().isoformat(), status="done")
    assert active_reminders() == []


def test_reminder_without_due_is_hidden():
    # No due date -> it can never pass or expire, so it used to sit in every
    # prompt and the UI banner forever. It must be skipped (the record itself
    # stays in the store, visible in the Memory Manager).
    store_memory(
        text="do the thing sometime",
        category="reminder",
        importance=5,
        extra={"status": "open"},
    )
    assert active_reminders() == []


def test_reminder_with_unparseable_due_is_hidden():
    _reminder("June 27")   # not ISO — unparseable, would never expire
    assert active_reminders() == []


def test_reminders_sorted_by_due_date():
    d1 = date.today().isoformat()
    d2 = (date.today() + timedelta(days=5)).isoformat()
    _reminder(d2)
    _reminder(d1)
    dues = [r["due"] for r in active_reminders()]
    assert dues == sorted(dues)
