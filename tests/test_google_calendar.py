"""Google Calendar reads and approval-gated mutation contract."""
from __future__ import annotations

import config
import pytest
from brain import tools
from brain.toolkit import google_calendar


@pytest.fixture(autouse=True)
def _direct_google_provider(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_CALENDAR_PROVIDER", "google")


class FakeRequest:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.headers = {}

    def execute(self):
        if self.error:
            raise self.error
        return self.result


class FakeEvents:
    def __init__(self):
        self.calls = []
        self.list_result = {"items": []}
        self.insert_result = {}
        self.patch_result = {}
        self.delete_error = None
        self.last_request = None

    def list(self, **kwargs):
        self.calls.append(("list", kwargs))
        self.last_request = FakeRequest(self.list_result)
        return self.last_request

    def insert(self, **kwargs):
        self.calls.append(("insert", kwargs))
        self.last_request = FakeRequest(self.insert_result)
        return self.last_request

    def patch(self, **kwargs):
        self.calls.append(("patch", kwargs))
        self.last_request = FakeRequest(self.patch_result)
        return self.last_request

    def delete(self, **kwargs):
        self.calls.append(("delete", kwargs))
        self.last_request = FakeRequest(None, self.delete_error)
        return self.last_request


class FakeService:
    def __init__(self):
        self.event_api = FakeEvents()

    def events(self):
        return self.event_api


def _event(event_id="evt_1", title="Planning", etag='"v1"'):
    return {
        "id": event_id,
        "etag": etag,
        "status": "confirmed",
        "summary": title,
        "start": {"dateTime": "2026-07-20T09:00:00-07:00"},
        "end": {"dateTime": "2026-07-20T10:00:00-07:00"},
        "htmlLink": "https://calendar.google.com/event?eid=redacted",
    }


def test_google_calendar_status_is_disconnected_without_token(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_CALENDAR_TOKEN_PATH", tmp_path / "missing-token.json")

    out = tools.dispatch("google_calendar_status", {})

    assert out["connected"] is False
    assert "connect_google_calendar.py" in out["connect_command"]


def test_google_calendar_live_read_executes_without_approval(monkeypatch):
    service = FakeService()
    service.event_api.list_result = {"items": [_event()], "nextPageToken": "next"}
    monkeypatch.setattr(google_calendar, "_service", lambda: service)

    out = tools.dispatch(
        "list_google_calendar_events",
        {"from_date": "2026-07-20", "to_date": "2026-07-21", "query": "Planning"},
    )

    assert out["events"][0]["id"] == "evt_1"
    assert out["events"][0]["etag"] == '"v1"'
    assert out["next_page_token"] == "next"
    name, kwargs = service.event_api.calls[0]
    assert name == "list"
    assert kwargs["calendarId"] == "primary"
    assert kwargs["q"] == "Planning"


def test_google_calendar_create_waits_for_approval(monkeypatch):
    service = FakeService()
    service.event_api.insert_result = _event()
    monkeypatch.setattr(google_calendar, "_service", lambda: service)

    queued = tools.dispatch(
        "create_google_calendar_event",
        {
            "title": "Planning",
            "start": "2026-07-20T09:00:00-07:00",
            "end": "2026-07-20T10:00:00-07:00",
        },
    )

    assert queued["pending"] is True
    assert queued["action"]["policy"] == tools.EXTERNAL_WRITE
    assert queued["action"]["preview"]["kind"] == "google_calendar_create"
    assert service.event_api.calls == []

    approved = tools.approve_pending(queued["action"]["id"])
    assert approved["result"]["created"] is True
    assert service.event_api.calls[0][0] == "insert"


def test_google_calendar_update_uses_reviewed_etag_after_approval(monkeypatch):
    service = FakeService()
    service.event_api.patch_result = _event(title="Updated", etag='"v2"')
    monkeypatch.setattr(google_calendar, "_service", lambda: service)

    queued = tools.dispatch(
        "update_google_calendar_event",
        {
            "event_id": "evt_1",
            "expected_etag": '"v1"',
            "expected_title": "Planning",
            "title": "Updated",
        },
    )

    assert service.event_api.calls == []
    approved = tools.approve_pending(queued["action"]["id"])
    assert approved["result"]["updated"] is True
    assert service.event_api.last_request.headers["If-Match"] == '"v1"'
    assert service.event_api.calls[0][1]["body"] == {"summary": "Updated"}


def test_google_calendar_delete_is_destructive_and_approval_gated(monkeypatch):
    service = FakeService()
    monkeypatch.setattr(google_calendar, "_service", lambda: service)

    queued = tools.dispatch(
        "delete_google_calendar_event",
        {"event_id": "evt_1", "expected_etag": '"v1"', "expected_title": "Planning"},
    )

    assert queued["pending"] is True
    assert queued["action"]["policy"] == tools.DESTRUCTIVE
    assert queued["action"]["preview"]["kind"] == "google_calendar_delete"
    assert service.event_api.calls == []

    approved = tools.approve_pending(queued["action"]["id"])
    assert approved["result"]["deleted"] is True
    assert service.event_api.last_request.headers["If-Match"] == '"v1"'


def test_google_calendar_stale_event_fails_without_retry(monkeypatch):
    class StaleError(Exception):
        class Response:
            status = 412

        resp = Response()

    service = FakeService()
    service.event_api.delete_error = StaleError()
    monkeypatch.setattr(google_calendar, "_service", lambda: service)
    queued = tools.dispatch(
        "delete_google_calendar_event",
        {"event_id": "evt_1", "expected_etag": '"v1"', "expected_title": "Planning"},
    )

    approved = tools.approve_pending(queued["action"]["id"])

    assert approved["execution_failed"] is True
    assert approved["result"]["stale_event"] is True
    assert approved["action"]["status"] == "failed"


def test_macos_calendar_fallback_reads_live_events(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_CALENDAR_PROVIDER", "macos")
    monkeypatch.setattr(
        google_calendar,
        "_run_macos",
        lambda payload: {
            "calendar_name": "me@example.com",
            "events": [
                {
                    "id": "mac_evt_1",
                    "status": "confirmed",
                    "title": "Local bridge event",
                    "start": {"dateTime": "2026-07-20T16:00:00.000Z"},
                    "end": {"dateTime": "2026-07-20T17:00:00.000Z"},
                    "description": None,
                    "location": None,
                    "all_day": False,
                }
            ],
        },
    )

    out = tools.dispatch("list_google_calendar_events", {"from_date": "2026-07-20", "to_date": "2026-07-21"})

    assert out["provider"] == "macos_calendar"
    assert out["calendar_id"] == "me@example.com"
    assert out["timeZone"] == "America/Phoenix"
    assert out["events"][0]["id"] == "mac_evt_1"
    assert out["events"][0]["start"] == {
        "dateTime": "2026-07-20T09:00:00-07:00",
        "timeZone": "America/Phoenix",
    }
    assert out["events"][0]["etag"].startswith('"macos-')


def test_macos_calendar_create_still_waits_for_approval(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_CALENDAR_PROVIDER", "macos")
    calls = []

    def fake_run(payload):
        calls.append(payload)
        return {
            "calendar_name": "me@example.com",
            "event": {
                "id": "mac_evt_2",
                "status": "confirmed",
                "title": payload["title"],
                "start": {"dateTime": "2026-07-20T16:00:00.000Z"},
                "end": {"dateTime": "2026-07-20T17:00:00.000Z"},
                "description": None,
                "location": None,
                "all_day": False,
            },
        }

    monkeypatch.setattr(google_calendar, "_run_macos", fake_run)
    queued = tools.dispatch(
        "create_google_calendar_event",
        {"title": "Approved event", "start": "2026-07-20T09:00:00-07:00", "end": "2026-07-20T10:00:00-07:00"},
    )

    assert queued["pending"] is True
    assert calls == []
    approved = tools.approve_pending(queued["action"]["id"])
    assert approved["result"]["created"] is True
    assert approved["result"]["provider"] == "macos_calendar"
    assert calls[0]["operation"] == "create"


def test_macos_calendar_update_rejects_stale_review(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_CALENDAR_PROVIDER", "macos")
    current = {
        "id": "mac_evt_1",
        "status": "confirmed",
        "title": "Changed elsewhere",
        "start": {"dateTime": "2026-07-20T16:00:00.000Z"},
        "end": {"dateTime": "2026-07-20T17:00:00.000Z"},
        "description": None,
        "location": None,
        "all_day": False,
    }
    monkeypatch.setattr(
        google_calendar,
        "_run_macos",
        lambda payload: {"calendar_name": "me@example.com", "event": current},
    )
    queued = tools.dispatch(
        "update_google_calendar_event",
        {"event_id": "mac_evt_1", "expected_etag": '"macos-old"', "expected_title": "Old title", "title": "New title"},
    )

    approved = tools.approve_pending(queued["action"]["id"])

    assert approved["execution_failed"] is True
    assert approved["result"]["stale_event"] is True


_MAC_EVENT = {
    "id": "mac_evt_1",
    "status": "confirmed",
    "title": "Reviewed title",
    "start": {"dateTime": "2026-07-20T16:00:00.000Z"},
    "end": {"dateTime": "2026-07-20T17:00:00.000Z"},
    "description": None,
    "location": None,
    "all_day": False,
}
_MAC_EXPECTED = {key: _MAC_EVENT.get(key) for key in ("title", "start", "end", "description", "location", "all_day")}


def test_macos_calendar_mutation_carries_the_reviewed_snapshot(monkeypatch):
    """The etag pre-check runs in one osascript call and the mutation in
    another, so the mutation payload must carry the reviewed raw fields for the
    JXA script to re-verify atomically — otherwise a concurrent change between
    the two calls is silently overwritten (TOCTOU)."""
    monkeypatch.setattr(config, "GOOGLE_CALENDAR_PROVIDER", "macos")
    etag = google_calendar._mac_record(_MAC_EVENT)["etag"]
    calls = []

    def fake_run(payload):
        calls.append(payload)
        if payload["operation"] == "get":
            return {"calendar_name": "me@example.com", "event": dict(_MAC_EVENT)}
        if payload["operation"] == "update":
            return {"calendar_name": "me@example.com", "event": {**_MAC_EVENT, "title": payload["title"]}}
        return {"deleted": True, "event_id": payload["event_id"], "calendar_name": "me@example.com"}

    monkeypatch.setattr(google_calendar, "_run_macos", fake_run)

    queued = tools.dispatch(
        "update_google_calendar_event",
        {"event_id": "mac_evt_1", "expected_etag": etag, "title": "New title"},
    )
    approved = tools.approve_pending(queued["action"]["id"])
    assert approved["result"]["updated"] is True
    update_payload = [p for p in calls if p["operation"] == "update"][0]
    assert update_payload["expected"] == _MAC_EXPECTED

    queued = tools.dispatch(
        "delete_google_calendar_event",
        {"event_id": "mac_evt_1", "expected_etag": etag, "expected_title": "Reviewed title"},
    )
    approved = tools.approve_pending(queued["action"]["id"])
    assert approved["result"]["deleted"] is True
    delete_payload = [p for p in calls if p["operation"] == "delete"][0]
    assert delete_payload["expected"] == _MAC_EXPECTED


def test_macos_calendar_change_between_check_and_mutation_is_refused(monkeypatch):
    """Simulate the race: the pre-check 'get' still matches the reviewed etag,
    but by the time the mutation osascript runs the event has changed — the
    script's own snapshot re-check refuses instead of overwriting."""
    monkeypatch.setattr(config, "GOOGLE_CALENDAR_PROVIDER", "macos")
    etag = google_calendar._mac_record(_MAC_EVENT)["etag"]

    def fake_run(payload):
        if payload["operation"] == "get":
            return {"calendar_name": "me@example.com", "event": dict(_MAC_EVENT)}
        # What the JXA script returns when matchesExpected() fails.
        return {
            "error": "the calendar event changed after it was reviewed; read it again and request a new approval",
            "stale_event": True,
        }

    monkeypatch.setattr(google_calendar, "_run_macos", fake_run)

    queued = tools.dispatch(
        "update_google_calendar_event",
        {"event_id": "mac_evt_1", "expected_etag": etag, "title": "New title"},
    )
    approved = tools.approve_pending(queued["action"]["id"])

    assert approved["execution_failed"] is True
    assert approved["result"]["stale_event"] is True
