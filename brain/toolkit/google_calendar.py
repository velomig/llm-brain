"""Google Calendar tools with local OAuth credentials.

Reads execute immediately. Mutations are registered with approval-requiring
policies in ``registry.py`` and therefore cannot reach these handlers until the
user approves the pending action.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import uuid
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import config
from brain.toolkit.email import EMAIL_RE

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
_TOKEN_LOCK = threading.Lock()
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_MACOS_JXA = r'''
const input = JSON.parse(__INPUT__);
const app = Application("Calendar");

function calendarFor(name) {
  const calendars = app.calendars();
  if (name) {
    const exact = calendars.find(c => c.name() === name);
    if (exact) return exact;
  }
  const emailNamed = calendars.filter(c => /^[^@\s]+@[^@\s]+$/.test(c.name()));
  if (emailNamed.length === 1) return emailNamed[0];
  return null;
}

function iso(value) {
  if (!value) return null;
  const parsed = new Date(value);
  return isNaN(parsed.getTime()) ? null : parsed.toISOString();
}

function eventDate(value) {
  const raw = String(value || "");
  return /^\d{4}-\d{2}-\d{2}$/.test(raw) ? new Date(raw + "T00:00:00") : new Date(raw);
}

function record(event) {
  let notes = null;
  try { notes = event.description(); } catch (_) {}
  return {
    id: event.uid(),
    status: "confirmed",
    title: event.summary() || "(untitled)",
    start: {dateTime: iso(event.startDate())},
    end: {dateTime: iso(event.endDate())},
    description: notes,
    location: event.location() || null,
    attendees: [],
    organizer: null,
    html_link: null,
    updated: null,
    all_day: !!event.alldayEvent()
  };
}

// The reviewed-event snapshot the caller captured when it verified the etag.
// Re-checking it here, inside the same osascript run as the mutation, is what
// makes the stale-event guard hold: a concurrent calendar change between the
// caller's read and this run flips a field and the mutation is refused.
function matchesExpected(event, expected) {
  if (!expected) return true;
  const current = record(event);
  return ["title", "start", "end", "description", "location", "all_day"].every(
    key => JSON.stringify(current[key] === undefined ? null : current[key]) ===
           JSON.stringify(expected[key] === undefined ? null : expected[key])
  );
}

const STALE = {error: "the calendar event changed after it was reviewed; read it again and request a new approval", stale_event: true};

const calendar = calendarFor(input.calendar_name || "");
if (!calendar) {
  JSON.stringify({error: "no unique email-named calendar was found; set NEURAL_MAP_GOOGLE_CALENDAR_MACOS_NAME"});
} else if (input.operation === "status") {
  JSON.stringify({connected: true, calendar_name: calendar.name()});
} else {
  const event = input.event_id ? calendar.events().find(e => e.uid() === input.event_id) : null;
  if (input.operation === "get") {
    JSON.stringify(event ? {event: record(event), calendar_name: calendar.name()} : {error: "event not found"});
  } else if (input.operation === "list") {
    const from = new Date(input.from);
    const to = new Date(input.to);
    const query = String(input.query || "").toLocaleLowerCase();
    const rows = calendar.events().filter(e => {
      const start = e.startDate();
      const text = String(e.summary() || "").toLocaleLowerCase();
      return start >= from && start <= to && (!query || text.includes(query));
    }).sort((a, b) => a.startDate() - b.startDate()).slice(0, input.max_results || 50).map(record);
    JSON.stringify({events: rows, calendar_name: calendar.name()});
  } else if (input.operation === "create") {
    const props = {
      summary: input.title,
      startDate: eventDate(input.start),
      endDate: eventDate(input.end),
      alldayEvent: /^\d{4}-\d{2}-\d{2}$/.test(String(input.start || ""))
    };
    if (input.description !== undefined) props.description = input.description || "";
    if (input.location !== undefined) props.location = input.location || "";
    const created = app.Event(props);
    calendar.events.push(created);
    JSON.stringify({event: record(created), calendar_name: calendar.name()});
  } else if (input.operation === "update") {
    if (!event) {
      JSON.stringify({error: "event not found"});
    } else if (!matchesExpected(event, input.expected)) {
      JSON.stringify(STALE);
    } else {
      if (input.title !== undefined) event.summary = input.title;
      if (input.description !== undefined) event.description = input.description || "";
      if (input.location !== undefined) event.location = input.location || "";
      if (input.start !== undefined) event.startDate = eventDate(input.start);
      if (input.end !== undefined) event.endDate = eventDate(input.end);
      JSON.stringify({event: record(event), calendar_name: calendar.name()});
    }
  } else if (input.operation === "delete") {
    if (!event) {
      JSON.stringify({error: "event not found"});
    } else if (!matchesExpected(event, input.expected)) {
      JSON.stringify(STALE);
    } else {
      app.delete(event);
      JSON.stringify({deleted: true, event_id: input.event_id, calendar_name: calendar.name()});
    }
  } else {
    JSON.stringify({error: "unsupported macOS Calendar operation"});
  }
}
'''


class GoogleCalendarUnavailable(RuntimeError):
    """Raised for missing setup/dependencies without exposing credential data."""


def _google_imports():
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise GoogleCalendarUnavailable(
            "Google Calendar dependencies are not installed; run "
            "./.venv/bin/python -m pip install -r requirements.txt"
        ) from exc
    return Request, Credentials, InstalledAppFlow, build


def _write_token(credentials) -> None:
    path = config.GOOGLE_CALENDAR_TOKEN_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(credentials.to_json(), encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def authorize_interactive(*, open_browser: bool = True) -> dict:
    """Run Google's installed-app OAuth flow and persist a private refresh token."""
    _, _, InstalledAppFlow, _ = _google_imports()
    secrets = config.GOOGLE_CALENDAR_CLIENT_SECRET_PATH
    if not secrets.is_file():
        raise GoogleCalendarUnavailable(
            f"OAuth client file not found at {secrets}. Download an OAuth Desktop app "
            "client JSON from Google Cloud and save it there."
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets), SCOPES)
    credentials = flow.run_local_server(
        host="127.0.0.1",
        port=0,
        open_browser=open_browser,
        authorization_prompt_message="Open Google authorization here:\n{url}",
        success_message="Google Calendar is connected. You may close this window.",
    )
    with _TOKEN_LOCK:
        _write_token(credentials)
    return {"connected": True, "token_path": str(config.GOOGLE_CALENDAR_TOKEN_PATH)}


def _credentials():
    Request, Credentials, _, _ = _google_imports()
    token = config.GOOGLE_CALENDAR_TOKEN_PATH
    if not token.is_file():
        raise GoogleCalendarUnavailable(
            "Google Calendar is not connected; run ./.venv/bin/python connect_google_calendar.py"
        )
    try:
        credentials = Credentials.from_authorized_user_file(str(token), SCOPES)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise GoogleCalendarUnavailable(
            "The Google Calendar token is unreadable; reconnect with connect_google_calendar.py"
        ) from exc
    if credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
        except Exception as exc:
            raise GoogleCalendarUnavailable(
                "Google Calendar authorization expired or was revoked; reconnect with "
                "connect_google_calendar.py"
            ) from exc
        with _TOKEN_LOCK:
            _write_token(credentials)
    if not credentials.valid:
        raise GoogleCalendarUnavailable(
            "Google Calendar authorization is invalid; reconnect with connect_google_calendar.py"
        )
    return credentials


def _service():
    _, _, _, build = _google_imports()
    return build("calendar", "v3", credentials=_credentials(), cache_discovery=False)


def _error(exc: Exception) -> dict:
    status = getattr(getattr(exc, "resp", None), "status", None)
    if status == 412:
        return {
            "error": "the Google Calendar event changed after it was reviewed; read it again and request a new approval",
            "stale_event": True,
        }
    if status in {401, 403}:
        return {
            "error": "Google Calendar authorization was rejected; reconnect or verify Calendar API access",
            "connection_required": True,
        }
    return {"error": f"Google Calendar request failed ({status or type(exc).__name__})"}


def _calendar_id(args: dict) -> str:
    return str(args.get("calendar_id") or config.GOOGLE_CALENDAR_ID or "primary").strip()


def _zone() -> ZoneInfo:
    try:
        return ZoneInfo(config.GOOGLE_CALENDAR_TIMEZONE)
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC")


def _rfc3339(value: str | None, default: datetime) -> str:
    raw = str(value or "").strip()
    if not raw:
        parsed = default
    elif _DATE_RE.fullmatch(raw):
        parsed = datetime.combine(date.fromisoformat(raw), time.min, tzinfo=_zone())
    else:
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("calendar bounds must be ISO dates or RFC3339 datetimes") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_zone())
    return parsed.isoformat()


def _event_time(value: str, timezone_name: str) -> dict:
    raw = str(value or "").strip()
    if _DATE_RE.fullmatch(raw):
        date.fromisoformat(raw)
        return {"date": raw}
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("event times must be ISO dates or RFC3339 datetimes") from exc
    result = {"dateTime": parsed.isoformat()}
    if parsed.tzinfo is None:
        try:
            ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone must be an IANA timezone name") from exc
        result["timeZone"] = timezone_name
    return result


def _attendees(value) -> list[dict]:
    rows = []
    for item in value or []:
        email = str(item or "").strip().lower()
        if not EMAIL_RE.match(email):
            raise ValueError(f"invalid attendee email: {email}")
        rows.append({"email": email})
    return rows


def _event_body(args: dict, *, partial: bool) -> dict:
    body: dict = {}
    if not partial or "title" in args:
        title = str(args.get("title") or "").strip()
        if not title:
            raise ValueError("title is required")
        body["summary"] = title[:300]
    if "description" in args:
        body["description"] = str(args.get("description") or "")[:12000]
    if "location" in args:
        body["location"] = str(args.get("location") or "")[:1000]
    changing_time = "start" in args or "end" in args
    if not partial or changing_time:
        if not args.get("start") or not args.get("end"):
            raise ValueError("start and end are both required when setting event time")
        timezone_name = str(args.get("timezone") or config.GOOGLE_CALENDAR_TIMEZONE)
        start = _event_time(str(args["start"]), timezone_name)
        end = _event_time(str(args["end"]), timezone_name)
        if ("date" in start) != ("date" in end):
            raise ValueError("start and end must both be dates or both be datetimes")
        body.update({"start": start, "end": end})
    if "attendees" in args:
        body["attendees"] = _attendees(args.get("attendees"))
    return body


def _event_record(event: dict) -> dict:
    all_day = bool(event.get("start", {}).get("date"))
    return {
        "id": event.get("id"),
        "etag": event.get("etag"),
        "status": event.get("status"),
        "title": event.get("summary") or "(untitled)",
        "start": _local_time_block(event.get("start"), all_day=all_day),
        "end": _local_time_block(event.get("end"), all_day=all_day),
        "description": event.get("description"),
        "location": event.get("location"),
        "attendees": [
            {key: row.get(key) for key in ("email", "displayName", "responseStatus") if row.get(key) is not None}
            for row in (event.get("attendees") or [])
        ],
        "organizer": event.get("organizer"),
        "html_link": event.get("htmlLink"),
        "updated": event.get("updated"),
    }


def _local_time_block(value: dict | None, *, all_day: bool = False) -> dict | None:
    """Normalize event timestamps before the model reasons about calendar dates."""
    if not isinstance(value, dict):
        return value
    if value.get("date"):
        return {"date": str(value["date"]), "timeZone": config.GOOGLE_CALENDAR_TIMEZONE}
    raw = str(value.get("dateTime") or "").strip()
    if not raw:
        return value
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_zone())
    local = parsed.astimezone(_zone())
    if all_day:
        return {"date": local.date().isoformat(), "timeZone": config.GOOGLE_CALENDAR_TIMEZONE}
    return {"dateTime": local.isoformat(), "timeZone": config.GOOGLE_CALENDAR_TIMEZONE}


def _google_connection_status(args: dict | None = None) -> dict:
    if not config.GOOGLE_CALENDAR_TOKEN_PATH.is_file():
        return {
            "connected": False,
            "connect_command": "./.venv/bin/python connect_google_calendar.py",
            "client_secret_path": str(config.GOOGLE_CALENDAR_CLIENT_SECRET_PATH),
        }
    try:
        service = _service()
        service.events().list(calendarId=_calendar_id(args or {}), maxResults=1).execute()
    except GoogleCalendarUnavailable as exc:
        return {"connected": False, "error": str(exc)}
    except Exception as exc:
        return {"connected": False, **_error(exc)}
    return {"connected": True, "calendar_id": _calendar_id(args or {})}


def _google_list_events(args: dict | None = None) -> dict:
    args = args or {}
    now = datetime.now(timezone.utc)
    try:
        kwargs = {
            "calendarId": _calendar_id(args),
            "timeMin": _rfc3339(args.get("from_date"), now - timedelta(days=1)),
            "timeMax": _rfc3339(args.get("to_date"), now + timedelta(days=30)),
            "singleEvents": True,
            "orderBy": "startTime",
            "maxResults": min(100, max(1, int(args.get("max_results") or 50))),
        }
        if str(args.get("query") or "").strip():
            kwargs["q"] = str(args["query"]).strip()[:500]
        result = _service().events().list(**kwargs).execute()
        return {
            "calendar_id": _calendar_id(args),
            "events": [_event_record(row) for row in result.get("items") or []],
            "next_page_token": result.get("nextPageToken"),
        }
    except (GoogleCalendarUnavailable, ValueError) as exc:
        return {"error": str(exc), "connection_required": isinstance(exc, GoogleCalendarUnavailable)}
    except Exception as exc:
        return _error(exc)


def _google_create_event(args: dict) -> dict:
    try:
        body = _event_body(args or {}, partial=False)
        event = _service().events().insert(
            calendarId=_calendar_id(args or {}),
            body=body,
            sendUpdates="all" if body.get("attendees") else "none",
        ).execute()
        return {"created": True, "calendar_id": _calendar_id(args or {}), "event": _event_record(event)}
    except (GoogleCalendarUnavailable, ValueError) as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return _error(exc)


def _conditional(request, etag: str):
    if not etag:
        raise ValueError("expected_etag is required; read the event immediately before changing it")
    request.headers["If-Match"] = etag
    return request


def _google_update_event(args: dict) -> dict:
    args = args or {}
    event_id = str(args.get("event_id") or "").strip()
    if not event_id:
        return {"error": "event_id is required"}
    try:
        body = _event_body(args, partial=True)
        if not body:
            raise ValueError("at least one event field must be changed")
        request = _service().events().patch(
            calendarId=_calendar_id(args),
            eventId=event_id,
            body=body,
            sendUpdates="all" if "attendees" in body else "none",
        )
        event = _conditional(request, str(args.get("expected_etag") or "")).execute()
        return {"updated": True, "calendar_id": _calendar_id(args), "event": _event_record(event)}
    except (GoogleCalendarUnavailable, ValueError) as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return _error(exc)


def _google_delete_event(args: dict) -> dict:
    args = args or {}
    event_id = str(args.get("event_id") or "").strip()
    if not event_id:
        return {"error": "event_id is required"}
    try:
        request = _service().events().delete(
            calendarId=_calendar_id(args), eventId=event_id, sendUpdates="all"
        )
        _conditional(request, str(args.get("expected_etag") or "")).execute()
        return {
            "deleted": True,
            "calendar_id": _calendar_id(args),
            "event_id": event_id,
            "title": str(args.get("expected_title") or ""),
        }
    except (GoogleCalendarUnavailable, ValueError) as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return _error(exc)


def _provider() -> str:
    configured = str(config.GOOGLE_CALENDAR_PROVIDER or "auto").strip().lower()
    if configured in {"google", "macos"}:
        return configured
    return "google" if config.GOOGLE_CALENDAR_TOKEN_PATH.is_file() else "macos"


def _run_macos(payload: dict) -> dict:
    script = _MACOS_JXA.replace("__INPUT__", json.dumps(json.dumps(payload)))
    try:
        completed = subprocess.run(
            ["/usr/bin/osascript", "-l", "JavaScript", "-e", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": f"macOS Calendar request failed ({type(exc).__name__})"}
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().lower()
        if "not authorized" in detail or "not permitted" in detail:
            return {"error": "macOS Calendar access is not allowed; enable Calendar/Automation access in System Settings", "connection_required": True}
        return {"error": "macOS Calendar request failed"}
    try:
        result = json.loads((completed.stdout or "").strip())
    except (TypeError, json.JSONDecodeError):
        return {"error": "macOS Calendar returned an invalid response"}
    return result if isinstance(result, dict) else {"error": "macOS Calendar returned an invalid response"}


def _mac_payload(operation: str, args: dict | None = None) -> dict:
    return {
        "operation": operation,
        "calendar_name": str((args or {}).get("calendar_id") or config.GOOGLE_CALENDAR_MACOS_NAME or "").strip(),
    }


def _mac_etag(event: dict) -> str:
    selected = {key: event.get(key) for key in ("id", "title", "start", "end", "description", "location", "all_day")}
    digest = hashlib.sha256(json.dumps(selected, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:24]
    return f'"macos-{digest}"'


def _mac_record(event: dict) -> dict:
    normalized = {
        **event,
        "start": _local_time_block(event.get("start"), all_day=bool(event.get("all_day"))),
        "end": _local_time_block(event.get("end"), all_day=bool(event.get("all_day"))),
    }
    return {**normalized, "etag": _mac_etag(normalized), "provider": "macos_calendar"}


def _mac_status(args: dict | None = None) -> dict:
    result = _run_macos(_mac_payload("status", args))
    if result.get("error"):
        return {"connected": False, **result, "provider": "macos_calendar"}
    return {**result, "provider": "macos_calendar", "calendar_id": result.get("calendar_name")}


def _mac_list(args: dict | None = None) -> dict:
    args = args or {}
    now = datetime.now(timezone.utc)
    try:
        payload = {
            **_mac_payload("list", args),
            "from": _rfc3339(args.get("from_date"), now - timedelta(days=1)),
            "to": _rfc3339(args.get("to_date"), now + timedelta(days=30)),
            "query": str(args.get("query") or "")[:500],
            "max_results": min(100, max(1, int(args.get("max_results") or 50))),
        }
    except ValueError as exc:
        return {"error": str(exc)}
    result = _run_macos(payload)
    if result.get("error"):
        return result
    return {
        "calendar_id": result.get("calendar_name"),
        "provider": "macos_calendar",
        "timeZone": config.GOOGLE_CALENDAR_TIMEZONE,
        "events": [_mac_record(event) for event in result.get("events") or []],
        "next_page_token": None,
    }


# Fields the macOS etag is computed over (minus the id, which the mutation
# already selects by). The reviewed values are re-checked inside the mutation's
# own osascript run, so a concurrent change between the etag pre-check below
# and the mutation cannot be silently overwritten.
_MAC_EXPECTED_KEYS = ("title", "start", "end", "description", "location", "all_day")


def _mac_check_etag(event_id: str, args: dict) -> tuple[dict | None, dict]:
    """Verify the reviewed etag and return the raw current event.

    This read gives a friendly early error, but it is NOT the guard: the
    returned raw fields travel with the mutation payload and the JXA script
    compares them against the live event in the same osascript run before
    changing anything (see matchesExpected). Raw-field equality implies etag
    equality, since the etag is a pure function of these fields."""
    expected = str(args.get("expected_etag") or "")
    if not expected:
        return {"error": "expected_etag is required; read the event immediately before changing it"}, {}
    result = _run_macos({**_mac_payload("get", args), "event_id": event_id})
    if result.get("error"):
        return result, {}
    raw = result.get("event") or {}
    if _mac_record(raw).get("etag") != expected:
        return {"error": "the calendar event changed after it was reviewed; read it again and request a new approval", "stale_event": True}, {}
    return None, raw


def _mac_create(args: dict) -> dict:
    if args.get("attendees"):
        return {"error": "attendee invitations are not supported by the macOS Calendar bridge"}
    try:
        _event_body(args or {}, partial=False)
    except ValueError as exc:
        return {"error": str(exc)}
    payload = {**_mac_payload("create", args)}
    for key in ("title", "start", "end", "description", "location"):
        if key in args:
            payload[key] = args.get(key)
    result = _run_macos(payload)
    if result.get("error"):
        return result
    return {"created": True, "calendar_id": result.get("calendar_name"), "provider": "macos_calendar", "event": _mac_record(result.get("event") or {})}


def _mac_update(args: dict) -> dict:
    event_id = str((args or {}).get("event_id") or "").strip()
    if not event_id:
        return {"error": "event_id is required"}
    if args.get("attendees") is not None:
        return {"error": "attendee changes are not supported by the macOS Calendar bridge"}
    stale, reviewed = _mac_check_etag(event_id, args)
    if stale:
        return stale
    try:
        body = _event_body(args or {}, partial=True)
    except ValueError as exc:
        return {"error": str(exc)}
    if not body:
        return {"error": "at least one event field must be changed"}
    payload = {
        **_mac_payload("update", args),
        "event_id": event_id,
        "expected": {key: reviewed.get(key) for key in _MAC_EXPECTED_KEYS},
    }
    for key in ("title", "start", "end", "description", "location"):
        if key in args:
            payload[key] = args.get(key)
    result = _run_macos(payload)
    if result.get("error"):
        return result
    return {"updated": True, "calendar_id": result.get("calendar_name"), "provider": "macos_calendar", "event": _mac_record(result.get("event") or {})}


def _mac_delete(args: dict) -> dict:
    event_id = str((args or {}).get("event_id") or "").strip()
    if not event_id:
        return {"error": "event_id is required"}
    stale, reviewed = _mac_check_etag(event_id, args)
    if stale:
        return stale
    result = _run_macos({
        **_mac_payload("delete", args),
        "event_id": event_id,
        "expected": {key: reviewed.get(key) for key in _MAC_EXPECTED_KEYS},
    })
    if result.get("error"):
        return result
    return {**result, "calendar_id": result.get("calendar_name"), "provider": "macos_calendar", "title": str(args.get("expected_title") or "")}


def connection_status(args: dict | None = None) -> dict:
    return _google_connection_status(args) if _provider() == "google" else _mac_status(args)


def list_events(args: dict | None = None) -> dict:
    return _google_list_events(args) if _provider() == "google" else _mac_list(args)


def create_event(args: dict) -> dict:
    return _google_create_event(args) if _provider() == "google" else _mac_create(args)


def update_event(args: dict) -> dict:
    return _google_update_event(args) if _provider() == "google" else _mac_update(args)


def delete_event(args: dict) -> dict:
    return _google_delete_event(args) if _provider() == "google" else _mac_delete(args)
