"""Local reminders, contacts, and calendar-style events."""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import config
from brain import store
from brain.embeddings import EmbeddingUnavailable, embed_document, text_only_embedding
from brain.memory import MemoryUnit, now_iso
from brain.toolkit.email import EMAIL_RE
from brain.utils import file_lock, replace_private

CONTACTS_FILE = config.DB_DIR / ".neural_map_contacts.json"
CALENDAR_FILE = config.DB_DIR / ".neural_map_calendar.json"

# Writers and undo share reentrant thread/process locks. Plain reads see atomic files.


def _contacts_lock():
    return file_lock(CONTACTS_FILE.with_name(".neural_map_contacts.lock"))


def _calendar_lock():
    return file_lock(CALENDAR_FILE.with_name(".neural_map_calendar.lock"))


def _store_memory_unit(unit: MemoryUnit, model: str | None = None) -> tuple[str, str]:
    try:
        embedding = embed_document(unit.text, model=model)
    except EmbeddingUnavailable:
        store.add(unit, text_only_embedding(), model=model, text_only=True)
        return unit.id, "text_only"
    store.add(unit, embedding, model=model)
    return unit.id, "vector"


def create_reminder(args: dict) -> dict:
    action = str((args or {}).get("action", "")).strip()
    due = str((args or {}).get("due", "")).strip()
    importance = int((args or {}).get("importance", 5) or 5)
    if not action:
        return {"error": "action is required"}
    try:
        parsed_due = datetime.fromisoformat(due).date().isoformat()
    except ValueError:
        return {"error": "due must be an ISO date, YYYY-MM-DD"}
    unit = MemoryUnit(
        text=f"[due {parsed_due}] {action}",
        category="reminder",
        importance=importance,
        extra={"due": parsed_due, "status": "open", "source_type": "tool_created"},
    )
    try:
        mid, mode = _store_memory_unit(unit)
    except Exception as exc:
        return {"error": str(exc)}
    return {"id": mid, "due": parsed_due, "text": unit.text, "mode": mode}


def list_reminders(args: dict | None = None) -> dict:
    include_done = bool((args or {}).get("include_done", False))
    rows = []
    today = date.today()
    for rec in store.list_records_all():
        meta = rec.get("meta") or {}
        if meta.get("category") != "reminder":
            continue
        status = meta.get("status") or "open"
        if status != "open" and not include_done:
            continue
        due_raw = meta.get("due")
        try:
            due_date = datetime.fromisoformat(str(due_raw)).date() if due_raw else None
        except ValueError:
            due_date = None
        rows.append(
            {
                "id": rec["id"],
                "text": rec["text"],
                "due": due_raw,
                "status": status,
                "overdue": bool(due_date and due_date < today),
                "mode": rec.get("mode"),
            }
        )
    rows.sort(key=lambda r: (r.get("status") != "open", r.get("due") or "9999", r.get("id") or ""))
    return {"reminders": rows}


def complete_reminder(args: dict) -> dict:
    reminder_id = str((args or {}).get("id", "")).strip()
    if not reminder_id:
        return {"error": "id is required"}
    n = store.update_metadata_all([reminder_id], {"status": "done", "completed_at": now_iso()})
    if n <= 0:
        return {"id": reminder_id, "error": "reminder not found"}
    return {"id": reminder_id, "status": "done"}


def _read_json_file(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json_file(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    replace_private(path, json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True))


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.strip().lower().encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def upsert_contact(args: dict) -> dict:
    name = str((args or {}).get("name", "")).strip()
    if not name:
        return {"error": "name is required"}
    email = str((args or {}).get("email", "")).strip()
    if email and not EMAIL_RE.match(email):
        return {"error": "invalid email"}
    cid = _stable_id("contact", name)
    contact = {
        "id": cid,
        "name": name[:160],
        "email": email[:240],
        "phone": str((args or {}).get("phone", "")).strip()[:80],
        "notes": str((args or {}).get("notes", "")).strip()[:1000],
        "updated_at": now_iso(),
    }
    with _contacts_lock():
        data = _read_json_file(CONTACTS_FILE)
        contacts = data.setdefault("contacts", [])
        contacts[:] = [c for c in contacts if c.get("id") != cid]
        contacts.append(contact)
        _write_json_file(CONTACTS_FILE, data)
    return contact


def lookup_contact(args: dict) -> dict:
    query = str((args or {}).get("query", "")).strip().casefold()
    if not query:
        return {"error": "query is required"}
    data = _read_json_file(CONTACTS_FILE)
    rows = []
    for contact in data.get("contacts") or []:
        blob = " ".join(str(contact.get(k) or "") for k in ("name", "email", "phone", "notes")).casefold()
        if query in blob:
            rows.append(contact)
    rows.sort(key=lambda c: str(c.get("name") or "").lower())
    return {"contacts": rows[:20]}


def _parse_event_dt(value: str | None, default: datetime | None = None) -> datetime | None:
    if not value:
        return default
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(value + "T00:00:00")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def create_calendar_event(args: dict) -> dict:
    title = str((args or {}).get("title", "")).strip()
    start_dt = _parse_event_dt(str((args or {}).get("start", "")).strip())
    if not title:
        return {"error": "title is required"}
    if start_dt is None:
        return {"error": "start must be an ISO date or datetime"}
    end_raw = str((args or {}).get("end", "")).strip()
    end_dt = _parse_event_dt(end_raw, start_dt + timedelta(hours=1)) if end_raw else start_dt + timedelta(hours=1)
    if end_dt is None:
        return {"error": "end must be an ISO date or datetime"}
    if end_dt < start_dt:
        return {"error": "end must be after start"}
    eid = _stable_id("event", f"{title}:{start_dt.isoformat()}")
    event = {
        "id": eid,
        "title": title[:200],
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "notes": str((args or {}).get("notes", "")).strip()[:1000],
        "updated_at": now_iso(),
    }
    with _calendar_lock():
        data = _read_json_file(CALENDAR_FILE)
        events = data.setdefault("events", [])
        events[:] = [e for e in events if e.get("id") != eid]
        events.append(event)
        _write_json_file(CALENDAR_FILE, data)
    return event


def list_calendar_events(args: dict | None = None) -> dict:
    args = args or {}
    start = _parse_event_dt(args.get("from_date"), datetime.now(timezone.utc) - timedelta(days=1))
    end = _parse_event_dt(args.get("to_date"), datetime.now(timezone.utc) + timedelta(days=30))
    data = _read_json_file(CALENDAR_FILE)
    rows = []
    for event in data.get("events") or []:
        event_start = _parse_event_dt(event.get("start"))
        if event_start is None:
            continue
        if start and event_start < start:
            continue
        if end and event_start > end:
            continue
        rows.append(event)
    rows.sort(key=lambda e: e.get("start") or "")
    return {"events": rows}
