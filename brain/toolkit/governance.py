"""Tool permission, preview, audit, and undo helpers."""
from __future__ import annotations

import base64
import difflib
import json
import os
import threading
import uuid
from pathlib import Path

import config
from brain import store
from brain.embeddings import EmbeddingUnavailable, embed_document, text_only_embedding
from brain.memory import MemoryUnit, now_iso
from brain.toolkit import local_state, memory_tools, source_truth
from brain.toolkit.paths import resolve_tool_path, trusted_prompt_path, trusted_prompt_update_target
from brain.utils import file_lock

PERMISSIONS_FILE = config.DB_DIR / ".neural_map_tool_permissions.json"
AUDIT_FILE = config.DB_DIR / ".neural_map_tool_audit.json"
_LOCK = threading.Lock()


def _permissions_lock():
    """Cross-process exclusive lock for the permissions file.

    The threading.Lock only serializes threads within one process; permission
    updates are read-modify-write, so two processes (web server + background
    jobs, or multiple workers) that interleave read/write silently drop each
    other's changes. Same pattern as pending._state_lock()."""
    return file_lock(PERMISSIONS_FILE.with_name(".neural_map_tool_permissions.lock"))


def _audit_lock():
    """Cross-process exclusive lock for the audit log (append is read-modify-write)."""
    return file_lock(AUDIT_FILE.with_name(".neural_map_tool_audit.lock"))

TOOL_GROUPS = {
    "get_current_datetime": "basic",
    "will_status": "system_status",
    "calculate": "basic",
    "get_stock_quotes": "external_read",
    "web_search": "external_read",
    "list_files": "files",
    "read_file": "files",
    "write_file": "files",
    "update_source_of_truth": "source_of_truth",
    "update_will_perspective": "will_perspective",
    "review_action_package": "action_packages",
    "search_local_files": "files",
    "inspect_document": "files",
    "transcribe_media": "files",
    "execute_python_code": "sandbox",
    "send_email": "external_send",
    "create_reminder": "local_state",
    "list_reminders": "local_state",
    "complete_reminder": "local_state",
    "upsert_contact": "local_state",
    "lookup_contact": "local_state",
    "create_calendar_event": "local_state",
    "list_calendar_events": "local_state",
    "google_calendar_status": "google_calendar",
    "list_google_calendar_events": "google_calendar",
    "create_google_calendar_event": "google_calendar",
    "update_google_calendar_event": "google_calendar",
    "delete_google_calendar_event": "google_calendar",
    "record_lesson": "self_model",
    "update_memory": "memory",
    "pin_memory": "memory",
    "mark_memory_wrong": "memory",
    "merge_memories": "memory",
    "cite_sources": "memory",
    "git_status": "project",
    "git_log": "project",
    "git_diff": "project",
    "run_tests": "project",
}


def _empty_permissions() -> dict:
    return {"tools": {}, "updated_at": None}


def _read_json(path: Path, default: dict) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return default
    return data if isinstance(data, dict) else default


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def tool_enabled(name: str) -> bool:
    with _LOCK:
        data = _read_json(PERMISSIONS_FILE, _empty_permissions())
    return bool((data.get("tools") or {}).get(name, True))


def set_tool_permission(name: str, enabled: bool) -> dict:
    with _LOCK, _permissions_lock():
        data = _read_json(PERMISSIONS_FILE, _empty_permissions())
        data.setdefault("tools", {})[name] = bool(enabled)
        data["updated_at"] = now_iso()
        _write_json(PERMISSIONS_FILE, data)
    return {"tool": name, "enabled": bool(enabled), "updated_at": data["updated_at"]}


def list_permissions(tool_policies: dict[str, str]) -> dict:
    with _LOCK:
        data = _read_json(PERMISSIONS_FILE, _empty_permissions())
    overrides = data.get("tools") or {}
    tools = []
    for name in sorted(tool_policies):
        tools.append(
            {
                "tool": name,
                "policy": tool_policies[name],
                "group": TOOL_GROUPS.get(name, "other"),
                "enabled": bool(overrides.get(name, True)),
                "default_enabled": True,
            }
        )
    return {"tools": tools, "updated_at": data.get("updated_at")}


def append_audit(action: dict, decision: str, result: dict | None = None) -> dict:
    event = {
        "id": f"audit_{uuid.uuid4().hex[:12]}",
        "action_id": action.get("id"),
        "tool": action.get("tool"),
        "policy": action.get("policy"),
        "args": action.get("args") or {},
        "decision": decision,
        "result": result,
        "created_at": now_iso(),
    }
    with _LOCK, _audit_lock():
        data = _read_json(AUDIT_FILE, {"events": []})
        events = data.setdefault("events", [])
        events.append(event)
        del events[:-config.MAX_TOOL_AUDIT_EVENTS]
        _write_json(AUDIT_FILE, data)
    return event


def list_audit(limit: int = 100) -> list[dict]:
    with _LOCK:
        events = list((_read_json(AUDIT_FILE, {"events": []}).get("events") or []))
    events.sort(key=lambda e: e.get("created_at") or "", reverse=True)
    return events[: max(1, min(limit, config.MAX_TOOL_AUDIT_EVENTS))]


def _snippet(text: str | None, limit: int = 1000) -> str:
    text = text or ""
    return text[:limit] + ("..." if len(text) > limit else "")


def _file_text(path: Path) -> str | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:4096]:
        return None
    return raw.decode("utf-8", errors="replace")


def _file_snapshot(path: Path) -> dict:
    snap = {"kind": "file", "path": str(path), "exists": path.exists(), "undoable": True}
    if not path.exists():
        return snap
    if not path.is_file():
        return {**snap, "undoable": False, "reason": "path is not a file"}
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return {**snap, "undoable": False, "reason": str(exc)}
    if len(raw) > config.MAX_TOOL_UNDO_BYTES:
        return {**snap, "undoable": False, "reason": "file is too large to snapshot"}
    return {**snap, "bytes_b64": base64.b64encode(raw).decode("ascii")}


def _restore_file_snapshot(snapshot: dict) -> dict:
    path = Path(str(snapshot.get("path") or ""))
    if not snapshot.get("undoable", False):
        return {"error": snapshot.get("reason") or "file snapshot is not undoable"}
    if snapshot.get("exists"):
        raw = base64.b64decode(str(snapshot.get("bytes_b64") or ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return {"ok": True, "path": str(path), "restored": True}
    if path.exists():
        path.unlink()
    return {"ok": True, "path": str(path), "removed": True}


def _memory_model(args: dict | None = None) -> str | None:
    model = (args or {}).get("__model")
    if model is None:
        model = (args or {}).get("model")
    if model is None:
        return None
    return str(model).strip() or None


def _memory_records(ids: list[str], model: str | None = None) -> list[dict]:
    wanted = set(ids)
    return [r for r in store.list_records_all(model) if r.get("id") in wanted]


def _restore_memory_records(records: list[dict], model: str | None = None) -> dict:
    restored = []
    for rec in records:
        memory_id = str(rec.get("id") or "")
        if not memory_id:
            continue
        meta = rec.get("meta") or {}
        store.delete_all([memory_id], model=model)
        extra = {k: v for k, v in meta.items() if k not in memory_tools.BASE_META_KEYS}
        unit = MemoryUnit(
            id=memory_id,
            text=str(rec.get("text") or ""),
            category=str(meta.get("category") or "note")[: config.MAX_CATEGORY_CHARS],
            importance=meta.get("importance", 3),
            date=meta.get("date") or now_iso(),
            source=meta.get("source"),
            access_count=int(meta.get("access_count", 0) or 0),
            extra=extra,
        )
        try:
            embedding = embed_document(unit.text, model=model)
            store.add(unit, embedding, model=model)
        except (EmbeddingUnavailable, Exception):
            store.add(unit, text_only_embedding(), model=model, text_only=True)
        restored.append(memory_id)
    return {"ok": True, "restored": restored}


def _json_file_snapshot(path: Path) -> dict:
    snap = {"kind": "json_file", "path": str(path), "exists": path.exists(), "undoable": True}
    if path.exists():
        try:
            raw = path.read_bytes()
        except OSError as exc:
            return {**snap, "undoable": False, "reason": str(exc)}
        if len(raw) > config.MAX_TOOL_UNDO_BYTES:
            return {**snap, "undoable": False, "reason": "state file is too large to snapshot"}
        snap["bytes_b64"] = base64.b64encode(raw).decode("ascii")
    return snap


def _restore_json_file_snapshot(snapshot: dict) -> dict:
    return _restore_file_snapshot(snapshot)


def preview_action(tool: str, args: dict | None = None) -> dict:
    args = args or {}
    try:
        if tool == "write_file":
            path = resolve_tool_path(str(args.get("filename") or ""))
            if trusted_prompt_path(path):
                target = trusted_prompt_update_target(path)
                hint = (
                    f"use update_source_of_truth with target={target}"
                    if target
                    else "use its dedicated update tool"
                )
                return {
                    "kind": "file_write",
                    "path": str(path),
                    "blocked": f"trusted prompt file; write_file will refuse this path; {hint}",
                }
            mode = str(args.get("mode") or "create")
            content = "" if args.get("content") is None else str(args.get("content"))
            old = _file_text(path) if path.exists() and path.is_file() else ""
            new = old + content if mode == "append" else content
            diff = ""
            if old is not None:
                diff = "".join(
                    difflib.unified_diff(
                        old.splitlines(True),
                        new.splitlines(True),
                        fromfile=f"{path} (before)",
                        tofile=f"{path} (after)",
                        n=3,
                    )
                )
            return {"kind": "file_write", "path": str(path), "mode": mode, "chars": len(content), "diff": _snippet(diff, 6000)}
        if tool == "update_source_of_truth":
            path = source_truth.target_path(args)
            target = str(args.get("target") or "source_of_truth").strip().lower()
            old_text = str(args.get("old_text") or "")
            new_text = str(args.get("new_text") or "")
            current = _file_text(path)
            diff = ""
            matches = 0
            match_kind = None
            if current is not None and old_text:
                plan = source_truth.plan_replacement(current, old_text, new_text)
                matches = plan["matches"]
                match_kind = plan["match_kind"]
                candidate = plan["updated"]
                diff = "".join(
                    difflib.unified_diff(
                        current.splitlines(True),
                        candidate.splitlines(True),
                        fromfile=f"{path} (before)",
                        tofile=f"{path} (after)",
                        n=3,
                    )
                )
            return {
                "kind": "source_of_truth_update",
                "target": target,
                "path": str(path),
                "matches": matches,
                "match_kind": match_kind,
                "old_chars": len(old_text),
                "new_chars": len(new_text),
                "diff": _snippet(diff, 6000),
            }
        if tool == "update_will_perspective":
            path = config.WILL_PERSPECTIVE_PATH.resolve(strict=False)
            old_text = str(args.get("old_text") or "")
            new_text = str(args.get("new_text") or "")
            current = _file_text(path)
            diff = ""
            matches = 0
            match_kind = None
            if current is not None and old_text:
                plan = source_truth.plan_replacement(current, old_text, new_text)
                matches = plan["matches"]
                match_kind = plan["match_kind"]
                diff = "".join(
                    difflib.unified_diff(
                        current.splitlines(True),
                        plan["updated"].splitlines(True),
                        fromfile=f"{path} (before)",
                        tofile=f"{path} (after)",
                        n=3,
                    )
                )
            return {
                "kind": "will_perspective_update",
                "path": str(path),
                "matches": matches,
                "match_kind": match_kind,
                "old_chars": len(old_text),
                "new_chars": len(new_text),
                "reason": _snippet(str(args.get("reason") or ""), 600),
                "diff": _snippet(diff, 6000),
                "identity_level": True,
            }
        if tool == "send_email":
            return {
                "kind": "email",
                "recipient": args.get("recipient"),
                "subject": args.get("subject"),
                "body": _snippet(str(args.get("body") or ""), 4000),
                "will_send": bool(config.EMAIL_SEND_ENABLED),
            }
        if tool == "review_action_package":
            return {
                "kind": "action_package",
                "title": _snippet(str(args.get("title") or "Action Package"), 300),
                "body": _snippet(str(args.get("body") or ""), 8000),
                "source_initiative_id": args.get("source_initiative_id"),
                "priority": args.get("priority"),
                "will_execute_external_action": False,
            }
        if tool == "record_lesson":
            return {
                "kind": "lesson_record",
                "text": _snippet(str(args.get("text") or ""), 600),
                "source": args.get("source"),
                "note": "Approval commits this as a standing lesson in the self-model prompt.",
            }
        if tool == "create_reminder":
            return {"kind": "reminder_create", "action": args.get("action"), "due": args.get("due"), "importance": args.get("importance")}
        if tool == "complete_reminder":
            rec = memory_tools.memory_record(str(args.get("id") or ""))
            return {"kind": "reminder_complete", "id": args.get("id"), "before": rec}
        if tool == "upsert_contact":
            name = str(args.get("name") or "")
            cid = local_state._stable_id("contact", name)
            existing = [c for c in (local_state._read_json_file(local_state.CONTACTS_FILE).get("contacts") or []) if c.get("id") == cid]
            return {"kind": "contact_upsert", "before": existing[0] if existing else None, "after": args}
        if tool == "create_calendar_event":
            return {"kind": "calendar_event_create", "event": args}
        if tool == "create_google_calendar_event":
            return {
                "kind": "google_calendar_create",
                "calendar_id": args.get("calendar_id") or config.GOOGLE_CALENDAR_ID,
                "event": args,
                "external_write": True,
            }
        if tool == "update_google_calendar_event":
            changes = {key: value for key, value in args.items() if key not in {"event_id", "expected_etag", "expected_title", "calendar_id"}}
            return {
                "kind": "google_calendar_update",
                "calendar_id": args.get("calendar_id") or config.GOOGLE_CALENDAR_ID,
                "event_id": args.get("event_id"),
                "current_title": args.get("expected_title"),
                "changes": changes,
                "external_write": True,
            }
        if tool == "delete_google_calendar_event":
            return {
                "kind": "google_calendar_delete",
                "calendar_id": args.get("calendar_id") or config.GOOGLE_CALENDAR_ID,
                "event_id": args.get("event_id"),
                "current_title": args.get("expected_title"),
                "external_write": True,
                "destructive": True,
            }
        if tool in {"update_memory", "pin_memory", "mark_memory_wrong"}:
            model = _memory_model(args)
            rec = memory_tools.memory_record(str(args.get("id") or ""), model=model)
            return {"kind": tool, "before": rec, "after": args}
        if tool == "merge_memories":
            ids = [str(i) for i in (args.get("ids") or [])]
            model = _memory_model(args)
            return {
                "kind": "memory_merge",
                "before": _memory_records(ids, model=model),
                "after": {"text": args.get("text"), "category": args.get("category")},
            }
        if tool == "run_tests":
            return {"kind": "project_tests", "target": args.get("target") or "full suite"}
        return {"kind": "generic", "args": args}
    except Exception as exc:
        return {"kind": "preview_error", "error": str(exc), "args": args}


def snapshot_action(tool: str, args: dict | None = None) -> dict:
    args = args or {}
    try:
        if tool == "write_file":
            return _file_snapshot(resolve_tool_path(str(args.get("filename") or "")))
        if tool == "update_source_of_truth":
            return _file_snapshot(source_truth.target_path(args))
        if tool == "update_will_perspective":
            return _file_snapshot(config.WILL_PERSPECTIVE_PATH.resolve(strict=False))
        if tool in {"upsert_contact"}:
            return _json_file_snapshot(local_state.CONTACTS_FILE)
        if tool in {"create_calendar_event"}:
            return _json_file_snapshot(local_state.CALENDAR_FILE)
        if tool == "complete_reminder":
            return {"kind": "memory_records", "records": _memory_records([str(args.get("id") or "")]), "undoable": True}
        if tool in {"update_memory", "pin_memory", "mark_memory_wrong"}:
            model = _memory_model(args)
            return {
                "kind": "memory_records",
                "model": model,
                "records": _memory_records([str(args.get("id") or "")], model=model),
                "undoable": True,
            }
        if tool == "merge_memories":
            ids = [str(i) for i in (args.get("ids") or [])]
            model = _memory_model(args)
            return {"kind": "memory_merge", "model": model, "records": _memory_records(ids, model=model), "undoable": True}
        if tool in {"create_reminder", "review_action_package"}:
            return {"kind": "created_memory", "undoable": True}
        if tool == "record_lesson":
            return {"kind": "created_lesson", "undoable": True}
        return {"kind": "none", "undoable": False, "reason": "tool has no automatic undo"}
    except Exception as exc:
        return {"kind": "snapshot_error", "undoable": False, "reason": str(exc)}


# Tools whose preview/snapshot reflect the CURRENT state of their target — a
# file's contents, a memory record, a contact row. If that target changes while
# the action is pending, the reviewed preview is no longer what would execute.
# Tools absent here have previews that are pure functions of their args
# (create_reminder, send_email, google calendar tools carry their own etag
# preconditions), so there is nothing to go stale.
_FRESHNESS_TOOLS = {
    "write_file",
    "update_source_of_truth",
    "update_will_perspective",
    "complete_reminder",
    "upsert_contact",
    "update_memory",
    "pin_memory",
    "mark_memory_wrong",
    "merge_memories",
}
# upsert_contact is compared on preview only: its snapshot is the whole
# contacts file, and an unrelated concurrent contact must not invalidate this
# approval. The preview captures the one contact this action would overwrite.
_FRESHNESS_PREVIEW_ONLY = {"upsert_contact"}
# Bookkeeping bumped by ordinary retrieval (store.reinforce): a memory that was
# merely recalled since review is not stale, only one whose content changed.
_VOLATILE_META_KEYS = {"access_count", "last_accessed"}


def _stable_view(value):
    if isinstance(value, dict):
        return {k: _stable_view(v) for k, v in value.items() if k not in _VOLATILE_META_KEYS}
    if isinstance(value, list):
        return [_stable_view(v) for v in value]
    return value


def refresh_if_stale(action: dict) -> dict | None:
    """Refresh approval details when the reviewed target's state changed.

    State-dependent actions show the target's current content (diff, before
    record) at review time. If the target changes while the action is pending,
    the old preview is no longer what the user is approving. Keep the action
    pending with a fresh preview/snapshot and make the caller ask for approval
    again.
    """
    tool = str(action.get("tool") or "")
    if tool not in _FRESHNESS_TOOLS:
        return None
    args = action.get("args") or {}
    preview = preview_action(tool, args)
    snapshot = snapshot_action(tool, args)
    if tool in _FRESHNESS_PREVIEW_ONLY:
        fresh = _stable_view(preview)
        reviewed = _stable_view(action.get("preview") or {})
    else:
        fresh = (_stable_view(preview), _stable_view(snapshot))
        reviewed = (_stable_view(action.get("preview") or {}), _stable_view(action.get("snapshot") or {}))
    if fresh == reviewed:
        return None
    return {
        "preview": preview,
        "snapshot": snapshot,
        "stale_preview": True,
        "stale_at": now_iso(),
    }


def undo_action(action: dict) -> dict:
    if action.get("status") != "approved":
        return {"error": "only approved actions can be undone"}
    if action.get("undo_result"):
        return {"error": "action already has an undo result"}
    tool = str(action.get("tool") or "")
    result = action.get("result") or {}
    snapshot = action.get("approval_snapshot") or action.get("snapshot") or {}
    if tool == "update_will_perspective":
        restored = _restore_file_snapshot(snapshot)
        if restored.get("ok"):
            try:
                from brain import perspective

                old_hash = str(result.get("new_hash") or "unknown")
                new_hash = perspective.perspective_hash()
                perspective.append_identity_change(
                    old_hash,
                    new_hash,
                    "undo of approved perspective update",
                )
            except Exception:
                pass
        return restored
    if tool in {"write_file", "update_source_of_truth"}:
        return _restore_file_snapshot(snapshot)
    if tool in {"upsert_contact", "create_calendar_event"}:
        return _restore_json_file_snapshot(snapshot)
    if tool in {"complete_reminder", "update_memory", "pin_memory", "mark_memory_wrong"}:
        model = snapshot.get("model") if "model" in snapshot else _memory_model(action.get("args") or {})
        return _restore_memory_records(snapshot.get("records") or [], model=model)
    if tool == "merge_memories":
        model = snapshot.get("model") if "model" in snapshot else _memory_model(action.get("args") or {})
        merged_id = result.get("id")
        if merged_id:
            store.delete_all([str(merged_id)], model=model)
        restored = _restore_memory_records(snapshot.get("records") or [], model=model)
        return {**restored, "removed": [merged_id] if merged_id else []}
    if tool in {"create_reminder", "review_action_package"}:
        created_id = result.get("id")
        if not created_id:
            return {"error": "approved action did not record created memory id"}
        store.delete_all([str(created_id)])
        return {"ok": True, "removed": [created_id]}
    if tool == "record_lesson":
        created_id = result.get("id")
        if not created_id:
            return {"error": "approved action did not record created lesson id"}
        from brain import self_model

        return self_model.remove_lesson(str(created_id))
    return {"error": snapshot.get("reason") or "this action is not undoable"}
