"""Tool schemas, policies, and handler registry."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable

import config
from brain import log
from brain.execution import context_aware_execution
from brain.memory import now_iso
from brain.toolkit import action_packages, basic, email, files, google_calendar, governance, lessons, local_state, memory_tools, pending, perspective_update, project, sandbox, source_truth, system_status

READ_ONLY = "read_only"
WRITE_SAFE = "write_safe"
DESTRUCTIVE = "destructive"
EXTERNAL_SEND = "external_send"
EXTERNAL_WRITE = "external_write"


@dataclass(frozen=True)
class ToolDef:
    handler: Callable[[dict], dict]
    policy: str


class ToolMetadataConflict(RuntimeError):
    """A registered tool's safety label contradicts its trusted capability profile."""


# This is deliberately independent of ToolDef.policy.  ToolDef is mutable at
# the registry boundary (plugins/tests can replace an entry), so accepting its
# label as its own proof of safety would let a dangerous operation borrow a
# weaker label.  The manifest is grouped by effective capability and frozen at
# import; policy is derived from capability rather than copied from the entry.
CAPABILITY_READ = "read_or_compute"
CAPABILITY_LOCAL_WRITE = "local_write"
CAPABILITY_EXTERNAL_WRITE = "external_write"
CAPABILITY_EXTERNAL_SEND = "external_send"
CAPABILITY_DESTRUCTIVE = "destructive"

_TRUSTED_CAPABILITY_GROUPS = {
    CAPABILITY_READ: frozenset(
        {
            "get_current_datetime",
            "will_status",
            "calculate",
            "get_stock_quotes",
            "web_search",
            "list_files",
            "read_file",
            "search_local_files",
            "inspect_document",
            "transcribe_media",
            "execute_python_code",
            "list_reminders",
            "lookup_contact",
            "list_calendar_events",
            "google_calendar_status",
            "list_google_calendar_events",
            "cite_sources",
            "git_status",
            "git_log",
            "git_diff",
        }
    ),
    CAPABILITY_LOCAL_WRITE: frozenset(
        {
            "write_file",
            "update_source_of_truth",
            "update_will_perspective",
            "review_action_package",
            "create_reminder",
            "complete_reminder",
            "upsert_contact",
            "create_calendar_event",
            "record_lesson",
            "update_memory",
            "pin_memory",
            "run_tests",
        }
    ),
    CAPABILITY_EXTERNAL_WRITE: frozenset(
        {"create_google_calendar_event", "update_google_calendar_event"}
    ),
    CAPABILITY_EXTERNAL_SEND: frozenset({"send_email"}),
    CAPABILITY_DESTRUCTIVE: frozenset(
        {"delete_google_calendar_event", "mark_memory_wrong", "merge_memories"}
    ),
}


def _build_capability_manifest() -> MappingProxyType:
    manifest: dict[str, str] = {}
    for capability, names in _TRUSTED_CAPABILITY_GROUPS.items():
        duplicates = set(names) & set(manifest)
        if duplicates:
            raise RuntimeError(f"tools have multiple trusted capabilities: {sorted(duplicates)}")
        manifest.update({name: capability for name in names})
    return MappingProxyType(manifest)


TRUSTED_CAPABILITY_MANIFEST = _build_capability_manifest()

_CAPABILITY_POLICY = MappingProxyType(
    {
        CAPABILITY_READ: READ_ONLY,
        CAPABILITY_LOCAL_WRITE: WRITE_SAFE,
        CAPABILITY_EXTERNAL_WRITE: EXTERNAL_WRITE,
        CAPABILITY_EXTERNAL_SEND: EXTERNAL_SEND,
        CAPABILITY_DESTRUCTIVE: DESTRUCTIVE,
    }
)


def _spec(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    params = {"type": "object", "properties": properties or {}}
    if required:
        params["required"] = required
    return {"type": "function", "function": {"name": name, "description": description, "parameters": params}}


TOOL_SPECS = [
    _spec("get_current_datetime", "Get the current local date and time."),
    _spec("will_status", "Get authoritative runtime status for Will's latest scheduled or manual review, including its timestamp and outcome."),
    _spec("calculate", "Evaluate an exact arithmetic expression.", {"expression": {"type": "string"}}, ["expression"]),
    _spec("get_stock_quotes", "Get current market prices for ticker symbols.", {"tickers": {"type": "array", "items": {"type": "string"}}}, ["tickers"]),
    _spec(
        "web_search",
        "Search the public web for any user-requested topic, including general research, news, products, travel, technical information, and current facts. The query must name the user's actual topic.",
        {"query": {"type": "string"}},
        ["query"],
    ),
    _spec("list_files", "List files and folders inside allowed local roots.", {"path": {"type": "string"}}, ["path"]),
    _spec("read_file", "Read a UTF-8-ish text file from allowed local roots.", {"filename": {"type": "string"}}, ["filename"]),
    _spec("execute_python_code", "Run a small sandboxed Python script.", {"script": {"type": "string"}}, ["script"]),
    _spec("send_email", "Draft or send a plain-text email when SMTP sending is configured.", {"recipient": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, ["recipient", "subject", "body"]),
    _spec("write_file", "Create, overwrite, or append a UTF-8 text file inside allowed roots.", {"filename": {"type": "string"}, "content": {"type": "string"}, "mode": {"type": "string", "enum": ["create", "overwrite", "append"]}}, ["filename", "content"]),
    _spec(
        "update_source_of_truth",
        "Patch the trusted Source of Truth or a domain calibration file by replacing exact old_text with new_text. Use target=finance for finance.md or target=nursing for nursing.md; omit target for the main Source of Truth. Requires approval and shows a diff.",
        {
            "target": {"type": "string", "enum": ["source_of_truth", "nursing", "finance"]},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
        },
        ["old_text", "new_text"],
    ),
    _spec("update_will_perspective", "Propose a small identity-level refinement to Will's core perspective by replacing exact old_text with new_text. Requires approval, shows a diff, and journals approved changes.", {"old_text": {"type": "string"}, "new_text": {"type": "string"}, "reason": {"type": "string"}}, ["old_text", "new_text"]),
    _spec("review_action_package", "Queue or record a review-only Action Package. Approval stores the package locally and does not execute any external action.", {"title": {"type": "string"}, "body": {"type": "string"}, "source_initiative_id": {"type": "string"}, "priority": {"type": "integer"}}, ["title", "body"]),
    _spec("search_local_files", "Search text snippets in local workspace files on disk under an allowed root. This does not search Neural Map memory-manager notes.", {"query": {"type": "string"}, "path": {"type": "string"}, "max_results": {"type": "integer"}}, ["query"]),
    _spec("inspect_document", "Read or search a local text/markdown/PDF document.", {"filename": {"type": "string"}, "query": {"type": "string"}}, ["filename"]),
    _spec("transcribe_media", "Extract available local audio metadata or transcript-like summary.", {"filename": {"type": "string"}}, ["filename"]),
    _spec("create_reminder", "Create a dated reminder in Neural Map memory.", {"action": {"type": "string"}, "due": {"type": "string"}, "importance": {"type": "integer"}}, ["action", "due"]),
    _spec("list_reminders", "List reminders.", {"include_done": {"type": "boolean"}}),
    _spec("complete_reminder", "Mark a reminder done by id.", {"id": {"type": "string"}}, ["id"]),
    _spec("upsert_contact", "Create or update a local contact.", {"name": {"type": "string"}, "email": {"type": "string"}, "phone": {"type": "string"}, "notes": {"type": "string"}}, ["name"]),
    _spec("lookup_contact", "Search local contacts.", {"query": {"type": "string"}}, ["query"]),
    _spec("create_calendar_event", "Create a private local calendar-style event.", {"title": {"type": "string"}, "start": {"type": "string"}, "end": {"type": "string"}, "notes": {"type": "string"}}, ["title", "start"]),
    _spec("list_calendar_events", "List private local calendar-style events.", {"from_date": {"type": "string"}, "to_date": {"type": "string"}}),
    _spec("google_calendar_status", "Check whether Will is connected to Google Calendar.", {"calendar_id": {"type": "string"}}),
    _spec(
        "list_google_calendar_events",
        "Read live Google Calendar events, normalized to America/Phoenix local time. Use this, not list_calendar_events, whenever the user refers to Google Calendar or their real schedule. Returned event ids and etags are required for safe updates/deletes.",
        {"from_date": {"type": "string"}, "to_date": {"type": "string"}, "query": {"type": "string"}, "max_results": {"type": "integer"}, "calendar_id": {"type": "string"}},
    ),
    _spec(
        "create_google_calendar_event",
        "Propose creating a real Google Calendar event. This always waits for explicit user approval before writing.",
        {"title": {"type": "string"}, "start": {"type": "string"}, "end": {"type": "string"}, "timezone": {"type": "string"}, "description": {"type": "string"}, "location": {"type": "string"}, "attendees": {"type": "array", "items": {"type": "string"}}, "calendar_id": {"type": "string"}},
        ["title", "start", "end"],
    ),
    _spec(
        "update_google_calendar_event",
        "Propose changing a real Google Calendar event. First read it with list_google_calendar_events, then pass its current id, etag, and title. This always waits for explicit user approval.",
        {"event_id": {"type": "string"}, "expected_etag": {"type": "string"}, "expected_title": {"type": "string"}, "title": {"type": "string"}, "start": {"type": "string"}, "end": {"type": "string"}, "timezone": {"type": "string"}, "description": {"type": "string"}, "location": {"type": "string"}, "attendees": {"type": "array", "items": {"type": "string"}}, "calendar_id": {"type": "string"}},
        ["event_id", "expected_etag", "expected_title"],
    ),
    _spec(
        "delete_google_calendar_event",
        "Propose deleting a real Google Calendar event. First read it and pass its current id, etag, and title. This always waits for explicit user approval.",
        {"event_id": {"type": "string"}, "expected_etag": {"type": "string"}, "expected_title": {"type": "string"}, "calendar_id": {"type": "string"}},
        ["event_id", "expected_etag", "expected_title"],
    ),
    _spec("record_lesson", "Propose a durable lesson-learned for the self-model. Requires approval; only an approved lesson is committed.", {"text": {"type": "string"}, "source": {"type": "string"}}, ["text"]),
    _spec("update_memory", "Update the text/category/importance of an existing memory.", {"id": {"type": "string"}, "text": {"type": "string"}, "category": {"type": "string"}, "importance": {"type": "integer"}}, ["id", "text"]),
    _spec("pin_memory", "Pin or unpin a memory.", {"id": {"type": "string"}, "pinned": {"type": "boolean"}}, ["id"]),
    _spec("mark_memory_wrong", "Retire a memory that is wrong or stale.", {"id": {"type": "string"}, "reason": {"type": "string"}}, ["id"]),
    _spec("merge_memories", "Merge several memories and supersede the originals.", {"ids": {"type": "array", "items": {"type": "string"}}, "text": {"type": "string"}, "category": {"type": "string"}}, ["ids", "text"]),
    _spec("cite_sources", "Search Neural Map memories by query, or return provenance for specific memory ids. Use this for pinned memories and memory-manager notes.", {"memory_ids": {"type": "array", "items": {"type": "string"}}, "query": {"type": "string"}}),
    _spec("git_status", "Show concise git status."),
    _spec("git_log", "Show recent commit history (one line per commit), optionally scoped to one file.", {"count": {"type": "integer"}, "path": {"type": "string"}}),
    _spec("git_diff", "Show git diff, optionally scoped to one file.", {"path": {"type": "string"}}),
    _spec("run_tests", "Run the project pytest suite or a specific target.", {"target": {"type": "string"}}),
]

# SECURITY-SENSITIVE: these are declared labels, not proof of capability.
# TRUSTED_CAPABILITY_MANIFEST independently derives the required label and is
# checked at registration, dispatch, and execution. Any edit here (new tool,
# changed policy) must update that manifest and the reviewed contract test.
HANDLERS: dict[str, ToolDef] = {
    "get_current_datetime": ToolDef(basic.current_datetime, READ_ONLY),
    "will_status": ToolDef(system_status.will_status, READ_ONLY),
    "calculate": ToolDef(basic.calculate, READ_ONLY),
    "get_stock_quotes": ToolDef(basic.get_stock_quotes, READ_ONLY),
    "web_search": ToolDef(basic.web_search, READ_ONLY),
    "list_files": ToolDef(files.list_files, READ_ONLY),
    "read_file": ToolDef(files.read_file, READ_ONLY),
    "search_local_files": ToolDef(files.search_local_files, READ_ONLY),
    "inspect_document": ToolDef(files.inspect_document, READ_ONLY),
    "transcribe_media": ToolDef(files.transcribe_media, READ_ONLY),
    "execute_python_code": ToolDef(sandbox.execute_python_code, READ_ONLY),
    "send_email": ToolDef(email.send_email, EXTERNAL_SEND),
    "write_file": ToolDef(files.write_file, WRITE_SAFE),
    "update_source_of_truth": ToolDef(source_truth.update_source_of_truth, WRITE_SAFE),
    "update_will_perspective": ToolDef(perspective_update.update_will_perspective, WRITE_SAFE),
    "review_action_package": ToolDef(action_packages.review_action_package, WRITE_SAFE),
    "create_reminder": ToolDef(local_state.create_reminder, WRITE_SAFE),
    "list_reminders": ToolDef(local_state.list_reminders, READ_ONLY),
    "complete_reminder": ToolDef(local_state.complete_reminder, WRITE_SAFE),
    "upsert_contact": ToolDef(local_state.upsert_contact, WRITE_SAFE),
    "lookup_contact": ToolDef(local_state.lookup_contact, READ_ONLY),
    "create_calendar_event": ToolDef(local_state.create_calendar_event, WRITE_SAFE),
    "list_calendar_events": ToolDef(local_state.list_calendar_events, READ_ONLY),
    "google_calendar_status": ToolDef(google_calendar.connection_status, READ_ONLY),
    "list_google_calendar_events": ToolDef(google_calendar.list_events, READ_ONLY),
    "create_google_calendar_event": ToolDef(google_calendar.create_event, EXTERNAL_WRITE),
    "update_google_calendar_event": ToolDef(google_calendar.update_event, EXTERNAL_WRITE),
    "delete_google_calendar_event": ToolDef(google_calendar.delete_event, DESTRUCTIVE),
    "record_lesson": ToolDef(lessons.record_lesson, WRITE_SAFE),
    "update_memory": ToolDef(memory_tools.update_memory, WRITE_SAFE),
    "pin_memory": ToolDef(memory_tools.pin_memory, WRITE_SAFE),
    "mark_memory_wrong": ToolDef(memory_tools.mark_memory_wrong, DESTRUCTIVE),
    "merge_memories": ToolDef(memory_tools.merge_memories, DESTRUCTIVE),
    "cite_sources": ToolDef(memory_tools.cite_sources, READ_ONLY),
    "git_status": ToolDef(project.git_status, READ_ONLY),
    "git_log": ToolDef(project.git_log, READ_ONLY),
    "git_diff": ToolDef(project.git_diff, READ_ONLY),
    "run_tests": ToolDef(project.run_tests, WRITE_SAFE),
}


def _metadata_conflict(
    name: str,
    entry: ToolDef | None,
    *,
    phase: str,
) -> dict | None:
    capability = TRUSTED_CAPABILITY_MANIFEST.get(name)
    declared_policy = entry.policy if entry is not None else None

    if capability is None:
        reason = "tool has no trusted capability profile"
        effective_policy = None
    elif entry is None:
        reason = "trusted capability profile has no registered tool"
        effective_policy = _CAPABILITY_POLICY[capability]
    else:
        effective_policy = _CAPABILITY_POLICY[capability]
        if declared_policy == effective_policy:
            return None
        reason = "declared policy contradicts independently classified capability"

    return {
        "type": "tool_metadata_integrity",
        "phase": phase,
        "tool": name,
        "declared_policy": declared_policy,
        "effective_policy": effective_policy,
        "capability": capability,
        "reason": reason,
    }


def _conflict_result(conflict: dict, *, emit_alert: bool) -> dict:
    result = {
        "error": f"tool metadata integrity conflict: {conflict['tool']} was denied",
        "error_kind": "arbiter_conflict",
        "kind": "arbiter_conflict",
        "conflict": conflict,
    }
    if not emit_alert:
        return result

    # Never include tool arguments here: the integrity alert needs identity and
    # policy provenance, not potentially sensitive payload data.
    log.fallback(
        "tool_metadata_integrity_conflict",
        level=logging.CRITICAL,
        phase=conflict.get("phase"),
        tool=conflict.get("tool"),
        declared_policy=conflict.get("declared_policy"),
        effective_policy=conflict.get("effective_policy"),
        capability=conflict.get("capability"),
    )
    try:
        governance.append_audit(
            {
                "id": None,
                "tool": conflict.get("tool"),
                "policy": conflict.get("declared_policy"),
                "args": {},
            },
            "arbiter_conflict",
            result,
        )
    except Exception as exc:
        # Alert persistence may fail, but denial must remain fail-closed.
        log.fallback(
            "tool_metadata_integrity_audit_failed",
            exc,
            level=logging.CRITICAL,
            tool=conflict.get("tool"),
        )
    return result


def validate_registry(entries: dict[str, ToolDef] | None = None) -> list[dict]:
    """Return registration conflicts against the independent capability manifest."""
    entries = HANDLERS if entries is None else entries
    names = set(entries) | set(TRUSTED_CAPABILITY_MANIFEST)
    return [
        conflict
        for name in sorted(names)
        if (conflict := _metadata_conflict(name, entries.get(name), phase="registration"))
        is not None
    ]


def _assert_registry_integrity(entries: dict[str, ToolDef] | None = None) -> None:
    conflicts = validate_registry(entries)
    if not conflicts:
        return
    for conflict in conflicts:
        _conflict_result(conflict, emit_alert=True)
    tools = ", ".join(str(conflict.get("tool")) for conflict in conflicts)
    raise ToolMetadataConflict(f"tool registration denied by metadata sanity check: {tools}")


def register_tool(name: str, entry: ToolDef, entries: dict[str, ToolDef] | None = None) -> None:
    """Register one tool only when its label matches its trusted capability."""
    target = HANDLERS if entries is None else entries
    conflict = _metadata_conflict(name, entry, phase="registration")
    if conflict is not None:
        _conflict_result(conflict, emit_alert=True)
        raise ToolMetadataConflict(
            f"tool registration denied by metadata sanity check: {name}"
        )
    target[name] = entry


def _runtime_metadata_guard(name: str, *, phase: str) -> dict | None:
    entry = HANDLERS.get(name)
    if entry is None and name not in TRUSTED_CAPABILITY_MANIFEST:
        return None
    conflict = _metadata_conflict(name, entry, phase=phase)
    return _conflict_result(conflict, emit_alert=True) if conflict is not None else None


# Static HANDLERS construction is the application's registration phase.  A bad
# checked-in label prevents the registry from loading instead of becoming an
# authorization bypass.  Runtime guards below catch later replacement/drift.
_assert_registry_integrity()

ACTION_TOOL_NAMES = set(HANDLERS) - {"get_current_datetime", "calculate", "get_stock_quotes", "web_search"}


def active_tool_specs() -> list[dict]:
    # Revalidate immediately before exposing operations to a model. This also
    # catches registry mutation that occurred after module import.
    _assert_registry_integrity()
    specs = [
        spec
        for spec in TOOL_SPECS
        if governance.tool_enabled(spec.get("function", {}).get("name", ""))
    ]
    if config.ACTION_TOOLS_ENABLED:
        return specs
    return [spec for spec in specs if spec.get("function", {}).get("name") not in ACTION_TOOL_NAMES]


def policy_for(name: str, args: dict | None = None) -> str:
    if name == "send_email" and not config.EMAIL_SEND_ENABLED:
        return READ_ONLY
    entry = HANDLERS.get(name)
    return entry.policy if entry else READ_ONLY


def requires_approval(name: str, args: dict | None = None) -> bool:
    # Callers that ask only this predicate still fail closed. dispatch() and
    # execute() return the richer arbiter_conflict before taking any action.
    if _metadata_conflict(name, HANDLERS.get(name), phase="policy_check") is not None:
        return True
    return policy_for(name, args) in {WRITE_SAFE, DESTRUCTIVE, EXTERNAL_SEND, EXTERNAL_WRITE}


@context_aware_execution(track_context=False)
def execute(name: str, args: dict | None = None) -> dict:
    entry = HANDLERS.get(name)
    if entry is None:
        return {"error": f"unknown tool: {name}"}
    conflict = _runtime_metadata_guard(name, phase="execution")
    if conflict is not None:
        return conflict
    if not governance.tool_enabled(name):
        return {"error": f"tool is disabled: {name}"}
    if name in ACTION_TOOL_NAMES and not config.ACTION_TOOLS_ENABLED:
        return {"error": "action tools are disabled"}
    return entry.handler(args or {})


@context_aware_execution(track_context=False)
def dispatch(name: str, args: dict | None = None, taint: list[str] | None = None) -> dict:
    """Run a tool, or queue it for approval.

    `taint` names the untrusted content in context when the model asked for
    this call (retrieved memories, web results, earlier file reads this turn).
    It is recorded on any pending action so approval flows can treat a
    possibly-injected proposal differently from a clean one.
    """
    if name not in HANDLERS:
        return {"error": f"unknown tool: {name}"}
    conflict = _runtime_metadata_guard(name, phase="dispatch")
    if conflict is not None:
        return conflict
    if not governance.tool_enabled(name):
        return {"error": f"tool is disabled: {name}"}
    if name in ACTION_TOOL_NAMES and not config.ACTION_TOOLS_ENABLED:
        return {"error": "action tools are disabled"}
    if requires_approval(name, args):
        preview = governance.preview_action(name, args or {})
        if preview.get("blocked"):
            return {
                "error": preview["blocked"],
                "path": preview.get("path"),
                "error_kind": "blocked_preflight",
            }
        action = pending.create(
            name,
            args or {},
            policy_for(name, args),
            preview=preview,
            snapshot=governance.snapshot_action(name, args or {}),
            taint=taint,
        )
        return {"pending": True, "action": action, "message": f"{name} requires approval before it runs"}
    return execute(name, args or {})


def approve_pending(action_id: str) -> dict:
    action = pending.get(action_id)
    if action is None:
        return {"error": "pending action not found"}
    if action.get("status") != "pending":
        return {"error": f"pending action is already {action.get('status')}"}
    # Atomically claim the action FIRST. Without this, two racing approvals
    # (double-click, retried request) both pass the status check above and both
    # execute — a duplicated external send. Only the claim winner runs. Claiming
    # before the freshness check also means no other approval's side effects can
    # slip between that check and execution; only a non-approval writer can
    # still race the (now milliseconds-wide) check-to-execute window.
    if pending.claim(action_id, {"approval_started_at": now_iso()}) is None:
        current = pending.get(action_id) or {}
        return {"error": f"pending action is already {current.get('status', 'resolved')}"}
    refreshed = governance.refresh_if_stale(action)
    if refreshed is not None:
        # Release the claim: the action stays pending with a fresh preview and
        # must be reviewed and approved again.
        updated = pending.update_action(
            action_id, {**refreshed, "status": "pending", "approval_started_at": None}
        ) or action
        return {
            "error": "pending action preview changed; review the updated preview and approve again",
            "error_kind": "stale_preview",
            "action": updated,
        }
    # Refuse an action whose reviewed preview could not display a change-defining
    # field in full. The whole taint/approval design funnels risky actions to the
    # Actions panel *so the reviewer can see them*; approving one whose preview was
    # cut short would let an injection-crafted payload hide past the visible
    # portion. Release the claim so it stays pending rather than wedged.
    if governance.preview_truncated(action):
        pending.update_action(action_id, {"status": "pending", "approval_started_at": None})
        return {
            "error": (
                "pending action preview is too large to display in full, so it cannot be "
                "approved as-is; inspect the target directly or split the action into smaller "
                "changes"
            ),
            "error_kind": "preview_truncated",
            "action": pending.get(action_id) or action,
        }
    snapshot = governance.snapshot_action(action.get("tool", ""), action.get("args") or {})
    pending.update_action(action_id, {"approval_snapshot": snapshot})
    try:
        result = execute(action.get("tool", ""), action.get("args") or {})
    except Exception:
        # Leave a terminal record rather than a wedged 'approving' action; do
        # not fall back to 'pending', which would invite a retry of a call that
        # may have partially taken effect.
        pending.resolve(action_id, "failed")
        raise
    execution_failed = bool(
        isinstance(result, dict)
        and (result.get("error") or result.get("returncode") not in (None, 0))
    )
    if execution_failed:
        pending.resolve(action_id, "failed", result)
        resolved = pending.get(action_id) or {}
        governance.append_audit(resolved, "failed", result)
        return {"action": resolved, "result": result, "execution_failed": True}
    pending.resolve(action_id, "approved", result)
    resolved = pending.get(action_id) or {}
    governance.append_audit(resolved, "approved", result)
    return {"action": resolved, "result": result}


def reject_pending(action_id: str) -> dict:
    # Reject only from "pending", atomically: an unconditional resolve here
    # could overwrite an action a racing approver already executed, recording
    # a completed side effect as "rejected" and breaking undo (which requires
    # "approved").
    action = pending.resolve(action_id, "rejected", expected_status="pending")
    if action is None:
        current = pending.get(action_id)
        if current is None:
            return {"error": "pending action not found"}
        return {"error": f"pending action is already {current.get('status')}", "action": current}
    governance.append_audit(action, "rejected")
    return {"action": action}


def undo_pending(action_id: str) -> dict:
    action = pending.get(action_id)
    if action is None:
        return {"error": "pending action not found"}
    result = governance.undo_action(action)
    if result.get("error"):
        return result
    updated = pending.record_undo(action_id, result)
    governance.append_audit(updated or action, "undone", result)
    return {"action": updated, "result": result}


def list_audit(limit: int = 100) -> list[dict]:
    return governance.list_audit(limit=limit)


def list_permissions() -> dict:
    return governance.list_permissions({name: policy_for(name) for name in HANDLERS})


def set_permission(name: str, enabled: bool) -> dict:
    if name not in HANDLERS:
        return {"error": f"unknown tool: {name}"}
    return governance.set_tool_permission(name, enabled)
