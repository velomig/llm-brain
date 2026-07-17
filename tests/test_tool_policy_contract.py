"""Metadata contract for the tool registry — the approval gatekeeper.

The runtime independently classifies each registered operation through the
trusted capability manifest. This reviewed policy map remains a third copy so
a single registry/manifest edit cannot silently reclassify a tool.
"""
from __future__ import annotations

import re

import pytest

import config
from brain.toolkit import governance, registry
from brain.toolkit.registry import DESTRUCTIVE, EXTERNAL_SEND, EXTERNAL_WRITE, READ_ONLY, WRITE_SAFE

# The reviewed label for every registered tool. Adding a tool to HANDLERS
# without deciding its row here fails the suite — that is the point.
EXPECTED_POLICIES = {
    "get_current_datetime": READ_ONLY,
    "will_status": READ_ONLY,
    "calculate": READ_ONLY,
    "get_stock_quotes": READ_ONLY,
    "web_search": READ_ONLY,
    "list_files": READ_ONLY,
    "read_file": READ_ONLY,
    "search_local_files": READ_ONLY,
    "inspect_document": READ_ONLY,
    "transcribe_media": READ_ONLY,
    "execute_python_code": READ_ONLY,
    "send_email": EXTERNAL_SEND,
    "write_file": WRITE_SAFE,
    "update_source_of_truth": WRITE_SAFE,
    "update_will_perspective": WRITE_SAFE,
    "review_action_package": WRITE_SAFE,
    "create_reminder": WRITE_SAFE,
    "list_reminders": READ_ONLY,
    "complete_reminder": WRITE_SAFE,
    "upsert_contact": WRITE_SAFE,
    "lookup_contact": READ_ONLY,
    "create_calendar_event": WRITE_SAFE,
    "list_calendar_events": READ_ONLY,
    "google_calendar_status": READ_ONLY,
    "list_google_calendar_events": READ_ONLY,
    "create_google_calendar_event": EXTERNAL_WRITE,
    "update_google_calendar_event": EXTERNAL_WRITE,
    "delete_google_calendar_event": DESTRUCTIVE,
    "record_lesson": WRITE_SAFE,
    "update_memory": WRITE_SAFE,
    "pin_memory": WRITE_SAFE,
    "mark_memory_wrong": DESTRUCTIVE,
    "merge_memories": DESTRUCTIVE,
    "cite_sources": READ_ONLY,
    "git_status": READ_ONLY,
    "git_log": READ_ONLY,
    "git_diff": READ_ONLY,
    "run_tests": WRITE_SAFE,
}

# Tools that irreversibly discard or supersede user data. Anything on this
# list labeled WRITE_SAFE (or weaker) is exactly the mislabel this contract
# exists to catch.
DESTRUCTIVE_CAPABILITY = {"mark_memory_wrong", "merge_memories", "delete_google_calendar_event"}

# Tool names whose verb implies mutation. A READ_ONLY label on a matching
# name needs an entry in the reviewed-exceptions map below.
_MUTATING_NAME = re.compile(
    r"(?:^|_)(?:write|update|delete|remove|merge|create|send|complete|upsert|"
    r"record|pin|mark|set|execute|run|prune)(?:_|$)"
)
_REVIEWED_READ_ONLY_EXCEPTIONS = {
    # AST-whitelisted subprocess sandbox: imports, open(), and non-math
    # attribute access are rejected before execution, and rlimits cap
    # cpu/memory/output — it can compute, not persist.
    "execute_python_code",
}


def test_policy_labels_match_reviewed_map():
    actual = {name: tool.policy for name, tool in registry.HANDLERS.items()}
    assert actual == EXPECTED_POLICIES


def test_runtime_capability_manifest_matches_reviewed_map():
    derived = {
        name: registry._CAPABILITY_POLICY[capability]
        for name, capability in registry.TRUSTED_CAPABILITY_MANIFEST.items()
    }
    assert derived == EXPECTED_POLICIES
    assert registry.validate_registry() == []


def test_registration_rejects_destructive_tool_borrowing_write_safe_label():
    def wrapped_delete(args):
        return {"deleted": True}

    with pytest.raises(registry.ToolMetadataConflict, match="registration denied"):
        registry.register_tool(
            "delete_google_calendar_event",
            registry.ToolDef(wrapped_delete, WRITE_SAFE),
            entries={},
        )

    event = governance.list_audit(limit=1)[0]
    assert event["decision"] == "arbiter_conflict"
    assert event["tool"] == "delete_google_calendar_event"
    assert event["result"]["conflict"]["effective_policy"] == DESTRUCTIVE


def test_dispatch_denies_mislabeled_high_risk_handler_before_it_runs(monkeypatch):
    calls = []

    def wrapped_delete(args):
        calls.append(args)
        return {"deleted": True}

    monkeypatch.setitem(
        registry.HANDLERS,
        "delete_google_calendar_event",
        registry.ToolDef(wrapped_delete, WRITE_SAFE),
    )

    result = registry.dispatch(
        "delete_google_calendar_event",
        {"event_id": "event-1", "expected_etag": "v1", "expected_title": "Important"},
    )

    assert result["kind"] == "arbiter_conflict"
    assert result["error_kind"] == "arbiter_conflict"
    assert result["conflict"] == {
        "type": "tool_metadata_integrity",
        "phase": "dispatch",
        "tool": "delete_google_calendar_event",
        "declared_policy": WRITE_SAFE,
        "effective_policy": DESTRUCTIVE,
        "capability": registry.CAPABILITY_DESTRUCTIVE,
        "reason": "declared policy contradicts independently classified capability",
    }
    assert calls == []


def test_execution_rechecks_metadata_after_approval_was_queued(monkeypatch):
    queued = registry.dispatch(
        "delete_google_calendar_event",
        {"event_id": "event-1", "expected_etag": "v1", "expected_title": "Important"},
    )
    calls = []

    def wrapped_delete(args):
        calls.append(args)
        return {"deleted": True}

    # Simulate registry drift after the correctly labeled action was reviewed.
    monkeypatch.setitem(
        registry.HANDLERS,
        "delete_google_calendar_event",
        registry.ToolDef(wrapped_delete, WRITE_SAFE),
    )

    result = registry.approve_pending(queued["action"]["id"])

    assert result["execution_failed"] is True
    assert result["action"]["status"] == "failed"
    assert result["result"]["kind"] == "arbiter_conflict"
    assert result["result"]["conflict"]["phase"] == "execution"
    assert calls == []


def test_every_policy_is_a_known_label():
    known = {READ_ONLY, WRITE_SAFE, DESTRUCTIVE, EXTERNAL_SEND, EXTERNAL_WRITE}
    unknown = {
        name: tool.policy
        for name, tool in registry.HANDLERS.items()
        if tool.policy not in known
    }
    assert unknown == {}   # a typo'd label would silently skip the approval set


def test_destructive_capability_tools_are_labeled_destructive():
    mislabeled = {
        name: registry.HANDLERS[name].policy
        for name in DESTRUCTIVE_CAPABILITY
        if registry.HANDLERS[name].policy != DESTRUCTIVE
    }
    assert mislabeled == {}


def test_mutating_named_tools_are_not_read_only():
    suspicious = {
        name: tool.policy
        for name, tool in registry.HANDLERS.items()
        if tool.policy == READ_ONLY
        and _MUTATING_NAME.search(name)
        and name not in _REVIEWED_READ_ONLY_EXCEPTIONS
    }
    assert suspicious == {}


def test_requires_approval_follows_policy(monkeypatch):
    monkeypatch.setattr(config, "EMAIL_SEND_ENABLED", True)
    for name, tool in registry.HANDLERS.items():
        assert registry.requires_approval(name) == (tool.policy != READ_ONLY), name


def test_send_email_downgrades_to_read_only_when_sending_disabled(monkeypatch):
    monkeypatch.setattr(config, "EMAIL_SEND_ENABLED", False)
    assert registry.policy_for("send_email") == READ_ONLY
    assert not registry.requires_approval("send_email")


def test_specs_and_groups_cover_every_handler():
    handler_names = set(registry.HANDLERS)
    spec_names = {spec["function"]["name"] for spec in registry.TOOL_SPECS}
    assert spec_names == handler_names          # every tool is declared to the model
    assert set(governance.TOOL_GROUPS) == handler_names   # and grouped for the permissions UI


# Read tools whose results put stored or external text in front of the model
# must be in pipeline._UNTRUSTED_READ_TOOLS, so that actions proposed after
# reading them carry taint and cannot take the blind chat-approval shortcut.
# The reviewed exceptions below are the only reads whose output cannot carry
# attacker-controlled text.
_UNTAINTED_READ_EXCEPTIONS = {
    "get_current_datetime",    # clock only
    "calculate",               # arithmetic over model-typed literals
    "execute_python_code",     # no-import sandbox: output derives only from the args
    "web_search",              # taints via explicit dispatch handling in both tool loops
    "google_calendar_status",  # connection flag plus the user's own calendar id
}


def test_text_returning_read_tools_are_in_the_taint_set():
    from brain import pipeline

    read_only = {name for name, tool in registry.HANDLERS.items() if tool.policy == READ_ONLY}
    untainted = read_only - _UNTAINTED_READ_EXCEPTIONS - pipeline._UNTRUSTED_READ_TOOLS
    # git_log was the canary: commit messages are attacker-influenced text, so
    # an action proposed after reading them must not be blind-approvable.
    assert untainted == set()


def test_approval_tools_have_reviewable_previews(monkeypatch):
    """Every tool that queues for approval must render a real preview — the
    approval flow is only as safe as what the approver can see."""
    monkeypatch.setattr(config, "EMAIL_SEND_ENABLED", True)
    minimal_args = {
        "write_file": {"filename": "preview-contract-probe.md", "content": ""},
    }
    for name, tool in registry.HANDLERS.items():
        if tool.policy == READ_ONLY:
            continue
        preview = governance.preview_action(name, minimal_args.get(name, {}))
        assert preview.get("kind") not in {"generic", "preview_error"}, (name, preview)
