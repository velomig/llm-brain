"""Regression coverage for resumable tasks and safe memory undo."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import config
from brain import llm, store, tasks, tools
from brain.toolkit import governance, memory_tools, pending
from util import store_memory


@pytest.fixture(autouse=True)
def task_state():
    tasks._STATE_FILE.unlink(missing_ok=True)
    yield
    tasks._STATE_FILE.unlink(missing_ok=True)


def completed(summary="Finished"):
    return json.dumps({"status": "completed", "summary": summary,
                       "evidence": ["Requested output observed"]})


def approve(tool, args):
    action = tools.dispatch(tool, args)["action"]
    result = tools.approve_pending(action["id"])
    assert result["action"]["status"] == "approved", result
    return action["id"], result["result"]


@pytest.mark.parametrize("text_only", [False, True])
def test_memory_undo_refuses_newer_correction(text_only):
    memory = store_memory("original", text_only=text_only)
    action_id, _ = approve("update_memory", {"id": memory.id, "text": "approved edit"})
    memory_tools.update_memory({"id": memory.id, "text": "newer correction"})
    assert "changed" in tools.undo_pending(action_id)["error"]
    assert memory_tools.memory_record(memory.id)["text"] == "newer correction"


@pytest.mark.parametrize("target", ["source", "merged"])
def test_merge_undo_refuses_newer_changes(target):
    first, second = store_memory("first"), store_memory("second")
    action_id, result = approve("merge_memories", {"ids": [first.id, second.id], "text": "merged"})
    edited_id = first.id if target == "source" else result["id"]
    memory_tools.update_memory({"id": edited_id, "text": "newer correction"})
    assert "changed" in tools.undo_pending(action_id)["error"]
    assert memory_tools.memory_record(edited_id)["text"] == "newer correction"
    assert memory_tools.memory_record(result["id"]) is not None


def test_created_reminder_undo_preserves_later_changes():
    action_id, result = approve("create_reminder", {"action": "test", "due": "2099-01-01"})
    store.update_metadata_all([result["id"]], {"status": "done"})
    assert tools.undo_pending(action_id).get("error")
    assert memory_tools.memory_record(result["id"]) is not None


def test_memory_undo_allows_recall_only_changes():
    memory = store_memory("original")
    action_id, _ = approve("update_memory", {"id": memory.id, "text": "approved edit"})
    store.reinforce_all([memory.id])
    assert tools.undo_pending(action_id)["result"]["ok"]
    assert memory_tools.memory_record(memory.id)["text"] == "original"


def test_legacy_memory_action_without_post_snapshot_fails_closed():
    memory = store_memory("original")
    action_id, _ = approve("pin_memory", {"id": memory.id, "pinned": True})
    pending.update_action(action_id, {"memory_post_snapshot": None})
    assert "snapshot" in tools.undo_pending(action_id)["error"]


def test_memory_writer_cannot_interleave_undo_check_and_restore(monkeypatch):
    memory = store_memory("original")
    action_id, _ = approve("update_memory", {"id": memory.id, "text": "approved edit"})
    restoring, writer_started, release = threading.Event(), threading.Event(), threading.Event()
    original_restore = governance._restore_memory_records

    def paused_restore(*args, **kwargs):
        restoring.set()
        assert release.wait(2)
        return original_restore(*args, **kwargs)

    def edit():
        writer_started.set()
        return memory_tools.update_memory({"id": memory.id, "text": "newer correction"})

    monkeypatch.setattr(governance, "_restore_memory_records", paused_restore)
    with ThreadPoolExecutor(max_workers=2) as pool:
        undo = pool.submit(tools.undo_pending, action_id)
        assert restoring.wait(2)
        writer = pool.submit(edit)
        try:
            assert writer_started.wait(2)
            assert not writer.done()
        finally:
            release.set()
        assert undo.result(timeout=2)["result"]["ok"]
        assert not writer.result(timeout=2).get("error")
    assert memory_tools.memory_record(memory.id)["text"] == "newer correction"


def expire_claim(task_id):
    with tasks._STATE_THREAD_LOCK, tasks._lock():
        rows = tasks._read_unlocked()
        tasks._patch_unlocked(rows, task_id, {"heartbeat_at": "2000-01-01T00:00:00+00:00"})
        tasks._write_unlocked(rows)


def test_old_worker_cannot_finish_or_dispatch_after_reclaim(monkeypatch):
    task = tasks.enqueue("Make a reminder")
    old = tasks._claim_next(task["id"])
    expire_claim(task["id"])
    new = tasks._claim_next(task["id"])
    assert old["attempt_id"] != new["attempt_id"]
    assert tasks._finish(task["id"], {"status": "done"}, old["attempt_id"]) is None
    assert tasks._finish(task["id"], {"status": "done"}) is None

    def stale_loop(system, user, specs, dispatch, **kwargs):
        dispatch("create_reminder", {"action": "test", "due": "2099-01-01"})

    monkeypatch.setattr(llm, "chat_with_tools", stale_loop)
    with pytest.raises(tasks.LostClaim):
        tasks._execute(old["prompt"], None, task=old)
    assert not pending.list_actions()
    assert tasks.get(task["id"])["attempt_id"] == new["attempt_id"]
    assert tasks.get(task["id"])["status"] == "running"


def test_heartbeat_refreshes_claim_during_model_work(monkeypatch):
    task = tasks.enqueue("Long model call")
    claimed = tasks._claim_next(task["id"])
    touched = threading.Event()
    original = tasks._finish

    def finish(*args, **kwargs):
        result = original(*args, **kwargs)
        if result:
            touched.set()
        return result

    monkeypatch.setattr(tasks, "_finish", finish)
    monkeypatch.setattr(config, "BACKGROUND_TASK_LEASE_SECONDS", 1)
    with tasks._heartbeat(claimed):
        assert touched.wait(2)
        live = tasks.get(task["id"])
        assert live["status"] == "running"
        assert live["attempt_id"] == claimed["attempt_id"]
        assert live["heartbeat_at"] != claimed["heartbeat_at"]


def test_exception_after_proposal_keeps_task_link_and_checkpoint(monkeypatch):
    task = tasks.enqueue("Make a reminder")

    def fail_after_proposal(system, user, specs, dispatch, **kwargs):
        dispatch("create_reminder", {"action": "test", "due": "2099-01-01"})
        raise RuntimeError("model disconnected")

    monkeypatch.setattr(llm, "chat_with_tools", fail_after_proposal)
    result = tasks.run_one()
    assert result["status"] == "awaiting_approval"
    assert len(result["pending_action_ids"]) == 1
    action = pending.get(result["pending_action_ids"][0])
    assert action["task_id"] == task["id"]
    assert result["observations"][0]["result"]["action"]["id"] == action["id"]


def test_crash_between_proposal_and_checkpoint_recovers_link():
    task = tasks.enqueue("Make a reminder")
    claimed = tasks._claim_next(task["id"])
    with pending.task_context(task["id"], claimed["attempt_id"]):
        action = tools.dispatch("create_reminder", {"action": "test", "due": "2099-01-01"})["action"]
    expire_claim(task["id"])
    recovered = tasks.get(task["id"])
    assert recovered["pending_action_ids"] == [action["id"]]
    assert recovered["status"] == "awaiting_approval"
    assert tasks._claim_next(task["id"]) is None


def test_resume_uses_approved_result_without_duplicate_proposal(monkeypatch):
    tasks.enqueue("Create a reminder and inspect its record")
    args = {"action": "test", "due": "2099-01-01"}

    def first_loop(system, user, specs, dispatch, **kwargs):
        dispatch("create_reminder", args)
        return "waiting", []

    monkeypatch.setattr(llm, "chat_with_tools", first_loop)
    first = tasks.run_one()
    action_id = first["pending_action_ids"][0]
    actual = tools.approve_pending(action_id)["result"]

    def second_loop(system, user, specs, dispatch, **kwargs):
        assert actual["id"] in kwargs["history"][0]["content"]
        replayed = dispatch("create_reminder", args)
        assert replayed["id"] == actual["id"]
        record = dispatch("cite_sources", {"memory_ids": [replayed["id"]]})
        assert record["sources"][0]["id"] == actual["id"]
        return completed(), []

    monkeypatch.setattr(llm, "chat_with_tools", second_loop)
    assert tasks.run_one()["status"] == "done"
    assert len(pending.list_actions(include_resolved=True)) == 1


@pytest.mark.parametrize("reply,expected", [
    ("I could not complete the task.", "blocked"), ("", "blocked"),
    ('{"status":"completed","summary":"done"}', "blocked"),
    ('{"status":"blocked","summary":"Need a filename"}', "blocked"),
    ('{"status":"failed","summary":"Read failed"}', "failed"),
])
def test_incomplete_reports_never_become_done(monkeypatch, reply, expected):
    tasks.enqueue("Do the requested work")
    monkeypatch.setattr(llm, "chat_with_tools", lambda *a, **k: (reply, []))
    result = tasks.run_one()
    assert result["status"] == expected
    assert result["error"]


@pytest.mark.parametrize("tool_result", [{"error": "failed"}, {"ok": False}, {"returncode": 1}])
def test_all_tools_failed_overrides_completion_claim(monkeypatch, tool_result):
    tasks.enqueue("Inspect the project")
    monkeypatch.setattr(tools, "dispatch", lambda *a, **k: tool_result)

    def loop(system, user, specs, dispatch, **kwargs):
        dispatch("git_status", {})
        return completed(), []

    monkeypatch.setattr(llm, "chat_with_tools", loop)
    result = tasks.run_one()
    assert result["status"] == "failed"
    assert result["observations"][0]["result"] == tool_result


def test_tool_iteration_limit_is_not_success(monkeypatch):
    tasks.enqueue("A multi-step task")
    monkeypatch.setattr(config, "TOOL_CALL_MAX_ITERS", 1)
    monkeypatch.setattr(llm, "_chat_options", lambda *a: {})

    def message(name, messages, options, tools=None):
        if tools:
            return "", [{"function": {"name": "get_current_datetime", "arguments": {}}}]
        return completed(), []

    monkeypatch.setattr(llm, "_chat_message", message)
    result = tasks.run_one()
    assert result["status"] == "budget_exhausted"
    assert result["stop_reason"] == "budget_exhausted"
    assert result["observations"]


def test_real_tool_loop_stops_before_dependent_action(monkeypatch):
    tasks.enqueue("Create then inspect a reminder")
    monkeypatch.setattr(llm, "_chat_options", lambda *a: {})
    calls = []

    def message(*args, **kwargs):
        calls.append(True)
        return "", [
            {"function": {"name": "create_reminder", "arguments": {"action": "test", "due": "2099-01-01"}}},
            {"function": {"name": "get_current_datetime", "arguments": {}}},
        ]

    monkeypatch.setattr(llm, "_chat_message", message)
    result = tasks.run_one()
    assert result["status"] == "awaiting_approval"
    assert len(calls) == 1
    assert len(result["observations"]) == 1


def test_explicit_retry_resumes_blocked_task_without_automatic_loop(monkeypatch):
    task = tasks.enqueue("Do the requested work")
    monkeypatch.setattr(llm, "chat_with_tools", lambda *a, **k: ("Cannot finish.", []))
    assert tasks.run_one()["status"] == "blocked"
    assert tasks.run_one() is None
    monkeypatch.setattr(llm, "chat_with_tools", lambda *a, **k: (completed(), []))
    assert tasks.run_one(task_id=task["id"])["status"] == "done"


def test_legacy_approval_resumes_using_saved_action_ids(monkeypatch):
    task = tasks.enqueue("Create a reminder")
    args = {"action": "test", "due": "2099-01-01"}
    action = tools.dispatch("create_reminder", args)["action"]
    tasks._finish(task["id"], {"status": "awaiting_approval", "pending_action_ids": [action["id"]]})
    actual = tools.approve_pending(action["id"])["result"]

    def loop(system, user, specs, dispatch, **kwargs):
        assert actual["id"] in kwargs["history"][0]["content"]
        assert dispatch("create_reminder", args)["id"] == actual["id"]
        return completed(), []

    monkeypatch.setattr(llm, "chat_with_tools", loop)
    assert tasks.run_one()["status"] == "done"
    assert len(pending.list_actions(include_resolved=True)) == 1


def test_resume_keeps_web_attempt_budget(monkeypatch):
    task = tasks.enqueue("Research Example City weather")
    monkeypatch.setattr(tools, "dispatch", lambda *a, **k: {"error": "provider unavailable"})

    def first(system, user, specs, dispatch, **kwargs):
        assert dispatch("web_search", {"query": "Example City weather"}).get("error")
        return '{"status":"blocked","summary":"Provider unavailable"}', []

    monkeypatch.setattr(llm, "chat_with_tools", first)
    assert tasks.run_one()["status"] == "blocked"
    monkeypatch.setattr(tools, "dispatch", lambda *a, **k: pytest.fail("search budget reset on resume"))

    def second(system, user, specs, dispatch, **kwargs):
        result = dispatch("web_search", {"query": "Example City weather"})
        assert result.get("error")
        return '{"status":"blocked","summary":"Search budget used"}', []

    monkeypatch.setattr(llm, "chat_with_tools", second)
    assert tasks.run_one(task_id=task["id"])["status"] == "blocked"
