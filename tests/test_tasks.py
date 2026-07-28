"""Background task runner: graduated-trust autonomy.

The runner must (a) auto-run read-only tools, (b) NEVER auto-execute a
write/send/destructive tool — instead parking it in the pending-approval queue,
(c) survive an exploding tool loop, and (d) claim each task exactly once.
"""
from __future__ import annotations

import pytest

import config
from brain import tasks
from brain.toolkit import pending


@pytest.fixture(autouse=True)
def _clean_task_state():
    """The task queue lives in its own JSON file, which the shared conftest
    fixture doesn't wipe. Start every test with an empty queue."""
    tasks._STATE_FILE.unlink(missing_ok=True)
    yield
    tasks._STATE_FILE.unlink(missing_ok=True)


def test_enqueue_creates_queued_task():
    t = tasks.enqueue("Summarize the finance notes.")
    assert t["status"] == "queued"
    assert t["prompt"] == "Summarize the finance notes."
    assert tasks.get(t["id"])["status"] == "queued"


def test_empty_prompt_rejected():
    try:
        tasks.enqueue("   ")
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty prompt")


def test_read_only_task_runs_and_completes(monkeypatch):
    tasks.enqueue("What time is it?")

    def fake_loop(system, user, specs, dispatch, **kwargs):
        result = dispatch("get_current_datetime", {})
        assert "iso" in result  # read-only tool actually executed
        return "It is now.", [{"tool": "get_current_datetime"}]

    monkeypatch.setattr(tasks.llm, "chat_with_tools", fake_loop)

    done = tasks.run_one()
    assert done["status"] == "done"
    assert done["tools_used"] == ["get_current_datetime"]
    assert done["pending_action_ids"] == []
    assert done["result"] == "It is now."


def test_write_tool_is_parked_not_executed(monkeypatch):
    tasks.enqueue("Remind me to call SSA tomorrow.")

    def fake_loop(system, user, specs, dispatch, **kwargs):
        result = dispatch("create_reminder", {"action": "call SSA", "due": "2099-01-01"})
        # A WRITE_SAFE tool must come back as pending, never executed.
        assert result.get("pending") is True
        return "Drafted a reminder; it needs your approval.", []

    monkeypatch.setattr(tasks.llm, "chat_with_tools", fake_loop)

    before = len(pending.list_actions(include_resolved=False))
    done = tasks.run_one()
    assert done["status"] == "awaiting_approval"
    assert len(done["pending_action_ids"]) == 1
    # The parked action exists in the approval queue and is still 'pending'.
    after = pending.list_actions(include_resolved=False)
    assert len(after) == before + 1
    parked = pending.get(done["pending_action_ids"][0])
    assert parked["status"] == "pending"
    assert parked["tool"] == "create_reminder"


def test_run_one_on_empty_queue_returns_none(monkeypatch):
    monkeypatch.setattr(tasks.llm, "chat_with_tools", lambda *a, **k: ("", []))
    assert tasks.run_one() is None


def test_task_failure_is_recorded(monkeypatch):
    t = tasks.enqueue("Do the thing.")

    def boom(*a, **k):
        raise RuntimeError("model down")

    monkeypatch.setattr(tasks.llm, "chat_with_tools", boom)
    done = tasks.run_one()
    assert done["status"] == "failed"
    assert "model down" in done["error"]
    assert tasks.get(t["id"])["status"] == "failed"


def test_claimed_task_is_not_run_twice(monkeypatch):
    tasks.enqueue("only once")
    calls = []

    def fake_loop(system, user, specs, dispatch, **kwargs):
        calls.append(user)
        return "ok", []

    monkeypatch.setattr(tasks.llm, "chat_with_tools", fake_loop)
    first = tasks.run_one()
    second = tasks.run_one()  # queue now empty
    assert first["status"] == "done"
    assert second is None
    assert calls == ["only once"]


def test_web_search_after_file_read_is_refused(monkeypatch):
    """The exfiltration shape: a prompt-injected file tells the model to search
    the web for the file's private contents. The background runner must apply
    the same guards as chat — the query is not derived from the task prompt AND
    another tool result already entered the turn, so the search never leaves
    the machine."""
    from brain import websearch

    searched = []
    monkeypatch.setattr(websearch, "search_web", lambda q: searched.append(q) or {"query": q, "results": []})
    tasks.enqueue("Summarize the finance notes.")

    def fake_loop(system, user, specs, dispatch, **kwargs):
        content = dispatch("read_file", {"filename": "README.md"})
        assert "content" in content
        leaked = dispatch("web_search", {"query": "PRIVATE-CONTENT-FROM-FILE"})
        assert leaked.get("error")
        return "done", [{"tool": "read_file"}]

    monkeypatch.setattr(tasks.llm, "chat_with_tools", fake_loop)

    done = tasks.run_one()
    assert done["status"] == "done"
    assert searched == []   # the network query never executed


def test_stock_quotes_must_be_named_in_the_task_prompt(monkeypatch):
    """The same exfiltration shape as web_search, through the quote provider:
    a prompt-injected file tells the model to fetch "quotes" for ticker-shaped
    chunks of private text. Only symbols named in the user-authored task prompt
    may reach the provider URL; anything else is refused without a network call."""
    from brain import market

    fetched = []
    monkeypatch.setattr(market, "_fetch_chart_quote", lambda s: fetched.append(s) or {"symbol": s, "price": 1.0})
    tasks.enqueue("Check the VTI price.")

    def fake_loop(system, user, specs, dispatch, **kwargs):
        ok = dispatch("get_stock_quotes", {"tickers": ["VTI"]})
        assert not ok.get("error")
        content = dispatch("read_file", {"filename": "README.md"})
        assert "content" in content
        leaked = dispatch("get_stock_quotes", {"tickers": ["XK7Q2", "VTI"]})
        assert "XK7Q2" in leaked["error"]
        return "done", [{"tool": "get_stock_quotes"}]

    monkeypatch.setattr(tasks.llm, "chat_with_tools", fake_loop)

    done = tasks.run_one()
    assert done["status"] == "done"
    assert fetched == ["VTI"]   # the injected symbols never left the machine


def test_web_search_underived_query_is_refused_even_first(monkeypatch):
    from brain import websearch

    searched = []
    monkeypatch.setattr(websearch, "search_web", lambda q: searched.append(q) or {"query": q, "results": []})
    tasks.enqueue("Summarize the finance notes.")

    def fake_loop(system, user, specs, dispatch, **kwargs):
        refused = dispatch("web_search", {"query": "secret project codename ORCHID"})
        assert refused.get("error")
        return "done", []

    monkeypatch.setattr(tasks.llm, "chat_with_tools", fake_loop)
    tasks.run_one()
    assert searched == []


def test_web_search_first_with_prompt_derived_query_is_allowed(monkeypatch):
    from brain import websearch

    searched = []
    monkeypatch.setattr(
        websearch,
        "search_web",
        lambda q: searched.append(q) or {"query": q, "results": [{"title": "t", "url": "u"}]},
    )
    tasks.enqueue("Research the best BSN programs in Colorado.")

    def fake_loop(system, user, specs, dispatch, **kwargs):
        first = dispatch("web_search", {"query": "best BSN programs in Colorado"})
        assert not first.get("error")
        second = dispatch("web_search", {"query": "best BSN programs in Colorado"})
        assert second.get("error")   # one search per task
        return "done", [{"tool": "web_search"}]

    monkeypatch.setattr(tasks.llm, "chat_with_tools", fake_loop)

    done = tasks.run_one()
    assert done["status"] == "done"
    assert searched == ["best BSN programs in Colorado"]
    assert done["tools_used"] == ["web_search"]


def test_failed_tool_does_not_disable_background_web_search(monkeypatch):
    from brain import websearch

    searched = []
    monkeypatch.setattr(
        websearch,
        "search_web",
        lambda q: searched.append(q) or {"query": q, "results": []},
    )
    tasks.enqueue("Research Sierra Vista weather.")

    def fake_loop(system, user, specs, dispatch, **kwargs):
        failed = dispatch("execute_python_code", {"script": "import requests"})
        assert failed.get("error")
        result = dispatch("web_search", {"query": "Sierra Vista weather"})
        assert not result.get("error")
        return "done", [{"tool": "web_search"}]

    monkeypatch.setattr(tasks.llm, "chat_with_tools", fake_loop)

    done = tasks.run_one()
    assert done["status"] == "done"
    assert searched == ["Sierra Vista weather"]


def test_interrupted_claim_requeues_once_then_fails(monkeypatch):
    """A runner that dies after claiming leaves the task 'running' with no one
    to finish it. The claim is a lease: on expiry the task is re-queued for one
    automatic retry, and a second interruption fails it — either way it never
    stays running (and unclaimable) forever."""
    t = tasks.enqueue("die midway")
    claimed = tasks._claim_next(None)   # simulate claim, then process exit
    assert claimed["id"] == t["id"]
    assert tasks.get(t["id"])["status"] == "running"   # lease still fresh

    monkeypatch.setattr(config, "BACKGROUND_TASK_LEASE_SECONDS", 0)
    recovered = tasks.get(t["id"])
    assert recovered["status"] == "queued"             # retried automatically
    assert recovered["requeued_after_interruption"] is True
    assert recovered["error"] is None

    reclaimed = tasks._claim_next(None)                # retry crashes too
    assert reclaimed["id"] == t["id"]
    failed = tasks.get(t["id"])
    assert failed["status"] == "failed"
    assert failed["recovered_stale_claim"] is True
    assert "interrupted twice" in failed["error"]
    assert tasks.run_one() is None   # queue is drained, not stuck


def test_approval_resolution_completes_the_parent_task(monkeypatch):
    """Approving the parked action must reconcile the task that queued it —
    otherwise the task stays 'awaiting_approval' forever."""
    from brain import tools as tools_facade

    tasks.enqueue("Remind me to call SSA tomorrow.")

    def fake_loop(system, user, specs, dispatch, **kwargs):
        dispatch("create_reminder", {"action": "call SSA", "due": "2099-01-01"})
        return "drafted", []

    monkeypatch.setattr(tasks.llm, "chat_with_tools", fake_loop)
    done = tasks.run_one()
    assert done["status"] == "awaiting_approval"
    action_id = done["pending_action_ids"][0]

    tools_facade.approve_pending(action_id)
    after = tasks.get(done["id"])
    assert after["status"] == "done"
    assert after["approval_outcomes"] == {action_id: "approved"}


def test_rejection_marks_the_parent_task_rejected(monkeypatch):
    from brain import tools as tools_facade

    tasks.enqueue("Remind me to call SSA tomorrow.")

    def fake_loop(system, user, specs, dispatch, **kwargs):
        dispatch("create_reminder", {"action": "call SSA", "due": "2099-01-01"})
        return "drafted", []

    monkeypatch.setattr(tasks.llm, "chat_with_tools", fake_loop)
    done = tasks.run_one()
    action_id = done["pending_action_ids"][0]

    tools_facade.reject_pending(action_id)
    after = [t for t in tasks.list_tasks() if t["id"] == done["id"]][0]
    assert after["status"] == "rejected"
    assert "rejected" in after["error"]
    assert after["approval_outcomes"] == {action_id: "rejected"}


def test_failed_action_marks_the_parent_task_failed(monkeypatch):
    tasks.enqueue("Remind me to call SSA tomorrow.")

    def fake_loop(system, user, specs, dispatch, **kwargs):
        dispatch("create_reminder", {"action": "call SSA", "due": "2099-01-01"})
        return "drafted", []

    monkeypatch.setattr(tasks.llm, "chat_with_tools", fake_loop)
    done = tasks.run_one()
    action_id = done["pending_action_ids"][0]

    pending.resolve(action_id, "failed", {"error": "execution failed"})
    after = tasks.get(done["id"])

    assert after["status"] == "failed"
    assert "failed" in after["error"]
    assert after["approval_outcomes"] == {action_id: "failed"}


def test_missing_action_state_fails_closed():
    task = tasks.enqueue("Do a reviewed action.")
    tasks._finish(
        task["id"],
        {"status": "awaiting_approval", "pending_action_ids": ["act_missing"]},
    )

    after = tasks.get(task["id"])

    assert after["status"] == "failed"
    assert "act_missing=missing" in after["error"]
    assert after["approval_outcomes"] == {"act_missing": "missing"}


def test_disabled_runner_is_noop(monkeypatch):
    tasks.enqueue("won't run")
    monkeypatch.setattr(config, "BACKGROUND_TASKS_ENABLED", False)
    monkeypatch.setattr(tasks.llm, "chat_with_tools", lambda *a, **k: ("nope", []))
    assert tasks.run_one() is None
