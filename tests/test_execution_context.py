"""Execution boundaries must propagate tracking and Stop state automatically."""
from __future__ import annotations

import asyncio
import threading

import pytest

import server
from brain import llm, pipeline, tasks
from brain.execution import bind_context, context_aware_execution, execution_depth
from brain.toolkit import registry


def test_execution_entry_points_are_decorated():
    entry_points = {
        "llm.chat": llm.chat,
        "llm.chat_with_tools": llm.chat_with_tools,
        "pipeline.answer": pipeline.answer,
        "tasks._execute": tasks._execute,
        "registry.dispatch": registry.dispatch,
        "registry.execute": registry.execute,
        "server._execute_chat_request": server._execute_chat_request,
        "server._capture_background": server._capture_background,
    }

    missing = [
        name for name, func in entry_points.items()
        if not getattr(func, "__context_aware_execution__", None)
    ]
    assert missing == []


def test_decorator_starts_attaches_and_cleans_context(monkeypatch):
    monkeypatch.setattr(llm, "normalize_num_ctx", lambda value, model=None: int(value or 4096))
    seen = {}

    @context_aware_execution(attach_usage=True)
    def work(model=None, num_ctx=None):
        seen["usage"] = llm.context_usage()
        seen["depth"] = execution_depth()
        return {"ok": True}

    result = work(model="test-model", num_ctx=8192)

    assert seen["usage"]["configured_tokens"] == 8192
    assert seen["depth"] == 1
    assert result["context_usage"]["model"] == "test-model"
    assert llm.context_usage() is None
    assert execution_depth() == 0


def test_nested_decorators_reuse_one_context_tracker(monkeypatch):
    monkeypatch.setattr(llm, "normalize_num_ctx", lambda value, model=None: 8192)
    original_begin = llm.begin_context_tracking
    original_reset = llm.reset_context_tracking
    calls = {"begin": 0, "reset": 0}

    def begin(*args, **kwargs):
        calls["begin"] += 1
        return original_begin(*args, **kwargs)

    def reset(token):
        calls["reset"] += 1
        return original_reset(token)

    monkeypatch.setattr(llm, "begin_context_tracking", begin)
    monkeypatch.setattr(llm, "reset_context_tracking", reset)

    @context_aware_execution
    def inner(model=None, num_ctx=None):
        assert execution_depth() == 2
        assert llm.context_usage()["configured_tokens"] == 8192

    @context_aware_execution
    def outer(model=None, num_ctx=None):
        inner(model=model, num_ctx=num_ctx)
        assert execution_depth() == 1

    outer(model="nested", num_ctx=8192)

    assert calls == {"begin": 1, "reset": 1}
    assert llm.context_usage() is None


def test_decorator_attaches_cancel_event_and_checks_both_boundaries():
    event = threading.Event()
    called = []

    @context_aware_execution(track_context=False)
    def cancelled_before(cancel_event=None):
        called.append(True)

    event.set()
    with pytest.raises(llm.GenerationCancelled):
        cancelled_before(cancel_event=event)
    assert called == []
    assert llm.active_cancel_event() is None

    event.clear()

    @context_aware_execution(track_context=False)
    def cancelled_during(cancel_event=None):
        event.set()

    with pytest.raises(llm.GenerationCancelled):
        cancelled_during(cancel_event=event)
    assert llm.active_cancel_event() is None


def test_async_decorator_preserves_context_across_await(monkeypatch):
    monkeypatch.setattr(llm, "normalize_num_ctx", lambda value, model=None: 4096)
    event = threading.Event()

    @context_aware_execution(attach_usage=True)
    async def work(model=None, num_ctx=None, cancel_event=None):
        before = llm.context_usage()
        await asyncio.sleep(0)
        assert llm.context_usage() == before
        assert llm.active_cancel_event() is event
        return {"ok": True}

    result = asyncio.run(work(model="async", num_ctx=4096, cancel_event=event))

    assert result["context_usage"]["configured_tokens"] == 4096
    assert llm.context_usage() is None
    assert llm.active_cancel_event() is None


def test_bind_context_propagates_state_to_a_new_thread(monkeypatch):
    monkeypatch.setattr(llm, "normalize_num_ctx", lambda value, model=None: 2048)
    event = threading.Event()
    seen = {}

    def worker():
        llm._record_context_usage(
            [{"role": "user", "content": "thread work"}],
            {"num_ctx": 2048},
            exact_tokens=123,
        )
        seen["usage"] = llm.context_usage()
        seen["cancel_event"] = llm.active_cancel_event()
        seen["depth"] = execution_depth()

    @context_aware_execution
    def schedule(model=None, num_ctx=None, cancel_event=None):
        thread = threading.Thread(target=bind_context(worker))
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive()
        seen["parent_usage"] = llm.context_usage()

    schedule(model="threaded", num_ctx=2048, cancel_event=event)

    assert seen["usage"]["configured_tokens"] == 2048
    assert seen["usage"]["used_tokens"] == 123
    assert seen["parent_usage"]["used_tokens"] == 123
    assert seen["parent_usage"]["calls"] == 1
    assert seen["cancel_event"] is event
    assert seen["depth"] == 1


def test_tool_dispatch_honors_inherited_cancellation():
    event = threading.Event()
    event.set()
    token = llm.set_cancel_event(event)
    try:
        with pytest.raises(llm.GenerationCancelled):
            registry.dispatch("get_current_datetime", {})
    finally:
        llm.reset_cancel_event(token)


def test_decorator_cleans_context_when_wrapped_code_fails(monkeypatch):
    monkeypatch.setattr(llm, "normalize_num_ctx", lambda value, model=None: 4096)
    event = threading.Event()

    @context_aware_execution
    def fail(model=None, num_ctx=None, cancel_event=None):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        fail(model="failure", num_ctx=4096, cancel_event=event)

    assert llm.context_usage() is None
    assert llm.active_cancel_event() is None
    assert execution_depth() == 0
