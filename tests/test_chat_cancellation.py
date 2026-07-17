"""Stop must cancel the generation itself, not just the client's wait.

The llm layer switches to Ollama streaming when a cancel event is active and
checks the event between chunks (closing the stream is what makes Ollama abort
server-side). Without an event — CLI, cron jobs, capture — calls stay
non-streaming and byte-for-byte the old behavior. GenerationCancelled derives
from BaseException so the pipeline's broad `except Exception` fallbacks can
never swallow a user's Stop."""
from __future__ import annotations

import threading
import time

import config
from fastapi.testclient import TestClient

import server
from brain import llm
from brain.llm import GenerationCancelled


class FakeStreamingClient:
    """Yields scripted responses; each call consumes the next script entry.

    A script entry is either a plain content string (returned whole for
    non-streaming calls, chunked per character group for streaming ones) or a
    list of message dicts to stream as chunks.
    """

    def __init__(self, script, on_first_chunk=None):
        self.script = list(script)
        self.calls = []
        self.on_first_chunk = on_first_chunk

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        entry = self.script.pop(0)
        messages = (
            [{"content": part} for part in entry]
            if isinstance(entry, list) and all(isinstance(p, str) for p in entry)
            else entry if isinstance(entry, list) else [{"content": entry}]
        )
        if not kwargs.get("stream"):
            merged = "".join(m.get("content", "") for m in messages)
            tool_calls = [tc for m in messages for tc in m.get("tool_calls", [])]
            return {"message": {"content": merged, "tool_calls": tool_calls or None}}

        def gen():
            for i, msg in enumerate(messages):
                yield {"message": msg}
                if i == 0 and self.on_first_chunk:
                    self.on_first_chunk()

        return gen()

    def show(self, name):
        return {"modelinfo": {"llama.context_length": 131072}}


def _with_event(monkeypatch, client, event):
    monkeypatch.setattr(llm, "_client", client)
    token = llm.set_cancel_event(event)
    return token


def test_chat_without_event_stays_non_streaming(monkeypatch):
    client = FakeStreamingClient(["plain answer"])
    monkeypatch.setattr(llm, "_client", client)

    assert llm.chat("sys", "user") == "plain answer"
    assert not client.calls[0].get("stream")


def test_chat_with_event_streams_and_assembles_chunks(monkeypatch):
    client = FakeStreamingClient([["Hello ", "from ", "chunks"]])
    token = _with_event(monkeypatch, client, threading.Event())
    try:
        assert llm.chat("sys", "user") == "Hello from chunks"
        assert client.calls[0]["stream"] is True
    finally:
        llm.reset_cancel_event(token)


def test_chat_stops_repetitive_stream_before_it_poison_history(monkeypatch):
    repeated = "Hh4e" * 25
    client = FakeStreamingClient([[repeated, repeated, repeated, "never reached"]])
    token = _with_event(monkeypatch, client, threading.Event())
    try:
        reply = llm.chat("sys", "user", model="looping-model")
    finally:
        llm.reset_cancel_event(token)

    assert "repetitive-output loop" in reply
    assert "Hh4e" not in reply
    assert len(reply) <= config.MAX_ASSISTANT_REPLY_CHARS


def test_chat_bounds_streamed_output_by_characters(monkeypatch):
    monkeypatch.setattr(config, "MAX_ASSISTANT_REPLY_CHARS", 64)
    client = FakeStreamingClient([["A" * 40, "B" * 40, "never reached"]])
    token = _with_event(monkeypatch, client, threading.Event())
    try:
        reply = llm.chat("sys", "user", model="rambling-model")
    finally:
        llm.reset_cancel_event(token)

    assert len(reply) <= 64
    assert "safety limit" in reply


def test_chat_replaces_repetitive_nonstreaming_output(monkeypatch):
    client = FakeStreamingClient(["Hh4e" * 100])
    monkeypatch.setattr(llm, "_client", client)

    reply = llm.chat("sys", "user", model="background-model")

    assert "repetitive-output loop" in reply
    assert not client.calls[0].get("stream")


def test_chat_raises_when_cancelled_before_the_call(monkeypatch):
    client = FakeStreamingClient(["never used"])
    event = threading.Event()
    event.set()
    token = _with_event(monkeypatch, client, event)
    try:
        try:
            llm.chat("sys", "user")
            raised = False
        except GenerationCancelled:
            raised = True
        assert raised
        assert client.calls == []   # cancelled before any Ollama traffic
    finally:
        llm.reset_cancel_event(token)


def test_chat_aborts_mid_stream_when_event_is_set(monkeypatch):
    event = threading.Event()
    client = FakeStreamingClient([["first ", "second ", "third"]], on_first_chunk=event.set)
    token = _with_event(monkeypatch, client, event)
    try:
        try:
            llm.chat("sys", "user")
            raised = False
        except GenerationCancelled:
            raised = True
        assert raised
    finally:
        llm.reset_cancel_event(token)


def test_chat_with_tools_streams_tool_calls_and_finishes(monkeypatch):
    tool_call = {"function": {"name": "calculate", "arguments": {"expression": "1+1"}}}
    client = FakeStreamingClient(
        [
            [{"content": "", "tool_calls": [tool_call]}],
            ["the answer is 2"],
        ]
    )
    token = _with_event(monkeypatch, client, threading.Event())
    try:
        reply, used = llm.chat_with_tools(
            "sys", "user", [{"function": {"name": "calculate"}}],
            dispatch=lambda name, args: {"result": 2},
        )
        assert reply == "the answer is 2"
        assert used == [{"tool": "calculate", "args": {"expression": "1+1"}, "result": {"result": 2}}]
        assert all(call["stream"] is True for call in client.calls)
    finally:
        llm.reset_cancel_event(token)


def test_chat_with_tools_retries_blank_final_after_tools(monkeypatch):
    tool_call = {"function": {"name": "calculate", "arguments": {"expression": "1+1"}}}
    client = FakeStreamingClient(
        [
            [{"content": "", "tool_calls": [tool_call]}],
            [""],
            ["the answer is 2"],
        ]
    )
    token = _with_event(monkeypatch, client, threading.Event())
    try:
        reply, used = llm.chat_with_tools(
            "sys", "user", [{"function": {"name": "calculate"}}],
            dispatch=lambda name, args: {"result": 2},
        )
        assert reply == "the answer is 2"
        assert used == [{"tool": "calculate", "args": {"expression": "1+1"}, "result": {"result": 2}}]
        assert client.calls[-1]["messages"][-1]["role"] == "user"
        assert "answer the user's last request" in client.calls[-1]["messages"][-1]["content"]
        assert "tools" not in client.calls[-1]
    finally:
        llm.reset_cancel_event(token)


def test_cancel_endpoint_unknown_request_is_a_noop():
    client = TestClient(server.app, base_url="http://127.0.0.1")
    res = client.post("/api/chat/cancel", json={"request_id": "no-such-id"})
    assert res.status_code == 200
    assert res.json()["cancelled"] is False


def test_pending_cancel_aborts_chat_that_registers_later(monkeypatch):
    monkeypatch.setattr(
        server,
        "answer",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("generation should not start")),
    )
    with server._INFLIGHT_LOCK:
        server._INFLIGHT_CANCEL.clear()
        server._PENDING_CANCEL.clear()
    try:
        client = TestClient(server.app, base_url="http://127.0.0.1")
        cancel_res = client.post(
            "/api/chat/cancel",
            json={"request_id": "soon-to-start", "allow_pending": True},
        )
        chat_res = client.post(
            "/api/chat",
            json={"message": "long question", "request_id": "soon-to-start"},
        )

        assert cancel_res.status_code == 200
        assert cancel_res.json()["cancelled"] is True
        assert cancel_res.json()["pending"] is True
        body = chat_res.json()
        assert body["cancelled"] is True
        assert body["reply"] is None
        with server._INFLIGHT_LOCK:
            assert "soon-to-start" not in server._INFLIGHT_CANCEL
            assert "soon-to-start" not in server._PENDING_CANCEL
    finally:
        with server._INFLIGHT_LOCK:
            server._INFLIGHT_CANCEL.clear()
            server._PENDING_CANCEL.clear()


def test_chat_rejects_when_cancel_registry_is_full(monkeypatch):
    monkeypatch.setattr(server, "_INFLIGHT_MAX", 1)
    monkeypatch.setattr(
        server,
        "answer",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("generation should not start")),
    )
    with server._INFLIGHT_LOCK:
        server._INFLIGHT_CANCEL.clear()
        server._INFLIGHT_CANCEL["already-running"] = threading.Event()
    try:
        client = TestClient(server.app, base_url="http://127.0.0.1")
        res = client.post("/api/chat", json={"message": "hello", "request_id": "new-request"})

        assert res.status_code == 429
        assert "too many in-flight" in res.json()["detail"]
        with server._INFLIGHT_LOCK:
            assert set(server._INFLIGHT_CANCEL) == {"already-running"}
    finally:
        with server._INFLIGHT_LOCK:
            server._INFLIGHT_CANCEL.clear()


def test_chat_rejects_duplicate_request_id(monkeypatch):
    monkeypatch.setattr(
        server,
        "answer",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("generation should not start")),
    )
    with server._INFLIGHT_LOCK:
        server._INFLIGHT_CANCEL.clear()
        server._INFLIGHT_CANCEL["same-id"] = threading.Event()
    try:
        client = TestClient(server.app, base_url="http://127.0.0.1")
        res = client.post("/api/chat", json={"message": "hello", "request_id": "same-id"})

        assert res.status_code == 409
        assert "already in flight" in res.json()["detail"]
    finally:
        with server._INFLIGHT_LOCK:
            server._INFLIGHT_CANCEL.clear()


def test_chat_endpoint_cancel_aborts_and_skips_capture(monkeypatch):
    """Full loop: POST /api/chat blocks in generation, POST /api/chat/cancel
    flips the event, the chat response comes back cancelled with no reply."""

    def fake_answer(message, mems, **kwargs):
        for _ in range(500):            # ~5s ceiling so a broken cancel can't hang the suite
            llm.raise_if_cancelled()
            time.sleep(0.01)
        return {"reply": "finished uncancelled"}

    captured = []
    monkeypatch.setattr(server, "answer", fake_answer)
    monkeypatch.setattr(server, "capture", lambda *a, **k: captured.append(a))

    result = {}

    def do_chat():
        c = TestClient(server.app, base_url="http://127.0.0.1")
        result["res"] = c.post(
            "/api/chat",
            json={"message": "long question", "request_id": "cancel-me"},
        )

    t = threading.Thread(target=do_chat)
    t.start()
    deadline = time.time() + 2.0
    while time.time() < deadline:       # wait for the request to register
        with server._INFLIGHT_LOCK:
            if "cancel-me" in server._INFLIGHT_CANCEL:
                break
        time.sleep(0.01)

    canceller = TestClient(server.app, base_url="http://127.0.0.1")
    cancel_res = canceller.post("/api/chat/cancel", json={"request_id": "cancel-me"})
    t.join(timeout=10)

    assert cancel_res.json()["cancelled"] is True
    body = result["res"].json()
    assert body["cancelled"] is True
    assert body["reply"] is None
    assert captured == []               # a stopped turn is never captured
    with server._INFLIGHT_LOCK:         # registry cleaned up
        assert "cancel-me" not in server._INFLIGHT_CANCEL
