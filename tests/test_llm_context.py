from __future__ import annotations

import threading

import config
import server
from fastapi.testclient import TestClient
from brain import evolution, llm, metacognition, pipeline, reflect
from util import store_memory


class FakeOllamaClient:
    def __init__(
        self,
        max_context: int | None = config.DEFAULT_NUM_CTX,
        prompt_eval_count: int | None = None,
    ):
        self.calls = []
        self.max_context = max_context
        self.prompt_eval_count = prompt_eval_count

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        user = "\n".join(m.get("content", "") for m in kwargs.get("messages", []) if m.get("role") == "user")
        if "Audit this completed turn" in user:
            content = '{"note": null}'
        elif "Review these facts" in user:
            content = '{"insights":[],"reminders":[],"initiatives":[]}'
        elif "Draft a human-reviewable mutation proposal" in user:
            content = (
                '{"title":"No mutation recommended","rationale":"ok","target_failure":"none",'
                '"expected_tradeoff":"none","prompt_overrides":{},"hyperparameters":{},'
                '"canary_impact":[],"risk_notes":[]}'
            )
        elif "Critique this draft" in user:
            content = "critique"
        elif "Rewrite the answer" in user:
            content = "final"
        else:
            content = "ok"
        response = {"message": {"content": content}}
        if self.prompt_eval_count is not None:
            response["prompt_eval_count"] = self.prompt_eval_count
        return response

    def show(self, name):
        if self.max_context is None:
            return {"modelinfo": {}}
        return {"modelinfo": {"llama.context_length": self.max_context}}


def test_chat_defaults_to_configured_context_window(monkeypatch):
    client = FakeOllamaClient()
    monkeypatch.setattr(llm, "_client", client)

    assert llm.chat("system", "user") == "ok"

    assert client.calls[0]["options"]["num_ctx"] == config.DEFAULT_NUM_CTX
    assert client.calls[0]["options"]["num_predict"] == config.MAX_GENERATION_TOKENS


def test_chat_preserves_explicit_context_window(monkeypatch):
    client = FakeOllamaClient()
    monkeypatch.setattr(llm, "_client", client)

    llm.chat("system", "user", num_ctx=8192)

    assert client.calls[0]["options"]["num_ctx"] == 8192


def test_chat_with_tools_defaults_to_configured_context_window(monkeypatch):
    client = FakeOllamaClient()
    monkeypatch.setattr(llm, "_client", client)

    reply, used = llm.chat_with_tools("system", "user", [], lambda name, args: {})

    assert reply == "ok"
    assert used == []
    assert client.calls[0]["options"]["num_ctx"] == config.DEFAULT_NUM_CTX


def test_chat_clamps_unknown_model_context_to_conservative_cap(monkeypatch):
    client = FakeOllamaClient(max_context=None)
    monkeypatch.setattr(llm, "_client", client)

    llm.chat("system", "user")

    assert client.calls[0]["options"]["num_ctx"] == config.UNKNOWN_MODEL_NUM_CTX


def test_chat_clamps_requested_context_to_known_model_max(monkeypatch):
    client = FakeOllamaClient(max_context=8192)
    monkeypatch.setattr(llm, "_client", client)

    llm.chat("system", "user", num_ctx=config.DEFAULT_NUM_CTX)

    assert client.calls[0]["options"]["num_ctx"] == 8192


def test_context_tracker_uses_exact_ollama_prompt_count(monkeypatch):
    client = FakeOllamaClient(max_context=8192, prompt_eval_count=321)
    monkeypatch.setattr(llm, "_client", client)
    token = llm.begin_context_tracking(8192, "tracked-model")
    try:
        llm.chat("system prompt", "user prompt", model="tracked-model", num_ctx=8192)
        usage = llm.context_usage()
    finally:
        llm.reset_context_tracking(token)

    assert usage["used_tokens"] == 321
    assert usage["configured_tokens"] == 8192
    assert usage["measurement"] == "exact"
    assert usage["calls"] == 1
    assert usage["percent"] == 3.92


def test_context_tracker_accumulates_turn_total_across_calls(monkeypatch):
    client = FakeOllamaClient(max_context=8192, prompt_eval_count=200)
    monkeypatch.setattr(llm, "_client", client)
    token = llm.begin_context_tracking(8192, "tracked-model")
    try:
        llm.chat("system prompt", "first call", model="tracked-model", num_ctx=8192)
        llm.chat("system prompt", "second call", model="tracked-model", num_ctx=8192)
        usage = llm.context_usage()
    finally:
        llm.reset_context_tracking(token)

    assert usage["used_tokens"] == 200          # peak stays per-call
    assert usage["turn_total_tokens"] == 400    # total sums every call in the turn
    assert usage["calls"] == 2


def test_context_tracker_falls_back_to_labeled_estimate(monkeypatch):
    client = FakeOllamaClient(max_context=8192)
    monkeypatch.setattr(llm, "_client", client)
    token = llm.begin_context_tracking(8192, "tracked-model")
    try:
        llm.chat("system " * 100, "user " * 100, model="tracked-model", num_ctx=8192)
        usage = llm.context_usage()
    finally:
        llm.reset_context_tracking(token)

    assert usage["used_tokens"] > 0
    assert usage["measurement"] == "estimated"
    assert usage["exact_calls"] == 0


def test_streamed_chat_tracks_final_prompt_count(monkeypatch):
    class StreamClient(FakeOllamaClient):
        def chat(self, **kwargs):
            assert kwargs["stream"] is True
            return iter(
                [
                    {"message": {"content": "hel"}},
                    {"message": {"content": "lo"}, "prompt_eval_count": 654},
                ]
            )

    monkeypatch.setattr(llm, "_client", StreamClient(max_context=8192))
    context_token = llm.begin_context_tracking(8192, "stream-model")
    cancel_token = llm.set_cancel_event(threading.Event())
    try:
        assert llm.chat("system", "user", model="stream-model", num_ctx=8192) == "hello"
        usage = llm.context_usage()
    finally:
        llm.reset_cancel_event(cancel_token)
        llm.reset_context_tracking(context_token)

    assert usage["used_tokens"] == 654
    assert usage["measurement"] == "exact"


def test_chat_api_returns_context_usage(monkeypatch):
    client = FakeOllamaClient(max_context=8192, prompt_eval_count=456)
    monkeypatch.setattr(llm, "_client", client)

    def fake_answer(message, mems, **kwargs):
        reply = llm.chat("system", message, model=kwargs.get("model"), num_ctx=kwargs.get("num_ctx"))
        return {"reply": reply, "tools_used": []}

    monkeypatch.setattr(server, "answer", fake_answer)
    response = TestClient(server.app, base_url="http://127.0.0.1").post(
        "/api/chat",
        json={"message": "show usage", "model": "tracked-model", "num_ctx": 8192},
    )

    assert response.status_code == 200
    usage = response.json()["context_usage"]
    assert usage["used_tokens"] == 456
    assert usage["configured_tokens"] == 8192
    assert usage["measurement"] == "exact"


def test_long_context_stress_preserves_known_large_window_for_system2(monkeypatch):
    long_ctx = 131_072
    client = FakeOllamaClient(max_context=long_ctx)
    monkeypatch.setattr(llm, "_client", client)
    monkeypatch.setattr(config, "METACOGNITION_ENABLED", True)
    monkeypatch.setattr(config, "EVOLUTION_ENABLED", True)

    long_history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} " + ("detail " * 500)}
        for i in range(12)
    ]
    pipeline._slow_reasoning_answer(
        "hard question " + ("context " * 1000),
        [],
        {"reply": "draft " + ("analysis " * 1000)},
        [],
        [],
        [],
        None,
        history=long_history,
        model="stress-model",
        num_ctx=long_ctx,
    )
    metacognition.supervise(
        "user " + ("history " * 1000),
        "reply " + ("reasoning " * 1000),
        thinking={"path": "system2"},
        model="stress-model",
    )
    store_memory("durable fact for reflection stress", category="fact")
    reflect.reflect(model="stress-model")
    evolution.propose({"preamble": "p", "tools_note": "t", "inject_note": "i"}, model="stress-model")

    assert client.calls
    assert all(call["options"]["num_ctx"] == long_ctx for call in client.calls)
