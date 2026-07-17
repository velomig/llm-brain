from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import config
import server
from util import store_memory


def test_normalize_history_preserves_all_valid_completed_turns():
    req = server.ChatReq(
        message="new question",
        history=[
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "newer question"},
            {"role": "assistant", "content": "previous answer"},
        ],
    )

    assert server._normalize_history(req) == [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "newer question"},
        {"role": "assistant", "content": "previous answer"},
    ]


@pytest.mark.parametrize(
    "history",
    [
        [{"role": "assistant", "content": "orphaned answer"}],
        [{"role": "user", "content": "unanswered prior message"}],
        [
            {"role": "user", "content": "first question"},
            {"role": "user", "content": "second question"},
        ],
    ],
)
def test_chat_request_rejects_incomplete_history_protocol(history):
    with pytest.raises(ValidationError, match="completed user/assistant pairs"):
        server.ChatReq(message="current message", history=history)


def test_chat_request_rejects_oversized_history_item_instead_of_silently_dropping_it():
    with pytest.raises(ValidationError) as exc_info:
        server.ChatReq(
            message="new question",
            history=[
                {"role": "user", "content": "question before the bad response"},
                {"role": "assistant", "content": "Hh4e" * (config.MAX_MESSAGE_CHARS // 4 + 1)},
            ],
        )

    error = exc_info.value.errors()[0]
    assert error["loc"] == ("history", 1, "content")
    assert error["type"] == "string_too_long"


def test_chat_request_rejects_excess_history_instead_of_silently_truncating_it():
    history = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"message {index}",
        }
        for index in range(config.HISTORY_TURNS * 2 + 1)
    ]

    with pytest.raises(ValidationError) as exc_info:
        server.ChatReq(message="new question", history=history)

    error = exc_info.value.errors()[0]
    assert error["loc"] == ("history",)
    assert error["type"] == "too_long"


def test_full_history_window_reaches_answer_pipeline_unchanged(monkeypatch):
    history = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"prior message {index}",
        }
        for index in range(config.HISTORY_TURNS * 2)
    ]
    received = {}

    def fake_answer(message, memories, **kwargs):
        received["message"] = message
        received["history"] = kwargs["history"]
        return {"reply": "pipeline reply"}

    monkeypatch.setattr(server, "answer", fake_answer)
    monkeypatch.setattr(server, "_store_count", lambda model=None: 0)
    monkeypatch.setattr(server, "_capture_background", lambda *args, **kwargs: None)
    monkeypatch.setattr(server.llm, "normalize_num_ctx", lambda value, model=None: 8192)
    monkeypatch.setattr(server.llm, "begin_context_tracking", lambda value, model=None: object())
    monkeypatch.setattr(server.llm, "reset_context_tracking", lambda token: None)
    monkeypatch.setattr(server.llm, "context_usage", lambda: None)

    response = TestClient(server.app, base_url="http://127.0.0.1").post(
        "/api/chat",
        json={
            "message": "current message",
            "history": history,
            "memory_enabled": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["reply"] == "pipeline reply"
    assert received == {"message": "current message", "history": history}


def test_activated_memories_are_prompt_context():
    keep = store_memory("This is the memory to recollect.", category="note", date="2026-07-03T00:00:00+00:00")
    other = store_memory("This one stays ordinary.", category="note")

    activated = server._activated_memories([keep.id])
    merged = server._merge_activated_memories(
        [{"id": keep.id, "text": "duplicate"}, {"id": other.id, "text": other.text}],
        activated,
    )

    assert activated[0]["id"] == keep.id
    assert activated[0]["activated"] is True
    assert activated[0]["text"] == "This is the memory to recollect."
    assert [m["id"] for m in merged] == [keep.id, other.id]


def test_chat_request_cleans_activated_memory_ids():
    req = server.ChatReq(
        message="hello",
        activated_memory_ids=[" m1 ", "m1", "", "x" * 200, "m2"],
    )

    assert req.activated_memory_ids == ["m1", "m2"]


def test_activated_superseded_memory_is_not_resurrected():
    # mark_memory_wrong / merges retire a memory everywhere else; a stale
    # activation chip (localStorage can outlive the correction by weeks) must
    # not re-inject it into the prompt with score 1.0.
    from brain import store

    retired = store_memory("FNDZ position is 40 shares.", category="note")
    current = store_memory("FNDZ position is 12 shares.", category="note")
    store.mark_superseded([retired.id])

    activated = server._activated_memories([retired.id, current.id])

    assert [m["id"] for m in activated] == [current.id]
