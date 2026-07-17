from fastapi.testclient import TestClient

import server
from brain import pipeline, self_model, store
from util import store_memory


def test_memory_disabled_chat_skips_retrieval_and_preserves_store(monkeypatch):
    store_memory("Alex's durable long-term fact", category="fact")
    captured = {}

    def fake_answer(message, mems, **kwargs):
        captured["mems"] = mems
        captured["memory_enabled"] = kwargs.get("memory_enabled")
        return {"reply": "fresh response", "tools_used": []}

    monkeypatch.setattr(server, "answer", fake_answer)
    monkeypatch.setattr(server, "capture", lambda *args, **kwargs: None)
    response = TestClient(server.app, base_url="http://127.0.0.1").post(
        "/api/chat",
        json={"message": "What do you know about me?", "memory_enabled": False},
    )

    assert response.status_code == 200
    assert captured == {"mems": [], "memory_enabled": False}
    assert response.json()["memories"] == []
    assert response.json()["memory_enabled"] is False
    assert store.count_all() == 1


def test_fresh_system_prompt_omits_persistent_model_blocks():
    self_model.update_from_turn(
        "Let's discuss consciousness",
        "I will remember this relationship-specific promise.",
    )

    system = pipeline.build_system([], memory_enabled=False)

    assert "relationship-specific promise" not in system
    assert "fresh_session_memory_disabled" in system
    assert "(disabled for fresh session)" in system


def test_destructive_memory_purge_route_no_longer_exists():
    response = TestClient(server.app, base_url="http://127.0.0.1").post(
        "/api/memory/purge",
        json={"confirmation": "PURGE_ALL_MEMORY"},
    )

    assert response.status_code in {404, 405}
