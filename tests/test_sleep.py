"""Sleep cycle: cold storage + speculative dreaming."""
from __future__ import annotations

import json

import server
from brain import sleep, store
from util import store_memory


def test_sleep_dry_run_does_not_mutate(monkeypatch):
    old = store_memory("old throwaway episode", category="episode", importance=1, date="2020-01-01T00:00:00+00:00")
    store_memory("coffee preference", category="preference", importance=4)
    store_memory("project plan", category="fact", importance=4)
    monkeypatch.setattr(sleep.llm, "chat", lambda *a, **k: "Speculative hunch: x")

    out = sleep.run(dry_run=True, max_dreams=1)

    assert out["archivable"] == 1
    assert out["dreamable"] == 1
    assert out["archived"] == 0
    assert out["dreamed"] == 0
    assert store.get_by_ids([old.id])[old.id]["text"] == "old throwaway episode"
    assert not sleep._COLD_DIR.exists()


def test_sleep_archives_weak_episode_and_stores_speculative_dream(monkeypatch):
    old = store_memory("User: passing small talk", category="episode", importance=1, date="2020-01-01T00:00:00+00:00")
    pinned = store_memory("pinned old episode", category="episode", importance=1, date="2020-01-01T00:00:00+00:00")
    store.update_metadata_all([pinned.id], {"pinned": True})
    store_memory("Alex likes black coffee.", category="preference", importance=4)
    store_memory("Alex is planning a product launch.", category="fact", importance=4)
    monkeypatch.setattr(sleep.llm, "chat", lambda *a, **k: "Speculative hunch: coffee ritual mirrors launch planning.")

    out = sleep.run(max_dreams=1)

    assert out["archived"] == 1
    assert old.id not in store.get_by_ids([old.id])
    assert store.get_by_ids([pinned.id])[pinned.id]["text"] == "pinned old episode"
    archived = sleep._COLD_DIR.joinpath(out["archive_path"].split("/")[-1]).read_text(encoding="utf-8")
    assert json.loads(archived)["id"] == old.id
    dreams = [r for r in store.list_records_all() if (r["meta"] or {}).get("category") == "dream"]
    assert len(dreams) == 1
    assert dreams[0]["text"].startswith("Speculative hunch:")
    assert dreams[0]["meta"]["source_type"] == "generated_dream"
    assert dreams[0]["meta"]["speculative"] is True
    assert out["dreamed"] == 1
    assert sleep.status()["last_run"]


def test_sleep_status_uses_file_lock_and_atomic_write(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sleep, "_STATUS_FILE", tmp_path / "sleep_state.json")
    from brain import utils as brain_utils

    monkeypatch.setattr(brain_utils.fcntl, "flock", lambda fd, op: calls.append(op))

    sleep._write_status({"last_run": "now"})

    assert sleep.status()["last_run"] == "now"
    assert brain_utils.fcntl.LOCK_EX in calls
    assert brain_utils.fcntl.LOCK_UN in calls
    assert not list(tmp_path.glob("*.tmp"))


def test_sleep_api_preview_and_run_creates_backup(monkeypatch):
    store_memory("old weak episode", category="episode", importance=1, date="2020-01-01T00:00:00+00:00")
    store_memory("coffee", category="preference", importance=4)
    store_memory("launch", category="fact", importance=4)
    monkeypatch.setattr(sleep.llm, "chat", lambda *a, **k: "Speculative hunch: coffee and launch both use ritual.")

    preview = server.sleep_cycle(server.SleepReq(dry_run=True, max_dreams=1))
    assert preview["archivable"] == 1
    assert preview["backup"] is None

    result = server.sleep_cycle(server.SleepReq(max_dreams=1))
    assert result["backup"]
    assert result["archived"] == 1
    assert result["dreamed"] == 1
