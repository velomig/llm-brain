"""Regression coverage for the third audit round.

Covers: prompt-injection taint on prefetched web/market data, the injection
fallback honoring the web privacy guard without re-searching, private
calibration files keeping owner-only permissions across atomic updates,
owner-only manual memory exports, and serialized metadata read-modify-write
cycles in the store.
"""
from __future__ import annotations

import stat
import threading
from types import SimpleNamespace

import backup
import config
from brain import pipeline, store
from brain.embeddings import embed_document
from brain.memory import MemoryUnit
from brain.toolkit import perspective_update, source_truth


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# --- prefetched external data taints the turn -----------------------------------

def _tool_path_setup(monkeypatch, captured):
    monkeypatch.setattr(pipeline.config, "TOOL_CALLING_ENABLED", True)
    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda name: True)

    def fake_chat_with_tools(system, user_text, tool_specs, dispatch, **kwargs):
        dispatch("list_reminders", {})
        return "done", [{"tool": "list_reminders"}]

    monkeypatch.setattr(pipeline.llm, "chat_with_tools", fake_chat_with_tools)

    def fake_dispatch(name, args=None, taint=None):
        captured["taint"] = list(taint or [])
        return {"ok": True}

    monkeypatch.setattr(pipeline.tools, "dispatch", fake_dispatch)


def test_prefetched_web_data_taints_dispatched_actions(monkeypatch):
    captured = {}
    _tool_path_setup(monkeypatch, captured)
    monkeypatch.setattr(pipeline, "_needs_web_prefetch", lambda *a, **k: True)
    monkeypatch.setattr(pipeline, "_needs_financial_prefetch", lambda *a, **k: False)
    monkeypatch.setattr(
        pipeline,
        "web_context_for",
        lambda text: {"provider": "DuckDuckGo", "results": [{"title": "x", "snippet": "y"}], "errors": []},
    )

    result = pipeline._single_pass_answer("what happened in AI news today", [], [], [], [], None)

    assert "web_search" in captured["taint"]
    assert result["reply"] == "done"


def test_prefetched_market_data_taints_dispatched_actions(monkeypatch):
    captured = {}
    _tool_path_setup(monkeypatch, captured)
    monkeypatch.setattr(pipeline, "_needs_web_prefetch", lambda *a, **k: False)
    monkeypatch.setattr(pipeline, "_needs_financial_prefetch", lambda *a, **k: True)
    monkeypatch.setattr(
        pipeline,
        "market_context_for",
        lambda *a, **k: {"provider": "Yahoo Finance", "quotes": [{"symbol": "VT", "price": 120.0}], "errors": []},
    )

    pipeline._single_pass_answer("how is my portfolio doing", [], [], [], [], None)

    assert "market_data" in captured["taint"]


def test_no_prefetch_leaves_taint_clean(monkeypatch):
    captured = {}
    _tool_path_setup(monkeypatch, captured)
    monkeypatch.setattr(pipeline, "_needs_web_prefetch", lambda *a, **k: False)
    monkeypatch.setattr(pipeline, "_needs_financial_prefetch", lambda *a, **k: False)

    pipeline._single_pass_answer("summarize my week", [], [], [], [], None)

    assert captured["taint"] == []


# --- injection fallback honors the web privacy guard -----------------------------

def _injection_path_setup(monkeypatch, search_calls):
    monkeypatch.setattr(pipeline.config, "TOOL_CALLING_ENABLED", False)
    monkeypatch.setattr(pipeline.llm, "chat", lambda *a, **k: "reply")

    def counting_search(text):
        search_calls.append(text)
        return {"provider": "DuckDuckGo", "results": [{"title": "hit"}], "errors": []}

    monkeypatch.setattr(pipeline, "web_context_for", counting_search)


def test_fallback_never_searches_when_web_guard_is_active(monkeypatch):
    search_calls: list[str] = []
    _injection_path_setup(monkeypatch, search_calls)
    monkeypatch.setattr(
        pipeline, "_web_search_context_guard_reason", lambda mems: "retrieved private context is present"
    )

    result = pipeline._single_pass_answer("search the web for that topic", [], [], [], [], None)

    assert search_calls == []
    assert result["web_data"] is None


def test_fallback_reuses_prefetched_web_data_instead_of_searching_twice(monkeypatch):
    search_calls: list[str] = []
    _injection_path_setup(monkeypatch, search_calls)
    monkeypatch.setattr(pipeline, "_needs_web_prefetch", lambda *a, **k: True)

    result = pipeline._single_pass_answer("what happened in AI news today", [], [], [], [], None)

    assert len(search_calls) == 1
    assert result["web_data"] == {
        "provider": "DuckDuckGo",
        "results": [{"title": "hit"}],
        "errors": [],
    }


# --- private files keep owner-only permissions across updates --------------------

def test_source_of_truth_update_heals_widened_permissions(tmp_path, monkeypatch):
    target = tmp_path / "finance.md"
    target.write_text("balance: OLD", encoding="utf-8")
    target.chmod(0o644)   # a previous buggy update already widened it
    monkeypatch.setattr(config, "FINANCE_PATH", target)
    monkeypatch.setattr(config, "DB_DIR", tmp_path)

    result = source_truth.update_source_of_truth(
        {"target": "finance", "old_text": "OLD", "new_text": "NEW"}
    )

    assert result.get("ok") is True
    assert target.read_text(encoding="utf-8") == "balance: NEW"
    assert _mode(target) == 0o600


def test_perspective_update_keeps_file_owner_only(tmp_path, monkeypatch):
    target = tmp_path / "will_perspective.md"
    target.write_text("I value OLD things.", encoding="utf-8")
    target.chmod(0o600)
    monkeypatch.setattr(config, "WILL_PERSPECTIVE_PATH", target)
    monkeypatch.setattr(config, "DB_DIR", tmp_path)
    monkeypatch.setattr(
        perspective_update.perspective,
        "append_identity_change",
        lambda old, new, reason: {"written": True},
    )

    result = perspective_update.update_will_perspective({"old_text": "OLD", "new_text": "NEW"})

    assert result.get("ok") is True
    assert _mode(target) == 0o600


def test_manual_export_file_is_owner_only(tmp_path, monkeypatch):
    monkeypatch.setattr(
        backup.portable,
        "export_data",
        lambda model=None: {"neural_map_export": {"memories": [], "graph": [], "count": 0}},
    )
    out = tmp_path / "memories.json"

    backup._export(SimpleNamespace(model=None, format="json", out=str(out)))

    assert out.exists()
    assert _mode(out) == 0o600


# --- store metadata: read-modify-write is serialized ------------------------------

def test_concurrent_metadata_updates_do_not_lose_fields():
    unit = MemoryUnit(id="meta_race_1", text="metadata race probe", category="note", importance=3)
    store.add(unit, embed_document(unit.text))

    workers, per_worker = 4, 10
    start = threading.Barrier(workers + 1)
    errors: list[Exception] = []

    def worker(n: int) -> None:
        try:
            start.wait(timeout=5)
            for i in range(per_worker):
                store.update_metadata(["meta_race_1"], {f"probe_{n}_{i}": 1})
                store.reinforce(["meta_race_1"])
        except Exception as exc:   # pragma: no cover - surfaced via assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(workers)]
    for thread in threads:
        thread.start()
    start.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=60)
        assert not thread.is_alive()
    assert errors == []

    meta = store.get_by_ids(["meta_race_1"])["meta_race_1"]["meta"]
    missing = [
        f"probe_{n}_{i}"
        for n in range(workers)
        for i in range(per_worker)
        if f"probe_{n}_{i}" not in meta
    ]
    assert missing == []
    assert int(meta.get("access_count", 0)) == workers * per_worker
