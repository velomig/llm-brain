"""Text-only mode is first-class: counting, searching, listing, categorizing, and
managing memories all span BOTH the vector and the text-only collections.

These exercise the centralized store.*_all helpers and the paths that use them, so
a memory written in degraded (embeddings-down) mode is never invisible."""
from __future__ import annotations

from datetime import date

import pytest

from brain import reflect, store, summarize
from brain.retriever import retrieve_weighted
from util import ids, store_memory


def _vec(text, **kw):
    return store_memory(text, text_only=False, **kw)


def _txt(text, **kw):
    return store_memory(text, text_only=True, **kw)


# --- store-level both-store helpers -------------------------------------------

def test_count_all_spans_both_stores():
    _vec("vector one")
    _txt("text-only one")
    _txt("text-only two")
    assert store.count_all() == 3
    assert store.count(text_only=False) == 1
    assert store.count(text_only=True) == 2


def test_categories_all_includes_text_only_only_category():
    _vec("a", category="vec_cat")
    _txt("b", category="txt_cat")   # exists ONLY in the text-only store
    assert store.categories_all() == ["txt_cat", "vec_cat"]


def test_list_records_all_tags_mode_and_covers_both():
    v = _vec("vector row")
    t = _txt("text-only row")
    rows = {r["id"]: r for r in store.list_records_all()}
    assert rows[v.id]["mode"] == "vector"
    assert rows[t.id]["mode"] == "text_only"


def test_get_by_category_all_merges_both_with_mode():
    _vec("v note", category="shared")
    _txt("t note", category="shared")
    got = store.get_by_category_all("shared")
    assert sorted(got["documents"]) == ["t note", "v note"]
    assert set(got["modes"]) == {"vector", "text_only"}


def test_delete_all_removes_a_text_only_record_by_id():
    t = _txt("doomed text-only memory")
    store.delete_all([t.id])   # caller doesn't know which store holds it
    assert store.count_all() == 0


def test_mark_superseded_all_retires_text_only_record():
    t = _txt("stale text-only fact")
    assert store.mark_superseded_all([t.id]) == 1
    res = retrieve_weighted("stale text-only fact", w_sim=0, w_keyword=1, w_graph=0)
    assert t.id not in ids(res)   # superseded → filtered from recall


def test_reinforce_all_bumps_text_only_record():
    t = _txt("recalled text-only memory")
    assert store.reinforce_all([t.id]) == 1


def test_delete_source_all_clears_text_only_chunks():
    _txt("chunk from a file", source="src:file")
    assert store.count_all() == 1
    store.delete_source_all("src:file")
    assert store.count_all() == 0


# --- the named higher-level paths ---------------------------------------------

def test_reminders_surface_from_text_only_store():
    today = date.today().isoformat()
    _txt(f"[due {today}] text-only reminder", category="reminder",
         extra={"due": today, "status": "open"})
    rems = reflect.active_reminders()
    assert any("text-only reminder" in r["text"] for r in rems)


def test_summarize_spans_text_only_memories(monkeypatch):
    _vec("Dog likes long walks at dawn.", category="dog")
    _txt("Dog is afraid of thunderstorms.", category="dog")
    captured = {}

    def _fake_chat(system, user, model=None):
        captured["user"] = user
        return "Dog profile."

    monkeypatch.setattr(summarize.llm, "chat", _fake_chat)
    out = summarize.summarize_category("dog")
    assert out == "Dog profile."
    assert "long walks" in captured["user"]       # vector memory included
    assert "thunderstorms" in captured["user"]     # text-only memory included


def test_importance_producer_reaches_text_only_memories():
    # A high-importance memory written in degraded mode must still surface via the
    # importance producer with no semantic/keyword/graph signal.
    t = _txt("critical text-only note", importance=5)
    res = retrieve_weighted(
        "totally unrelated query zzz",
        w_sim=0, w_keyword=0, w_graph=0, w_recency=0, w_importance=5,
    )
    assert t.id in ids(res)
