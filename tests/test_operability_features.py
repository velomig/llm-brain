"""Operability features: provenance, previews, pinning, backups, and diagnostics."""
from __future__ import annotations

import json

import server
from brain import graph, integrity, log, observability, portable, sleep, store, summarize
from util import store_memory


def test_memory_detail_and_pin_endpoint():
    unit = store_memory("keep me", category="episode")

    detail = server.memory_detail(unit.id)
    assert detail["text"] == "keep me"
    assert detail["source_type"] == "user_entered"
    assert detail["pinned"] is False

    out = server.pin_memory(unit.id, server.PinReq(pinned=True))
    assert out == {"id": unit.id, "pinned": True}
    assert server.memory_detail(unit.id)["pinned"] is True


def test_pinned_memories_sort_before_newer_unpinned_memories():
    old = store_memory("old pinned", category="note", date="2020-01-01T00:00:00+00:00")
    new = store_memory("new unpinned", category="note", date="2026-01-01T00:00:00+00:00")
    server.pin_memory(old.id, server.PinReq(pinned=True))

    rows = server.memories(q=None, category=None, source=None, model=None, limit=200)["memories"]

    assert rows[0]["id"] == old.id
    assert [row["id"] for row in rows[:2]] == [old.id, new.id]


def test_pinned_episode_is_not_pruned():
    unit = store_memory("old but important", category="episode", importance=1, date="2020-01-01T00:00:00+00:00")
    server.pin_memory(unit.id, server.PinReq(pinned=True))

    result = server.prune(server.PruneReq(dry_run=True))

    assert result["pruned"] == 0
    assert store.get_by_ids([unit.id])[unit.id]["text"] == "old but important"


def test_delete_source_preview_and_backup():
    store_memory("chunk one", source="upload:abc:file.md")
    store_memory("chunk two", source="upload:abc:file.md")

    preview = server.preview_delete_source(server.DeleteSourceReq(source="upload:abc:file.md"))
    assert preview["count"] == 2

    result = server.delete_source(server.DeleteSourceReq(source="upload:abc:file.md"))
    assert result["deleted"] == 2
    assert result["backup"]
    backup = json.loads(portable.BACKUP_DIR.joinpath(result["backup"].split("/")[-1]).read_text(encoding="utf-8"))
    assert backup["neural_map_export"]["count"] == 2


def test_filtered_export_only_includes_requested_category():
    store_memory("fact one", category="fact")
    store_memory("note one", category="note")

    resp = server.export(format="json", category="fact")
    payload = json.loads(resp.body.decode("utf-8"))["neural_map_export"]

    assert payload["count"] == 1
    assert payload["memories"][0]["category"] == "fact"


def test_logs_endpoint_exposes_recent_fallbacks():
    log.fallback("test_event", RuntimeError("boom"), phase="unit")

    events = server.logs(limit=5)["events"]

    assert events[0]["event"] == "test_event"
    assert events[0]["fields"] == {"phase": "unit"}
    assert "RuntimeError: boom" == events[0]["error"]


def test_traces_endpoint_exposes_recent_cognitive_traces():
    trace = observability.start("unit", model="test-model", user_text="private text", memories=2)
    observability.event(trace, "intuition", "system1", reason="unit")
    observability.finish(trace, "ok", path="system1")

    traces = server.traces(limit=3)["traces"]

    assert traces[0]["id"] == trace["id"]
    assert traces[0]["input"]["chars"] == len("private text")
    assert "private text" not in str(traces[0])
    assert traces[0]["events"][-1]["layer"] == "trace"
    assert traces[0]["metrics"]["event_count"] == len(traces[0]["events"])
    assert traces[0]["metrics"]["complexity_score"] == len(traces[0]["events"])
    assert traces[0]["metrics"]["latency_delta_ms"] >= 0


def test_trace_finish_adds_derived_telemetry(monkeypatch):
    timestamps = iter(
        [
            "2026-07-04T00:00:00+00:00",
            "2026-07-04T00:00:00.050000+00:00",
            "2026-07-04T00:00:01.250000+00:00",
            "2026-07-04T00:00:01.250000+00:00",
            "2026-07-04T00:00:01.250000+00:00",
        ]
    )
    monkeypatch.setattr(observability, "_now", lambda: next(timestamps))

    trace = observability.start("unit")
    observability.event(trace, "retrieval", "ok")
    finished = observability.finish(trace, "ok")

    assert finished["metrics"] == {
        "latency_delta_ms": 1250.0,
        "event_count": 3,
        "complexity_score": 3,
    }


def test_compact_old_episodes_creates_digest_and_backup(monkeypatch):
    old = store_memory("User: durable plan\nAssistant: ok", category="episode", date="2020-01-01T00:00:00+00:00")
    pinned = store_memory("pinned old episode", category="episode", date="2020-01-01T00:00:00+00:00")
    server.pin_memory(pinned.id, server.PinReq(pinned=True))
    monkeypatch.setattr(server.compact_old_episodes.__globals__["llm"], "chat", lambda system, user, model=None: "Durable plan digest.")

    preview = server.compact(server.CompactReq(dry_run=True))
    assert preview["eligible"] == 1

    result = server.compact(server.CompactReq())
    assert result["compacted"] == 1
    assert result["backup"]
    assert old.id not in store.get_by_ids([old.id])
    assert store.get_by_ids([pinned.id])[pinned.id]["text"] == "pinned old episode"
    digest = [r for r in store.list_records_all() if (r["meta"] or {}).get("category") == "episode_digest"]
    assert digest[0]["text"] == "Durable plan digest."


def test_summarize_replace_deletes_graph_edges_with_the_originals(monkeypatch):
    fact = store_memory("Alex likes black coffee.", category="fact")
    graph.add_relations(
        [{"subject": "Alex", "relation": "likes", "object": "black coffee"}],
        fact_id=fact.id,
        collection=store.collection_name(None),
    )
    monkeypatch.setattr(summarize.llm, "chat", lambda *a, **k: "Coffee profile.")

    assert summarize.summarize_category("fact", replace=True) == "Coffee profile."

    assert fact.id not in store.get_by_ids([fact.id])
    report = integrity.run(fallback_limit=0)
    assert not [
        issue
        for issue in report["checks"]["graph_edges"]["issues"]
        if issue["code"] == "dangling_graph_fact"
    ]


def test_compact_old_episodes_deletes_graph_edges_with_the_originals(monkeypatch):
    old = store_memory("User: durable plan\nAssistant: ok", category="episode", date="2020-01-01T00:00:00+00:00")
    graph.add_relations(
        [{"subject": "Alex", "relation": "planned", "object": "durable plan"}],
        fact_id=old.id,
        collection=store.collection_name(None),
    )
    monkeypatch.setattr(summarize.llm, "chat", lambda *a, **k: "Durable plan digest.")

    result = summarize.compact_old_episodes()

    assert result["compacted"] == 1
    report = integrity.run(fallback_limit=0)
    assert not [
        issue
        for issue in report["checks"]["graph_edges"]["issues"]
        if issue["code"] == "dangling_graph_fact"
    ]


def test_sleep_archive_deletes_graph_edges_with_the_originals():
    old = store_memory("User: passing small talk", category="episode", importance=1, date="2020-01-01T00:00:00+00:00")
    graph.add_relations(
        [{"subject": "Alex", "relation": "mentioned", "object": "small talk"}],
        fact_id=old.id,
        collection=store.collection_name(None),
    )

    result = sleep.run(max_dreams=0)

    assert result["archived"] == 1
    assert old.id not in store.get_by_ids([old.id])
    report = integrity.run(fallback_limit=0)
    assert not [
        issue
        for issue in report["checks"]["graph_edges"]["issues"]
        if issue["code"] == "dangling_graph_fact"
    ]


def test_empty_summary_never_replaces_the_category(monkeypatch):
    # Regression: with replace=True an empty model answer used to be stored as
    # the "summary" while the authoritative originals were deleted.
    kept = store_memory("dog fact one", category="dog")
    store_memory("dog fact two", category="dog")
    monkeypatch.setattr(summarize.llm, "chat", lambda *a, **k: "  \n ")

    out = summarize.summarize_category("dog", replace=True)

    assert out is None
    assert kept.id in store.get_by_ids([kept.id])   # originals untouched
    assert not [r for r in store.list_records_all() if (r["meta"] or {}).get("category") == "dog-summary"]


def test_empty_digest_keeps_old_episodes(monkeypatch):
    old = store_memory("User: durable plan\nAssistant: ok", category="episode", date="2020-01-01T00:00:00+00:00")
    monkeypatch.setattr(summarize.llm, "chat", lambda *a, **k: "")

    result = summarize.compact_old_episodes()

    assert result["compacted"] == 0
    assert "empty digest" in result["error"]
    assert old.id in store.get_by_ids([old.id])     # episodes untouched
    assert not [r for r in store.list_records_all() if (r["meta"] or {}).get("category") == "episode_digest"]
