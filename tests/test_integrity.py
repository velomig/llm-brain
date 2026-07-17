"""Integrity audit catches boundary failures before they cascade."""
from __future__ import annotations

import server
from brain import graph, integrity, store
from util import store_memory


def test_integrity_clean_store_passes():
    fact = store_memory("Alex likes black coffee.", category="fact", importance=4)
    graph.add_relations(
        [{"subject": "Alex", "relation": "likes", "object": "black coffee"}],
        fact_id=fact.id,
        collection=store.collection_name(None),
    )

    report = integrity.run(fallback_limit=0)

    assert report["ok"] is True
    assert report["summary"]["errors"] == 0
    assert report["checks"]["store_metadata"]["count"] == 1
    assert report["checks"]["graph_edges"]["count"] == 1


def test_integrity_allows_multiple_graph_edges_for_one_fact():
    fact = store_memory("Alex likes coffee and tea.", category="fact", importance=4)
    graph.add_relations(
        [
            {"subject": "Alex", "relation": "likes", "object": "coffee"},
            {"subject": "Alex", "relation": "likes", "object": "tea"},
        ],
        fact_id=fact.id,
        collection=store.collection_name(None),
    )

    report = integrity.run(fallback_limit=0)

    assert report["ok"] is True
    assert report["checks"]["graph_edges"]["count"] == 2


def test_integrity_flags_dangling_graph_fact():
    graph.add_relations(
        [{"subject": "Alex", "relation": "holds", "object": "FNDX"}],
        fact_id="missing-fact",
        collection=store.collection_name(None),
    )

    report = integrity.run(fallback_limit=0)

    assert report["ok"] is False
    issues = report["checks"]["graph_edges"]["issues"]
    assert any(issue["code"] == "dangling_graph_fact" and issue["fact_id"] == "missing-fact" for issue in issues)


def test_integrity_repair_removes_dangling_graph_facts():
    graph.add_relations(
        [{"subject": "Alex", "relation": "holds", "object": "FNDX"}],
        fact_id="missing-fact",
        collection=store.collection_name(None),
    )

    preview = integrity.repair(dry_run=True)
    repaired = integrity.repair(dry_run=False)
    report = integrity.run(fallback_limit=0)

    assert preview["repairable"] == 1
    assert preview["deleted"] == 0
    assert repaired["deleted"] == 1
    assert report["ok"] is True


def test_deleting_graph_backed_memory_preserves_integrity():
    # Regression: DELETE /api/memory removed the record but left its graph
    # edges, so a clean report flipped to dangling_graph_fact immediately after
    # a normal, supported deletion.
    fact = store_memory("Alex likes black coffee.", category="fact", importance=4)
    graph.add_relations(
        [{"subject": "Alex", "relation": "likes", "object": "black coffee"}],
        fact_id=fact.id,
        collection=store.collection_name(None),
    )
    assert integrity.run(fallback_limit=0)["ok"] is True

    server._delete_memory_all(fact.id)

    report = integrity.run(fallback_limit=0)
    assert report["ok"] is True
    assert report["checks"]["graph_edges"]["count"] == 0


def test_deleting_source_preserves_integrity():
    # Same hole in bulk form: delete_source removed every record of a source
    # but left all their graph edges dangling.
    kept = store_memory("Alex walks daily.", category="fact", importance=4)
    graph.add_relations(
        [{"subject": "Alex", "relation": "does", "object": "daily walks"}],
        fact_id=kept.id,
        collection=store.collection_name(None),
    )
    doomed = store_memory(
        "Alex holds FNDX.", category="fact", importance=4, source="upload:abc:portfolio.pdf"
    )
    graph.add_relations(
        [{"subject": "Alex", "relation": "holds", "object": "FNDX"}],
        fact_id=doomed.id,
        collection=store.collection_name(None),
    )

    server._delete_source_all("upload:abc:portfolio.pdf")

    report = integrity.run(fallback_limit=0)
    assert report["ok"] is True
    # Only the deleted source's edges are gone; unrelated edges survive.
    assert report["checks"]["graph_edges"]["count"] == 1
    assert store.get_by_ids([kept.id])[kept.id]["text"] == "Alex walks daily."


def test_integrity_flags_generated_record_without_source_type():
    unit = store_memory("Generated-looking insight without provenance.", category="insight", importance=4)
    row = store.get_by_ids([unit.id])[unit.id]
    meta = dict(row["meta"])
    meta["source_type"] = ""
    store._collection().update(ids=[unit.id], metadatas=[meta])

    report = integrity.run(fallback_limit=0)

    assert report["ok"] is False
    issues = report["checks"]["store_metadata"]["issues"]
    assert any(issue["code"] == "generated_missing_source_type" for issue in issues)


def test_integrity_endpoint_returns_report():
    report = server.integrity_check(fallback_limit=0)

    assert "checks" in report
    assert "store_metadata" in report["checks"]


def test_integrity_repair_endpoint_dry_run():
    graph.add_relations(
        [{"subject": "Alex", "relation": "holds", "object": "FNDX"}],
        fact_id="missing-fact",
        collection=store.collection_name(None),
    )

    report = server.integrity_repair(dry_run=True)

    assert report["dry_run"] is True
    assert report["repairable"] == 1
    assert integrity.run(fallback_limit=0)["ok"] is False
