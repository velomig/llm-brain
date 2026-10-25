"""Portable backup: export -> wipe -> import restores memories (text + all
metadata) and graph edges, independent of Chroma/SQLite. Embeddings are recomputed
on import, so a backup round-trips even with the vector store emptied or embeddings
down. JSON is canonical; Markdown is a readable archive."""
from __future__ import annotations

import json

import pytest

from brain import graph, store
from brain.portable import EXPORT_VERSION, export_data, import_data, to_markdown
from util import store_memory

ACTIVE = None  # default model in tests


def _coll():
    return store.collection_name(None)


# --- export shape -------------------------------------------------------------

def test_export_envelope_and_modes():
    store_memory("alpha", category="fact")
    store_memory("beta", category="note", text_only=True)

    exp = export_data()["neural_map_export"]
    assert exp["version"] == EXPORT_VERSION
    assert exp["count"] == 2
    assert exp["embedding_model"]
    assert {m["mode"] for m in exp["memories"]} == {"vector", "text_only"}


# --- round-trip fidelity ------------------------------------------------------

def test_roundtrip_preserves_metadata_extras_and_graph():
    fact = store_memory("Alex prefers Ollama", category="fact", importance=5, source="upload:abc:notes.pdf")
    rem = store_memory("pay rent", category="reminder", importance=4, extra={"due": "2026-07-01", "status": "open"})
    graph.add_relations(
        [{"subject": "Alex", "relation": "uses", "object": "Ollama"}],
        fact_id=fact.id, date=fact.date, collection=_coll(),
    )

    data = export_data()

    store.clear_all()
    graph.clear()
    assert export_data()["neural_map_export"]["count"] == 0

    result = import_data(data)
    assert result == {"memories": 2, "edges": 1, "mode": "vector"}

    got = store.get_by_ids([fact.id, rem.id])
    assert got[fact.id]["meta"]["importance"] == 5
    assert got[fact.id]["meta"]["source"] == "upload:abc:notes.pdf"
    assert got[rem.id]["meta"]["due"] == "2026-07-01"        # extra field survives
    assert got[rem.id]["meta"]["status"] == "open"
    assert graph.fact_ids_for({"ollama"}, collection=_coll()) == [fact.id]


# --- export keeps the graph consistent with the memory payload ----------------

def test_export_excludes_dangling_and_foreign_edges():
    fact = store_memory("Alex uses Ollama", category="fact")
    graph.add_relations(                                   # good: fact is in the payload
        [{"subject": "Alex", "relation": "uses", "object": "Ollama"}],
        fact_id=fact.id, date=fact.date, collection=_coll(),
    )
    graph.add_relations(                                   # dangling: fact_id not stored
        [{"subject": "Ghost", "relation": "haunts", "object": "House"}],
        fact_id="not-a-real-memory", date=None, collection=_coll(),
    )
    graph.add_relations(                                   # foreign collection's fact
        [{"subject": "Other", "relation": "rel", "object": "Thing"}],
        fact_id="lives-in-model-b", date=None, collection="neural_map_vec_model_b",
    )

    exported = export_data()["neural_map_export"]["graph"]
    assert [e["fact_id"] for e in exported] == [fact.id]   # only the resolvable edge


def test_export_excludes_foreign_edge_with_same_fact_id():
    # Imported backups preserve memory ids, so the same fact id can exist as an
    # edge in another model collection. Export must keep only this model's edge.
    fact = store_memory("Alex uses Ollama", category="fact")
    graph.add_relations(
        [{"subject": "Alex", "relation": "uses", "object": "Ollama"}],
        fact_id=fact.id,
        date=fact.date,
        collection=_coll(),
    )
    graph.add_relations(
        [{"subject": "Alex", "relation": "foreign", "object": "OtherModel"}],
        fact_id=fact.id,
        date=fact.date,
        collection=FOREIGN_COLLECTION,
    )

    exported = export_data()["neural_map_export"]["graph"]
    assert len(exported) == 1
    assert exported[0]["collection"] == _coll()
    assert exported[0]["object"] == "Ollama"


def test_export_graph_fact_ids_are_subset_of_memories():
    a = store_memory("alpha fact", category="fact")
    b = store_memory("beta fact", category="fact")
    graph.add_relations([{"subject": "A", "relation": "r", "object": "B"}], fact_id=a.id, collection=_coll())
    graph.add_relations([{"subject": "C", "relation": "r", "object": "D"}], fact_id=b.id, collection=_coll())
    graph.add_relations([{"subject": "E", "relation": "r", "object": "F"}], fact_id="orphan", collection=_coll())

    payload = export_data()["neural_map_export"]
    mem_ids = {m["id"] for m in payload["memories"]}
    assert mem_ids >= {a.id, b.id}
    # The invariant: every exported edge resolves to an exported memory.
    assert all(e["fact_id"] in mem_ids for e in payload["graph"])


# --- cross-model portability: edges remap to the destination collection -------

FOREIGN_COLLECTION = "neural_map_vec_some_other_model_deadbeef"


def test_import_remaps_edges_to_destination_collection():
    # A backup made under a DIFFERENT embedding model: the edge's collection points
    # at a foreign store. After restore here, recall must find it under THIS store.
    fact = store_memory("Alex uses Ollama", category="fact")
    data = export_data()
    data["neural_map_export"]["graph"] = [
        {
            "subject": "Alex",
            "relation": "uses",
            "object": "Ollama",
            "fact_id": fact.id,
            "collection": FOREIGN_COLLECTION,
            "date": fact.date,
        }
    ]

    store.clear_all()
    graph.clear()
    import_data(data)   # restore under the default (destination) model

    dest = _coll()
    assert graph.fact_ids_for({"ollama"}, collection=dest) == [fact.id]          # now scoped here
    assert graph.fact_ids_for({"ollama"}, collection=FOREIGN_COLLECTION) == []   # no longer foreign


def test_import_rejects_graph_edges_for_unimported_facts():
    data = {
        "neural_map_export": {
            "version": EXPORT_VERSION, "exported_at": "x", "schema_version": 1,
            "embedding_model": "m", "count": 0, "memories": [],
            "graph": [{"subject": "A", "relation": "rel", "object": "B",
                       "fact_id": "ghost-not-imported", "collection": FOREIGN_COLLECTION, "date": None}],
        }
    }
    with pytest.raises(ValueError, match="outside this export"):
        import_data(data)
    assert graph.export_edges() == []


# --- replace vs merge ---------------------------------------------------------

def test_replace_is_authoritative():
    keep = store_memory("from the backup")
    data = export_data()
    extra = store_memory("created after the backup")

    import_data(data, replace=True)

    ids = {r["id"] for r in store.list_records_all()}
    assert keep.id in ids
    assert extra.id not in ids       # wiped — the backup is authoritative


def test_replace_preserves_foreign_graph_edges():
    keep = store_memory("from the backup", category="fact")
    data = export_data()
    graph.add_relations(
        [{"subject": "Foreign", "relation": "keeps", "object": "Edge"}],
        fact_id="foreign-fact",
        collection=FOREIGN_COLLECTION,
    )

    import_data(data, replace=True)

    assert graph.fact_ids_for({"foreign"}, collection=FOREIGN_COLLECTION) == ["foreign-fact"]
    assert graph.fact_ids_for({"foreign"}, collection=_coll()) == []
    assert keep.id in {r["id"] for r in store.list_records_all()}


def test_replace_clears_legacy_unscoped_graph_edges():
    keep = store_memory("from the backup", category="fact")
    data = export_data()
    graph.add_relations(
        [{"subject": "Legacy", "relation": "stale", "object": "Edge"}],
        fact_id=keep.id,
        collection=None,
    )

    import_data(data, replace=True)

    assert graph.fact_ids_for({"legacy"}, collection=_coll()) == []


def test_merge_overwrites_by_id_and_keeps_others():
    a = store_memory("original", category="fact")
    data = export_data()             # snapshot holding only `a`
    other = store_memory("separate") # created after the snapshot
    data["neural_map_export"]["memories"][0]["text"] = "rewritten"

    import_data(data, replace=False)

    got = store.get_by_ids([a.id, other.id])
    assert got[a.id]["text"] == "rewritten"      # overwritten by id
    assert got[other.id]["text"] == "separate"   # untouched by a merge


def test_merge_hard_embedding_failure_preserves_existing(monkeypatch):
    existing = store_memory("original", category="fact")
    data = export_data()
    data["neural_map_export"]["memories"][0]["text"] = "rewritten"

    def _boom(_texts, model=None):
        raise ConnectionError("ollama down")

    monkeypatch.setattr("brain.portable.embed_documents", _boom)

    with pytest.raises(ConnectionError):
        import_data(data, replace=False)

    got = store.get_by_ids([existing.id])
    assert got[existing.id]["text"] == "original"


def test_merge_write_failure_restores_existing(monkeypatch):
    existing = store_memory("original", category="fact")
    data = export_data()
    data["neural_map_export"]["memories"][0]["text"] = "rewritten"
    original_add_many = store.add_many
    calls = 0

    def _partial_write_then_fail(units, embeddings, **kwargs):
        nonlocal calls
        calls += 1
        original_add_many(units, embeddings, **kwargs)
        if calls == 1:
            raise RuntimeError("chroma write failed after partial insert")

    monkeypatch.setattr("brain.portable.store.add_many", _partial_write_then_fail)

    with pytest.raises(RuntimeError):
        import_data(data, replace=False)

    got = store.get_by_ids([existing.id])
    assert got[existing.id]["text"] == "original"
    assert store.count_all() == 1


def test_merge_replaces_graph_edges_of_overwritten_memories():
    a = store_memory("Alex likes coffee", category="fact")
    graph.add_relations(
        [{"subject": "Alex", "relation": "likes", "object": "coffee"}],
        fact_id=a.id,
        collection=_coll(),
    )
    data = export_data()   # snapshot: the memory plus its coffee edge
    # After the export, the memory's local associations change.
    graph.delete_fact_ids([a.id], collections=[_coll()])
    graph.add_relations(
        [{"subject": "Alex", "relation": "likes", "object": "tea"}],
        fact_id=a.id,
        collection=_coll(),
    )

    import_data(data, replace=False)

    # The overwritten memory carries only the imported association; the stale
    # local one must not survive alongside it.
    edges = [e for e in graph.export_edges() if e.get("fact_id") == a.id]
    assert {e["object"] for e in edges} == {"coffee"}


def test_merge_write_failure_restores_existing_graph(monkeypatch):
    existing = store_memory("original", category="fact")
    graph.add_relations(
        [{"subject": "Alex", "relation": "remembers", "object": "Original"}],
        fact_id=existing.id,
        collection=_coll(),
    )
    data = export_data()
    data["neural_map_export"]["memories"][0]["text"] = "rewritten"
    data["neural_map_export"]["graph"][0]["object"] = "Rewritten"

    original_add_many = store.add_many
    calls = 0

    def _fail_first_write(units, embeddings, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("chroma write failed")
        original_add_many(units, embeddings, **kwargs)

    monkeypatch.setattr("brain.portable.store.add_many", _fail_first_write)

    with pytest.raises(RuntimeError):
        import_data(data, replace=False)

    got = store.get_by_ids([existing.id])
    assert got[existing.id]["text"] == "original"
    edges = [e for e in graph.export_edges() if e.get("fact_id") == existing.id]
    assert {e["object"] for e in edges} == {"Original"}


def test_replace_write_failure_restores_existing_store_and_graph(monkeypatch):
    existing = store_memory("original", category="fact")
    graph.add_relations(
        [{"subject": "Alex", "relation": "remembers", "object": "Original"}],
        fact_id=existing.id,
        collection=_coll(),
    )
    data = export_data()
    data["neural_map_export"]["memories"][0]["text"] = "rewritten"
    original_add_many = store.add_many
    calls = 0

    def _partial_write_then_fail(units, embeddings, **kwargs):
        nonlocal calls
        calls += 1
        original_add_many(units, embeddings, **kwargs)
        if calls == 1:
            raise RuntimeError("chroma write failed after partial insert")

    monkeypatch.setattr("brain.portable.store.add_many", _partial_write_then_fail)

    with pytest.raises(RuntimeError):
        import_data(data, replace=True)

    got = store.get_by_ids([existing.id])
    assert got[existing.id]["text"] == "original"
    assert store.count_all() == 1
    assert graph.fact_ids_for({"alex"}, collection=_coll()) == [existing.id]


def test_merge_does_not_duplicate_across_stores():
    m = store_memory("re-imported twice")
    data = export_data()
    import_data(data)
    import_data(data)
    assert store.count_all() == 1                 # still single residence


# --- portability: embeddings recomputed on import -----------------------------

def test_import_falls_back_to_text_only_when_embeddings_down(fake_embed):
    fact = store_memory("Alex uses Ollama", category="fact")
    graph.add_relations(
        [{"subject": "Alex", "relation": "uses", "object": "Ollama"}],
        fact_id=fact.id,
        date=fact.date,
        collection=_coll(),
    )
    data = export_data()
    store.clear_all()
    graph.clear()

    fake_embed.available = False                  # simulate Ollama embedder down
    result = import_data(data)

    assert result["mode"] == "text_only"
    assert store.count(text_only=True) == 1
    assert graph.fact_ids_for({"ollama"}, collection=store.collection_name(None, text_only=True)) == [fact.id]
    assert graph.fact_ids_for({"ollama"}, collection=store.collection_name(None)) == []


# --- markdown archive ---------------------------------------------------------

def test_markdown_groups_by_category_and_lists_edges():
    fact = store_memory("Alex prefers Ollama", category="fact", importance=5)
    graph.add_relations(
        [{"subject": "Alex", "relation": "uses", "object": "Ollama"}],
        fact_id=fact.id, date=fact.date, collection=_coll(),
    )
    md = to_markdown(export_data())

    assert "# Neural Map Memory Export" in md
    assert "## fact" in md
    assert "Alex prefers Ollama" in md
    assert "Alex — *uses* → Ollama" in md


# --- validation ---------------------------------------------------------------

def test_import_rejects_non_export():
    with pytest.raises(ValueError):
        import_data({"not": "an export"})


# --- API download endpoint ----------------------------------------------------

def test_api_export_json_is_an_attachment():
    import server

    store_memory("downloadable")
    resp = server.export(format="json")
    assert resp.media_type == "application/json"
    assert "attachment" in resp.headers["content-disposition"]
    body = json.loads(resp.body)
    assert body["neural_map_export"]["count"] == 1


def test_api_export_markdown():
    import server

    store_memory("md memory", category="note")
    resp = server.export(format="md")
    assert resp.media_type == "text/markdown"
    assert b"# Neural Map Memory Export" in resp.body
