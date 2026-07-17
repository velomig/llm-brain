"""Graph edges are model-scoped: each edge records the collection (embedding
model) holding its fact, recall filters to the active store, and re-asserting a
triple repoints fact_id AND collection together. Legacy edges (no collection)
stay best-effort until backfilled."""
from __future__ import annotations

from brain import graph, store
from brain.memory import now_iso
from brain.retriever import retrieve_weighted
from util import ids, store_memory

COLL_A = "neural_map_vec_modelA"
COLL_B = "neural_map_vec_modelB"


def _link(subject, relation, obj, fact_id, collection=None):
    graph.add_relations(
        [{"subject": subject, "relation": relation, "object": obj}],
        fact_id=fact_id,
        date=now_iso(),
        collection=collection,
    )


# --- scoping at the fact_ids level --------------------------------------------

def test_collection_recorded_and_scopes_recall():
    _link("Alex", "holds", "FNDW", "fact-A", collection=COLL_A)
    assert graph.fact_ids_for({"fndw"}, collection=COLL_A) == ["fact-A"]
    assert graph.fact_ids_for({"fndw"}, collection=COLL_B) == []     # foreign store
    assert graph.fact_ids_for({"fndw"}) == ["fact-A"]               # unscoped sees all


def test_legacy_null_edge_resolves_under_any_collection():
    _link("Alice", "knows", "Bob", "fact-legacy", collection=None)
    # NULL collection = "unknown / any model": kept as best-effort under any scope.
    assert graph.fact_ids_for({"bob"}, collection=COLL_A) == ["fact-legacy"]
    assert graph.fact_ids_for({"bob"}, collection=COLL_B) == ["fact-legacy"]


def test_same_triple_in_two_collections_is_independent():
    # P1 fix: the same relationship asserted in two model stores keeps a SEPARATE
    # edge each, so a later assert in one collection never steals the other's link.
    _link("Alex", "holds", "FNDZ", "fact-A", collection=COLL_A)
    _link("Alex", "holds", "FNDZ", "fact-B", collection=COLL_B)
    assert graph.stats()["edges"] == 2                                # one per collection
    assert graph.fact_ids_for({"fndz"}, collection=COLL_A) == ["fact-A"]   # A keeps its own
    assert graph.fact_ids_for({"fndz"}, collection=COLL_B) == ["fact-B"]


def test_reassert_within_same_collection_repoints():
    # Within ONE collection, restating a triple still upserts to the newest fact.
    _link("Alex", "holds", "FNDZ", "fact-old", collection=COLL_A)
    _link("Alex", "holds", "FNDZ", "fact-new", collection=COLL_A)
    assert graph.stats()["edges"] == 1
    assert graph.fact_ids_for({"fndz"}, collection=COLL_A) == ["fact-new"]


def test_null_incoming_collection_does_not_clobber():
    _link("Ada", "wrote", "Notes", "fact-1", collection=COLL_A)
    # A later assert to the SAME collection with no fact keeps the recorded one (COALESCE).
    _link("Ada", "wrote", "Notes", None, collection=COLL_A)
    assert graph.fact_ids_for({"ada"}, collection=COLL_A) == ["fact-1"]


# --- backfill -----------------------------------------------------------------

def test_backfill_stamps_only_null_edges():
    _link("X", "rel", "Y", "fact-null", collection=None)
    _link("P", "rel", "Q", "fact-tagged", collection=COLL_A)

    n = graph.backfill_collection(COLL_B)
    assert n == 1                                                     # only the NULL one
    assert graph.fact_ids_for({"y"}, collection=COLL_B) == ["fact-null"]
    assert graph.fact_ids_for({"y"}, collection=COLL_A) == []        # now scoped to B
    assert graph.backfill_collection(COLL_B) == 0                    # idempotent


def test_backfill_dry_run_changes_nothing():
    _link("X", "rel", "Y", "fact-null", collection=None)
    assert graph.backfill_collection(COLL_B, dry_run=True) == 1
    # Still NULL → still resolves under any collection.
    assert graph.fact_ids_for({"y"}, collection=COLL_A) == ["fact-null"]


# --- end-to-end through the retriever -----------------------------------------

def test_retrieval_scopes_graph_to_active_collection():
    # A real fact in the default (active) collection, linked under that collection,
    # is recalled; an edge pointing into a different model's collection is not.
    fact = store_memory("yield target is fifteen percent", category="fact", importance=4)
    graph.add_relations(
        [{"subject": "Alex", "relation": "holds", "object": "FNDW"}],
        fact_id=fact.id,
        date=fact.date,
        collection=store.collection_name(None),
    )
    _link("Alex", "holds", "VOOG", "ghost-other-model", collection=COLL_B)

    res = retrieve_weighted("tell me about FNDW and VOOG", w_sim=0, w_keyword=0, w_graph=1)
    got = ids(res)
    assert fact.id in got
    assert "ghost-other-model" not in got


def test_graph_expansion_does_not_bridge_through_foreign_collection():
    # Regression: graph.expand() used to match entities and walk neighbors across
    # all collections, then only scope the final fact-id lookup. That let a
    # foreign Alex -> Bridge edge pull in a local Bridge fact for a Alex query,
    # even though the active collection never linked Alex to Bridge.
    bridge_fact = store_memory("Bridge-only local fact", category="fact", importance=4)
    graph.add_relations(
        [{"subject": "Bridge", "relation": "mentions", "object": "LocalFact"}],
        fact_id=bridge_fact.id,
        date=bridge_fact.date,
        collection=store.collection_name(None),
    )
    _link("Alex", "foreign-link", "Bridge", "foreign-fact", collection=COLL_B)

    scoped = [
        store.collection_name(None),
        store.collection_name(None, text_only=True),
    ]
    assert graph.expand("what about Alex?", collection=scoped, limit=None) == []
