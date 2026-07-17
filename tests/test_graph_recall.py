"""Graph-based associative recall: entities in the question pull in linked fact
memories that semantic/keyword would miss — plus the P2 upsert/supersede edges."""
from __future__ import annotations

import config
from brain import graph, store
from brain.retriever import retrieve_weighted
from util import ids, store_memory


def _link(fact: str, subject: str, relation: str, obj: str):
    """Store a fact and link it to a (subject, relation, object) edge."""
    unit = store_memory(fact, category="fact", importance=4)
    graph.add_relations(
        [{"subject": subject, "relation": relation, "object": obj}],
        fact_id=unit.id,
        date=unit.date,
    )
    return unit


def test_graph_pulls_linked_fact_when_other_signals_off():
    # The fact text shares no words with the question, so only the graph edge
    # (question mentions "FNDW" -> linked fact) can surface it.
    fact = _link("Distribution yield target is fifteen percent", "Alex", "holds", "FNDW")
    res = retrieve_weighted(
        "tell me about FNDW",
        w_sim=0, w_keyword=0, w_graph=1, w_recency=0, w_importance=0,
    )
    assert fact.id in ids(res)
    assert next(r for r in res if r["id"] == fact.id)["graph_linked"] is True


def test_graph_expand_returns_linked_fact_ids():
    fact = _link("some durable fact", "Alice", "knows", "Bob")
    assert fact.id in graph.expand("what about Bob")


def test_superseded_linked_fact_is_dropped_from_recall():
    # Even though the edge still points at it, a retired fact must not surface.
    fact = _link("retired position detail", "Alex", "held", "SHLD")
    store.mark_superseded([fact.id])
    res = retrieve_weighted("tell me about SHLD", w_sim=0, w_keyword=0, w_graph=1)
    assert fact.id not in ids(res)


def test_superseded_linked_facts_do_not_starve_active_graph_recall(monkeypatch):
    # Retired facts linked to the same entity would fill GRAPH_RECALL_K before the
    # active fact (linked last) is reached, and expand()'s fact ids carry no
    # active/retired ordering. The robust fix pulls the FULL scoped id set and
    # admits only active facts — so MANY more retired facts than any bounded
    # over-fetch window must not starve the live one. Distinct triples (not the
    # same one restated) so each keeps its own edge rather than repointing.
    monkeypatch.setattr(config, "GRAPH_RECALL_K", 1)
    for i in range(8):   # deliberately deep — would overflow a bounded pre-filter cap
        old = _link(f"analyst{i} retired view", f"Analyst{i}", "rates", "NVDA")
        store.mark_superseded([old.id])
    active = _link("current NVDA thesis", "Alex", "holds", "NVDA")
    # Isolate the graph signal — otherwise the recency/importance producers would
    # surface the active fact on their own and mask graph starvation.
    res = retrieve_weighted(
        "tell me about NVDA",
        w_sim=0, w_keyword=0, w_graph=1, w_recency=0, w_importance=0,
    )
    assert active.id in ids(res)


def test_reasserting_triple_repoints_edge_to_newest_fact():
    # P2 regression: a newer fact restating the same triple must win, and the
    # edge must not duplicate.
    old = _link("old yield note", "Alex", "holds", "FNDZ")
    new = store_memory("new yield note", category="fact", importance=4)
    graph.add_relations(
        [{"subject": "Alex", "relation": "holds", "object": "FNDZ"}],
        fact_id=new.id,
        date=new.date,
    )
    linked = graph.fact_ids_for({"fndz"})
    assert linked == [new.id]
    assert old.id not in linked
    assert graph.stats()["edges"] == 1
