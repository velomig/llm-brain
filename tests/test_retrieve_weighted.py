"""retrieve_weighted: candidate production (semantic, keyword, graph, AND the
recency/importance pulls), ranking, and the all-weights-zero empty guarantee."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import config
from brain import store
from brain.retriever import retrieve_weighted
from util import ids, store_memory

# Disjoint vocabularies so a query that overlaps exactly one memory ranks it top.
TRAVEL = "retirement travel nursing plan income 2033 switch"
COFFEE = "espresso machine grinder beans roast morning"
BIKE = "mountain bicycle trail suspension pedals helmet"


def _seed_three():
    return (
        store_memory(TRAVEL),
        store_memory(COFFEE),
        store_memory(BIKE),
    )


def test_semantic_retrieval_ranks_relevant_first():
    travel, _, _ = _seed_three()
    # Pure semantic: isolate the vector signal from keyword/recency/importance.
    res = retrieve_weighted(
        "what is the travel nursing retirement plan",
        w_sim=1, w_keyword=0, w_recency=0, w_importance=0, w_graph=0,
    )
    assert res, "semantic retrieval returned nothing"
    assert res[0]["id"] == travel.id


def test_keyword_finds_exact_token_semantic_might_miss():
    # BM25's IDF only rewards RARE tokens, so seed filler docs that lack the code;
    # this mirrors the real corpus where an order id appears in just one memory.
    for i in range(6):
        store_memory(f"unrelated filler note {i} about gardening and weather")
    target = store_memory("Order ticket reference ACME-ZZ9 is on file")
    res = retrieve_weighted(
        "ACME-ZZ9",
        w_sim=0, w_keyword=1, w_recency=0, w_importance=0, w_graph=0,
    )
    assert target.id in ids(res)


def test_superseded_memory_is_excluded():
    m = store_memory(TRAVEL)
    store.mark_superseded([m.id])
    res = retrieve_weighted("travel nursing retirement plan", w_sim=1, w_keyword=1)
    assert m.id not in ids(res)


def test_superseded_do_not_starve_active_semantic_recall(monkeypatch):
    # Retired duplicates are the NEAREST vectors; with a tight candidate_k they
    # would fill the ANN pull and the active match (a hair farther, one extra
    # token) would never enter the candidate set. Filtering superseded inside the
    # producer — not just after the union — is what keeps the active one alive.
    # Balanced recall off so the single global candidate_k pull is the only path
    # (otherwise the per-category pull would surface the active one regardless).
    monkeypatch.setattr(config, "BALANCED_RECALL", False)
    for _ in range(3):
        old = store_memory(TRAVEL)
        store.mark_superseded([old.id])
    active = store_memory(TRAVEL + " current")
    res = retrieve_weighted(
        TRAVEL,
        w_sim=1, w_keyword=0, w_recency=0, w_importance=0, w_graph=0,
        candidate_k=3,
    )
    assert active.id in ids(res)


def test_recency_only_surfaces_most_recent():
    # Recency is now a real candidate source: with semantic/keyword/graph all off
    # and a query that matches nothing lexically, the newest memory still surfaces.
    store_memory(COFFEE, date="2020-01-01T00:00:00+00:00")
    newest = store_memory(BIKE, date="2026-06-01T00:00:00+00:00")
    res = retrieve_weighted(
        "completely unrelated words zzz",
        w_sim=0, w_keyword=0, w_graph=0, w_recency=5, w_importance=0,
    )
    assert res, "recency-only retrieval returned nothing"
    assert res[0]["id"] == newest.id


def test_continuous_recency_reranks_semantic_candidates_toward_fresh_fact():
    # The stale wording is an exact semantic match; the fresh record has extra
    # tokens and is therefore slightly farther away in embedding space. The
    # continuous forgetting curve must overcome that tiny ANN advantage.
    stale = store_memory(TRAVEL, date="2020-01-01T00:00:00+00:00")
    fresh = store_memory(TRAVEL + " updated current record")

    res = retrieve_weighted(
        TRAVEL,
        top_k=2,
        w_sim=1,
        w_keyword=0,
        w_recency=config.W_RECENCY,
        w_importance=0,
        w_graph=0,
        w_sensory=0,
    )
    by_id = {row["id"]: row for row in res}

    assert by_id[stale.id]["similarity"] > by_id[fresh.id]["similarity"]
    assert by_id[fresh.id]["recency"] > by_id[stale.id]["recency"]
    assert (
        by_id[fresh.id]["score_components"]["recency"]
        > by_id[stale.id]["score_components"]["recency"]
    )
    assert res[0]["id"] == fresh.id


def test_recency_halflife_materially_changes_final_score(monkeypatch):
    dated = store_memory(
        "time decay calibration note",
        date=(datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
    )

    monkeypatch.setattr(config, "RECENCY_HALFLIFE_DAYS", 30.0)
    normal = retrieve_weighted(
        "unrelated",
        w_sim=0,
        w_keyword=0,
        w_recency=1,
        w_importance=0,
        w_graph=0,
        w_sensory=0,
    )[0]
    monkeypatch.setattr(config, "RECENCY_HALFLIFE_DAYS", 10.0)
    fast_decay = retrieve_weighted(
        "unrelated",
        w_sim=0,
        w_keyword=0,
        w_recency=1,
        w_importance=0,
        w_graph=0,
        w_sensory=0,
    )[0]

    assert normal["id"] == fast_decay["id"] == dated.id
    assert 0.49 < normal["recency"] < 0.51
    assert normal["score_components"]["recency"] > fast_decay["score_components"]["recency"]
    assert normal["score"] > fast_decay["score"]


def test_importance_only_surfaces_most_important():
    store_memory(COFFEE, importance=1)
    top = store_memory(BIKE, importance=5)
    res = retrieve_weighted(
        "completely unrelated words zzz",
        w_sim=0, w_keyword=0, w_graph=0, w_recency=0, w_importance=5,
    )
    assert res, "importance-only retrieval returned nothing"
    assert res[0]["id"] == top.id


def test_finance_and_nursing_memories_are_opt_in_for_general_queries():
    finance = store_memory(
        "Portfolio shares stock ticker allocation note",
        category="episode",
        importance=5,
        extra={"pinned": True},
        date="2026-07-09T00:00:00+00:00",
    )
    nursing = store_memory(
        "Nursing clinical NCLEX medication patient note",
        category="episode",
        importance=5,
        date="2026-07-09T00:00:00+00:00",
    )
    general = store_memory(
        "Election commission news and president briefing",
        category="wisdom",
        importance=2,
        date="2020-01-01T00:00:00+00:00",
    )

    res = retrieve_weighted(
        "president election commission news",
        top_k=10,
        w_sim=0,
        w_keyword=0,
        w_recency=5,
        w_importance=5,
        w_graph=0,
        w_sensory=0,
    )

    assert finance.id not in ids(res)
    assert nursing.id not in ids(res)
    assert general.id in ids(res)


def test_finance_and_nursing_memories_return_when_prompted():
    finance = store_memory("Portfolio shares stock ticker allocation note", category="episode")
    nursing = store_memory("Nursing clinical NCLEX medication patient note", category="episode")

    finance_res = retrieve_weighted(
        "portfolio stock allocation",
        top_k=10,
        w_sim=0,
        w_keyword=0,
        w_recency=1,
        w_importance=1,
        w_graph=0,
        w_sensory=0,
    )
    nursing_res = retrieve_weighted(
        "nursing clinical medication",
        top_k=10,
        w_sim=0,
        w_keyword=0,
        w_recency=1,
        w_importance=1,
        w_graph=0,
        w_sensory=0,
    )

    assert finance.id in ids(finance_res)
    assert nursing.id in ids(nursing_res)


def test_current_news_queries_do_not_recall_old_raw_episodes_by_default():
    stale_episode = store_memory(
        "User: any news today? Assistant: no specific news items were found.",
        category="episode",
        importance=5,
        date="2026-07-09T00:00:00+00:00",
    )
    durable = store_memory(
        "When web search fails, say the provider failed instead of claiming no news exists.",
        category="wisdom",
        importance=4,
        date="2026-07-09T00:00:00+00:00",
    )

    news = ids(retrieve_weighted(
        "president election news today",
        top_k=10,
        w_sim=0,
        w_keyword=0,
        w_recency=5,
        w_importance=5,
        w_graph=0,
        w_sensory=0,
    ))
    recall = ids(retrieve_weighted(
        "recall recent memories about president election news today",
        top_k=10,
        w_sim=0,
        w_keyword=0,
        w_recency=5,
        w_importance=5,
        w_graph=0,
        w_sensory=0,
    ))

    assert stale_episode.id not in news
    assert durable.id in news
    assert stale_episode.id in recall


def test_first_person_recall_with_today_keeps_episodes():
    # Regression: "today" used to be taken as proof of an external news query,
    # dropping every episode — but "what did I say today about the dentist?"
    # is episodic recall, and episodes are the only records that can answer it.
    episode = store_memory(
        "User: my dentist moved the crown appointment to Friday. Assistant: noted.",
        category="episode",
        importance=3,
        date="2026-07-10T09:00:00+00:00",
    )

    said = ids(retrieve_weighted(
        "what did I say today about the dentist?",
        top_k=10,
        w_sim=0,
        w_keyword=0,
        w_recency=5,
        w_importance=5,
        w_graph=0,
        w_sensory=0,
    ))
    told = ids(retrieve_weighted(
        "what did you tell me today about the dentist?",
        top_k=10,
        w_sim=0,
        w_keyword=0,
        w_recency=5,
        w_importance=5,
        w_graph=0,
        w_sensory=0,
    ))

    assert episode.id in said
    assert episode.id in told


def test_importance_producer_respects_category_scope():
    keep = store_memory("alpha note", category="keep", importance=5)
    drop = store_memory("beta note", category="other", importance=5)
    got = ids(retrieve_weighted(
        "anything", w_sim=0, w_keyword=0, w_graph=0, w_importance=5, category="keep",
    ))
    assert keep.id in got
    assert drop.id not in got


def test_pinned_general_memory_is_prompt_visible_without_word_match():
    pinned = store_memory(
        "Current household note: the garage keypad battery needs replacement.",
        category="note",
        importance=3,
        date="2024-01-01T00:00:00+00:00",
    )
    store.update_metadata_all([pinned.id], {"pinned": True})
    for i in range(6):
        store_memory(
            f"fresh unrelated gardening task {i}",
            importance=5,
            date=f"2026-01-0{(i % 6) + 1}T00:00:00+00:00",
        )

    got = ids(retrieve_weighted("can you look at my note?", top_k=5))

    assert pinned.id in got


def test_pinned_finance_memory_still_requires_finance_prompt():
    pinned = store_memory(
        "Current portfolio note: FNDX 40%, FNDY 30%, FNDZ 30%.",
        category="note",
        importance=3,
        date="2024-01-01T00:00:00+00:00",
    )
    store.update_metadata_all([pinned.id], {"pinned": True})

    general = ids(retrieve_weighted("can you look at my note?", top_k=5))
    finance = ids(retrieve_weighted("can you look at my portfolio note?", top_k=5))

    assert pinned.id not in general
    assert pinned.id in finance


def test_pinned_producer_respects_category_scope():
    keep = store_memory("pinned scoped note", category="keep")
    drop = store_memory("pinned other note", category="other")
    store.update_metadata_all([keep.id, drop.id], {"pinned": True})

    got = ids(retrieve_weighted("anything", category="keep"))

    assert keep.id in got
    assert drop.id not in got


def test_important_distractor_does_not_outrank_relevant_at_defaults():
    # The noise guard for making importance a producer: at DEFAULT weights, a
    # high-importance but irrelevant memory must NOT displace the genuinely
    # relevant semantic hit. (Importance is now a candidate source, but RRF keeps
    # it subordinate to a real match.)
    travel = store_memory(TRAVEL, importance=3)
    store_memory(COFFEE, importance=5)   # important, but irrelevant to the query
    store_memory(BIKE, importance=5)     # ditto
    res = retrieve_weighted("what is the travel nursing retirement plan")  # all defaults
    assert res[0]["id"] == travel.id


def test_precision_filter_drops_unmatched_recent_and_important_noise_at_defaults():
    relevant = store_memory(
        "ACME ticker decline catalyst news includes Starship FAA launch update",
        importance=3,
        date="2024-01-01T00:00:00+00:00",
    )
    noisy = store_memory(
        "Hospital nursing medication schedule and clinical staffing checklist",
        importance=5,
        date="2026-07-08T00:00:00+00:00",
    )

    got = ids(retrieve_weighted("ACME stock decline catalyst news", top_k=10))

    assert relevant.id in got
    assert noisy.id not in got


def test_precision_filter_keeps_broad_memory_sweeps():
    durable = store_memory(
        "Emergency contact and durable personal profile note",
        importance=5,
        date="2024-01-01T00:00:00+00:00",
    )

    got = ids(retrieve_weighted("what do you remember about me?", top_k=5))

    assert durable.id in got


def test_all_weights_zero_yields_no_candidates():
    # With every signal off, nothing produces — the result is empty. (The API layer
    # rejects this up front; the retriever still handles it safely.)
    _seed_three()
    res = retrieve_weighted(
        "travel nursing retirement plan",
        w_sim=0, w_keyword=0, w_graph=0, w_recency=0, w_importance=0,
    )
    assert res == []


def test_top_k_limits_result_count():
    for i in range(8):
        store_memory(f"shared marker token entry number {i}")
    res = retrieve_weighted("shared marker token", top_k=3, w_sim=1, w_keyword=1)
    assert len(res) == 3


def test_tied_results_have_stable_order():
    first = store_memory("alpha tied note", importance=3, date="2026-01-01T00:00:00+00:00")
    second = store_memory("beta tied note", importance=3, date="2026-01-01T00:00:00+00:00")

    res = retrieve_weighted(
        "unrelated query zzz",
        top_k=2,
        w_sim=0,
        w_keyword=0,
        w_graph=0,
        w_recency=0,
        w_importance=1,
    )

    assert ids(res) == sorted([first.id, second.id])


def test_tied_recency_and_importance_score_identically():
    # Recency and importance are coarse signals (same-second dates, 1-5 scale),
    # so ties are the NORM. Tied values must share an RRF rank: splitting them
    # into distinct ranks hands out score differences bigger than a real
    # semantic/keyword edge, ordered only by random memory id — which made a
    # genuinely-matching memory lose to noise in ~1/15 runs of the PDF eval.
    same_date = "2026-01-01T00:00:00+00:00"
    a = store_memory("alpha equal note", importance=3, date=same_date)
    b = store_memory("beta equal note", importance=3, date=same_date)

    res = retrieve_weighted(
        "unrelated query zzz",
        top_k=5,
        w_sim=0,
        w_keyword=0,
        w_graph=0,
        w_recency=1,
        w_importance=1,
    )

    by_id = {r["id"]: r["score"] for r in res}
    assert by_id[a.id] == by_id[b.id]


def test_empty_store_returns_empty():
    assert retrieve_weighted("anything at all") == []
