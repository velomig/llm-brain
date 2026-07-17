"""Hybrid retrieval with RRF plus calibrated continuous time decay.

Four core signals, fused without mixing raw cosine and BM25 score scales:
  1. Semantic  — ChromaDB ANN over the HNSW index (efficient; no full load)
  2. Keyword   — BM25 (cached; rebuilt only when the collection changes)
  3. Recency   — normalized exponential decay over the candidate union
  4. Importance — manual 1-5 rank over the candidate union

Ranked signals contribute ``w / (RRF_K + rank)``. Recency is already normalized
to [0, 1], so it contributes ``w_recency * decay / RRF_K``: the same scale as a
top-ranked RRF signal while preserving the actual half-life magnitude.

RRF needs no score normalization (it only uses ranks), which makes fusing a
cosine similarity with a BM25 score robust. The four config weights still apply
as relative emphasis on each signal.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import numpy as np
from rank_bm25 import BM25Okapi

import config
from brain import domains as domain_routing, evolution, graph, log, sensory, store
from brain.embeddings import EmbeddingUnavailable, embed_query
from brain.memory import clamp_importance, recency_score


def _tok(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


_QUESTION_STOPWORDS = {
    "a", "about", "after", "again", "all", "also", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "between", "but",
    "by", "can", "could", "did", "do", "does", "for", "from", "get", "give",
    "had", "has", "have", "help", "how", "i", "if", "in", "into", "is", "it",
    "its", "just", "know", "let", "like", "look", "me", "my", "need", "now",
    "of", "on", "or", "our", "please", "should", "so", "tell", "than", "that",
    "the", "their", "them", "then", "there", "these", "this", "those", "to",
    "today", "try", "use", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "would", "you", "your",
}

_BROAD_MEMORY_PATTERNS = (
    "what do you remember",
    "what do you know about me",
    "what memories",
    "show me memories",
    "recall memories",
    "recent memories",
    "important memories",
    "old memory",
    "old memories",
    "my memories",
)
_CURRENT_EXTERNAL_TERMS = {
    "breaking",
    "current",
    "headline",
    "headlines",
    "latest",
    "live",
    "news",
    "recent",
    "today",
}

_DOMAIN_OPT_IN_TERMS = {
    "finance": {
        "allocation", "btc", "buy", "cbrs", "dividend", "etf", "fidelity",
        "finance", "finances", "financial", "fxaix", "holding", "holdings",
        "investment", "investments", "market", "markets", "money", "portfolio",
        "price", "prices", "quote", "quotes", "schd", "sgov", "shares", "spcx",
        "stock", "stocks", "ticker", "tickers", "trade", "trading",
    },
    "nursing": {
        "clinical", "clinicals", "exam", "lab", "labs", "medication",
        "medications", "nclex", "nurse", "nurses", "nursing", "patient",
        "patients", "pharmacology", "rn", "semester",
    },
}


def _question_terms(question: str) -> set[str]:
    return {
        tok
        for tok in _tok(question)
        if len(tok) > 2 and tok not in _QUESTION_STOPWORDS
    }


def _memory_terms(row: dict) -> set[str]:
    parts = [
        row.get("text") or "",
        row.get("category") or "",
        row.get("source") or "",
    ]
    return {tok for tok in _tok(" ".join(parts)) if len(tok) > 2}


def _domain_opt_ins(question: str) -> set[str]:
    """Fallback when the caller didn't supply a turn-level domain decision.

    Shares brain.domains with prompt construction, in the permissive mode:
    for retrieval visibility a single weak term opts in, because hiding the
    user's own domain memories costs more than letting them compete in rank."""
    return domain_routing.detect(question, strict=False)


def _row_protected_domains(row: dict) -> set[str]:
    terms = _memory_terms(row)
    category = str(row.get("category") or "").casefold()
    source = str(row.get("source") or "").casefold()
    domains = set()
    for domain, domain_terms in _DOMAIN_OPT_IN_TERMS.items():
        if domain in category or domain in source or terms & domain_terms:
            domains.add(domain)
    return domains


def _is_broad_memory_question(question: str) -> bool:
    text = " ".join(_tok(question))
    if any(pattern in text for pattern in _BROAD_MEMORY_PATTERNS):
        return True
    terms = set(_tok(question))
    return bool({"memory", "memories", "remember", "recall", "notes"} & terms) and bool(
        {"all", "recent", "important", "old", "about", "know", "me"} & terms
    )


_CONVERSATION_RECALL_PRONOUNS = {"i", "we", "you"}
_CONVERSATION_RECALL_VERBS_PAST = {"asked", "chatted", "discussed", "mentioned", "said", "talked", "told", "wrote"}
_CONVERSATION_RECALL_VERBS_BASE = {"ask", "chat", "discuss", "mention", "say", "talk", "tell", "write"}


def _is_conversation_recall_question(question: str) -> bool:
    """First-person recall of a prior conversation — "what did I say today about
    the dentist?". A recency word in such a question scopes the recall window,
    not an external news query, so it must not suppress episodic memory. Past
    tense is the discriminator: "tell me today's headlines" (imperative) stays
    external, "what did you tell me today" (recall) does not."""
    terms = set(_tok(question))
    if not terms & _CONVERSATION_RECALL_PRONOUNS:
        return False
    if terms & _CONVERSATION_RECALL_VERBS_PAST:
        return True
    return "did" in terms and bool(terms & _CONVERSATION_RECALL_VERBS_BASE)


def _is_current_external_question(question: str) -> bool:
    if not set(_tok(question)) & _CURRENT_EXTERNAL_TERMS:
        return False
    return not _is_conversation_recall_question(question)


def _rrf(rank: int) -> float:
    return 1.0 / (config.RRF_K + rank)


def _competition_rank(cand: set[str], value_by_id: dict[str, float]) -> dict[str, int]:
    """Rank ids by value (descending); tied values share a rank (1224-style)."""
    ranks: dict[str, int] = {}
    rank = 0
    prev_value: float | None = None
    for index, cid in enumerate(sorted(cand, key=lambda x: (-value_by_id[x], x))):
        value = value_by_id[cid]
        if prev_value is None or value != prev_value:
            rank = index
            prev_value = value
        ranks[cid] = rank
    return ranks


# Retired memories carry superseded=True; active records omit the key entirely.
# `{"$ne": True}` matches both the omitted key and an explicit False, so it keeps
# active records and drops retired ones. Pushing this into the ANN `where` makes
# Chroma return candidate_k ACTIVE neighbors, instead of filling the cap with
# retired vectors that get dropped later and starve active memories below the
# cutoff (verified on chromadb 1.5.9).
_ACTIVE_WHERE: dict = {"superseded": {"$ne": True}}


def _active_where(extra: dict | None = None) -> dict:
    """The not-superseded filter, AND-combined with an optional extra clause."""
    if not extra:
        return dict(_ACTIVE_WHERE)
    return {"$and": [dict(_ACTIVE_WHERE), extra]}


# --- cached BM25/category indexes keyed by cheap persisted store revisions ------
_RECORDS_CACHE: dict[str, tuple] = {}   # model key -> (signature, ids, texts, metas)
_BM25_CACHE: dict[str, tuple] = {}      # model key -> (signature, bm25|None, ids, texts, metas)
_CATEGORY_CACHE: dict[str, tuple] = {}  # model key -> (signature, categories)


def _model_cache_key(model: str | None = None) -> str:
    return store.collection_name(model)


def _collection_signature(model: str | None = None, text_only: bool = False) -> tuple[int, int]:
    return store.count(model, text_only=text_only), store.collection_version(model, text_only=text_only)


def _combined_signature(model: str | None = None) -> tuple[tuple[int, int], tuple[int, int]]:
    return _collection_signature(model), _collection_signature(model, text_only=True)


def _combined_records(model: str | None = None):
    """All (ids, texts, metas) across both collections, cached by store revision.

    Shared by the BM25 index and the recency/importance candidate producers, so
    the whole store is loaded into RAM at most once per change, per query path.
    """
    signature = _combined_signature(model)
    key = _model_cache_key(model)
    cached = _RECORDS_CACHE.get(key)
    if cached and cached[0] == signature:
        return cached[1:]
    vector_data = store.list_records(model)
    text_data = store.list_records(model, text_only=True)
    ids = (vector_data.get("ids") or []) + (text_data.get("ids") or [])
    texts = (vector_data.get("documents") or []) + (text_data.get("documents") or [])
    metas = (vector_data.get("metadatas") or []) + (text_data.get("metadatas") or [])
    _RECORDS_CACHE[key] = (signature, ids, texts, metas)
    return ids, texts, metas


def _combined_keyword_index(model: str | None = None):
    signature = _combined_signature(model)
    key = _model_cache_key(model)
    cached = _BM25_CACHE.get(key)
    if cached and cached[0] == signature:
        return cached[1:]

    ids, texts, metas = _combined_records(model)
    bm25 = BM25Okapi([_tok(t) for t in texts]) if texts else None
    _BM25_CACHE[key] = (signature, bm25, ids, texts, metas)
    return bm25, ids, texts, metas


def _scan_candidates(model: str | None, category: str | None, key, k: int) -> list[tuple[str, str, dict]]:
    """The top-`k` memories by `key(meta)` (descending), as a real candidate source.

    Used to make recency and importance genuine PRODUCERS — the most-recent or
    highest-importance memories enter the pool even when no semantic/keyword/graph
    signal matches them. Honors the optional category scope.
    """
    if k <= 0:
        return []
    ids, texts, metas = _combined_records(model)
    rows = [
        (cid, txt, meta or {})
        for cid, txt, meta in zip(ids, texts, metas)
        if not (meta or {}).get("superseded")
        and not (category and (meta or {}).get("category") != category)
    ]
    rows.sort(key=lambda r: key(r[2]), reverse=True)
    return rows[:k]


def _pinned_candidates(model: str | None, category: str | None, k: int) -> list[tuple[str, str, dict]]:
    """Pinned memories deliberately stay prompt-visible, independent of wording.

    This is bounded separately from ordinary top_k so "pinned" means durable
    context without letting a large pinned set crowd out all query-specific
    recall. Category scope still applies when the caller asks for one.
    """
    if k <= 0:
        return []
    ids, texts, metas = _combined_records(model)
    rows = [
        (cid, txt, meta or {})
        for cid, txt, meta in zip(ids, texts, metas)
        if (meta or {}).get("pinned")
        and not (meta or {}).get("superseded")
        and not (category and (meta or {}).get("category") != category)
    ]
    rows.sort(
        key=lambda r: (
            clamp_importance(r[2].get("importance", 3)),
            r[2].get("date") or "",
            r[0],
        ),
        reverse=True,
    )
    return rows[:k]


def _precision_filter(question: str, rows: list[dict], category: str | None, direct_signal_enabled: bool) -> list[dict]:
    """Trim broad recall producers unless they are connected to this prompt."""
    if not config.PRECISION_RECALL_ENABLED or category or not direct_signal_enabled:
        return rows
    if _is_broad_memory_question(question):
        return rows

    q_terms = _question_terms(question)
    if not q_terms:
        return rows

    keep: list[dict] = []
    for row in rows:
        semantic_match = (
            row["_sem_rank"] < config.PRECISION_SEMANTIC_RANK_LIMIT
            and row.get("similarity", 0.0) >= config.PRECISION_MIN_SEMANTIC_SIMILARITY
        )
        direct_match = (
            row["_kw_rank"] < 10**9
            or row["_graph_rank"] < 10**9
            or row["_sensory_rank"] < 10**9
            or semantic_match
        )
        lexical_match = bool(q_terms & _memory_terms(row))
        if direct_match or lexical_match:
            keep.append(row)

    # Precision should reduce noise, not make the assistant amnesic.
    return keep or rows


def _domain_opt_in_filter(
    question: str, rows: list[dict], category: str | None, domains: set[str] | None = None
) -> list[dict]:
    if not config.MEMORY_DOMAIN_OPT_IN_ENABLED or category:
        return rows
    allowed = _domain_opt_ins(question) if domains is None else domains
    keep: list[dict] = []
    for row in rows:
        protected = _row_protected_domains(row)
        if not protected or protected & allowed:
            keep.append(row)
    return keep


def _current_query_episode_filter(question: str, rows: list[dict], category: str | None) -> list[dict]:
    if category or not _is_current_external_question(question) or _is_broad_memory_question(question):
        return rows
    return [
        row for row in rows
        if str(row.get("category") or "").casefold() != config.EPISODE_CATEGORY
    ]


def _categories(model: str | None = None) -> list[str]:
    """Distinct categories, ordered smallest-first.

    Order matters for balanced recall: when the per-category pull is capped, we
    keep the small categories (the ones a big category can starve out of the
    global ANN pool) and skip the large ones, which are already well represented.
    """
    signature = _collection_signature(model)
    key = _model_cache_key(model)
    cached = _CATEGORY_CACHE.get(key)
    if cached and cached[0] == signature:
        return cached[1]
    data = store.list_records(model)
    counts: dict[str, int] = {}
    for m in data.get("metadatas") or []:
        cat = (m or {}).get("category")
        if cat:
            counts[cat] = counts.get(cat, 0) + 1
    cats = sorted(counts, key=lambda c: (counts[c], c))
    _CATEGORY_CACHE[key] = (signature, cats)
    return cats


def _keyword_candidates(question: str, candidate_k: int, model: str | None, category: str | None) -> list[dict]:
    rows: list[dict] = []
    bm25, ids, texts, metas = _combined_keyword_index(model)
    if bm25 is None or not ids:
        return rows
    scores = bm25.get_scores(_tok(question))
    for pos in np.argsort(-scores):
        score = float(scores[pos])
        if score <= 0 or len(rows) >= candidate_k:
            break
        meta = metas[pos] or {}
        if meta.get("superseded"):
            continue
        if category and meta.get("category") != category:
            continue
        rows.append(
            {
                "id": ids[pos],
                "text": texts[pos],
                "meta": meta,
                "score": score,
            }
        )
    return rows


def _collect_semantic(res: dict, dist: dict[str, float], docs: dict[str, dict]) -> None:
    """Fold one ANN result into the running id→best-distance / id→doc maps."""
    ids = res["ids"][0] if res["ids"] else []
    for i, cid in enumerate(ids):
        d = float(res["distances"][0][i])
        if cid not in dist or d < dist[cid]:
            dist[cid] = d
        docs.setdefault(cid, {"text": res["documents"][0][i], "meta": res["metadatas"][0][i] or {}})


def _semantic_candidates(qvec, candidate_k: int, model: str | None, docs: dict[str, dict], category: str | None = None):
    """Balanced semantic recall.

    Beyond one global ANN pull, optionally pull a few candidates from each
    category and from high-importance memories, so a category that the global
    top-K would have starved still gets a fair shot. The union is re-ranked by
    TRUE cosine distance, so ranks stay meaningful and no irrelevant noise leaks
    in (every extra candidate is the closest match within its bucket).

    When `category` is given, recall is scoped to that one category and the
    balancing pulls are skipped (there's nothing to balance against).

    Returns (sem_rank, sim_by_id).
    """
    dist: dict[str, float] = {}

    if category:
        _collect_semantic(
            store.query(qvec, candidate_k, model=model, where=_active_where({"category": category})), dist, docs
        )
    else:
        _collect_semantic(store.query(qvec, candidate_k, model=model, where=_active_where()), dist, docs)
        if config.BALANCED_RECALL:
            if config.PER_CATEGORY_K > 0:
                cats = _categories(model)
                if config.MAX_BALANCED_CATEGORIES > 0:
                    cats = cats[:config.MAX_BALANCED_CATEGORIES]
                for cat in cats:
                    res = store.query(qvec, config.PER_CATEGORY_K, model=model, where=_active_where({"category": cat}))
                    _collect_semantic(res, dist, docs)
            if config.IMPORTANT_RECALL_K > 0:
                res = store.query(
                    qvec, config.IMPORTANT_RECALL_K, model=model,
                    where=_active_where({"importance": {"$gte": config.IMPORTANT_THRESHOLD}}),
                )
                _collect_semantic(res, dist, docs)

    sem_rank: dict[str, int] = {}
    sim_by_id: dict[str, float] = {}
    for rank, cid in enumerate(sorted(dist, key=lambda c: dist[c])):
        sem_rank[cid] = rank
        sim_by_id[cid] = round(1.0 - dist[cid], 4)
    return sem_rank, sim_by_id


def retrieve(
    question: str,
    top_k: int | None = None,
    candidate_k: int | None = None,
    model: str | None = None,
    category: str | None = None,
    domains: set[str] | None = None,
) -> list[dict]:
    return retrieve_weighted(
        question, top_k=top_k, candidate_k=candidate_k, model=model, category=category, domains=domains
    )


def retrieve_weighted(
    question: str,
    top_k: int | None = None,
    candidate_k: int | None = None,
    w_sim: float | None = None,
    w_keyword: float | None = None,
    w_recency: float | None = None,
    w_importance: float | None = None,
    w_graph: float | None = None,
    w_sensory: float | None = None,
    model: str | None = None,
    category: str | None = None,
    domains: set[str] | None = None,
) -> list[dict]:
    top_k = evolution.hyperparam("TOP_K", config.TOP_K) if top_k is None else top_k
    candidate_k = evolution.hyperparam("CANDIDATE_K", config.CANDIDATE_K) if candidate_k is None else candidate_k
    w_sim = evolution.hyperparam("W_SIM", config.W_SIM) if w_sim is None else w_sim
    w_keyword = evolution.hyperparam("W_KEYWORD", config.W_KEYWORD) if w_keyword is None else w_keyword
    w_recency = evolution.hyperparam("W_RECENCY", config.W_RECENCY) if w_recency is None else w_recency
    w_importance = evolution.hyperparam("W_IMPORTANCE", config.W_IMPORTANCE) if w_importance is None else w_importance
    w_graph = evolution.hyperparam("W_GRAPH", config.W_GRAPH) if w_graph is None else w_graph
    w_sensory = evolution.hyperparam("W_SENSORY", config.W_SENSORY) if w_sensory is None else w_sensory
    now = datetime.now(timezone.utc)  # one snapshot for production and re-ranking

    docs: dict[str, dict] = {}          # id -> {"text", "meta"}
    sem_rank: dict[str, int] = {}
    sim_by_id: dict[str, float] = {}
    kw_rank: dict[str, int] = {}
    kw_by_id: dict[str, float] = {}
    graph_rank: dict[str, int] = {}
    sensory_rank: dict[str, int] = {}
    sensory_by_id: dict[str, float] = {}
    pinned_rank: dict[str, int] = {}

    any_weight_enabled = any(
        w > 0 for w in (w_sim, w_keyword, w_recency, w_importance, w_graph, w_sensory)
    )

    # 0. PINNED — user-marked durable context. Pinned is a visibility signal,
    # not a match signal, so only produce it during normal retrieval turns.
    if any_weight_enabled and config.PINNED_RECALL_K > 0 and config.W_PINNED > 0:
        for rank, (cid, text, meta) in enumerate(_pinned_candidates(model, category, config.PINNED_RECALL_K)):
            pinned_rank[cid] = rank
            docs.setdefault(cid, {"text": text, "meta": meta})

    # 1. SEMANTIC — ChromaDB ANN with balanced, category/importance-aware recall
    if w_sim > 0:
        try:
            qvec = embed_query(question, model=model)
        except EmbeddingUnavailable:
            qvec = None
        if qvec is not None:
            sem_rank, sim_by_id = _semantic_candidates(qvec, candidate_k, model, docs, category=category)

    # 2. KEYWORD — cached BM25 across both vector and text-only collections.
    if w_keyword > 0:
        for rank, row in enumerate(_keyword_candidates(question, candidate_k, model, category)):
            cid = row["id"]
            kw_rank[cid] = rank
            kw_by_id[cid] = round(row["score"], 4)
            docs.setdefault(cid, {"text": row["text"], "meta": row["meta"]})

    # 2.5 GRAPH — associative recall: entities named in the question pull in their
    # one-hop-linked fact memories, even when semantic/keyword signals miss them.
    if w_graph > 0:
        # expand() can't see superseded (that lives in Chroma metadata) and its
        # fact ids carry no active/retired ordering, so any pre-filter cap could
        # be entirely filled by retired facts and starve active ones below it. Pull
        # the FULL scoped id set (limit=None) and admit only active facts, up to
        # GRAPH_RECALL_K, below — bounded by the entity neighborhood, which is far
        # smaller than the whole-store load the keyword/recency producers already do.
        try:
            gids = graph.expand(
                question,
                depth=config.GRAPH_RECALL_DEPTH,
                limit=None,
                # Scope to BOTH the vector and text-only collections for this model:
                # facts consolidated during an embedding outage live text-only, and
                # their edges are tagged accordingly. get_by_ids() reads both stores.
                collection=[
                    store.collection_name(model),
                    store.collection_name(model, text_only=True),
                ],
            )
        except Exception as exc:
            log.fallback("graph_failed", exc, phase="recall")
            gids = []
        missing = [g for g in gids if g not in docs]
        fetched = store.get_by_ids(missing, model=model) if missing else {}
        grank = 0
        for cid in gids:
            if grank >= config.GRAPH_RECALL_K:
                break
            if cid not in docs:
                rec = fetched.get(cid)
                if rec is None:
                    continue
                docs[cid] = {"text": rec["text"], "meta": rec["meta"]}
            meta = docs[cid]["meta"]
            if meta.get("superseded"):
                continue
            if category and meta.get("category") != category:
                continue
            graph_rank[cid] = grank
            grank += 1

    # 2.7 SENSORY — text-anchor recall over image/audio sensation records. These
    # are separate Chroma collections because raw sensory vectors do not share the
    # text embedding dimensions. The returned candidate represents the exact
    # remembered source ("I remember seeing/hearing ...").
    if w_sensory > 0:
        try:
            sense_rows = sensory.retrieve_text(question, top_k=config.SENSORY_RECALL_K, category=category)
        except Exception as exc:
            log.fallback("sensory_failed", exc, phase="recall")
            sense_rows = []
        for rank, row in enumerate(sense_rows):
            cid = row["id"]
            sensory_rank[cid] = rank
            sensory_by_id[cid] = row.get("sensory_similarity", 0.0)
            docs.setdefault(cid, {"text": row["text"], "meta": row["meta"]})

    # 2.6 RECENCY / IMPORTANCE as PRODUCERS — the most-recent and highest-importance
    # memories become candidates in their own right, so a recency- or importance-led
    # query surfaces fresh / high-value memories even with no semantic, keyword, or
    # graph hit. They're still re-ranked by the calibrated hybrid score below,
    # which keeps them subordinate unless the matching weight is high.
    recency_ids: set[str] = set()
    if w_recency > 0:
        for cid, text, meta in _scan_candidates(
            model,
            category,
            lambda m: recency_score(
                m.get("date"), config.RECENCY_HALFLIFE_DAYS, now=now
            ),
            config.RECENCY_RECALL_K,
        ):
            docs.setdefault(cid, {"text": text, "meta": meta})
            recency_ids.add(cid)

    importance_ids: set[str] = set()
    if w_importance > 0:
        for cid, text, meta in _scan_candidates(
            model, category, lambda m: clamp_importance(m.get("importance", 3)), config.IMPORTANCE_RECALL_K
        ):
            docs.setdefault(cid, {"text": text, "meta": meta})
            importance_ids.add(cid)

    # Candidates come from every PRODUCING signal: semantic, keyword, graph, and now
    # the recency/importance pulls above. (Recency/importance still ALSO re-rank the
    # whole union below.) With every weight at zero there are no producers and the
    # result is empty — the API layer rejects that all-zero request up front.
    cand = (
        set(sem_rank)
        | set(kw_rank)
        | set(graph_rank)
        | set(sensory_rank)
        | set(pinned_rank)
        | recency_ids
        | importance_ids
    )
    # Drop memories retired by belief-updating, so we answer from current truth.
    cand = {c for c in cand if not docs[c]["meta"].get("superseded")}
    if not cand:
        return []

    # 3 & 4. RECENCY + IMPORTANCE across the complete candidate union. Recency's
    # continuous value contributes directly below; its rank remains a stable
    # tie-break. Importance uses competition ranking so equal 1-5 values share a
    # rank rather than receiving arbitrary score differences from memory ids.
    rec_by_id = {c: recency_score(docs[c]["meta"].get("date"), config.RECENCY_HALFLIFE_DAYS, now=now) for c in cand}
    imp_by_id = {c: clamp_importance(docs[c]["meta"].get("importance", 3)) for c in cand}
    rec_rank = _competition_rank(cand, rec_by_id)
    imp_rank = _competition_rank(cand, imp_by_id)

    scored: list[dict] = []
    for cid in cand:
        components = {
            "semantic": w_sim * _rrf(sem_rank[cid]) if cid in sem_rank else 0.0,
            "keyword": w_keyword * _rrf(kw_rank[cid]) if cid in kw_rank else 0.0,
            "graph": w_graph * _rrf(graph_rank[cid]) if cid in graph_rank else 0.0,
            "sensory": w_sensory * _rrf(sensory_rank[cid]) if cid in sensory_rank else 0.0,
            "pinned": config.W_PINNED * _rrf(pinned_rank[cid]) if cid in pinned_rank else 0.0,
            # Calibrate normalized decay to one top-ranked RRF contribution.
            # Unlike rank(recency), this makes RECENCY_HALFLIFE_DAYS materially
            # change the result and strongly separates fresh facts from stale ones.
            "recency": w_recency * rec_by_id[cid] * _rrf(0) if w_recency > 0 else 0.0,
            "importance": w_importance * _rrf(imp_rank[cid]) if w_importance > 0 else 0.0,
        }
        score = sum(components.values())
        meta = docs[cid]["meta"]
        scored.append(
            {
                "_score_raw": score,
                "_sem_rank": sem_rank.get(cid, 10**9),
                "_kw_rank": kw_rank.get(cid, 10**9),
                "_graph_rank": graph_rank.get(cid, 10**9),
                "_sensory_rank": sensory_rank.get(cid, 10**9),
                "_rec_rank": rec_rank[cid],
                "_imp_rank": imp_rank[cid],
                "id": cid,
                "text": docs[cid]["text"],
                "category": meta.get("category"),
                "source": meta.get("source"),
                "date": meta.get("date"),
                "importance": meta.get("importance"),
                "pinned": bool(meta.get("pinned", False)),
                "similarity": sim_by_id.get(cid, 0.0),
                "keyword": kw_by_id.get(cid, 0.0),
                "recency": round(rec_by_id[cid], 4),
                "graph_linked": cid in graph_rank,
                "sensory": cid in sensory_rank,
                "sensory_kind": meta.get("kind") if cid in sensory_rank else None,
                "sensory_similarity": sensory_by_id.get(cid, 0.0),
                "score_components": {
                    name: round(value, 6) for name, value in components.items()
                },
                "score": round(score, 6),
            }
        )

    scored.sort(
        key=lambda x: (
            -x["_score_raw"],
            x["_sem_rank"],
            x["_kw_rank"],
            x["_graph_rank"],
            x["_sensory_rank"],
            0 if x.get("pinned") else 1,
            x["_rec_rank"],
            x["_imp_rank"],
            x["id"],
        )
    )
    scored = _precision_filter(
        question,
        scored,
        category,
        direct_signal_enabled=any(w > 0 for w in (w_sim, w_keyword, w_graph, w_sensory)),
    )
    scored = _domain_opt_in_filter(question, scored, category, domains=domains)
    scored = _current_query_episode_filter(question, scored, category)
    for row in scored:
        for key in ("_score_raw", "_sem_rank", "_kw_rank", "_graph_rank", "_sensory_rank", "_rec_rank", "_imp_rank"):
            row.pop(key, None)
    return scored[:top_k]
