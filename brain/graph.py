"""Entity knowledge graph — the associative layer that makes it a "Neural Map".

Flat vector search finds memories that *sound* like the query. A brain also
recalls by association: ask about Alex's retirement and SCHD, JEPQ, SGOV, the
travel-nursing plan and the 2033 income switch should all light up because they
are *linked*, even when a chunk doesn't lexically match.

During consolidation the model emits (subject, relation, object) triples; we
store them here as edges (SQLite, beside the vector store). At retrieval we find
entities mentioned in the question, walk one hop out, and pull the fact memories
attached to those edges as extra candidates. Tiny and personal-scale by design.
"""
from __future__ import annotations

import re
import sqlite3
import threading

import config

_DB_PATH = config.DB_DIR / "graph.sqlite3"
_LOCK = threading.Lock()
_conn: sqlite3.Connection | None = None


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        config.DB_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.execute(
            """CREATE TABLE IF NOT EXISTS edges (
                   subject     TEXT NOT NULL,
                   subject_lc  TEXT NOT NULL,
                   relation    TEXT NOT NULL,
                   object      TEXT NOT NULL,
                   object_lc   TEXT NOT NULL,
                   fact_id     TEXT,
                   collection  TEXT NOT NULL DEFAULT '',
                   date        TEXT,
                   UNIQUE(subject_lc, relation, object_lc, collection)
               )"""
        )
        _migrate_schema(_conn)
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_subj ON edges(subject_lc)")
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_obj ON edges(object_lc)")
        _conn.commit()
    return _conn


def _unique_includes_collection(conn: sqlite3.Connection) -> bool:
    """True once a UNIQUE index spans (subject_lc, relation, object_lc, collection)."""
    for _seq, name, unique, *_ in conn.execute("PRAGMA index_list(edges)").fetchall():
        if not unique:
            continue
        cols = {r[0] for r in conn.execute("SELECT name FROM pragma_index_info(?)", (name,))}
        if {"subject_lc", "relation", "object_lc", "collection"} <= cols:
            return True
    return False


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Bring older graphs up to the per-collection edge schema.

    A fact_id only resolves inside the collection (embedding model) that holds it,
    so the same (subject, relation, object) triple must be allowed ONCE PER
    collection — otherwise a later consolidation/import in one model store steals
    the edge from another. Two historical shapes are upgraded in place:
      v0: no `collection` column at all.
      v1: `collection` column but a GLOBAL UNIQUE(subject_lc, relation, object_lc).
    Both are rebuilt to UNIQUE(subject_lc, relation, object_lc, collection) with
    collection NOT NULL DEFAULT '' (where '' means "unknown / any model" — the
    best-effort sentinel old NULLs become). Edges are preserved; '' rows stay
    unique because the old global key already held one row per triple."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(edges)")}
    if "collection" not in cols:
        conn.execute("ALTER TABLE edges ADD COLUMN collection TEXT NOT NULL DEFAULT ''")
    if _unique_includes_collection(conn):
        return
    conn.execute(
        """CREATE TABLE edges_new (
               subject     TEXT NOT NULL,
               subject_lc  TEXT NOT NULL,
               relation    TEXT NOT NULL,
               object      TEXT NOT NULL,
               object_lc   TEXT NOT NULL,
               fact_id     TEXT,
               collection  TEXT NOT NULL DEFAULT '',
               date        TEXT,
               UNIQUE(subject_lc, relation, object_lc, collection)
           )"""
    )
    conn.execute(
        """INSERT INTO edges_new
               (subject, subject_lc, relation, object, object_lc, fact_id, collection, date)
           SELECT subject, subject_lc, relation, object, object_lc, fact_id,
                  COALESCE(collection, ''), date FROM edges"""
    )
    conn.execute("DROP TABLE edges")
    conn.execute("ALTER TABLE edges_new RENAME TO edges")


def _norm(value) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def add_relations(
    triples: list[dict],
    fact_id: str | None = None,
    date: str | None = None,
    collection: str | None = None,
) -> int:
    """Store (subject, relation, object) triples, one edge per triple PER COLLECTION.

    Uniqueness is scoped by collection, so the same triple can link to a different
    fact in each model store — a later assert/import in one store never steals the
    edge from another. Within a collection, re-asserting a triple UPSERTS: it
    repoints the edge at the newest fact (fact_id/date) instead of being ignored, so
    a superseded fact's edge moves to its replacement rather than dangling. A null
    incoming fact_id/date never clobbers an existing one (COALESCE keeps the good
    value). `collection` defaults to '' meaning "unknown / any model"."""
    rows = []
    for t in triples or []:
        if not isinstance(t, dict):
            continue
        subj, rel, obj = _norm(t.get("subject")), _norm(t.get("relation")), _norm(t.get("object"))
        if not subj or not rel or not obj:
            continue
        rows.append((subj, subj.lower(), rel, obj, obj.lower(), fact_id, collection or "", date))
    if not rows:
        return 0
    with _LOCK:
        db = _db()
        db.executemany(
            """INSERT INTO edges
               (subject, subject_lc, relation, object, object_lc, fact_id, collection, date)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(subject_lc, relation, object_lc, collection) DO UPDATE SET
                   fact_id = COALESCE(excluded.fact_id, fact_id),
                   date    = COALESCE(excluded.date, date),
                   subject = excluded.subject,
                   object  = excluded.object""",
            rows,
        )
        db.commit()
    return len(rows)


def _collection_names(collection: str | list[str] | None) -> list[str] | None:
    if collection is None:
        return None
    names = [collection] if isinstance(collection, str) else list(collection)
    if "" not in names:
        names.append("")  # always include best-effort untagged edges
    return names


def entities(collection: str | list[str] | None = None) -> list[str]:
    """Every distinct entity name in the graph (subjects and objects)."""
    names = _collection_names(collection)
    with _LOCK:
        if names is None:
            cur = _db().execute("SELECT subject FROM edges UNION SELECT object FROM edges")
        else:
            qs = ",".join("?" * len(names))
            cur = _db().execute(
                f"""SELECT subject FROM edges WHERE collection IN ({qs})
                    UNION SELECT object FROM edges WHERE collection IN ({qs})""",
                (*names, *names),
            )
        return [r[0] for r in cur.fetchall()]


def _entities_in(text: str, collection: str | list[str] | None = None) -> list[str]:
    """Known entities mentioned in the text (word-boundary, case-insensitive)."""
    low = text.lower()
    hits = []
    for ent in set(entities(collection=collection)):
        if not ent:
            continue
        if re.search(rf"(?<!\w){re.escape(ent.lower())}(?!\w)", low):
            hits.append(ent)
    return hits


def neighbors(entity: str, depth: int = 1, collection: str | list[str] | None = None) -> set[str]:
    """Entities reachable from `entity` within `depth` hops (excludes itself)."""
    seen = {entity.lower()}
    frontier = {entity.lower()}
    names = _collection_names(collection)
    for _ in range(max(1, depth)):
        if not frontier:
            break
        qs = ",".join("?" * len(frontier))
        scope = ""
        params: tuple = (*frontier, *frontier)
        if names is not None:
            cqs = ",".join("?" * len(names))
            scope = f" AND collection IN ({cqs})"
            params = (*frontier, *names, *frontier, *names)
        with _LOCK:
            cur = _db().execute(
                f"""SELECT object_lc, object FROM edges WHERE subject_lc IN ({qs}){scope}
                    UNION SELECT subject_lc, subject FROM edges WHERE object_lc IN ({qs}){scope}""",
                params,
            )
            rows = cur.fetchall()
        nxt = set()
        for lc, _name in rows:
            if lc not in seen:
                seen.add(lc)
                nxt.add(lc)
        frontier = nxt
    return {lc for lc in seen if lc != entity.lower()}


def fact_ids_for(
    entity_lcs: set[str], collection: str | list[str] | None = None
) -> list[str]:
    """Fact ids attached to edges touching any of the given (lowercased) entities.

    When `collection` is given, recall is scoped to that store: only edges whose
    fact lives in it — plus edges with no recorded collection ('', best-effort) —
    are returned, so the active model never gets fact ids that don't resolve in its
    own collection. Pass a list to scope to several stores at once (e.g. a model's
    vector AND text-only collections, so facts written during an embedding outage
    still surface). `collection=None` is unscoped (every edge)."""
    if not entity_lcs:
        return []
    qs = ",".join("?" * len(entity_lcs))
    scope = ""
    params: tuple = (*entity_lcs, *entity_lcs)
    names = _collection_names(collection)
    if names is not None:
        cqs = ",".join("?" * len(names))
        scope = f" AND collection IN ({cqs})"
        params = (*params, *names)
    with _LOCK:
        # GROUP BY + ORDER BY makes the id order deterministic and newest-first.
        # Without it SQLite returned rows in arbitrary storage order, so which
        # facts won the GRAPH_RECALL_K candidate slots (and their RRF ranks)
        # could differ between runs on identical data.
        cur = _db().execute(
            f"""SELECT fact_id FROM edges
                WHERE fact_id IS NOT NULL AND (subject_lc IN ({qs}) OR object_lc IN ({qs})){scope}
                GROUP BY fact_id
                ORDER BY MAX(COALESCE(date, '')) DESC, fact_id""",
            params,
        )
        return [r[0] for r in cur.fetchall() if r[0]]


def expand(
    question: str,
    depth: int = 1,
    limit: int | None = 12,
    collection: str | list[str] | None = None,
) -> list[str]:
    """Associative recall: question -> mentioned entities -> neighbors -> fact ids.

    `collection` scopes the fact ids to the active store(s) (see fact_ids_for).
    `limit=None` returns every scoped id (the ids carry no active/retired
    ordering, so a caller that must admit N *active* facts has to see them all)."""
    matched = _entities_in(question, collection=collection)
    if not matched:
        return []
    reach = set()
    for ent in matched:
        reach.add(ent.lower())
        reach |= neighbors(ent, depth=depth, collection=collection)
    ids = fact_ids_for(reach, collection=collection)
    return ids if limit is None else ids[:limit]


def backfill_collection(collection: str, dry_run: bool = False) -> int:
    """Stamp `collection` onto edges with no recorded collection (the '' sentinel).

    In the common single-embedding-model setup every existing fact lives in that
    one store, so tagging unscoped edges with it makes graph recall properly scoped.
    Idempotent — only '' edges are touched. Returns how many matched.

    Under the per-collection unique key an '' edge and a same-triple edge already
    tagged `collection` can coexist, so each '' edge is promoted where it doesn't
    collide (UPDATE OR IGNORE) and any remainder — a redundant duplicate of an
    already-tagged edge — is dropped, leaving exactly one edge per (triple)."""
    with _LOCK:
        db = _db()
        (n,) = db.execute("SELECT COUNT(*) FROM edges WHERE collection = ''").fetchone()
        if n and not dry_run:
            db.execute(
                "UPDATE OR IGNORE edges SET collection = ? WHERE collection = ''", (collection,)
            )
            db.execute("DELETE FROM edges WHERE collection = ''")
            db.commit()
    return int(n)


def export_edges() -> list[dict]:
    """Every edge as a portable row (for backup/export). Implementation-neutral:
    no SQLite rowids, just the (subject, relation, object) triple and its
    fact_id/collection/date."""
    with _LOCK:
        cur = _db().execute(
            "SELECT subject, relation, object, fact_id, collection, date FROM edges ORDER BY subject_lc, relation, object_lc"
        )
        # Emit the '' (unknown) sentinel as null, so the portable JSON reads cleanly
        # and round-trips (add_relations coerces a null collection back to '').
        return [
            {"subject": s, "relation": r, "object": o, "fact_id": f, "collection": c or None, "date": d}
            for s, r, o, f, c, d in cur.fetchall()
        ]


def import_edges(rows: list[dict]) -> int:
    """Restore edges from export_edges() output. Upserts (so a re-import is
    idempotent and the latest fact wins). Returns the number of edges written."""
    n = 0
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        n += add_relations(
            [{"subject": row.get("subject"), "relation": row.get("relation"), "object": row.get("object")}],
            fact_id=row.get("fact_id"),
            date=row.get("date"),
            collection=row.get("collection"),
        )
    return n


def stats() -> dict:
    with _LOCK:
        edges = _db().execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    return {"edges": edges, "entities": len(set(entities()))}


def clear() -> None:
    with _LOCK:
        _db().execute("DELETE FROM edges")
        _db().commit()


def clear_collections(collections: list[str], include_legacy: bool = False) -> int:
    """Delete edges scoped to the given collection names. Returns rows removed."""
    names = [c for c in dict.fromkeys(collections) if c]
    if not names and not include_legacy:
        return 0
    clauses: list[str] = []
    params: list[str] = []
    if names:
        qs = ",".join("?" * len(names))
        clauses.append(f"collection IN ({qs})")
        params.extend(names)
    if include_legacy:
        clauses.append("collection = ''")
    with _LOCK:
        db = _db()
        cur = db.execute(f"DELETE FROM edges WHERE {' OR '.join(clauses)}", tuple(params))
        db.commit()
        return int(cur.rowcount or 0)


def delete_fact_ids(fact_ids: list[str], collections: list[str] | None = None) -> int:
    """Delete graph edges attached to the given memory ids.

    Used by integrity repair for dangling edges whose backing memories were
    already deleted. When collections are supplied, deletion is limited to those
    collection names; include ``None`` to target legacy/null-scoped rows.
    """
    ids = [fid for fid in dict.fromkeys(fact_ids) if fid]
    if not ids:
        return 0
    clauses = [f"fact_id IN ({','.join('?' * len(ids))})"]
    params: list[str] = list(ids)
    if collections is not None:
        names = [c or "" for c in dict.fromkeys(collections)]
        clauses.append(f"collection IN ({','.join('?' * len(names))})")
        params.extend(names)
    with _LOCK:
        db = _db()
        cur = db.execute(f"DELETE FROM edges WHERE {' AND '.join(clauses)}", tuple(params))
        db.commit()
        return int(cur.rowcount or 0)
