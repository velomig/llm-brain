"""The graph edge schema migrates in place to the per-collection unique key.

Two historical shapes must upgrade without losing edges: v0 (no `collection`
column) and v1 (`collection` column but a GLOBAL UNIQUE over the triple). After
migration the same triple can coexist once per collection. Exercised on in-memory
DBs so it's independent of the module's cached connection."""
from __future__ import annotations

import sqlite3

from brain import graph

_NEW_SCHEMA = """CREATE TABLE edges (
    subject TEXT NOT NULL, subject_lc TEXT NOT NULL,
    relation TEXT NOT NULL, object TEXT NOT NULL, object_lc TEXT NOT NULL,
    fact_id TEXT, collection TEXT NOT NULL DEFAULT '', date TEXT,
    UNIQUE(subject_lc, relation, object_lc, collection))"""

_V1_SCHEMA = """CREATE TABLE edges (
    subject TEXT NOT NULL, subject_lc TEXT NOT NULL,
    relation TEXT NOT NULL, object TEXT NOT NULL, object_lc TEXT NOT NULL,
    fact_id TEXT, collection TEXT, date TEXT,
    UNIQUE(subject_lc, relation, object_lc))"""

_V0_SCHEMA = """CREATE TABLE edges (
    subject TEXT NOT NULL, subject_lc TEXT NOT NULL,
    relation TEXT NOT NULL, object TEXT NOT NULL, object_lc TEXT NOT NULL,
    fact_id TEXT, date TEXT,
    UNIQUE(subject_lc, relation, object_lc))"""

_COLS = "(subject, subject_lc, relation, object, object_lc, fact_id, collection, date)"


def _conn(schema: str) -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.execute(schema)
    return c


def test_migrate_v1_global_unique_to_per_collection():
    conn = _conn(_V1_SCHEMA)
    conn.execute(
        f"INSERT INTO edges {_COLS} VALUES (?,?,?,?,?,?,?,?)",
        ("Alex", "alex", "holds", "FNDZ", "fndz", "fact-1", None, "2026-01-01"),
    )
    assert graph._unique_includes_collection(conn) is False

    graph._migrate_schema(conn)

    assert graph._unique_includes_collection(conn) is True
    assert conn.execute("SELECT fact_id, collection FROM edges").fetchone() == ("fact-1", "")  # NULL -> ''
    # The whole point: the same triple now coexists in a second collection.
    conn.execute(
        f"INSERT INTO edges {_COLS} VALUES (?,?,?,?,?,?,?,?)",
        ("Alex", "alex", "holds", "FNDZ", "fndz", "fact-2", "collB", "2026-01-02"),
    )
    assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 2


def test_migrate_v0_adds_column_and_unique():
    conn = _conn(_V0_SCHEMA)
    conn.execute(
        "INSERT INTO edges (subject, subject_lc, relation, object, object_lc, fact_id, date) VALUES (?,?,?,?,?,?,?)",
        ("A", "a", "rel", "B", "b", "f1", "2026-01-01"),
    )

    graph._migrate_schema(conn)

    assert graph._unique_includes_collection(conn) is True
    assert conn.execute("SELECT collection FROM edges").fetchone() == ("",)


def test_migrate_is_noop_on_current_schema():
    conn = _conn(_NEW_SCHEMA)
    assert graph._unique_includes_collection(conn) is True
    graph._migrate_schema(conn)   # must not rebuild or raise
    assert graph._unique_includes_collection(conn) is True
