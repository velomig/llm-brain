"""Metadata is versioned, and a repair pass upgrades pre-versioned records.

New memories carry schema_version; migrate.py / store.repair_schema backfill it
(plus the defaults the ranking code assumes) onto older records, across BOTH the
vector and text-only stores, without re-embedding and idempotently."""
from __future__ import annotations

import uuid

from brain import store
from brain.embeddings import embed_document, text_only_embedding
from brain.memory import SCHEMA_VERSION, MemoryUnit, repaired_metadata
from util import store_memory


def _add_legacy(text_only: bool, meta: dict, text: str = "legacy memory") -> str:
    """Insert a record straight into a collection with pre-versioned metadata.

    Chroma's update() *merges* metadata (it can't drop a key), so a genuinely
    version-less record has to be written directly rather than by stripping one.
    """
    cid = uuid.uuid4().hex
    embedding = text_only_embedding() if text_only else embed_document(text)
    store._collection(text_only=text_only).add(
        ids=[cid], embeddings=[embedding], documents=[text], metadatas=[meta]
    )
    return cid


# --- new memories are stamped -------------------------------------------------

def test_new_memory_carries_current_schema_version():
    meta = MemoryUnit(text="hello", category="fact").metadata()
    assert meta["schema_version"] == SCHEMA_VERSION


def test_stored_memory_round_trips_schema_version():
    u = store_memory("a versioned memory")
    got = store.get_by_ids([u.id])
    assert got[u.id]["meta"]["schema_version"] == SCHEMA_VERSION


# --- repaired_metadata (pure, no store) ---------------------------------------

def test_repair_is_noop_for_current_record():
    assert repaired_metadata({"schema_version": SCHEMA_VERSION, "category": "fact"}) is None


def test_repair_stamps_version_and_backfills_defaults():
    upgraded = repaired_metadata({"category": "note"})   # legacy: no schema_version
    assert upgraded is not None
    assert upgraded["schema_version"] == SCHEMA_VERSION
    assert upgraded["access_count"] == 0          # backfilled
    assert upgraded["importance"] == 3            # backfilled + clamped default
    assert upgraded["category"] == "note"         # preserved


def test_repair_preserves_situational_fields_and_does_not_invent_date():
    legacy = {"category": "reminder", "due": "2026-07-01", "superseded": True, "access_count": 4}
    upgraded = repaired_metadata(legacy)
    assert upgraded["due"] == "2026-07-01"        # situational field untouched
    assert upgraded["superseded"] is True
    assert upgraded["access_count"] == 4          # existing value preserved
    assert "date" not in upgraded                 # never faked


# --- store.repair_schema (both stores, in place) ------------------------------

def test_repair_schema_upgrades_both_stores():
    v = _add_legacy(False, {"category": "fact", "importance": 4})
    t = _add_legacy(True, {"category": "note"})

    result = store.repair_schema()

    assert result["migrated"] == 2
    assert result["by_mode"] == {"vector": 1, "text_only": 1}
    rows = {r["id"]: r for r in store.list_records_all()}
    assert rows[v]["meta"]["schema_version"] == SCHEMA_VERSION
    assert rows[t]["meta"]["schema_version"] == SCHEMA_VERSION
    assert rows[v]["meta"]["importance"] == 4   # preserved through migration


def test_repair_schema_is_idempotent():
    store_memory("already current")
    first = store.repair_schema()
    assert first["migrated"] == 0          # freshly written → already versioned
    second = store.repair_schema()
    assert second["migrated"] == 0


def test_repair_schema_dry_run_writes_nothing():
    cid = _add_legacy(False, {"category": "fact"})

    planned = store.repair_schema(dry_run=True)
    assert planned["migrated"] == 1

    # Still unversioned after a dry run...
    rows = {r["id"]: r for r in store.list_records_all()}
    assert "schema_version" not in rows[cid]["meta"]
    # ...and a real run still finds it.
    assert store.repair_schema()["migrated"] == 1
