"""delete_by_source must remove exactly one source's chunks, across both the
vector and text-only collections, leaving other sources untouched."""
from __future__ import annotations

import server
from brain import sensory, store
from util import store_memory


def _count_source(src: str) -> int:
    n = 0
    for text_only in (False, True):
        metas = store.list_records(text_only=text_only).get("metadatas") or []
        n += sum(1 for m in (metas or []) if (m or {}).get("source") == src)
    return n


def test_delete_by_source_removes_only_that_source():
    store_memory("alpha one", source="src:A")
    store_memory("alpha two", source="src:A")
    store_memory("beta one", source="src:B")

    store.delete_by_source("src:A")

    assert _count_source("src:A") == 0
    assert _count_source("src:B") == 1


def test_delete_by_source_spans_both_collections():
    # The danger case: a source with chunks in BOTH stores (some embedded, some
    # text-only). A caller that deletes both modes must clear all of them.
    store_memory("vector chunk", source="src:mixed", text_only=False)
    store_memory("text only chunk", source="src:mixed", text_only=True)
    assert _count_source("src:mixed") == 2

    store.delete_by_source("src:mixed", text_only=False)
    store.delete_by_source("src:mixed", text_only=True)

    assert _count_source("src:mixed") == 0


def test_delete_unknown_source_is_noop():
    store_memory("keep me", source="src:keep")
    store.delete_by_source("src:does-not-exist")
    assert _count_source("src:keep") == 1


def test_delete_last_memory_for_source_removes_sensory_anchor():
    unit = store_memory("caption chunk", source="src:image")
    sid = sensory.add_media("image", b"image-bytes", "src:image", "image.png", "caption chunk", "image")
    assert sid
    assert sensory.list_records("src:image")

    server._delete_memory_all(unit.id)

    assert _count_source("src:image") == 0
    assert sensory.list_records("src:image") == []


def test_delete_one_chunk_keeps_sensory_until_source_is_gone():
    first = store_memory("caption chunk one", source="src:image")
    second = store_memory("caption chunk two", source="src:image")
    sid = sensory.add_media("image", b"image-bytes", "src:image", "image.png", "caption chunks", "image")
    assert sid

    server._delete_memory_all(first.id)

    assert _count_source("src:image") == 1
    assert sensory.list_records("src:image")

    server._delete_memory_all(second.id)

    assert _count_source("src:image") == 0
    assert sensory.list_records("src:image") == []
