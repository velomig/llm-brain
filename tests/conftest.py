"""Shared test harness for memory-behavior tests.

Two things make these tests hermetic and Ollama-free:

  1. NEURAL_MAP_DB is pointed at a throwaway temp dir BEFORE any brain.* import,
     so brain.store's module-level Chroma client (and the graph SQLite file) bind
     to the temp store instead of the real 8000-memory one.
  2. brain.embeddings._client is swapped for a deterministic fake. Every embed
     path funnels through that one client, so the real embed_document/embed_query
     functions keep working — they just produce a CRC-hashed bag-of-words vector.
     Shared words => high cosine similarity, which is all retrieval needs, and no
     network/model is touched. Flip `.available` to simulate "Ollama is down".

An autouse fixture wipes both collections and the graph between tests, so each
test starts from an empty, isolated brain.
"""
from __future__ import annotations

import os
import pathlib
import re
import shutil
import sys
import tempfile
import zlib

import pytest

# --- isolate the store BEFORE importing any brain module ----------------------
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_TEST_DB = tempfile.mkdtemp(prefix="neural_map_test_")
os.environ["NEURAL_MAP_DB"] = _TEST_DB
os.environ.setdefault("NEURAL_MAP_CONSOLIDATION", "0")   # no LLM in capture paths
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")   # quiet Chroma telemetry

import config  # noqa: E402
from brain import attention, awareness, cognitive_state, embeddings, emotion, evolution, graph, grounding, homeostasis, initiative, perspective, presence, self_model, sensory, sleep, store, will, workspace  # noqa: E402
from brain.toolkit import governance, local_state, pending  # noqa: E402

_EMBED_DIM = 64


class FakeEmbeddingClient:
    """Stand-in for ollama.Client used by brain.embeddings.

    Produces a deterministic bag-of-words vector: each token bumps one slot, so
    texts that share words land close in cosine space. No network, no model.
    """

    def __init__(self) -> None:
        self.available = True

    def _vec(self, text: str | None) -> list[float]:
        vec = [0.0] * _EMBED_DIM
        for tok in re.findall(r"\w+", (text or "").lower()):
            vec[zlib.crc32(tok.encode("utf-8")) % _EMBED_DIM] += 1.0
        if not any(vec):
            vec[0] = 1.0   # cosine is undefined for an all-zero vector
        return vec

    def _guard(self) -> None:
        if not self.available:
            # Message matches embeddings._is_embedding_unavailable so the real
            # code raises EmbeddingUnavailable and takes the text-only fallback.
            raise RuntimeError("model does not support embeddings")

    def embeddings(self, model=None, prompt=None):   # single: embed_document/query
        self._guard()
        return {"embedding": self._vec(prompt)}

    def embed(self, model=None, input=None):          # batch: embed_documents
        self._guard()
        return {"embeddings": [self._vec(t) for t in (input or [])]}


@pytest.fixture
def fake_embed():
    """The patched embedding client; set `.available = False` to fake Ollama down."""
    client = FakeEmbeddingClient()
    embeddings._client = client
    return client


def _wipe() -> None:
    for text_only in (False, True):
        col = store._collection(text_only=text_only)
        ids = col.get().get("ids") or []
        if ids:
            col.delete(ids=ids)
        store._bump_version(text_only=text_only)   # invalidate BM25/category caches
    graph.clear()
    sensory.clear_all()
    try:
        initiative._STATE_FILE.unlink()
    except FileNotFoundError:
        pass
    try:
        emotion._STATE_FILE.unlink()
    except FileNotFoundError:
        pass
    try:
        sleep._STATUS_FILE.unlink()
    except FileNotFoundError:
        pass
    shutil.rmtree(sleep._COLD_DIR, ignore_errors=True)
    try:
        will._STATE_FILE.unlink()
    except FileNotFoundError:
        pass
    try:
        evolution._STATE_FILE.unlink()
    except FileNotFoundError:
        pass
    try:
        cognitive_state._STATE_FILE.unlink()
    except FileNotFoundError:
        pass
    try:
        awareness._STATE_FILE.unlink()
    except FileNotFoundError:
        pass
    try:
        attention._STATE_FILE.unlink()
    except FileNotFoundError:
        pass
    try:
        homeostasis._STATE_FILE.unlink()
    except FileNotFoundError:
        pass
    try:
        self_model._STATE_FILE.unlink()
    except FileNotFoundError:
        pass
    try:
        perspective._STATE_FILE.unlink()
    except FileNotFoundError:
        pass
    for path in (
        pending.PENDING_FILE,
        governance.PERMISSIONS_FILE,
        governance.AUDIT_FILE,
        local_state.CONTACTS_FILE,
        local_state.CALENDAR_FILE,
        workspace._STATE_FILE,
        grounding._STATE_FILE,
        presence._STATE_FILE,
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


@pytest.fixture(autouse=True)
def isolated_brain(fake_embed, monkeypatch):
    """Clean store + graph and a fresh, available embedder for every test."""
    monkeypatch.setattr(config, "SERVER_TOKEN", "")
    _wipe()
    yield
    _wipe()


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TEST_DB, ignore_errors=True)
