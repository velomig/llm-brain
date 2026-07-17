"""Image ingestion: an image is captioned + OCR'd into text, then chunked,
stored, and made retrievable like any document — with the vision call mocked so
the test stays hermetic and Ollama-free."""
from __future__ import annotations

import math

import pytest

from brain import documents, sensory, store, vision
from brain.documents import ingest_file, is_supported_audio, is_supported_document, is_supported_image

PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
    "de0000000c4944415408d763f8cfc0f01f0005000100ffa3d4c40000000049454e44ae426082"
)

DESCRIPTION = (
    "A photo of a black cat sitting on a red doormat.\n"
    "Text in image: WELCOME"
)


@pytest.fixture
def fake_vision(monkeypatch):
    """Patch the vision model so no real model is needed. Returns a recorder."""
    calls = []

    def _describe(data, model=None, prompt=None):
        calls.append({"data": data, "model": model})
        return DESCRIPTION

    monkeypatch.setattr(vision, "describe_image", _describe)
    return calls


def test_image_suffixes_are_supported():
    assert is_supported_image("photo.JPG")
    assert is_supported_document("scan.png")
    assert not is_supported_image("notes.md")
    assert is_supported_audio("voice.wav")
    assert is_supported_document("voice.mp3")


def test_large_media_vector_is_stable_and_normalized():
    data = bytes(range(256)) * 100_000

    first = sensory.media_vector(data, "image")
    second = sensory.media_vector(data, "image")

    assert first == second
    assert math.isclose(sum(v * v for v in first), 1.0)


def test_ingest_image_stores_description(tmp_path, fake_vision):
    img = tmp_path / "doormat.png"
    img.write_bytes(PNG_1x1)

    n, cat = ingest_file(img)

    assert n >= 1
    assert cat == "doormat"
    texts = " ".join(store.list_records().get("documents") or [])
    assert "black cat" in texts
    assert "WELCOME" in texts          # OCR text made it into memory
    assert "[doormat.png]" in texts    # tagged with its filename like other docs
    senses = sensory.list_records()
    assert len(senses) == 1
    assert senses[0]["kind"] == "image"
    assert senses[0]["meta"]["source_type"] == "sensory_image"
    assert len(senses[0]["meta"]["media_sha256"]) == 64
    assert len(senses[0]["meta"]["source_sha256"]) == 64


def test_image_is_retrievable_by_its_contents(tmp_path, fake_vision):
    img = tmp_path / "doormat.png"
    img.write_bytes(PNG_1x1)
    ingest_file(img)

    from brain.retriever import retrieve_weighted

    hits = retrieve_weighted("welcome cat doormat", top_k=5)
    assert any("black cat" in (h.get("text") or "") for h in hits)
    assert any(h.get("sensory") and h.get("sensory_kind") == "image" for h in hits)


def test_sensory_retrieval_verifies_anchor_vector_hash_pair():
    sid = sensory.add_media(
        "image",
        PNG_1x1,
        "upload:abc:doormat.png",
        "doormat.png",
        DESCRIPTION,
        "doormat",
    )
    hits = sensory.retrieve_text("welcome cat doormat", top_k=5)
    matched = [h for h in hits if h["sensory_id"] == sid]
    assert matched
    assert matched[0]["meta"]["hash_verified"] is True

    col = sensory._collection("image")
    got = col.get(ids=[sid], include=["metadatas"])
    bad_meta = dict((got.get("metadatas") or [{}])[0] or {})
    bad_meta["media_sha256"] = "0" * 64
    col.update(ids=[sid], metadatas=[bad_meta])

    hits = sensory.retrieve_text("welcome cat doormat", top_k=5)
    assert all(h["sensory_id"] != sid for h in hits)


def test_audio_ingest_stores_sensation(tmp_path):
    import wave

    audio = tmp_path / "workspace.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 800)

    n, cat = ingest_file(audio)

    assert n >= 1
    assert cat == "workspace"
    senses = sensory.list_records()
    assert len(senses) == 1
    assert senses[0]["kind"] == "audio"
    assert "workspace.wav" in senses[0]["text"]
    assert senses[0]["meta"]["source_type"] == "sensory_audio"


def test_vision_model_override_is_passed_through(tmp_path, fake_vision):
    img = tmp_path / "x.webp"
    img.write_bytes(PNG_1x1)
    ingest_file(img, vision_model="some-other-vision")

    assert fake_vision[-1]["model"] == "some-other-vision"


def test_unavailable_vision_model_raises(tmp_path, monkeypatch):
    def _boom(data, model=None, prompt=None):
        raise vision.VisionUnavailable("vision model 'llama3.2-vision' is unavailable.")

    monkeypatch.setattr(vision, "describe_image", _boom)
    img = tmp_path / "y.png"
    img.write_bytes(PNG_1x1)

    with pytest.raises(vision.VisionUnavailable):
        ingest_file(img)


def test_failed_media_replacement_preserves_existing_record(monkeypatch):
    """A failed replacement must not destroy the record it was replacing: the
    old delete-first order left zero sensory records and an orphan vector when
    the second write failed."""
    source = "upload:abc:doormat.png"
    sid = sensory.add_media("image", PNG_1x1, source, "doormat.png", DESCRIPTION, "doormat")
    assert sid

    def boom(text):
        raise RuntimeError("anchor write failed")

    monkeypatch.setattr(sensory, "text_anchor_vector", boom)
    with pytest.raises(RuntimeError):
        sensory.add_media("image", b"new-image-bytes", source, "doormat.png", "new description", "doormat")

    # The prior record survives intact, and the half-written replacement vector
    # was rolled back rather than left orphaned.
    records = sensory.list_records()
    assert [r["id"] for r in records] == [sid]
    assert "black cat" in records[0]["text"]
    vec_ids = sensory._collection("image").get(where={"source": source}).get("ids") or []
    assert vec_ids == [sid]


def test_failed_identical_replacement_restores_prior_record(monkeypatch):
    """Identical bytes re-uploaded for the same source share the record id, so
    the replacement overwrites in place; if the anchor write then fails, the
    prior vector record must be restored, not left half-replaced."""
    source = "upload:abc:doormat.png"
    sid = sensory.add_media("image", PNG_1x1, source, "doormat.png", DESCRIPTION, "doormat")

    def boom(text):
        raise RuntimeError("anchor write failed")

    with monkeypatch.context() as patched:
        patched.setattr(sensory, "text_anchor_vector", boom)
        with pytest.raises(RuntimeError):
            sensory.add_media("image", PNG_1x1, source, "doormat.png", "different description", "doormat")

    records = sensory.list_records()
    assert [r["id"] for r in records] == [sid]
    got = sensory._collection("image").get(ids=[sid], include=["documents"])
    assert "black cat" in (got.get("documents") or [""])[0]
    # The restored pair still verifies (anchor and vector hashes agree).
    hits = sensory.retrieve_text("welcome cat doormat", top_k=5)
    assert any(h["sensory_id"] == sid and h["meta"]["hash_verified"] for h in hits)


def test_failed_sensory_write_on_reupload_preserves_prior_chunks(monkeypatch):
    """Endpoint-level: an identical re-upload REPLACES the existing chunks in
    place (same content-hashed source), so the failure cleanup must not delete
    that source — it would destroy the only remaining copy."""
    import base64

    from fastapi.testclient import TestClient

    import server

    monkeypatch.setattr(server, "describe_image", lambda data, model=None: DESCRIPTION)
    client = TestClient(server.app, base_url="http://127.0.0.1")
    payload = {"filename": "doormat.png", "data_base64": base64.b64encode(PNG_1x1).decode()}

    first = client.post("/api/ingest", json=payload)
    assert first.status_code == 200
    chunk_count = store.count_all(None)
    assert chunk_count > 0
    assert len(sensory.list_records()) == 1

    def _boom(*args, **kwargs):
        raise RuntimeError("sensory write failed")

    monkeypatch.setattr(server.sensory, "add_media", _boom)
    second = client.post("/api/ingest", json=payload)

    assert second.status_code == 400
    assert store.count_all(None) == chunk_count       # replaced chunks survive
    assert len(sensory.list_records()) == 1           # prior sensory record survives


def test_failed_sensory_write_rolls_back_committed_chunks(monkeypatch):
    """Endpoint-level: if sensory.add_media fails after the text chunks were
    committed, the 400 response must not leave orphaned chunks behind."""
    import base64

    from fastapi.testclient import TestClient

    import server

    monkeypatch.setattr(server, "describe_image", lambda data, model=None: DESCRIPTION)

    def _boom(*args, **kwargs):
        raise RuntimeError("sensory write failed")

    monkeypatch.setattr(server.sensory, "add_media", _boom)
    client = TestClient(server.app, base_url="http://127.0.0.1")
    before = store.count_all(None)

    response = client.post(
        "/api/ingest",
        json={"filename": "doormat.png", "data_base64": base64.b64encode(PNG_1x1).decode()},
    )

    assert response.status_code == 400
    assert "sensory write failed" in response.json()["detail"]
    assert store.count_all(None) == before   # committed chunks rolled back
    assert sensory.list_records() == []


def test_failed_sensory_write_on_reupload_keeps_the_old_caption(monkeypatch):
    """Endpoint-level: a failed identical re-upload must not change stored
    memory AT ALL. The vision caption differs between runs, so if the new
    chunks replaced the old ones before the sensory write failed, the store
    would hold the new caption alongside the old sensory description —
    a state change smuggled out of a 400 response."""
    import base64

    from fastapi.testclient import TestClient

    import server

    monkeypatch.setattr(server, "describe_image", lambda data, model=None: DESCRIPTION)
    client = TestClient(server.app, base_url="http://127.0.0.1")
    payload = {"filename": "doormat.png", "data_base64": base64.b64encode(PNG_1x1).decode()}

    assert client.post("/api/ingest", json=payload).status_code == 200
    chunk_count = store.count_all(None)

    # Second run: the (nondeterministic) vision model captions differently,
    # and the sensory write fails.
    monkeypatch.setattr(server, "describe_image", lambda data, model=None: "A totally different caption")

    def _boom(*args, **kwargs):
        raise RuntimeError("sensory write failed")

    monkeypatch.setattr(server.sensory, "add_media", _boom)
    second = client.post("/api/ingest", json=payload)

    assert second.status_code == 400
    assert store.count_all(None) == chunk_count
    texts = [rec["text"] for rec in store.list_records_all(None)]
    assert not any("totally different caption" in t for t in texts)   # new chunks rolled back
    assert any("WELCOME" in t for t in texts)                          # old caption intact


def test_identical_media_reupload_replaces_chunks_after_sensory_success(monkeypatch):
    """Endpoint-level: the deferred replacement must still replace — a
    successful identical re-upload swaps in the new caption's chunks without
    doubling the store or the sensory records."""
    import base64

    from fastapi.testclient import TestClient

    import server

    monkeypatch.setattr(server, "describe_image", lambda data, model=None: DESCRIPTION)
    client = TestClient(server.app, base_url="http://127.0.0.1")
    payload = {"filename": "doormat.png", "data_base64": base64.b64encode(PNG_1x1).decode()}

    assert client.post("/api/ingest", json=payload).status_code == 200
    chunk_count = store.count_all(None)

    monkeypatch.setattr(server, "describe_image", lambda data, model=None: "A fresh caption of the doormat")
    assert client.post("/api/ingest", json=payload).status_code == 200

    assert store.count_all(None) == chunk_count       # replaced, not doubled
    assert len(sensory.list_records()) == 1
    texts = [rec["text"] for rec in store.list_records_all(None)]
    assert any("fresh caption" in t for t in texts)
    assert not any("WELCOME" in t for t in texts)
