"""Regression coverage for the September 2026 code audit."""
from concurrent.futures import ThreadPoolExecutor
import logging
import threading
import subprocess
import sys
import time
from pathlib import Path

import ollama
import pytest
from fastapi.testclient import TestClient

import config
import server
from brain import documents, embeddings, graph, portable, sleep, store, tasks
from brain.toolkit import governance, local_state, registry
from util import store_memory


@pytest.mark.parametrize("filename", [
    "runner.sh", "start.command", "launch.applescript", "job.plist", "requirements-dev.txt",
    "pyproject.toml", "Makefile", "runner", "brain/policy.txt", "launchd/task.json",
    "Neural Map.app/Contents/Resources/data.json",
])
def test_code_guard_covers_execution_and_configuration_inputs(tmp_path, monkeypatch, filename):
    from brain.toolkit.paths import reserved_code_path

    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    monkeypatch.setattr(config, "TOOL_CODE_WRITES_ENABLED", False)
    assert reserved_code_path(tmp_path / filename)
    monkeypatch.setattr(config, "TOOL_CODE_WRITES_ENABLED", True)
    assert not reserved_code_path(tmp_path / filename)


@pytest.mark.parametrize("executable", [False, True])
def test_code_guard_recognizes_disguised_scripts(tmp_path, monkeypatch, executable):
    from brain.toolkit.paths import reserved_code_path

    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    monkeypatch.setattr(config, "TOOL_CODE_WRITES_ENABLED", False)
    target = tmp_path / "script.txt"
    target.write_text("payload" if executable else "#!/bin/sh\nexit 0\n")
    target.chmod(0o700 if executable else 0o600)
    assert reserved_code_path(target)


def approved(tool, args):
    queued = registry.dispatch(tool, args)
    assert queued.get("pending"), queued
    action_id = queued["action"]["id"]
    result = registry.approve_pending(action_id)
    assert result["action"]["status"] == "approved", result
    return action_id


def test_scheduled_shell_script_is_reserved(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    monkeypatch.setattr(config, "TOOL_ALLOWED_ROOTS", [tmp_path])
    monkeypatch.setattr(config, "TOOL_CODE_WRITES_ENABLED", False)
    target = tmp_path / "backup_job.sh"
    target.write_text("#!/bin/sh\nexit 0\n")
    result = registry.dispatch("write_file", {
        "filename": str(target), "mode": "overwrite",
        "content": "#!/bin/sh\n# audit placeholder; never executed\nexit 0\n",
    })
    assert result.get("error_kind") == "blocked_preflight", result


def test_undo_contact_preserves_later_contact():
    first = approved("upsert_contact", {"name": "Audit Alpha"})
    approved("upsert_contact", {"name": "Audit Beta"})
    result = registry.undo_pending(first)
    assert result.get("result", {}).get("ok"), result
    assert local_state.lookup_contact({"query": "Audit Beta"})["contacts"]


def test_undo_contact_refuses_newer_changes_to_same_record():
    first = approved("upsert_contact", {"name": "Audit Alpha", "notes": "original action"})
    approved("upsert_contact", {"name": "Audit Alpha", "notes": "newer action"})
    assert registry.undo_pending(first).get("error")
    assert local_state.lookup_contact({"query": "Audit Alpha"})["contacts"][0]["notes"] == "newer action"


def test_undo_calendar_preserves_later_event():
    first = approved("create_calendar_event", {"title": "Alpha", "start": "2099-01-01T12:00:00Z"})
    approved("create_calendar_event", {"title": "Beta", "start": "2099-01-02T12:00:00Z"})
    assert registry.undo_pending(first)["result"]["ok"]
    assert [row["title"] for row in local_state._read_json_file(local_state.CALENDAR_FILE)["events"]] == ["Beta"]


def test_undo_calendar_refuses_newer_changes():
    args = {"title": "Alpha", "start": "2099-01-01T12:00:00Z"}
    first = approved("create_calendar_event", args)
    local_state.create_calendar_event({**args, "notes": "newer details"})
    assert registry.undo_pending(first).get("error")
    assert local_state._read_json_file(local_state.CALENDAR_FILE)["events"][0]["notes"] == "newer details"


def test_invalid_calendar_end_is_reported():
    result = local_state.create_calendar_event({"title": "Alpha", "start": "2099-01-01", "end": "bad date"})
    assert "ISO" in result["error"]


def test_undo_file_preserves_subsequent_edit(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TOOL_ALLOWED_ROOTS", [tmp_path])
    target = tmp_path / "audit-note.txt"
    target.write_text("original")
    action = approved("write_file", {
        "filename": str(target), "mode": "overwrite", "content": "approved edit",
    })
    target.write_text("later user edit")
    result = registry.undo_pending(action)
    assert result.get("error") and target.read_text() == "later user edit", result


def test_concurrent_file_undo_runs_once_and_preserves_private_permissions(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TOOL_ALLOWED_ROOTS", [tmp_path])
    target = tmp_path / "note.txt"
    target.write_text("before")
    target.chmod(0o600)
    action = approved("write_file", {"filename": str(target), "mode": "overwrite", "content": "after"})
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(registry.undo_pending, [action, action]))
    assert sum(bool(r.get("result", {}).get("ok")) for r in results) == 1
    assert target.read_text() == "before"
    assert target.stat().st_mode & 0o777 == 0o600


def test_created_file_undo_refuses_changed_content(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TOOL_ALLOWED_ROOTS", [tmp_path])
    target = tmp_path / "new.txt"
    action = approved("write_file", {"filename": str(target), "content": "created"})
    target.write_text("later edit")
    assert registry.undo_pending(action).get("error")
    assert target.read_text() == "later edit"


def test_undo_memory_restores_pin_and_provenance():
    memory = store_memory("synthetic document fact", extra={
        "pinned": True, "source_type": "uploaded_doc",
    })
    action = approved("pin_memory", {"id": memory.id, "pinned": False})
    result = registry.undo_pending(action)
    assert result.get("result", {}).get("ok"), result
    meta = store.get_by_ids([memory.id])[memory.id]["meta"]
    assert (meta["pinned"], meta["source_type"]) == (True, "uploaded_doc"), meta


@pytest.mark.parametrize("text_only", [False, True])
def test_memory_undo_restores_exact_metadata_without_embedding(monkeypatch, text_only):
    memory = store_memory("preserve all fields", text_only=text_only,
                          extra={"pinned": True, "source_type": "uploaded_doc", "custom": "keep"})
    original = store.get_by_ids([memory.id])[memory.id]
    action = approved("mark_memory_wrong", {"id": memory.id, "reason": "test"})
    monkeypatch.setattr(governance, "embed_document", lambda *a, **kw: pytest.fail("undo must reuse the saved embedding"))
    assert registry.undo_pending(action)["result"]["ok"]
    assert store.get_by_ids([memory.id])[memory.id] == original


def test_concurrent_uploads_leave_one_complete_version(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    second_entered = threading.Event()
    original = server._ingest_upload_locked

    def hold_first(req):
        if req.text.startswith("First"):
            entered.set()
            assert release.wait(5)
        else:
            second_entered.set()
        return original(req)

    monkeypatch.setattr(server, "_ingest_upload_locked", hold_first)
    monkeypatch.setattr(server, "_backup_before", lambda *a, **kw: "audit-backup-placeholder")
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(server.ingest_markdown, server.IngestReq(filename="audit.txt", text="First complete version"))
        assert entered.wait(5)
        second = pool.submit(server.ingest_markdown, server.IngestReq(filename="audit.txt", text="Second complete version"))
        try:
            assert not second_entered.wait(0.2), "second upload entered the replacement transaction concurrently"
        finally:
            release.set()
        responses = [first.result(5), second.result(5)]
    assert all(response["chunks"] == 1 for response in responses)
    assert store.count_all() == 1, responses
    assert "Second complete version" in store.list_records_all()[0]["text"]


def test_direct_source_replacements_are_serialized(monkeypatch):
    entered, release, second_entered = threading.Event(), threading.Event(), threading.Event()
    original = documents._ingest_document_text_locked

    def hold_first(text, *args, **kwargs):
        if text == "first":
            entered.set()
            assert release.wait(5)
        else:
            second_entered.set()
        return original(text, *args, **kwargs)

    monkeypatch.setattr(documents, "_ingest_document_text_locked", hold_first)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(documents.ingest_document_text, "first", "note.txt", "shared-source")
        assert entered.wait(5)
        second = pool.submit(documents.ingest_document_text, "second", "note.txt", "shared-source")
        try:
            assert not second_entered.wait(0.2)
        finally:
            release.set()
        assert first.result(5)[0] == second.result(5)[0] == 1
    rows = store.list_records_all()
    assert len(rows) == 1 and "second" in rows[0]["text"]


def test_reentrant_file_lock_excludes_other_processes(tmp_path):
    from brain.utils import file_lock

    lock, ready, changed = (tmp_path / name for name in ("state.lock", "ready", "changed"))
    code = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from brain.utils import file_lock
Path(sys.argv[3]).write_text('ready')
with file_lock(Path(sys.argv[2])):
    Path(sys.argv[4]).write_text('changed')
"""
    child = None
    try:
        with file_lock(lock):
            with file_lock(lock):
                child = subprocess.Popen([sys.executable, "-I", "-c", code,
                                          str(Path(__file__).resolve().parents[1]), str(lock), str(ready), str(changed)])
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert ready.exists()
                with pytest.raises(subprocess.TimeoutExpired):
                    child.wait(timeout=0.1)
                assert not changed.exists()
            # Leaving the nested lock must not release the outer flock.
            with pytest.raises(subprocess.TimeoutExpired):
                child.wait(timeout=0.1)
            assert not changed.exists()
        assert child.wait(timeout=5) == 0
        assert changed.read_text() == "changed"
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait()


def test_task_capacity_never_silently_discards_pending_work(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "_STATE_FILE", tmp_path / "tasks.json")
    monkeypatch.setattr(config, "BACKGROUND_TASK_MAX", 2)
    first = tasks.enqueue("first unfinished task")
    tasks.enqueue("second unfinished task")
    try:
        tasks.enqueue("third unfinished task")
    except ValueError:
        pass  # Explicitly refusing a full queue would also be safe.
    assert tasks.get(first["id"]) is not None


def test_task_retention_prunes_terminal_history_before_active_work(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "_STATE_FILE", tmp_path / "tasks.json")
    monkeypatch.setattr(config, "BACKGROUND_TASK_MAX", 2)
    first = tasks.enqueue("keep this active task")
    done = tasks.enqueue("completed task")
    tasks._finish(done["id"], {"status": "done"})
    last = tasks.enqueue("new task")
    assert {t["id"] for t in tasks.list_tasks()} == {first["id"], last["id"]}


def test_reduced_task_capacity_keeps_every_unfinished_state(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "_STATE_FILE", tmp_path / "tasks.json")
    monkeypatch.setattr(config, "BACKGROUND_TASK_MAX", 1)
    rows = [{"id": str(i), "status": status} for i, status in enumerate(["queued", "running", "awaiting_approval"])]
    tasks._write_unlocked(rows)
    assert tasks._read_unlocked() == rows


def test_encoded_bootstrap_token_is_redacted(monkeypatch):
    token = "synthetic-audit-secret"
    monkeypatch.setattr(config, "SERVER_TOKEN", token)
    response = TestClient(server.app, base_url="http://127.0.0.1").get(
        "/?%74oken=" + token, follow_redirects=False,
    )
    assert response.status_code in (302, 303, 307)
    record = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1,
                               "%s", ("GET /?%74oken=" + token + " HTTP/1.1",), None)
    server._RedactTokenQueryFilter().filter(record)
    assert token not in record.getMessage()


@pytest.mark.parametrize("key", ["token", "%74oken", "to%6ben", "%74%6f%6b%65%6e"])
def test_redaction_handles_duplicates_and_mapping_log_arguments(key):
    record = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, "%(path)s",
                               ({"path": f"GET /?{key}=secret1&x=public&token=secret2 HTTP/1.1"},), None)
    server._RedactTokenQueryFilter().filter(record)
    assert "secret" not in record.getMessage()
    assert "x=public" in record.getMessage()


def test_missing_embedding_model_keeps_manual_memory(monkeypatch):
    def unavailable(**kwargs):
        raise ollama.ResponseError("model 'audit-embed' not found", status_code=404)

    monkeypatch.setattr(embeddings._client, "embeddings", unavailable)
    response = TestClient(server.app, base_url="http://127.0.0.1").post(
        "/api/memory", json={"text": "synthetic memory while embedder is unavailable"},
    )
    assert response.status_code == 200, response.json()
    assert store.count(text_only=True) == 1


@pytest.mark.parametrize("method", ["_embed", "_embed_batch"])
@pytest.mark.parametrize("error", [
    ConnectionError("offline"), TimeoutError("timed out"),
    ollama.ResponseError("model 'missing' not found", status_code=404),
    ollama.ResponseError("unavailable", status_code=503),
])
def test_embedding_transport_failures_allow_fallback(monkeypatch, method, error):
    def fail(**kwargs):
        raise error
    monkeypatch.setattr(embeddings._client, "embeddings" if method == "_embed" else "embed", fail)
    with pytest.raises(embeddings.EmbeddingUnavailable):
        getattr(embeddings, method)("text" if method == "_embed" else ["text"])


def test_embedding_validation_error_is_not_masked(monkeypatch):
    def fail(**kwargs):
        raise ollama.ResponseError("invalid dimensions", status_code=400)
    monkeypatch.setattr(embeddings._client, "embeddings", fail)
    with pytest.raises(ollama.ResponseError):
        embeddings._embed("text")


def test_malformed_replace_import_preserves_existing_memory():
    existing = store_memory("must survive an invalid backup")
    try:
        portable.import_data({"neural_map_export": {
            "version": 1, "count": 1, "memories": [{"text": "record without an id"}],
        }}, replace=True)
    except ValueError:
        pass
    assert existing.id in store.get_by_ids([existing.id])


@pytest.mark.parametrize("patch", [
    {"version": 99}, {"version": True}, {"schema_version": 99}, {"count": 0},
    {"memories": [None]}, {"memories": [{"id": "a", "text": ""}]},
    {"memories": [{"id": "a", "text": "ok", "pinned": "false"}]},
    {"memories": [{"id": "a", "text": "ok", "custom": {}}]},
    {"memories": [{"id": "a", "text": "ok", "importance": 9}]},
    {"memories": [{"id": "a", "text": "ok", "custom": float("nan")}]},
    {"count": 2, "memories": [{"id": "a", "text": "one"}, {"id": "a", "text": "two"}]},
    {"graph": "bad"}, {"graph": [{"subject": "a", "relation": "b", "object": "c", "fact_id": "missing"}]},
])
def test_invalid_archives_fail_before_embedding_or_writing(monkeypatch, patch):
    existing = store_memory("retain existing archive")
    before = portable.export_data()["neural_map_export"]["memories"]
    monkeypatch.setattr(portable, "embed_documents", lambda *a, **kw: pytest.fail("invalid archive reached embedding"))
    data = {"version": 1, "count": 1, "memories": [{"id": "a", "text": "valid"}], **patch}
    with pytest.raises(ValueError):
        portable.import_data({"neural_map_export": data}, replace=True)
    assert portable.export_data()["neural_map_export"]["memories"] == before
    assert existing.id in store.get_by_ids([existing.id])


def test_import_cannot_overwrite_foreign_graph_collection():
    triple = {"subject": "Audit entity", "relation": "owns", "object": "Audit item"}
    graph.add_relations([triple], fact_id="existing-foreign-fact", collection="other-model")
    try:
        portable.import_data({"neural_map_export": {
            "memories": [], "graph": [{**triple, "fact_id": "injected-fact", "collection": "other-model"}],
        }})
    except ValueError:
        pass
    assert graph.export_edges()[0]["fact_id"] == "existing-foreign-fact"


def test_oversized_snapshot_is_rejected_before_reading(tmp_path, monkeypatch):
    target = tmp_path / "large-audit-file.txt"
    target.write_bytes(b"x" * 1025)
    monkeypatch.setattr(config, "MAX_TOOL_UNDO_BYTES", 1024)
    original = Path.read_bytes
    reads = []

    def record_read(path):
        if path == target:
            reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", record_read)
    monkeypatch.setattr(config, "TOOL_ALLOWED_ROOTS", [tmp_path])
    snapshot = governance.snapshot_action("write_file", {"filename": str(target)})
    assert not snapshot["undoable"]
    assert reads == [], "oversized file was read into memory before the cap check"


def test_snapshot_reads_are_bounded_even_if_file_grows(tmp_path, monkeypatch):
    import io

    target = tmp_path / "growing.txt"
    target.write_bytes(b"a")
    monkeypatch.setattr(config, "MAX_TOOL_UNDO_BYTES", 16)
    real_open = Path.open
    reads = []

    class GrowingFile(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    monkeypatch.setattr(Path, "open", lambda path, *a, **kw: GrowingFile(b"x" * 100)
                        if path == target else real_open(path, *a, **kw))
    assert not governance._file_snapshot(target)["undoable"]
    assert not governance._json_file_snapshot(target)["undoable"]
    assert reads == [17, 17]


@pytest.mark.parametrize("dry_run", [False, True])
def test_zero_dream_limit_disables_generation(monkeypatch, dry_run):
    store_memory("synthetic fact A", category="audit-a", importance=5)
    store_memory("synthetic fact B", category="audit-b", importance=5)
    calls = []

    def synthetic_dream(*args, **kwargs):
        calls.append(True)
        return "Speculative hunch: synthetic audit connection."

    monkeypatch.setattr(sleep, "_dream", synthetic_dream)
    result = sleep.run(max_archive=0, max_dreams=0, dry_run=dry_run)
    assert result["dreamed"] == result["dreamable"] == 0 and not calls, result
