"""Regression coverage for the fourth audit round.

Covers: write_file refusing the project's own code paths (H1), portable
restore rolling back even when the destructive clears themselves fail (H2),
hidden-path exclusion for the file tools (M2), the calendar attendee
allowlist (M3), the metacognition active-note cap (M4), the system-prompt
budget (M6), the reflect() lock (M8), the evolution proposal cap (M9), and
audit-event truncation (M11).
"""
from __future__ import annotations

import pytest

import config
from brain import evolution, metacognition, pipeline, store
from brain import reflect as reflect_mod
from brain.portable import export_data, import_data
from brain.toolkit import files, governance
from brain.toolkit import google_calendar
from util import store_memory


# --- H1: write_file must not touch the project's own code ---------------------

def test_write_file_refuses_project_code_paths(tmp_path, monkeypatch):
    base = tmp_path / "proj"
    (base / "web").mkdir(parents=True)
    monkeypatch.setattr(config, "BASE_DIR", base)
    monkeypatch.setattr(config, "TOOL_ALLOWED_ROOTS", [tmp_path])

    for target in (base / "config.py", base / "brain_module.py", base / "web" / "app.js"):
        out = files.write_file({"filename": str(target), "content": "x", "mode": "create"})
        assert "reserved" in out["error"], target
        assert not target.exists()


def test_write_file_allows_python_outside_base_dir(tmp_path, monkeypatch):
    base = tmp_path / "proj"
    base.mkdir()
    monkeypatch.setattr(config, "BASE_DIR", base)
    monkeypatch.setattr(config, "TOOL_ALLOWED_ROOTS", [tmp_path])

    out = files.write_file({"filename": str(tmp_path / "scratch.py"), "content": "x", "mode": "create"})
    assert out.get("ok") is True


def test_write_file_code_guard_can_be_opted_out(tmp_path, monkeypatch):
    base = tmp_path / "proj"
    base.mkdir()
    monkeypatch.setattr(config, "BASE_DIR", base)
    monkeypatch.setattr(config, "TOOL_ALLOWED_ROOTS", [tmp_path])
    monkeypatch.setattr(config, "TOOL_CODE_WRITES_ENABLED", True)

    out = files.write_file({"filename": str(base / "patch.py"), "content": "x", "mode": "create"})
    assert out.get("ok") is True


def test_write_file_preview_surfaces_code_path_block(tmp_path, monkeypatch):
    base = tmp_path / "proj"
    base.mkdir()
    monkeypatch.setattr(config, "BASE_DIR", base)
    monkeypatch.setattr(config, "TOOL_ALLOWED_ROOTS", [tmp_path])

    preview = governance.preview_action("write_file", {"filename": str(base / "registry.py"), "content": "x"})
    assert "blocked" in preview


# --- H2: restore rollback survives failures of the clears themselves ----------

def test_replace_clear_failure_restores_snapshot(monkeypatch):
    existing = store_memory("original", category="fact")
    data = export_data()

    calls = 0
    original_clear = store.clear_all

    def _fail_once(model=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("database is locked")
        return original_clear(model)

    monkeypatch.setattr("brain.portable.store.clear_all", _fail_once)

    with pytest.raises(RuntimeError):
        import_data(data, replace=True)

    got = store.get_by_ids([existing.id])
    assert got[existing.id]["text"] == "original"


def test_replace_graph_clear_failure_restores_snapshot(monkeypatch):
    existing = store_memory("original", category="fact")
    data = export_data()

    def _always_locked(collections, include_legacy=False):
        raise RuntimeError("database is locked")

    monkeypatch.setattr("brain.portable.graph.clear_collections", _always_locked)

    with pytest.raises(RuntimeError):
        import_data(data, replace=True)

    # The graph clear fails during rollback too; the snapshot restore must
    # still run instead of the store staying wiped.
    got = store.get_by_ids([existing.id])
    assert got[existing.id]["text"] == "original"


# --- M2: file tools refuse hidden paths ---------------------------------------

def test_file_tools_refuse_hidden_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TOOL_ALLOWED_ROOTS", [tmp_path])
    secret_dir = tmp_path / ".claude"
    secret_dir.mkdir()
    (secret_dir / "settings.local.json").write_text('{"key": "s3cret"}', encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=abc", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("hello", encoding="utf-8")

    for hidden in (secret_dir / "settings.local.json", tmp_path / ".env"):
        out = files.read_file({"filename": str(hidden)})
        assert "hidden path is blocked" in out["error"], hidden

    out = files.write_file({"filename": str(tmp_path / ".env"), "content": "x", "mode": "overwrite"})
    assert "hidden path is blocked" in out["error"]

    listing = files.list_files({"path": str(tmp_path)})
    names = [row["name"] for row in listing["entries"]]
    assert "visible.txt" in names
    assert ".env" not in names
    assert ".claude" not in names


# --- M3: calendar attendees share the email recipient allowlist ---------------

def test_attendees_require_recipient_allowlist(monkeypatch):
    monkeypatch.setattr(config, "EMAIL_ALLOWED_RECIPIENTS", {"ok@example.com"})
    assert google_calendar._attendees(["ok@example.com"]) == [{"email": "ok@example.com"}]
    with pytest.raises(ValueError, match="ALLOWED_RECIPIENTS"):
        google_calendar._attendees(["evil@example.com"])


def test_attendees_fail_closed_without_allowlist(monkeypatch):
    monkeypatch.setattr(config, "EMAIL_ALLOWED_RECIPIENTS", set())
    with pytest.raises(ValueError, match="ALLOWED_RECIPIENTS"):
        google_calendar._attendees(["anyone@example.com"])


# --- M4: metacognition notes are capped and flagged unreviewed ----------------

def test_metacognition_notes_capped_and_marked_unreviewed(monkeypatch):
    monkeypatch.setattr(config, "METACOGNITION_MAX_ACTIVE", 3)

    for i in range(5):
        metacognition._store_note(f"wisdom note number {i}", bias="test", confidence=0.5)

    rows = [
        r
        for r in store.list_records_all(None)
        if (r.get("meta") or {}).get("category") == config.METACOGNITION_CATEGORY
    ]
    active = [r for r in rows if not (r.get("meta") or {}).get("superseded")]
    assert len(active) == 3
    assert all((r.get("meta") or {}).get("unreviewed") for r in active)


# --- M6: system prompt is budgeted, guardrails survive trimming ---------------

def test_build_system_enforces_prompt_budget(monkeypatch):
    fat_mems = [
        {"id": str(i), "text": f"memory {i} " + ("word " * 400), "category": "note",
         "source": None, "date": "2026-06-12"}
        for i in range(40)
    ]
    monkeypatch.setattr(config, "SYSTEM_PROMPT_MAX_CHARS", 0)
    unbounded = pipeline.build_system(fat_mems)
    budget = len(pipeline.build_system([])) + 3000
    assert len(unbounded) > budget

    monkeypatch.setattr(config, "SYSTEM_PROMPT_MAX_CHARS", budget)
    bounded = pipeline.build_system(fat_mems)
    assert len(bounded) <= budget
    assert "trimmed" in bounded
    # The guardrail-bearing preamble is never a trim candidate.
    assert "Do not mention that you are reading from memory unless asked" in bounded


# --- M8: reflect() runs under its serialization lock --------------------------

def test_reflect_serializes_via_lock(monkeypatch):
    seen = {}

    def _observe(model=None):
        seen["thread_lock_held"] = reflect_mod._REFLECT_THREAD_LOCK.locked()
        return {"insights": 0, "reminders": 0, "initiatives": 0}

    monkeypatch.setattr(reflect_mod, "_reflect_locked", _observe)
    reflect_mod.reflect()
    assert seen["thread_lock_held"] is True


# --- M9: evolution proposals are capped, pending ones kept --------------------

def test_evolution_proposals_are_capped_keeping_pending():
    state = {
        "proposals": [{"id": f"r{i}", "status": "rejected"} for i in range(evolution._MAX_PROPOSALS + 50)]
        + [{"id": "keep-me", "status": "pending"}],
        "snapshots": [],
    }
    with evolution._LOCK:
        with evolution._state_lock():
            evolution._write_state_unlocked(state)

    current = evolution.state()
    assert len(current["proposals"]) == evolution._MAX_PROPOSALS
    assert any(p.get("id") == "keep-me" for p in current["proposals"])
    # Oldest resolved proposals were the ones dropped (51 over the cap).
    assert current["proposals"][0]["id"] == "r51"


# --- M11: audit events store bounded args/results -----------------------------

def test_audit_events_truncate_large_args_and_results():
    event = governance.append_audit(
        {"id": "a1", "tool": "write_file", "policy": "approval", "args": {"content": "x" * 60000}},
        "executed",
        result={"stdout": "y" * 60000},
    )
    assert len(event["args"]["content"]) < 3000
    assert "chars total" in event["args"]["content"]
    assert len(event["result"]["stdout"]) < 3000
