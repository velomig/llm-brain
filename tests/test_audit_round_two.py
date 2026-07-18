"""Regression coverage for the 2026-07 whole-code audit fixes."""
from __future__ import annotations

import stat
import threading
import time

import config
from brain import portable
from brain.toolkit import governance, memory_tools, perspective_update, source_truth


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_private_storage_directory_is_owner_only(tmp_path):
    store_dir = tmp_path / "store"
    store_dir.mkdir(mode=0o755)
    store_dir.chmod(0o755)

    assert config.ensure_private_directory(store_dir) == store_dir
    assert _mode(store_dir) == 0o700


def test_safety_backup_directory_and_file_are_owner_only(tmp_path, monkeypatch):
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(mode=0o755)
    backup_dir.chmod(0o755)
    monkeypatch.setattr(portable, "BACKUP_DIR", backup_dir)
    monkeypatch.setattr(
        portable,
        "export_data",
        lambda model=None: {"neural_map_export": {"memories": [], "count": 0}},
    )

    path = portable.save_backup("permission-test")

    assert _mode(backup_dir) == 0o700
    assert _mode(path) == 0o600


def _run_concurrent_replacements(update, replacements):
    start = threading.Barrier(len(replacements) + 1)
    results = []

    def worker(old_text, new_text):
        start.wait(timeout=2)
        results.append(update({"old_text": old_text, "new_text": new_text}))

    threads = [threading.Thread(target=worker, args=pair) for pair in replacements]
    for thread in threads:
        thread.start()
    start.wait(timeout=2)
    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive()
    return results


def test_source_of_truth_concurrent_edits_do_not_lose_updates(tmp_path, monkeypatch):
    path = tmp_path / "source-of-truth.md"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    monkeypatch.setattr(config, "SOURCE_OF_TRUTH_PATH", path)

    original_plan = source_truth.plan_replacement
    active = 0
    max_active = 0
    guard = threading.Lock()

    def slow_plan(*args):
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.04)
        try:
            return original_plan(*args)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(source_truth, "plan_replacement", slow_plan)
    results = _run_concurrent_replacements(
        source_truth.update_source_of_truth,
        [("alpha", "ALPHA"), ("beta", "BETA")],
    )

    assert [result.get("ok") for result in results] == [True, True]
    assert path.read_text(encoding="utf-8") == "ALPHA\nBETA\n"
    assert max_active == 1


def test_perspective_concurrent_edits_do_not_lose_updates(tmp_path, monkeypatch):
    path = tmp_path / "will-perspective.md"
    path.write_text("first principle\nsecond principle\n", encoding="utf-8")
    monkeypatch.setattr(config, "WILL_PERSPECTIVE_PATH", path)
    monkeypatch.setattr(config, "WILL_PERSPECTIVE_MAX_CHARS", 10_000)
    monkeypatch.setattr(config, "WILL_PERSPECTIVE_MAX_EDIT_CHARS", 1_000)
    monkeypatch.setattr(
        perspective_update.perspective,
        "append_identity_change",
        lambda *args, **kwargs: {"written": True},
    )

    original_plan = perspective_update.plan_replacement
    active = 0
    max_active = 0
    guard = threading.Lock()

    def slow_plan(*args):
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.04)
        try:
            return original_plan(*args)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(perspective_update, "plan_replacement", slow_plan)
    results = _run_concurrent_replacements(
        perspective_update.update_will_perspective,
        [("first principle", "FIRST principle"), ("second principle", "SECOND principle")],
    )

    assert [result.get("ok") for result in results] == [True, True]
    assert path.read_text(encoding="utf-8") == "FIRST principle\nSECOND principle\n"
    assert max_active == 1


def test_absent_memory_tool_model_remains_none(monkeypatch):
    monkeypatch.setattr(config, "EMBED_MODEL", "__selected__")

    assert memory_tools._tool_model({}) is None
    assert governance._memory_model({}) is None
    assert config.embedding_model(memory_tools._tool_model({})) == config.CHAT_MODEL

