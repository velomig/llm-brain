"""Project-level tools."""
from __future__ import annotations

from pathlib import Path
import fcntl
import subprocess
import sys
import threading

import config
from brain.toolkit.paths import is_under, resolve_tool_path

FULL_SUITE_TARGETS = {
    "all",
    "all tests",
    "entire suite",
    "full",
    "full-suite",
    "full suite",
    "test suite",
    "tests",
    "the full suite",
    "whole suite",
}
_TEST_RUN_THREAD_LOCK = threading.Lock()


def _output_text(value, limit: int) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value or "")[-limit:]


def run_project_command(args: list[str], timeout: float = 30.0) -> dict:
    try:
        proc = subprocess.run(args, cwd=str(config.BASE_DIR), capture_output=True, text=True, timeout=timeout)
        return {
            "returncode": proc.returncode,
            "stdout": _output_text(proc.stdout, 8000),
            "stderr": _output_text(proc.stderr, 4000),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "error": f"command timed out after {timeout:g}s",
            "stdout": _output_text(exc.stdout, 4000),
            "stderr": _output_text(exc.stderr, 2000),
        }
    except Exception as exc:
        return {"error": str(exc)}


def git_status(args: dict | None = None) -> dict:
    return run_project_command(["git", "status", "--short"], timeout=10.0)


def git_log(args: dict | None = None) -> dict:
    """Recent commit history. Read-only; count is clamped so a runaway request
    can't dump the whole history into the prompt."""
    args = args or {}
    try:
        count = int(args.get("count") or 10)
    except (TypeError, ValueError):
        count = 10
    count = max(1, min(count, 50))
    cmd = ["git", "log", f"-n{count}", "--date=short", "--pretty=format:%h %ad %s"]
    path = args.get("path")
    if path:
        try:
            resolved = resolve_tool_path(path)
        except Exception as exc:
            return {"path": path, "error": str(exc)}
        cmd += ["--", str(resolved.relative_to(config.BASE_DIR)) if is_under(resolved, config.BASE_DIR) else str(resolved)]
    return run_project_command(cmd, timeout=10.0)


def git_diff(args: dict | None = None) -> dict:
    args = args or {}
    cmd = ["git", "diff", "--"]
    path = args.get("path")
    if path:
        try:
            resolved = resolve_tool_path(path)
        except Exception as exc:
            return {"path": path, "error": str(exc)}
        cmd.append(str(resolved.relative_to(config.BASE_DIR)) if is_under(resolved, config.BASE_DIR) else str(resolved))
    return run_project_command(cmd, timeout=10.0)


def run_tests(args: dict | None = None) -> dict:
    args = args or {}
    cmd = [sys.executable, "-m", "pytest"]
    target = str(args.get("target") or "").strip()
    if target.lower() in FULL_SUITE_TARGETS:
        target = ""
    if target:
        path_part, sep, node_part = target.partition("::")
        path = Path(path_part)
        resolved = (path if path.is_absolute() else config.BASE_DIR / path).resolve(strict=False)
        if (
            target.startswith("-")
            or not path_part
            or ".." in path.parts
            or not (resolved == config.BASE_DIR or is_under(resolved, config.BASE_DIR))
        ):
            return {"error": "invalid pytest target"}
        safe_target = str(resolved.relative_to(config.BASE_DIR))
        if sep:
            safe_target += sep + node_part
        cmd.append(safe_target)
    # Full suites are not safe to overlap: they exercise shared project files,
    # ports, and process-level fixtures. A second approval must fail immediately
    # instead of launching another pytest that can make both runs hang.
    if not _TEST_RUN_THREAD_LOCK.acquire(blocking=False):
        return {"error": "another test run is already in progress", "retryable": True}
    try:
        lock_path = config.DB_DIR / ".neural_map_run_tests.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = lock_path.open("a+", encoding="utf-8")
        try:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {
                    "error": "another test run is already in progress",
                    "retryable": True,
                }
            return run_project_command(cmd, timeout=120.0)
        finally:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                lock_handle.close()
    finally:
        _TEST_RUN_THREAD_LOCK.release()
