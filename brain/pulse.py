"""System Pulse: a shared heartbeat log of autonomous background activity.

The self-model can only reason about agency from what reaches its context. The
scheduled jobs (will review, task runner, reflection, sleep cycle) run under
cron/launchd where the model cannot see them, so its self-assessment degrades
into inference from protocol docs. Each job appends one line here; the chat
prompt injects the recent tail, turning "I have never initiated anything" into
evidence-based observation of its own background runs.

Format, one event per line (newest last):

    TIMESTAMP | PROCESS_NAME | ACTION_TAKEN | STATUS
"""
from __future__ import annotations

import threading

import config
from brain.utils import file_lock
from brain.memory import now_iso

PULSE_FILE = config.DB_DIR / "system_pulse.log"
# Cap the file so a years-long install stays a few KB; recent() never needs more.
MAX_LINES = 400
_FIELD_MAX_CHARS = 200
_THREAD_LOCK = threading.Lock()


def _lock():
    """Cross-process exclusive lock on the pulse file."""
    return file_lock(PULSE_FILE.with_name(".neural_map_system_pulse.lock"))


def _clean_field(value: object) -> str:
    # The pipe is the field separator and a newline would split the record, so
    # both must be stripped from free-text fields before writing.
    text = " ".join(str(value or "").replace("|", "/").split())
    return text[:_FIELD_MAX_CHARS] or "-"


def record(process: str, action: str, status: str) -> str:
    """Append one pulse line and return it. Never raises: a broken pulse must
    not take down the job it is reporting on."""
    line = f"{now_iso()} | {_clean_field(process)} | {_clean_field(action)} | {_clean_field(status)}"
    try:
        with _THREAD_LOCK:
            with _lock():
                lines: list[str] = []
                if PULSE_FILE.exists():
                    lines = PULSE_FILE.read_text(encoding="utf-8").splitlines()
                lines.append(line)
                if len(lines) > MAX_LINES:
                    lines = lines[-MAX_LINES:]
                PULSE_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        pass
    return line


def recent(limit: int = 20) -> list[str]:
    """Return the newest pulse lines, oldest first. Empty list if none yet."""
    try:
        with _THREAD_LOCK:
            with _lock():
                if not PULSE_FILE.exists():
                    return []
                lines = [ln for ln in PULSE_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        return []
    return lines[-max(0, limit):]
