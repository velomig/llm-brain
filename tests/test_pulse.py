from __future__ import annotations

from brain import pipeline, pulse


def _clear_pulse():
    if pulse.PULSE_FILE.exists():
        pulse.PULSE_FILE.unlink()


def test_record_appends_pipe_delimited_line():
    _clear_pulse()
    line = pulse.record("will.py", "Reviewed goals/reminders", "initiated 2 new item(s)")

    assert line.count("|") == 3
    assert "will.py" in line and "initiated 2 new item(s)" in line
    assert pulse.recent() == [line]


def test_record_sanitizes_separators_and_newlines():
    _clear_pulse()
    pulse.record("task|runner", "did\nthings", "ok | fine")

    (line,) = pulse.recent()
    # Fields may not smuggle extra separators or split the one-line record.
    assert line.count("|") == 3
    assert "\n" not in line
    assert "task/runner" in line


def test_record_trims_to_max_lines():
    _clear_pulse()
    for i in range(pulse.MAX_LINES + 25):
        pulse.record("job.py", f"run {i}", "ok")

    lines = pulse.PULSE_FILE.read_text(encoding="utf-8").splitlines()
    assert len(lines) == pulse.MAX_LINES
    assert f"run {pulse.MAX_LINES + 24}" in lines[-1]


def test_recent_returns_newest_last_with_limit():
    _clear_pulse()
    for i in range(5):
        pulse.record("job.py", f"run {i}", "ok")

    tail = pulse.recent(limit=2)
    assert len(tail) == 2
    assert "run 3" in tail[0] and "run 4" in tail[1]


def test_build_system_includes_system_pulse():
    _clear_pulse()
    pulse.record("will.py", "Reviewed goals/reminders", "initiated 1 new item(s)")

    system = pipeline.build_system([])

    assert "System Pulse" in system
    assert "Reviewed goals/reminders" in system


def test_build_system_pulse_placeholder_when_empty():
    _clear_pulse()

    system = pipeline.build_system([])

    assert "(no background runs recorded yet)" in system
