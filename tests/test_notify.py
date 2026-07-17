"""Local notification bridge (the assistant's outbound 'voice')."""
from __future__ import annotations

import json

import config
from brain import notify, will


def test_escape_applescript_neutralizes_quotes_and_backslashes():
    assert notify._escape_applescript('say "hi"\\end') == 'say \\"hi\\"\\\\end'


def test_notify_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(config, "NOTIFY_ENABLED", False)
    calls = []
    monkeypatch.setattr(notify.subprocess, "run", lambda *a, **k: calls.append(a))
    out = notify.notify(title="T", message="M")
    assert out["delivered"] is False
    assert calls == []


def test_notify_empty_message_is_noop(monkeypatch):
    monkeypatch.setattr(config, "NOTIFY_ENABLED", True)
    monkeypatch.setattr(notify.sys, "platform", "darwin")
    calls = []
    monkeypatch.setattr(notify.subprocess, "run", lambda *a, **k: calls.append(a))
    out = notify.notify(title="T", message="   ")
    assert out["delivered"] is False
    assert calls == []


def test_notify_non_darwin_is_noop(monkeypatch):
    monkeypatch.setattr(config, "NOTIFY_ENABLED", True)
    monkeypatch.setattr(notify.sys, "platform", "linux")
    calls = []
    monkeypatch.setattr(notify.subprocess, "run", lambda *a, **k: calls.append(a))
    out = notify.notify(title="T", message="M")
    assert out["delivered"] is False
    assert calls == []


def test_notify_delivery_never_raises_on_subprocess_error(monkeypatch):
    monkeypatch.setattr(config, "NOTIFY_ENABLED", True)
    monkeypatch.setattr(notify.sys, "platform", "darwin")
    monkeypatch.setattr(notify.shutil, "which", lambda name: "/usr/bin/osascript" if name == "osascript" else None)

    def boom(*a, **k):
        raise OSError("no gui session")

    monkeypatch.setattr(notify.subprocess, "run", boom)
    out = notify.notify(title="T", message="M")
    assert out["delivered"] is False


def test_will_maybe_notify_gated_by_priority(monkeypatch):
    monkeypatch.setattr(config, "NOTIFY_MIN_PRIORITY", 4)
    fired = []
    monkeypatch.setattr(will.notify, "notify", lambda **k: fired.append(k) or {"delivered": True})

    will._maybe_notify("low", {"priority": 3, "kind": "curiosity"})
    assert fired == []

    will._maybe_notify("high", {"priority": 5, "kind": "worry"})
    assert len(fired) == 1
    assert fired[0]["message"] == "high"
    assert fired[0]["subtitle"] == "Worry"


def test_will_maybe_notify_swallows_notifier_failure(monkeypatch):
    monkeypatch.setattr(config, "NOTIFY_MIN_PRIORITY", 1)

    def boom(**k):
        raise RuntimeError("notifier exploded")

    monkeypatch.setattr(will.notify, "notify", boom)
    # Must not raise — review() correctness cannot depend on a notification.
    will._maybe_notify("anything", {"priority": 5, "kind": "worry"})
