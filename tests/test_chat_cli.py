from __future__ import annotations

import builtins

import chat


def test_cli_capture_failure_warns_and_keeps_running(monkeypatch, capsys):
    prompts = iter(["hello", "/quit"])

    monkeypatch.setattr(chat.store, "count_all", lambda: 0)
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(prompts))
    monkeypatch.setattr(chat, "retrieve", lambda _text: [])
    monkeypatch.setattr(chat, "reinforce_recall", lambda _mems: None)
    monkeypatch.setattr(chat, "answer", lambda *_args, **_kwargs: {"reply": "hi"})

    def _boom(_user, _reply, **_kwargs):
        raise RuntimeError("store offline")

    monkeypatch.setattr(chat, "capture", _boom)

    chat.main()

    out = capsys.readouterr().out
    assert "brain" in out
    assert "hi" in out
    assert "warning: memory capture failed: store offline" in out
