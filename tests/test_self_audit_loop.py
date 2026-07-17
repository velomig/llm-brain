"""The Self-Auditing Loop is injected ONLY into the System 2 critique pass.

It used to ride in every system prompt, costing tokens and instruction-following
attention on every turn while only the critique actually used it. These tests
pin the subtraction: absent from build_system, present in the critique prompt,
and cleanly omitted when disabled or missing."""
from __future__ import annotations

import config
from brain import observability, pipeline


def _configure(monkeypatch, tmp_path, text="Check nursing and finance assumptions.", enabled=True):
    audit = tmp_path / "self-audit-loop.md"
    audit.write_text(f"# Audit\n\n{text}", encoding="utf-8")
    monkeypatch.setattr(config, "SELF_AUDIT_LOOP_ENABLED", enabled)
    monkeypatch.setattr(config, "SELF_AUDIT_LOOP_PATH", audit)
    monkeypatch.setattr(config, "SELF_AUDIT_LOOP_MAX_CHARS", 12000)


def _run_system2(monkeypatch):
    replies = iter(["critique", "final"])
    prompts = []

    def fake_chat(system, user, **kwargs):
        prompts.append(user)
        return next(replies)

    monkeypatch.setattr(pipeline.llm, "chat", fake_chat)
    out = pipeline._slow_reasoning_answer(
        "make a study plan", [], {"reply": "draft"}, [], [], [], None,
    )
    return out, prompts


def test_self_audit_loop_stays_out_of_the_system_prompt(tmp_path, monkeypatch):
    _configure(monkeypatch, tmp_path)

    system = pipeline.build_system([])

    assert "Self-Auditing Loop" not in system
    assert "Check nursing and finance assumptions." not in system


def test_system2_critique_receives_the_checklist(tmp_path, monkeypatch):
    _configure(monkeypatch, tmp_path)

    out, prompts = _run_system2(monkeypatch)

    assert out["reply"] == "final"
    assert "Check nursing and finance assumptions." in prompts[0]   # critique
    assert "never reveal it" in prompts[0]
    assert "Check nursing and finance assumptions." not in prompts[1]  # rewrite


def test_disabled_self_audit_loop_omits_the_section(tmp_path, monkeypatch):
    _configure(monkeypatch, tmp_path, text="Hidden audit", enabled=False)

    out, prompts = _run_system2(monkeypatch)

    assert out["reply"] == "final"
    assert "Hidden audit" not in prompts[0]
    assert "Self-Auditing Loop" not in prompts[0]


def test_build_system_records_prompt_budget_per_block(monkeypatch):
    trace = observability.start("chat_turn")

    system = pipeline.build_system([{"text": "a memory", "category": "note"}], trace=trace)

    budget = [e for e in trace["events"] if e["layer"] == "prompt_budget"]
    assert len(budget) == 1
    fields = budget[0]["fields"]
    assert fields["total_chars"] == len(system)
    for key in ("preamble_chars", "source_of_truth_chars", "cognitive_state_chars", "memories_chars"):
        assert fields[key] > 0
