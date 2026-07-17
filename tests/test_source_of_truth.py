from __future__ import annotations

import config
from brain import pipeline


def _configure_trusted_files(tmp_path, monkeypatch, source_text, agency_text, template_text):
    source = tmp_path / "source-of-truth.md"
    source.write_text(source_text, encoding="utf-8")
    agency = tmp_path / "agency_protocol.md"
    agency.write_text(agency_text, encoding="utf-8")
    template = tmp_path / "action-package-template.md"
    template.write_text(template_text, encoding="utf-8")
    monkeypatch.setattr(config, "SOURCE_OF_TRUTH_ENABLED", True)
    monkeypatch.setattr(config, "SOURCE_OF_TRUTH_PATH", source)
    monkeypatch.setattr(config, "SOURCE_OF_TRUTH_MAX_CHARS", 20000)
    monkeypatch.setattr(config, "AGENCY_PROTOCOL_PATH", agency)
    monkeypatch.setattr(config, "AGENCY_PROTOCOL_MAX_CHARS", 20000)
    monkeypatch.setattr(config, "ACTION_PACKAGE_TEMPLATE_PATH", template)
    monkeypatch.setattr(config, "ACTION_PACKAGE_TEMPLATE_MAX_CHARS", 8000)


def _configure_domain_files(tmp_path, monkeypatch):
    nursing = tmp_path / "nursing.md"
    nursing.write_text("# Nursing\n\nTRUSTED_NURSING_MARKER: prioritize patient safety.", encoding="utf-8")
    finance = tmp_path / "finance.md"
    finance.write_text("# Finance\n\nTRUSTED_FINANCE_MARKER: distinguish investing from speculation.", encoding="utf-8")
    monkeypatch.setattr(config, "NURSING_PATH", nursing)
    monkeypatch.setattr(config, "NURSING_MAX_CHARS", 30000)
    monkeypatch.setattr(config, "FINANCE_PATH", finance)
    monkeypatch.setattr(config, "FINANCE_MAX_CHARS", 30000)


def test_build_system_injects_configured_source_of_truth(tmp_path, monkeypatch):
    _configure_trusted_files(
        tmp_path,
        monkeypatch,
        "# Source\n\nDo not infer stale portfolio facts.",
        "# Agency\n\nPrepare action packages for high-stakes triggers.",
        "# Template\n\nApproval Gate",
    )

    system = pipeline.build_system([], tool_mode=True)

    assert "Source of Truth Markdown:" in system
    assert "Do not infer stale portfolio facts." in system
    assert "Agency Protocol Markdown" in system
    assert "Prepare action packages for high-stakes triggers." in system
    assert "Action Package Template Markdown" in system
    assert "Approval Gate" in system
    assert system.index("Source of Truth Markdown:") < system.index("Retrieved Memories JSON:")


def test_pinned_memory_overrides_are_before_source_of_truth(tmp_path, monkeypatch):
    _configure_trusted_files(
        tmp_path,
        monkeypatch,
        "# Source\n\nOlder portfolio snapshot: FNDX 1,234.567 shares.",
        "# Agency\n\nunused",
        "# Template\n\nunused",
    )

    system = pipeline.build_system(
        [
            {
                "id": "portfolio-note",
                "text": "I currently have 123.456 shares of FNDX, 45.678 shares of FNDY, 90.123 shares of FNDZ, 10 shares of ACME",
                "category": "note",
                "date": "2026-07-03T06:56:25+00:00",
                "pinned": True,
            }
        ]
    )

    assert system.index("User-Pinned Memory Overrides JSON:") < system.index("Source of Truth Markdown:")
    assert "123.456 shares of FNDX" in system
    assert "Older portfolio snapshot" in system


def test_agency_docs_stay_out_of_injection_mode_prompts(tmp_path, monkeypatch):
    # The Agency Protocol and Action Package Template govern tool actions; on
    # turns with no tools they were ~14k chars diluting every instruction.
    _configure_trusted_files(
        tmp_path,
        monkeypatch,
        "# Source\n\nDo not infer stale portfolio facts.",
        "# Agency\n\nPrepare action packages for high-stakes triggers.",
        "# Template\n\nApproval Gate",
    )

    system = pipeline.build_system([], tool_mode=False)

    assert "Do not infer stale portfolio facts." in system   # calibration stays
    assert "Agency Protocol Markdown" not in system
    assert "Action Package Template Markdown" not in system


def test_trusted_agency_blocks_stay_ordered_before_retrieved_memories(tmp_path, monkeypatch):
    _configure_trusted_files(
        tmp_path,
        monkeypatch,
        "# Source\n\nTrusted source marker.",
        "# Agency\n\nTrusted agency marker.",
        "# Template\n\nTrusted template marker.",
    )

    system = pipeline.build_system(
        [{"text": "Retrieved memory marker.", "meta": {"category": "fact"}}], tool_mode=True
    )

    assert system.index("Source of Truth Markdown:") < system.index("Trusted source marker.")
    assert system.index("Trusted source marker.") < system.index("Agency Protocol Markdown")
    assert system.index("Agency Protocol Markdown") < system.index("Trusted agency marker.")
    assert system.index("Trusted agency marker.") < system.index("Action Package Template Markdown")
    assert system.index("Action Package Template Markdown") < system.index("Trusted template marker.")
    assert system.index("Trusted template marker.") < system.index("Retrieved Memories JSON:")
    assert system.index("Retrieved Memories JSON:") < system.index("Retrieved memory marker.")


def test_agency_and_action_template_are_independently_bounded(tmp_path, monkeypatch):
    _configure_trusted_files(
        tmp_path,
        monkeypatch,
        "# Source\n\nFull source text stays visible.",
        "agency-abcdef",
        "template-abcdef",
    )
    monkeypatch.setattr(config, "AGENCY_PROTOCOL_MAX_CHARS", 8)
    monkeypatch.setattr(config, "ACTION_PACKAGE_TEMPLATE_MAX_CHARS", 11)

    system = pipeline.build_system([], tool_mode=True)

    assert "Full source text stays visible." in system
    assert "agency-a\n\n[Agency Protocol truncated by max character setting]" in system
    assert "agency-abcdef" not in system
    assert "template-ab\n\n[Action Package Template truncated by max character setting]" in system
    assert "template-abcdef" not in system


def test_source_of_truth_can_be_disabled(tmp_path, monkeypatch):
    source = tmp_path / "source-of-truth.md"
    source.write_text("Hidden calibration", encoding="utf-8")
    monkeypatch.setattr(config, "SOURCE_OF_TRUTH_ENABLED", False)
    monkeypatch.setattr(config, "SOURCE_OF_TRUTH_PATH", source)

    system = pipeline.build_system([])

    assert "(source of truth disabled)" in system
    assert "Hidden calibration" not in system


def test_source_of_truth_is_bounded(tmp_path, monkeypatch):
    source = tmp_path / "source-of-truth.md"
    source.write_text("abcdef", encoding="utf-8")
    monkeypatch.setattr(config, "SOURCE_OF_TRUTH_ENABLED", True)
    monkeypatch.setattr(config, "SOURCE_OF_TRUTH_PATH", source)
    monkeypatch.setattr(config, "SOURCE_OF_TRUTH_MAX_CHARS", 3)

    system = pipeline.build_system([])

    assert "abc\n\n[Source of Truth truncated by max character setting]" in system
    assert "abcdef" not in system


def test_nursing_request_injects_only_trusted_nursing_calibration(tmp_path, monkeypatch):
    _configure_domain_files(tmp_path, monkeypatch)

    system = pipeline.build_system([], user_text="Quiz me on nursing care for a patient with sepsis.")

    assert "Domain Calibration below is also trusted user-authored calibration" in system
    assert "## Nursing Calibration" in system
    assert "TRUSTED_NURSING_MARKER" in system
    assert "TRUSTED_FINANCE_MARKER" not in system
    assert system.index("Domain Calibration Markdown:") < system.index("Persistent Cognitive State JSON:")
    assert system.index("TRUSTED_NURSING_MARKER") < system.index("Retrieved Memories JSON:")


def test_finance_request_injects_only_trusted_finance_calibration(tmp_path, monkeypatch):
    _configure_domain_files(tmp_path, monkeypatch)

    system = pipeline.build_system([], user_text="Review my retirement portfolio and FNDX allocation.")

    assert "## Finance Calibration" in system
    assert "TRUSTED_FINANCE_MARKER" in system
    assert "TRUSTED_NURSING_MARKER" not in system


def test_cross_domain_request_injects_both_trusted_calibrations(tmp_path, monkeypatch):
    _configure_domain_files(tmp_path, monkeypatch)

    system = pipeline.build_system(
        [],
        user_text="How should nursing school affect my retirement portfolio contributions?",
    )

    assert "TRUSTED_NURSING_MARKER" in system
    assert "TRUSTED_FINANCE_MARKER" in system
    assert system.index("## Nursing Calibration") < system.index("## Finance Calibration")


def test_unrelated_request_loads_no_domain_calibration(tmp_path, monkeypatch):
    _configure_domain_files(tmp_path, monkeypatch)

    system = pipeline.build_system([], user_text="Help me refactor this Python function.")

    assert "(no domain-specific calibration selected)" in system
    assert "TRUSTED_NURSING_MARKER" not in system
    assert "TRUSTED_FINANCE_MARKER" not in system


def test_answer_path_passes_request_to_domain_selector(tmp_path, monkeypatch):
    _configure_domain_files(tmp_path, monkeypatch)
    captured = {}
    monkeypatch.setattr(config, "INTUITION_ENABLED", False)
    monkeypatch.setattr(pipeline.llm, "supports_tools", lambda _name: False)
    monkeypatch.setattr(pipeline, "market_context_for", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "web_context_for", lambda *a, **k: None)

    def fake_chat(system, user, **kwargs):
        captured["system"] = system
        return "ok"

    monkeypatch.setattr(pipeline.llm, "chat", fake_chat)

    result = pipeline.answer("Explain nursing delegation for an RN.", [])

    assert result["reply"] == "ok"
    assert "TRUSTED_NURSING_MARKER" in captured["system"]
    assert "TRUSTED_FINANCE_MARKER" not in captured["system"]
