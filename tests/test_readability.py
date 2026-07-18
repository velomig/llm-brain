from __future__ import annotations

from brain import pipeline, readability


def test_repairs_collapsed_headings_labels_and_bullets():
    dense = (
        "### Phase 2: Logic Audit — **Status: IN PROGRESS** This file delegates execution. "
        "#### **Audit Findings:** * **Structural Integrity:** The imports are stable. "
        "* **Registry Reliance:** Labels come from the registry."
    )

    result = readability.format_reply(dense)

    assert "###" not in result
    assert "**" not in result
    assert "Phase 2: Logic Audit" in result
    assert "\n\nAudit Findings:" in result
    assert "\n\n- Structural Integrity:" in result
    assert "\n\n- Registry Reliance:" in result


def test_preserves_fenced_code_exactly():
    code = "```markdown\n### Keep this\n**literal**\n```"

    result = readability.format_reply(f"### Example {code} After code.")

    assert code in result
    assert result == f"Example\n\n{code}\n\nAfter code."


def test_breaks_long_prose_at_sentence_boundaries():
    sentence = "This sentence contains enough detail to make the answer easier to test and understand."
    result = readability.format_reply(" ".join([sentence] * 5))

    assert "\n\n" in result
    assert result.replace("\n\n", " ") == " ".join([sentence] * 5)


def test_repairs_plain_text_audit_report_structure_from_regression():
    dense = (
        "Phase 2: Logic & Dependency Audit — Status: COMPLETED (Gatekeeper Layer) "
        "I am pleased to report that the authorization mechanism is structurally sound. "
        "Audit Findings (The Gatekeeper Layer): 1. Structural Integrity: High – The registry "
        "uses dual verification. It validates labels at runtime. 2. Resistance to Mislabeling: "
        "High – Metadata conflicts raise an error. 3. Verification of Grounding Mandate: The "
        "arbiter requires receipts. Verdict on Gatekeeper Layer: PASS / SECURE. The architecture "
        "fails closed. --- Next Steps: Phase 2 (Continued) — The Engine & Regression Testing "
        "The gatekeeper is secure. Current Plan: 1. Regression Testing: Execute the suite. "
        "2. Post-Test Analysis: Identify any failed path."
    )

    result = readability.format_reply(dense)

    assert result.startswith(
        "Phase 2: Logic & Dependency Audit — Status: COMPLETED (Gatekeeper Layer)\n\n"
    )
    assert "\n\nAudit Findings (The Gatekeeper Layer):\n\n1. Structural Integrity:" in result
    assert "\n\n2. Resistance to Mislabeling:" in result
    assert "\n\n3. Verification of Grounding Mandate:" in result
    assert "\n\nVerdict on Gatekeeper Layer: PASS / SECURE.\n\n" in result
    assert "\n\nNext Steps:\n\nPhase 2 (Continued) — The Engine & Regression Testing\n\n" in result
    assert "\n\nCurrent Plan:\n\n1. Regression Testing:" in result
    assert "\n\n2. Post-Test Analysis:" in result


def test_concise_plain_reply_is_unchanged():
    reply = "The service is running normally."

    assert readability.format_reply(reply) == reply


def test_system_prompt_has_persistent_readability_contract():
    system = pipeline.build_system([])

    assert "Readable Output Contract" in system
    assert "never collapse Markdown headings" in system
