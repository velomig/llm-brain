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


def test_concise_plain_reply_is_unchanged():
    reply = "The service is running normally."

    assert readability.format_reply(reply) == reply


def test_system_prompt_has_persistent_readability_contract():
    system = pipeline.build_system([])

    assert "Readable Output Contract" in system
    assert "never collapse Markdown headings" in system
