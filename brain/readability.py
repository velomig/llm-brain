"""Deterministic final-pass formatting for readable assistant replies.

Models occasionally collapse otherwise valid Markdown into one long line.  The
web UI intentionally renders replies as plain text, so repair the useful
structure here instead of trusting a renderer (or another model call).
"""
from __future__ import annotations

import re


_FENCED_BLOCK_RE = re.compile(r"(```[\s\S]*?(?:```|$))")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_HEADING_SENTINEL = "\x02NEURAL_MAP_HEADING\x02"
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘(]*[A-Z0-9])")


def _protect_inline_code(text: str) -> tuple[str, list[str]]:
    protected: list[str] = []

    def replace(match: re.Match[str]) -> str:
        index = len(protected)
        protected.append(match.group(0))
        return f"\x00NEURAL_MAP_CODE_{index}\x00"

    return _INLINE_CODE_RE.sub(replace, text), protected


def _restore_inline_code(text: str, protected: list[str]) -> str:
    for index, value in enumerate(protected):
        text = text.replace(f"\x00NEURAL_MAP_CODE_{index}\x00", value)
    return text


def _paragraphs(line: str, target: int = 220) -> list[str]:
    """Split a wall of prose at sentence boundaries without rewriting it."""
    if len(line) <= target or line.lstrip().startswith(("- ", "• ")) or " | " in line:
        return [line]
    sentences = _SENTENCE_BREAK_RE.split(line)
    if len(sentences) < 2:
        return [line]
    groups: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > target:
            groups.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups


def _format_prose(text: str) -> str:
    text, inline_code = _protect_inline_code(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Recover headings and list items that a model emitted inline.
    text = re.sub(r"(?<!\n)[ \t]+(?=#{1,6}[ \t]+\S)", "\n\n", text)
    text = re.sub(
        r"(?m)^[ \t]*#{1,6}[ \t]+",
        _HEADING_SENTINEL,
        text,
    )
    text = re.sub(r"[ \t]+---[ \t]+", "\n\n", text)
    text = re.sub(
        r"(?<=[.!?:;])[ \t]+(?=(?:[-+*]|•)[ \t]+(?:\*\*)?\S)",
        "\n\n",
        text,
    )
    text = re.sub(
        r"(?<=\*\*)[ \t]+(?=(?:[-+*]|•)[ \t]+(?:\*\*)?\S)",
        "\n\n",
        text,
    )
    text = re.sub(
        r"(?<=[.!?:;])[ \t]+(?=\d{1,2}[.)][ \t]+\S)",
        "\n\n",
        text,
    )

    # A status attached to a collapsed heading should not run into its prose.
    text = re.sub(
        r"(\*\*Status:[^*\n]{1,80}\*\*)[ \t]+(?=(?:\*\*)?[A-Z0-9])",
        r"\1\n\n",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(\*\*(?:IN PROGRESS|COMPLETE|COMPLETED|FAILED|BLOCKED)\*\*)[ \t]+(?=[A-Z0-9])",
        r"\1\n\n",
        text,
        flags=re.IGNORECASE,
    )

    # The client is a safe plain-text surface, so remove visual Markdown noise.
    text = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)
    text = re.sub(r"(?m)^[ \t]*(?:[-+*]|•)[ \t]+", "- ", text)

    output: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.rstrip()
        is_heading = line.startswith(_HEADING_SENTINEL)
        if is_heading:
            line = line[len(_HEADING_SENTINEL):].strip()
        if not line:
            if output and output[-1] != "":
                output.append("")
            continue

        pieces = _paragraphs(line)
        if is_heading and output and output[-1] != "":
            output.append("")
        for index, piece in enumerate(pieces):
            if index and output and output[-1] != "":
                output.append("")
            output.append(piece)
        if is_heading and output[-1] != "":
            output.append("")

    formatted = "\n".join(output).strip()
    formatted = re.sub(r"\n{3,}", "\n\n", formatted)
    return _restore_inline_code(formatted, inline_code)


def format_reply(reply: object) -> str:
    """Return a roomier plain-text reply while preserving fenced code exactly."""
    text = "" if reply is None else str(reply)
    if not text.strip():
        return text
    parts = _FENCED_BLOCK_RE.split(text)
    for index in range(0, len(parts), 2):
        parts[index] = _format_prose(parts[index])
    # Formatting prose trims its outer whitespace. Restore explicit visual
    # separation around code blocks so they cannot become glued to a paragraph.
    return "\n\n".join(part.strip("\n") for part in parts if part.strip()).strip()
