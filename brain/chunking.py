"""Split markdown into retrieval-sized chunks.

Strategy: cut the document on headings (so each chunk stays within one section
and keeps its heading "breadcrumb"), then greedily pack each section's
paragraphs up to a max character budget. A single oversized paragraph is hard
split. YAML frontmatter is stripped.

Returns a list of (chunk_text, heading_path) tuples.
"""
from __future__ import annotations

import re

_FRONTMATTER = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def strip_frontmatter(text: str) -> str:
    return _FRONTMATTER.sub("", text, count=1)


def _split_sections(text: str) -> list[tuple[str, str]]:
    """(heading_path, body) pairs, where heading_path is e.g. 'Title > Section'."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []   # (level, title)
    path = ""
    buf: list[str] = []
    fence: str | None = None    # the marker (``` or ~~~) if inside a fenced code block

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((path, body))

    for line in text.splitlines():
        stripped = line.lstrip()
        marker = stripped[:3] if stripped.startswith(("```", "~~~")) else None
        if fence is None and marker:                    # entering a code block
            fence = marker
            buf.append(line)
            continue
        if fence is not None:                           # inside one: '#' is not a heading
            buf.append(line)
            if marker == fence:
                fence = None
            continue
        m = _HEADING.match(line)
        if m:
            flush()
            buf = []
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            path = " > ".join(t for _, t in stack)
        else:
            buf.append(line)
    flush()
    return sections


def _pack(body: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    cur = ""
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:                       # oversized single paragraph
            if cur:
                chunks.append(cur)
                cur = ""
            for i in range(0, len(para), max_chars):
                chunks.append(para[i : i + max_chars])
            continue
        if cur and len(cur) + len(para) + 2 > max_chars:
            chunks.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        chunks.append(cur)
    return chunks


def chunk_markdown(text: str, max_chars: int = 1200) -> list[tuple[str, str]]:
    text = strip_frontmatter(text)
    out: list[tuple[str, str]] = []
    for path, body in _split_sections(text):
        for chunk in _pack(body, max_chars):
            out.append((chunk, path))
    return out
