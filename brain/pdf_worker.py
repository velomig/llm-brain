"""Isolated PDF worker. Invoked as a script; never imports application state."""
from __future__ import annotations

import json
import math
import re
import resource
import sys
from importlib.metadata import version
from pathlib import Path


def main() -> None:
    source, output = Path(sys.argv[1]), Path(sys.argv[2])
    seconds, memory_mb, page_cap, text_cap = map(int, sys.argv[3:])
    memory = memory_mb * 1024 * 1024
    for limit in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
        try:
            resource.setrlimit(limit, (memory, memory))
        except (ValueError, OSError):
            # macOS may reject address-space limits; the parent monitors RSS.
            pass
    resource.setrlimit(resource.RLIMIT_CPU, (math.ceil(seconds), math.ceil(seconds)))
    resource.setrlimit(resource.RLIMIT_FSIZE, (text_cap * 4 + 65536,) * 2)
    try:
        if tuple(int(n) for n in version("pypdf").split(".")[:3]) < (6, 15, 0):
            raise ValueError("PDF ingestion requires pypdf >= 6.15.0; install requirements.txt")
        from pypdf import PdfReader

        reader = PdfReader(source)
        if len(reader.pages) > page_cap:
            raise ValueError("PDF exceeds the page limit")
        pages, length = [], 0
        for i, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").strip()
            if not text or re.search(r"this page (?:is )?intentionally left blank", text, re.I):
                continue
            text = f"Page {i}\n{text}"
            length += len(text) + (2 if pages else 0)
            if length > text_cap:
                raise ValueError("PDF extraction exceeds the text limit")
            pages.append(text)
        result = {"text": "\n\n".join(pages)}
    except Exception as exc:
        result = {"error": f"PDF extraction failed: {type(exc).__name__}: {str(exc)[:500]}"}
    output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
