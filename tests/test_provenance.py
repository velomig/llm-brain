"""Memory provenance is citable: the model-facing system prompt carries each
memory's category, a human-friendly source, and date, and the preamble tells the
model to cite them (only) when asked — so it can say "based on your note from
June 12 and the uploaded PDF" without inventing or leaking storage keys."""
from __future__ import annotations

import json

from brain.pipeline import _provenance_source, build_system


# --- source labels read well in a citation -----------------------------------

def test_upload_source_reduces_to_filename():
    assert _provenance_source("upload:deadbeef0123:lighthouse.pdf") == "lighthouse.pdf"


def test_path_source_reduces_to_basename():
    assert _provenance_source("/Users/alex/docs/retirement-plan.md") == "retirement-plan.md"


def test_plain_source_and_none_pass_through():
    assert _provenance_source("portfolio tickers resolved from memory") == "portfolio tickers resolved from memory"
    assert _provenance_source(None) is None
    assert _provenance_source("") is None


# --- build_system exposes citable provenance, not raw storage keys ------------

def _memories_json(system: str) -> list[dict]:
    block = system.split("Retrieved Memories JSON:", 1)[1].split("Proactive Reminders JSON:", 1)[0].strip()
    return json.loads(block)


def test_build_system_carries_clean_provenance_fields():
    mems = [
        {"id": "1", "text": "Prefers Ollama", "category": "fact",
         "source": "upload:abc123:notes.pdf", "date": "2026-06-12T09:30:00+00:00", "pinned": True},
    ]
    rows = _memories_json(build_system(mems))
    assert rows[0]["id"] == "1"
    assert rows[0]["category"] == "fact"
    assert rows[0]["source"] == "notes.pdf"          # not the raw upload:hash: key
    assert rows[0]["date"] == "2026-06-12"           # day precision, citable
    assert rows[0]["pinned"] is True


def test_build_system_does_not_leak_raw_upload_key():
    mems = [{"id": "1", "text": "x", "category": "note", "source": "upload:abc123:secret.pdf", "date": "2026-06-12"}]
    system = build_system(mems)
    assert "upload:abc123" not in system
    assert "secret.pdf" in system


def test_preamble_instructs_cite_when_asked_and_not_to_invent():
    system = build_system([{"id": "1", "text": "x", "category": "note", "source": None, "date": "2026-06-12"}])
    # Default stays quiet about memory...
    assert "Do not mention that you are reading from memory unless asked" in system
    # ...but provenance citation is enabled on request, with an anti-hallucination guard.
    assert "cite" in system.lower()
    assert "never invent a source" in system.lower()
