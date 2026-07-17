"""Add a structured memory from the command line.

Examples:
    python ingest.py "I prefer my coffee black with a pinch of salt." \
        --category preference --importance 5
    python ingest.py "Project XJ-9 ships in Q3." --category project
"""
from __future__ import annotations

import argparse

from brain import store
from brain.embeddings import EmbeddingUnavailable, embed_document, text_only_embedding
from brain.memory import MemoryUnit


def main() -> None:
    ap = argparse.ArgumentParser(description="Add a memory to the Neural Map.")
    ap.add_argument("text", help="the memory text")
    ap.add_argument("--category", default="note")
    ap.add_argument("--importance", type=int, default=3, choices=range(1, 6))
    args = ap.parse_args()

    unit = MemoryUnit(
        text=args.text,
        category=args.category,
        importance=args.importance,
        extra={"source_type": "user_entered"},
    )
    try:
        embedding = embed_document(unit.text)
    except EmbeddingUnavailable:
        store.add(unit, text_only_embedding(), text_only=True)
    else:
        store.add(unit, embedding)
    print(f"saved {unit.id[:8]}  category={unit.category}  importance={unit.importance}")


if __name__ == "__main__":
    main()
