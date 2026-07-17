"""Schema migration / repair for the memory store.

As metadata evolves (schema_version, due, source, access_count, ...), memories
written by older code lack the newer fields. This stamps the current
schema_version on those records and backfills the defaults the ranking code
assumes (access_count, importance, category), so old memories carry the same
explicit metadata as new ones.

Metadata-only: it never re-embeds, never deletes, and is idempotent — records
already at the current schema are skipped, so it's safe to run repeatedly (e.g.
after pulling a new version). It walks BOTH the vector and text-only stores.

Usage:
    ./.venv/bin/python migrate.py              # apply the migration
    ./.venv/bin/python migrate.py --dry-run    # report what would change, write nothing
    ./.venv/bin/python migrate.py --model nomic-embed-text:latest   # specific embed model
"""
from __future__ import annotations

import argparse

from brain import graph, store
from brain.memory import SCHEMA_VERSION


def main() -> None:
    ap = argparse.ArgumentParser(description="Migrate/repair memory metadata to the current schema.")
    ap.add_argument("--dry-run", action="store_true", help="report changes without writing")
    ap.add_argument(
        "--model",
        default=None,
        help="embed model whose collections to migrate (default: the configured EMBED_MODEL)",
    )
    args = ap.parse_args()

    result = store.repair_schema(model=args.model, dry_run=args.dry_run)
    verb = "would migrate" if args.dry_run else "migrated"
    print(f"schema v{SCHEMA_VERSION}: scanned {result['scanned']}, {verb} {result['migrated']}")
    for mode, n in result["by_mode"].items():
        print(f"  {mode:9} {n}")

    # Graph: stamp the active store onto edges that predate model-scoping, so
    # associative recall resolves them under the current embedding model.
    collection = store.collection_name(args.model)
    edges = graph.backfill_collection(collection, dry_run=args.dry_run)
    gverb = "would tag" if args.dry_run else "tagged"
    print(f"graph: {gverb} {edges} legacy edge(s) with collection {collection!r}")

    if args.dry_run and (result["migrated"] or edges):
        print("(dry run — re-run without --dry-run to apply)")


if __name__ == "__main__":
    main()
