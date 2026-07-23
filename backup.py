"""Back up, export, and restore the Neural Map — your memories in a portable shape.

ChromaDB and the graph SQLite file are internal storage. This dumps the actual
content (text + metadata + entity edges) to JSON or Markdown, and restores it from
JSON. Import re-embeds with the active model, so a backup moves cleanly between
machines, Chroma versions, and even embedding models.

Usage:
    ./.venv/bin/python backup.py export                       # JSON to stdout
    ./.venv/bin/python backup.py export -o memories.json      # JSON to a file
    ./.venv/bin/python backup.py export --format md -o brain.md
    ./.venv/bin/python backup.py import -i memories.json      # merge (overwrite by id)
    ./.venv/bin/python backup.py import -i memories.json --replace   # full restore
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from brain import portable


def _write_owner_only(path: Path, text: str) -> None:
    """Write the export 0600: it is the entire memory archive, so it must not
    inherit a permissive umask (same pattern as portable.save_backup)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(text)
    finally:
        if fd >= 0:
            os.close(fd)


def _export(args: argparse.Namespace) -> None:
    data = portable.export_data(model=args.model)
    text = portable.to_markdown(data) if args.format == "md" else json.dumps(data, ensure_ascii=False, indent=2)
    if args.out:
        _write_owner_only(Path(args.out), text)
        print(f"exported {data['neural_map_export']['count']} memories "
              f"({len(data['neural_map_export']['graph'])} edges) -> {args.out}")
    else:
        sys.stdout.write(text + ("\n" if not text.endswith("\n") else ""))


def _import(args: argparse.Namespace) -> None:
    data = json.loads(Path(args.infile).read_text(encoding="utf-8"))
    backup_path = None
    if args.replace:
        backup_path = portable.save_backup("before-replace-import", model=args.model)
    result = portable.import_data(data, model=args.model, replace=args.replace)
    how = "restored (replaced)" if args.replace else "imported (merged)"
    print(f"{how} {result['memories']} memories, {result['edges']} edges "
          f"into the {result['mode']} store")
    if backup_path:
        print(f"pre-replace backup: {backup_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export/restore Neural Map memories in a portable shape.")
    ap.add_argument("--model", default=None, help="embedding model whose store to use (default: configured)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("export", help="write memories + graph to JSON or Markdown")
    pe.add_argument("-o", "--out", default=None, help="output file (default: stdout)")
    pe.add_argument("--format", choices=["json", "md"], default="json", help="export format (default: json)")
    pe.set_defaults(func=_export)

    pi = sub.add_parser("import", help="restore memories + graph from a JSON export")
    pi.add_argument("-i", "--in", dest="infile", required=True, help="JSON export file to restore")
    pi.add_argument("--replace", action="store_true", help="wipe existing memories first (authoritative restore)")
    pi.set_defaults(func=_import)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
