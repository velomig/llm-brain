"""Ingest document files (or whole folders) into the Neural Map.

Each file is split into retrieval-sized chunks, embedded, and stored — tagged with
its source path, so re-running on an edited file cleanly replaces its chunks.

Images (.png/.jpg/.gif/.webp/.bmp/.tiff) are captioned + OCR'd by a local vision
model (config.VISION_MODEL) and the resulting text is stored like any document.

Examples:
    python ingest_md.py notes.md
    python ingest_md.py screenshot.png               # captioned + OCR'd via vision model
    python ingest_md.py ~/Documents/notes            # whole folder, recursive (text + images)
    python ingest_md.py ./docs --category reference --importance 4
    python ingest_md.py ~/Documents/notes --watch
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import config
from brain import sensory, store
from brain.documents import find_documents, ingest_file


def memory_count() -> int:
    return store.count_all()


def _source(path: Path) -> str:
    return str(path.expanduser().resolve())


def ingest_files(files: list[Path], args: argparse.Namespace) -> tuple[int, set[Path]]:
    total = 0
    succeeded: set[Path] = set()
    for f in files:
        try:
            n, cat = ingest_file(
                f,
                category=args.category,
                importance=args.importance,
                max_chars=args.max_chars,
                replace=not args.no_replace,
                vision_model=args.vision_model,
            )
        except Exception as exc:
            print(f"  {f.name}: skipped ({exc})")
            continue
        total += n
        succeeded.add(f)
        print(f"  {f.name}: {n} chunks  (category={cat})")
    return total, succeeded


def watch(args: argparse.Namespace) -> None:
    seen: dict[Path, float] = {}
    print(f"watching {', '.join(args.paths)} every {args.interval:g}s. Ctrl-C to stop.")
    while True:
        files = find_documents(args.paths, recursive=not args.no_recursive)
        current = {f: f.stat().st_mtime for f in files if f.exists()}

        changed = [f for f, mtime in current.items() if seen.get(f) != mtime]
        if changed:
            total, succeeded = ingest_files(changed, args)
            for f in succeeded:
                seen[f] = current[f]
            print(f"  updated {len(changed)} file(s), {total} chunks. Store now holds {memory_count()} memories.\n")

        removed = set(seen) - set(current)
        for f in sorted(removed):
            if args.delete_removed:
                store.delete_by_source(_source(f))
                store.delete_by_source(_source(f), text_only=True)
                sensory.delete_source(_source(f))
                print(f"  removed {f.name} from store")
            else:
                print(f"  noticed {f.name} is gone; stored chunks were left intact")
            seen.pop(f, None)

        for f in current:
            if f not in changed:
                seen[f] = current[f]
        time.sleep(args.interval)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest .md, .txt, .pdf, and image files as long-term memory.")
    ap.add_argument("paths", nargs="+", help="document/image files and/or directories")
    ap.add_argument("--category", default=None, help="tag (default: each file's name)")
    ap.add_argument("--importance", type=int, default=3, choices=range(1, 6))
    ap.add_argument("--max-chars", type=int, default=config.CHUNK_MAX_CHARS)
    ap.add_argument("--vision-model", default=None, help=f"vision model for images (default: {config.VISION_MODEL})")
    ap.add_argument("--no-recursive", action="store_true", help="don't descend into subfolders")
    ap.add_argument("--no-replace", action="store_true", help="keep existing chunks for a re-ingested file")
    ap.add_argument("--watch", action="store_true", help="keep running and re-ingest changed files")
    ap.add_argument("--delete-removed", action="store_true", help="in --watch mode, delete stored chunks for files that disappear")
    ap.add_argument("--interval", type=float, default=2.0, help="watch polling interval in seconds")
    args = ap.parse_args()

    if args.watch:
        watch(args)
        return

    files = find_documents(args.paths, recursive=not args.no_recursive)
    if not files:
        print("no supported documents found.")
        return

    total, _ = ingest_files(files, args)

    print(f"\ningested {total} chunks from {len(files)} file(s). Store now holds {memory_count()} memories.")


if __name__ == "__main__":
    main()
