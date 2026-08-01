#!/usr/bin/env python3
"""Run the Neural Map sleep cycle.

Examples:
  ./.venv/bin/python sleep_cycle.py --dry-run
  ./.venv/bin/python sleep_cycle.py
"""
from __future__ import annotations

import argparse
import json

from brain import portable, pulse, sleep


def main() -> None:
    parser = argparse.ArgumentParser(description="Cold-archive weak episodes and synthesize speculative dreams.")
    parser.add_argument("--model", default=None, help="Use the selected model's memory collection.")
    parser.add_argument("--dry-run", action="store_true", help="Preview candidates without changing memory.")
    parser.add_argument("--older-than-days", type=float, default=None, help="Archive episodes older than this many days.")
    parser.add_argument("--max-archive", type=int, default=None, help="Maximum weak episodes to archive.")
    parser.add_argument("--max-dreams", type=int, default=None, help="Maximum speculative dreams to generate.")
    args = parser.parse_args()

    backup = None
    if not args.dry_run:
        preview = sleep.run(
            model=args.model,
            dry_run=True,
            older_than_days=args.older_than_days,
            max_archive=args.max_archive,
            max_dreams=args.max_dreams,
        )
        if preview.get("archivable", 0) > 0 or preview.get("dreamable", 0) > 0:
            backup = str(portable.save_backup("before-sleep-cycle", model=args.model))

    result = sleep.run(
        model=args.model,
        dry_run=args.dry_run,
        older_than_days=args.older_than_days,
        max_archive=args.max_archive,
        max_dreams=args.max_dreams,
    )
    result["backup"] = backup
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.dry_run:
        pulse.record(
            "sleep_cycle.py",
            "Consolidated memory (archive + dreams)",
            f"archived {result.get('archived', 0)}, dreamed {result.get('dreamed', 0)}",
        )


if __name__ == "__main__":
    main()
