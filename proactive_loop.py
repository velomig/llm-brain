#!/usr/bin/env python3
"""Run the Will layer on a timer.

Examples:
  ./.venv/bin/python proactive_loop.py --once
  ./.venv/bin/python proactive_loop.py --interval-hours 4

For launchd/cron, prefer --once on a schedule; for a terminal session, omit
--once and leave the loop running.
"""
from __future__ import annotations

import argparse
import json
import time

import config
from brain import pulse, will


def _run_once(model: str | None, mode: str | None, dry_run: bool) -> dict:
    result = will.review(model=model, mode=mode, dry_run=dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not dry_run:
        created = result.get("created", 0)
        status = f"initiated {created} new item(s)" if created else (result.get("reason") or "no action taken")
        pulse.record("will.py", "Reviewed goals/reminders/memories", status)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous curiosity/worry review loop.")
    parser.add_argument("--model", default=None, help="Use the selected model's memory collection.")
    parser.add_argument("--mode", default="helpful", choices=["quiet", "helpful", "active", "coach"])
    parser.add_argument("--dry-run", action="store_true", help="Preview without creating initiatives.")
    parser.add_argument("--once", action="store_true", help="Run one review and exit.")
    parser.add_argument("--interval-hours", type=float, default=config.WILL_INTERVAL_HOURS)
    args = parser.parse_args()

    while True:
        _run_once(args.model, args.mode, args.dry_run)
        if args.once:
            break
        time.sleep(max(0.1, args.interval_hours) * 3600)


if __name__ == "__main__":
    main()
