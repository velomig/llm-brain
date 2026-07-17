#!/usr/bin/env python3
"""Drain the Neural Map background task queue.

Read-only tools run automatically; any write/send/delete is parked in the
pending-approval queue for the user to review. Schedule this like the other
periodic jobs, e.g. in crontab during awake hours:

  */30 7-22 * * * cd '.../LLM Brain' && '.../.venv/bin/python' task_runner.py --once >> memory_store/task_runner.log 2>&1

Or enqueue and run one by hand:

  ./.venv/bin/python task_runner.py --add "Summarize this week's finance notes"
  ./.venv/bin/python task_runner.py --once
"""
from __future__ import annotations

import argparse
import json

from brain import tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run queued background tasks (read-only auto, writes parked for approval).")
    parser.add_argument("--add", metavar="PROMPT", default=None, help="Enqueue a task and exit.")
    parser.add_argument("--once", action="store_true", help="Drain the queue once and exit.")
    parser.add_argument("--list", action="store_true", help="List tasks and exit.")
    parser.add_argument("--model", default=None, help="Model whose memory collection to use.")
    parser.add_argument("--limit", type=int, default=25, help="Max tasks to run this pass.")
    args = parser.parse_args()

    if args.add:
        task = tasks.enqueue(args.add)
        print(json.dumps({"enqueued": task["id"], "prompt": task["prompt"]}, ensure_ascii=False))
        return
    if args.list:
        print(json.dumps(tasks.list_tasks(), ensure_ascii=False, indent=2))
        return

    finished = tasks.run_pending(model=args.model, limit=args.limit)
    print(json.dumps(
        {
            "ran": len(finished),
            "tasks": [
                {
                    "id": t["id"],
                    "status": t["status"],
                    "tools_used": t.get("tools_used"),
                    "pending_action_ids": t.get("pending_action_ids"),
                }
                for t in finished
            ],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
