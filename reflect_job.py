"""Standalone reflection job — run on a schedule (cron, launchd, or the
Claude Code `schedule` skill) so the Neural Map "sleeps" regularly: it distils
new insights, reminders, and optional initiative candidates from your stored facts.

Usage:
    .venv/bin/python reflect_job.py
    .venv/bin/python reflect_job.py --prune           # also forget faded episodes
    .venv/bin/python reflect_job.py --count-contact   # count printed initiatives
                                                      # against the daily budget

Example cron (every night at 3am):
    0 3 * * *  cd "/path/to/LLM Brain" && .venv/bin/python reflect_job.py --prune
"""
from __future__ import annotations

import sys

from brain import initiative, pulse, reflect, store


def main() -> None:
    if "--prune" in sys.argv:
        pruned = store.prune_episodes()
        print(f"pruned {pruned['pruned']} faded episodes")
    result = reflect.reflect()
    print(
        f"reflection: {result['insights']} insight(s), "
        f"{result['reminders']} reminder(s), {result.get('initiatives', 0)} initiative(s)"
    )
    pulse.record(
        "reflect_job.py",
        "Distilled insights/reminders from stored facts",
        f"{result['insights']} insight(s), {result['reminders']} reminder(s), "
        f"{result.get('initiatives', 0)} initiative(s)",
    )
    for r in reflect.active_reminders():
        flag = "OVERDUE" if r.get("overdue") else "upcoming"
        print(f"  [{flag}] {r.get('due')}  {r.get('text')}")
    # INITIATIVE LAYER: visible but non-intrusive scheduled-output hook. Disable
    # with NEURAL_MAP_INITIATIVE=0 to return this job to reminder-only output.
    scheduled_initiatives = initiative.active_initiatives(mode="helpful")
    for item in scheduled_initiatives:
        print(f"  [initiative/{item.get('kind')}] {item.get('text')}")
    # A cron log line isn't a user contact: counting it by default burned the
    # small daily interruption budget at 3am, suppressing UI initiatives all day.
    # Opt in with --count-contact when this job's output IS the delivery channel
    # (e.g. piped to a notification).
    if "--count-contact" in sys.argv:
        initiative.note_shown(scheduled_initiatives, mode="helpful")
    briefing = initiative.briefing(mode="helpful", respect_budget=True)
    printed_initiatives = {item.get("text") for item in scheduled_initiatives}
    if briefing.get("suggested_next_action") and briefing["suggested_next_action"] not in printed_initiatives:
        print(f"briefing next action: {briefing['suggested_next_action']}")


if __name__ == "__main__":
    main()
