# Agentic execution audit fixes — 2026-10-24

Implemented the four confirmed findings in AUDIT_REPORT_2026-10-24.md, plus continuation after approval. Existing uncommitted changes were preserved.

## Changes

- **Safe memory undo:** approval records a stable post-action snapshot for memory edits, merges, and created reminders/action packages. Undo checks all affected records and refuses to overwrite newer content or metadata. Recall counters do not invalidate undo. A reentrant, cross-process memory mutation lock covers comparison/restoration and store writes. Legacy actions without a post-action snapshot fail closed and require manual restoration.
- **Worker ownership:** every task claim gets a unique attempt ID and a renewable heartbeat. Completion and tool dispatch require the current running attempt. Expired workers cannot overwrite or dispatch under a replacement claim.
- **Durable task/action linkage:** pending proposals carry their parent task and attempt IDs at creation. Each tool observation is checkpointed. Recovery discovers proposals even if the process dies before saving their IDs in the task. Interrupted tasks with pending actions remain awaiting approval. Identical action proposals within a task reuse prior results instead of creating duplicates.
- **Explicit completion outcomes:** background model output must provide a structured status, summary, and completion evidence. Invalid/empty reports, all-tool failures, and iteration exhaustion cannot become successful completion. Task results preserve observations and a stop reason. Completion evidence remains model-reported; this is not an independent factual verifier.
- **Approval continuation:** the background tool loop pauses at the first pending action. After approval, the task is requeued and receives saved observations and actual action results. Existing tasks with legacy action-ID links are supported. Search budgets and provenance survive continuation.
- **Explicit retries:** blocked, failed, and budget-exhausted tasks can be retried by supplying their ID to POST /api/tasks/run. Scheduled drains do not automatically retry these states.

## Validation

- Full suite excluding the PDF module: **886 passed, 1 skipped**.
- After adding the final web-budget continuation regression: task and new regression modules **44 passed** (including all 26 new regression cases).
- Complete PDF module, outside the sandbox with Ollama pointed away from the live service: **8 passed**.
- Across these runs: **895 distinct passing cases, 1 skipped**.
- `git diff --check` passed.
- An earlier unrestricted full-suite attempt was stopped when it reached the live Ollama service. The completed broad run used the sandbox; only the isolated PDF module needed unrestricted process monitoring.

Tests cover stale-worker dispatch and completion, heartbeat renewal, interruption between proposal and checkpoint, approval/resume without duplicate actions, legacy task compatibility, explicit retries, malformed completion reports, tool errors, budget exhaustion, merge undo, recall-only changes, and concurrent memory editing during undo.

No external email/calendar action or live agent workflow was used as a validation scenario. Restart the application and scheduled workers to load the updated code. No migration is required.
