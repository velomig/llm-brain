# Agentic architecture and code audit — 2026-10-24

Scope: current working copy, including existing uncommitted September fixes. Focused review of background execution, LLM tool loops, approvals, recovery, and undo, with the full existing test suite run. This is not an exhaustive security certification or a live-model capability evaluation. No application source was changed.

## Assessment

The project has useful agent foundations: durable memory, a persistent task queue, bounded tool loops, centralized tool policy, reviewable approvals, provenance checks, and proactive candidate selection. It is currently strongest as a memory assistant that can perform short tool workflows and propose actions.

The next major improvement should be durable, goal-directed execution. The existing cognitive layers do not substitute for persisted steps, outcome checks, or continuation after interruption. In particular:

- `brain/will.py:769` reviews context and stores/notifies initiatives; it does not enqueue and execute a plan.
- `brain/tasks.py:333` starts a fresh tool conversation from the task prompt. It does not inject the chat retrieval context or persist the conversation for later continuation.
- `brain/tasks.py:162` marks a waiting task done when its proposed actions are approved. It never resumes the model using the actual action results. A workflow such as “create an event, then use its returned ID in a follow-up action” cannot reliably proceed across this boundary.
- `brain/llm.py:484` bounds the loop by iterations (default four); exhaustion produces a final answer but no structured indication that work remains.
- `brain/toolkit/registry.py:393` approval-gates all writes. There is no scoped standing authorization for a particular goal, directory, or class of reversible operation.

## Confirmed findings

### 1. P1 — Memory undo can erase a newer correction

Location: `brain/toolkit/governance.py:745–753`.

File/contact/calendar undo compares the current target with a post-action snapshot. Memory undo bypasses that check and unconditionally restores the pre-action records. Consequently, approving an update, making a later correction to the same memory, and undoing the older update silently discards the correction. Merge undo similarly deletes the merged record without checking for subsequent edits.

Reproduction: created an isolated memory with text `original`; approved an update to `first approved edit`; updated it to `newer user correction`; undid the first action. Undo reported success and the memory became `original`.

Fix: store a stable post-action version for each affected memory, and compare it under the same per-record mutation lock before restoring/deleting. Reject undo when meaningful content has changed. Exclude ordinary recall counters from this comparison.

### 2. P2 — Expired task workers can overwrite a replacement worker

Locations: `brain/tasks.py:110–122`, `255–266`.

A task lease is based on the initial claim time; there is no heartbeat or unique attempt token. A read of the task list can requeue a sufficiently long-running task even if its worker is alive. Another runner can claim it, and the original worker can still call `_finish`, which blindly patches by task ID. This can replace the newer attempt's status/results and disconnect its pending actions.

Reproduction: claim a task, age its claim timestamp, reclaim it, then finish using the old claim. The old worker's result is accepted while the replacement claim is active.

Fix: persist an attempt ID and heartbeat, require compare-and-set on attempt ID and running status for completion, and check ownership before each tool dispatch. Add idempotency keys to proposals so recovery cannot duplicate a logical action.

### 3. P2 — An exception after proposing an action loses the task/action relationship

Locations: `brain/tasks.py:325–344`, `356–360`.

Pending actions are persisted immediately, but their IDs remain only in `_execute`'s local list until the whole LLM loop returns. If the next model request fails, `run_one` marks the task failed without saving those IDs. The action remains approvable while its parent task shows no pending actions. Retrying the task can create another proposal for the same work.

Reproduction: a fake model loop dispatched a real `create_reminder` proposal and then raised an exception. The resulting task was failed with an empty action list, while the pending queue still contained the proposal.

Fix: persist task ID, attempt ID, and logical step ID on each proposal; durably associate it with the task when created. Recover linked actions after interruption before deciding whether to retry. Persist each tool result/checkpoint rather than only the final summary.

### 4. P2 — Task completion does not distinguish inability from success

Locations: `brain/tasks.py:318–344`, `361–369`; `brain/llm.py:503–507`.

Any normally returning tool loop without pending actions becomes `done`, with `error=None`. Tool error results do not become task failure, and the detailed tool trace returned by `chat_with_tools` is discarded. Empty responses, explicit inability, and iteration exhaustion are therefore indistinguishable from successful completion at the task-status level.

Reproduction: returned `I could not complete the task.` from the model loop. The task was persisted as `done` with no error.

Fix: return a structured execution outcome such as completed, blocked, failed, or budget_exhausted, with tool results and a stop reason. Add task-specific completion checks; do not infer failure or success solely from wording. An optional tool failure should not automatically fail a task that otherwise achieves its objective.

## Validation

- Full suite: **865 passed, 4 failed, 1 skipped** in 16.81 seconds within the sandbox.
- All four failures were in PDF tests because process-memory monitoring was unavailable in the sandbox.
- Re-ran the complete PDF test module outside the sandbox: **8 passed**. No PDF code defect was established by those failures.
- Four isolated, assertion-based reproductions confirmed findings 1–4. They used the existing fake embedding harness, a temporary database, and mocked model responses; no live email/calendar action was executed.
- Live Ollama reasoning, scheduler installation, browser E2E behavior, and real external-service integration were not validated.

## Recommended implementation sequence

1. **Make execution state reliable.** Fix the four findings above. Add regression tests that require safe undo, reject stale workers, preserve proposals after exceptions, and distinguish incomplete work.
2. **Add resumable runs.** Persist goal, success criteria, plan steps, observations, tool results, attempt ownership, budgets, and next step. Resume after approvals using actual execution results. A transactional SQLite run/event store would simplify atomic state transitions compared with multiple JSON files.
3. **Connect initiative to execution.** Let authorized initiatives create runs. Keep a stable goal-to-initiative-to-task relationship and return verified outcomes to the goal record. Preserve cooldowns and duplicate prevention.
4. **Introduce scoped standing permissions.** Allow configured reversible actions within a specific workspace and goal budget. Keep external sends, destructive operations, and trusted identity/policy changes explicitly controlled. Enforce these scopes in the dispatcher, not merely in prompts.
5. **Evaluate completed workflows.** Measure verified success, human interventions, recovery after restart, duplicate effects, elapsed time, and resource use on realistic multi-step tasks. Include approval/resume and tool-failure scenarios. Use these results to decide whether additional cognitive layers improve outcomes.

The architectural priority is a reliable loop of goal → next step → tool execution → observation → verification → continuation, with durable state at each boundary.
