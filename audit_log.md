# Neural Map Audit Log

Purpose: Keep a transparent, human-readable record of major system changes, verification results, and architectural drift checks. Use this for development history; keep `todo.md` focused on future work.

## Audit Summary Template

Use this format for each meaningful project change:

```md
## YYYY-MM-DD - Short Title

Phase:
Scope:
Intent:

Files changed:
- `path`

Behavior changed:
- ...

Verification:
- Command: `...`
- Result: ...

Architectural drift check:
- Status: `none` | `accepted` | `needs follow-up`
- Notes: ...

Open follow-ups:
- ...
```

## Architectural Drift Definition

Architectural drift means a change that makes the project harder to reason about because it bypasses, duplicates, or contradicts established structure.

Drift signals:
- A new file is added without a clear home in the existing project map.
- Runtime behavior is implemented in docs, memory, or prompt text when it belongs in code.
- Prompt or calibration logic is duplicated across files instead of using one loader/path.
- A new setting is added without a config entry, documentation, or default.
- A new cognitive layer does not report bounded, graceful behavior.
- A new data source enters the prompt without clear trust/provenance framing.
- A test-only workaround becomes production behavior.
- A feature silently changes nursing, finance, or user-preference behavior without checking `source-of-truth.md`.
- A generated artifact is mixed with user-authored source material without labeling.

Accepted drift:
- A change may intentionally create a new pattern when the old structure cannot support the requirement.
- Accepted drift must be named in the audit summary, documented in README or a protocol file, and include at least one follow-up to consolidate the pattern if needed.

## 2026-07-05 - Phase 1 Source of Truth

Phase: 1 - Foundation
Scope: Calibration files and long-term user context.
Intent: Convert `aboutme.md` and `nursing.md` into a cleaner operational source of truth that reduces hallucinated intent.

Files changed:
- `source-of-truth.md`
- `$HOME/Documents/Obsidian Vault/nursing.md`

Behavior changed:
- Created a master calibration file separating stable facts, dated facts, preferences, decision rules, "do not infer" guardrails, and verification needs.
- Cleaned stale nursing-school language in the Obsidian nursing file.
- Added a pointer from `nursing.md` to the master Source of Truth.

Verification:
- Manual readback of both files.
- Confirmed outdated "starting nursing school in August" language was removed from `nursing.md`.

Architectural drift check:
- Status: `accepted`
- Notes: `source-of-truth.md` is a new root-level file. This is accepted because it is a durable, user-authored calibration artifact referenced by runtime config in Phase 2.

Open follow-ups:
- Confirm current nursing semester, course list, and near-term exam schedule.
- Confirm stale financial items such as specific holdings and contribution allocation.

## 2026-07-05 - Phase 2 Runtime Calibration And Self-Audit Loop

Phase: 2 - Engine
Scope: Prompt construction and structured reasoning.
Intent: Move calibration and self-audit behavior from passive notes into deterministic runtime context.

Files changed:
- `config.py`
- `brain/pipeline.py`
- `README.md`
- `source-of-truth.md`
- `self-audit-loop.md`
- `tests/test_source_of_truth.py`
- `tests/test_self_audit_loop.py`

Behavior changed:
- Injects `source-of-truth.md` into every chat prompt before retrieved memories.
- Injects `self-audit-loop.md` into every chat prompt before retrieved memories.
- Uses the Self-Auditing Loop explicitly during System 2 critique and rewrite.
- Adds environment switches for both trusted Markdown blocks.

Verification:
- Command: `.venv/bin/python -m pytest -q`
- Result: `292 passed`, with one existing FastAPI/Starlette deprecation warning.
- Direct prompt sanity check confirmed order: Source of Truth, Self-Auditing Loop, Retrieved Memories.

Architectural drift check:
- Status: `none`
- Notes: The new prompt blocks use config-backed paths, bounded loaders, documentation, and focused tests.

Open follow-ups:
- Decide whether self-audit outcomes should remain fully silent or optionally appear in diagnostics.

## 2026-07-05 - Phase 3 Audit Log And Drift Criteria

Phase: 3 - Infrastructure
Scope: Development traceability and architecture governance.
Intent: Create a formal record of progress and define what counts as architectural drift.

Files changed:
- `audit_log.md`
- `todo.md`
- `README.md`

Behavior changed:
- Adds a standard Audit Summary format.
- Defines architectural drift signals and accepted-drift requirements.
- Keeps development history separate from the future-work roadmap.

Verification:
- Manual readback.

Architectural drift check:
- Status: `none`
- Notes: `audit_log.md` is a governance/documentation artifact and is referenced from the roadmap and README project map.

Open follow-ups:
- Consider adding a lightweight automated drift check later, such as validating that new root-level files are listed in README or `audit_log.md`.

## 2026-07-05 - Audit Fix Pass

Phase: Post-audit hardening
Scope: Backup privacy, cancellation, file tools, SMTP, and auth-token exposure.
Intent: Fix the five issues found in the whole-code audit.

Files changed:
- `backup_job.sh`
- `server.py`
- `web/index.html`
- `brain/tools.py`
- `tests/test_chat_cancellation.py`
- `tests/test_action_tools.py`
- `tests/test_auth_token.py`
- `README.md`
- `todo.md`
- `audit_log.md`

Behavior changed:
- Backup jobs now use `umask 077`, private backup directories, and `600` JSON files.
- Existing `/Users/Shared/neuralmap-backups` permissions were locked down to owner-only access.
- Stop requests can register a bounded pending cancel so early UI clicks still abort generation when `/api/chat` starts.
- `list_files` now reports a per-entry error for bad children instead of failing the whole directory listing.
- SMTP STARTTLS uses `ssl.create_default_context()`.
- Query-string tokens are limited to one-time GET bootstrap and redirect to a clean URL.

Verification:
- Command: `sh -n backup_job.sh`
- Result: passed.
- Command: `.venv/bin/python -m pytest tests/test_chat_cancellation.py -q`
- Result: passed.
- Command: `.venv/bin/python -m pytest tests/test_action_tools.py -q`
- Result: passed.
- Command: `.venv/bin/python -m pytest tests/test_auth_token.py tests/test_chat_cancellation.py -q`
- Result: passed.
- Command: `.venv/bin/python -m pytest -q`
- Result: 297 tests passed, with one existing FastAPI/Starlette deprecation warning.
- Command: `.venv/bin/python -m compileall -q .`
- Result: passed.
- Command: `git diff --check`
- Result: passed.

Architectural drift check:
- Status: `none`
- Notes: Fixes use existing config, middleware, tool, and test patterns.

Open follow-ups:
- Address the existing FastAPI/Starlette TestClient deprecation warning when convenient.

## 2026-07-05 - Pinned Memory Ordering Fix

Phase: Maintenance
Scope: Memory manager API ordering.
Intent: Ensure pinned memories appear at the top of the memory list.

Files changed:
- `server.py`
- `tests/test_operability_features.py`
- `audit_log.md`

Behavior changed:
- `/api/memories` now sorts pinned memories before unpinned memories, with date-descending order inside each group.

Verification:
- Command: `.venv/bin/python -m pytest tests/test_operability_features.py -q`
- Result: passed.
- Command: `.venv/bin/python -m pytest -q`
- Result: passed, with one existing FastAPI/Starlette deprecation warning.
- Command: `.venv/bin/python -m compileall -q server.py tests/test_operability_features.py`
- Result: passed.

Architectural drift check:
- Status: `none`
- Notes: The fix is in the existing server-side memory row ordering path and adds a focused regression test.

Open follow-ups:
- None.

## 2026-07-05 - Persistent Cognitive State Layer

Phase: AGI-like engine expansion
Scope: Structured current state, prompt construction, API, and tests.
Intent: Make Neural Map more proactive and AGI-like by giving it an explicit goal stack, world model, and belief ledger instead of relying only on retrieved memories.

Files changed:
- `config.py`
- `brain/cognitive_state.py`
- `brain/pipeline.py`
- `server.py`
- `tests/conftest.py`
- `tests/test_cognitive_state.py`
- `web/index.html`
- `README.md`
- `todo.md`
- `audit_log.md`

Behavior changed:
- Adds a persistent JSON cognitive-state store beside the Chroma memory store.
- Injects a bounded cognitive-state snapshot into every chat prompt before retrieved memories.
- Exposes API endpoints for reading state and upserting goals, world facts, and beliefs.
- Adds a browser State panel for viewing and updating goals, world facts, and beliefs.
- Reports cognitive state in `/api/health`.
- Filters inactive and expired records out of the prompt snapshot while preserving them in the full state view.

Verification:
- Command: `.venv/bin/python -m compileall -q brain/cognitive_state.py brain/pipeline.py server.py tests/test_cognitive_state.py`
- Result: passed.
- Command: `node -e "const fs=require('fs'); const html=fs.readFileSync('web/index.html','utf8'); const m=html.match(/<script>([\s\S]*)<\/script>/); new Function(m[1]);"`
- Result: passed.
- Command: `.venv/bin/python -m pytest tests/test_cognitive_state.py tests/test_self_audit_loop.py tests/test_source_of_truth.py tests/test_health.py -q`
- Result: 15 passed, with one existing FastAPI/Starlette deprecation warning.
- Command: `.venv/bin/python -m pytest -q`
- Result: 303 passed, with one existing FastAPI/Starlette deprecation warning after one transient uploaded-PDF recall eval failure passed on immediate rerun and the final full run passed.

Architectural drift check:
- Status: `none`
- Notes: The new layer has one clear module, config-backed limits, trusted prompt framing, health visibility, API tests, and README documentation.

Open follow-ups:
- Add UI controls for editing the goal stack, world model, and belief ledger from the browser.

## 2026-07-08 - Whole-Code Audit Fix Pass

Phase: Post-audit hardening
Scope: Retrieval tie-ranking determinism, per-turn Ollama probe caching, and test hermeticity.
Intent: Fix the three issues found in the 2026-07-08 whole-code audit.

Files changed:
- `brain/retriever.py`
- `brain/memory.py`
- `brain/llm.py`
- `tests/test_audit_fixes.py`
- `tests/test_retrieve_weighted.py`
- `audit_log.md`

Behavior changed:
- Recency and importance re-rank signals now use competition ranking: candidates with EQUAL values share an RRF rank. Previously ties were split into distinct ranks ordered by random memory id, handing out score differences larger than a real semantic/keyword edge — the root cause of the ~1/15 flake in `test_eval_recall_from_uploaded_pdf`.
- `recency_score` accepts a single `now` snapshot; the retriever passes one per ranking pass so identical dates tie exactly instead of differing by microseconds.
- `model_max_context` caches successful `ollama show` probes (bounded, invalidated when the client instance changes), removing ~4 blocking HTTP round trips per System 2 turn. Failed probes are not cached, mirroring `supports_tools`.
- The two SPCX two-question tests stub `pipeline.llm.chat`: their prompts route `answer()` into System 2, whose un-mocked critique/rewrite previously hit live Ollama — failing `test_tool_path_prefetches_web_results_and_skips_web_tool` whenever Ollama was up, and making the suite Ollama-dependent and slow.

Verification:
- Command: 60-trial in-process repro of the PDF-recall eval (scratchpad script)
- Result: 3/60 failures before the fix; 0/60 after.
- Command: `.venv/bin/python -m pytest -rN` (twice)
- Result: `378 passed` both runs; suite time dropped from ~61s to ~6s because the SPCX tests no longer generate through live Ollama.

Architectural drift check:
- Status: `none`
- Notes: Tie-ranking is a contained change inside the existing RRF fusion; the probe cache mirrors the established `_TOOLS_CAPABILITY` pattern; test changes only add stubs and one regression test (`test_tied_recency_and_importance_score_identically`).

Open follow-ups:
- Latent: `_tool_model`/`_memory_model` in `brain/toolkit/memory_tools.py` and `governance.py` return the string "None" when no model is set (`str(None)` is truthy). Harmless while a dedicated embed model is configured (collection naming ignores the chat model), but would mis-scope memory tools under `NEURAL_MAP_EMBED_MODEL=__selected__`.

## 2026-07-08 - Activation Hardening: Superseded Filter And CSRF Origin Gate

Phase: Post-audit hardening
Scope: Activated-memory provenance and cross-site request protection.
Intent: Close the two gaps found while reviewing the activated_memory_ids flow.

Files changed:
- `server.py`
- `tests/test_chat_history.py`
- `tests/test_host_validation.py`
- `audit_log.md`

Behavior changed:
- `_activated_memories` now skips records with `superseded` metadata, matching
  the current-truth invariant every retrieval producer already enforces. A stale
  activation chip (localStorage outlives corrections) can no longer resurrect a
  retired/wrong memory into the prompt at score 1.0, and no longer reinforces it
  each turn.
- New `require_allowed_origin` middleware rejects POST/PUT/PATCH/DELETE requests
  whose Origin header is a foreign web origin (403). Origins are checked with the
  same `_name_allowed` rules as the Host gate (IP literals, localhost, *.local,
  NEURAL_MAP_ALLOWED_HOSTS); absent Origin (curl, launchd, scripts) is allowed;
  "null" is rejected. This closes the cross-site "simple request" CSRF vector
  against the token-less loopback server (text/plain JSON POSTs from a malicious
  page, which skip the CORS preflight and carry an allowed Host of 127.0.0.1).

Verification:
- Command: `.venv/bin/python -m pytest tests/test_host_validation.py tests/test_chat_history.py -rN`
- Result: 15 passed (6 new Origin tests incl. end-to-end 403/200 through the
  middleware; 1 new superseded-activation regression test).
- Command: `.venv/bin/python -m pytest -rN`
- Result: `385 passed`.

Architectural drift check:
- Status: `none`
- Notes: The Origin gate reuses the existing Host-allowlist rules via a shared
  `_name_allowed` helper and the same middleware pattern; the activation filter
  applies an existing store invariant to one more producer.

Open follow-ups:
- There is no "un-retire" affordance: if a superseded memory should return,
  the flag must be cleared in metadata (update_memory preserves it). Decide
  whether that needs a tool/UI control.

## 2026-07-08 - Action Package Provenance Verification

Phase: Post-audit hardening
Scope: /api/action_packages/from_initiative trust model.
Intent: Stop client-asserted initiative content from becoming a trusted-looking
high-priority Action Package without verification.

Files changed:
- `brain/action_packages.py`
- `server.py`
- `tests/test_action_tools.py`
- `audit_log.md`

Behavior changed:
- `queue_from_initiative` now verifies the claimed source initiative against the
  store: the id must resolve to a stored, still-open `initiative` record whose
  text matches the submitted text. Verified packages behave exactly as before.
- Anything unverifiable — unknown id, dismissed initiative, edited text, or no
  id — is still queued for review but tainted `unverified_initiative` and its
  priority is capped at 3, so fabricated content cannot present as an urgent
  trusted package. The result reports `verified_initiative`.
- The one-word chat approval shortcut refuses unverified packages (mirroring the
  tainted-external-send rule): they must be read and approved in the Actions
  panel (`pending_decision: blocked_unverified_initiative`).

Verification:
- Command: `.venv/bin/python -m pytest tests/test_action_tools.py -rN`
- Result: 45 passed (4 new: unverified taint + priority cap, verified
  passthrough, edited-text downgrade, chat-approval block).
- Command: `.venv/bin/python -m pytest -rN`
- Result: `389 passed`.

Architectural drift check:
- Status: `none`
- Notes: Reuses the existing taint mechanism and the established
  panel-only-approval pattern; verification reads the same initiative records
  the initiative layer already writes.

Open follow-ups:
- `ActionPackageReq.source_ids` is accepted but unused by the package body;
  either wire the ids into the Confirmed Facts provenance table or drop the
  field from the schema.

## 2026-07-08 - Attention Momentum Blend

Phase: Post-audit hardening
Scope: Attention router score update dynamics.
Intent: Stop a single high-intensity message from instantly displacing an
established focus (jittery focused_domain flips).

Files changed:
- `brain/attention.py`
- `config.py`
- `tests/test_attention.py`
- `audit_log.md`

Behavior changed:
- `update_from_turn` now blends additively with momentum — per turn each domain
  scores `old * ATTENTION_DECAY + signal * ATTENTION_SIGNAL_BLEND` (clamped to
  [0, 1]) — instead of `max(old * decay, signal)`, which let one loud message
  jump straight to the top score.
- New config `ATTENTION_SIGNAL_BLEND` (env `NEURAL_MAP_ATTENTION_SIGNAL_BLEND`,
  default 0.5): 1.0 approximates the old jumpy behavior, smaller switches more
  slowly. Chosen at 0.5 so a cold start still focuses on the first clear
  message (blend * typical signal clears ATTENTION_FOCUS_THRESHOLD) while an
  established focus survives a one-message burst and a sustained topic shift
  takes over in ~2-4 turns.
- Unsignaled domains decay exactly as before (old * decay + 0).

Verification:
- Command: `.venv/bin/python -m pytest tests/test_attention.py -rN`
- Result: 10 passed (4 new: single-burst stability, sustained-shift takeover,
  cold-start focus, unsignaled decay).
- Command: `.venv/bin/python -m pytest -rN`
- Result: `393 passed`.

Architectural drift check:
- Status: `none`
- Notes: Same state file, locks, and clamping; one new config-backed knob with
  documentation and defaults preserving both responsiveness and stability.

Open follow-ups:
- None.

## 2026-07-08 - Learned Attention Vocabulary Via Evolution

Phase: Post-audit hardening
Scope: Attention router vocabulary blind spot.
Intent: Let the static _DOMAIN_TERMS dictionary grow through the existing
human-approved evolution pipeline instead of staying a permanent blind spot.

Files changed:
- `brain/evolution.py`
- `brain/attention.py`
- `config.py`
- `tests/test_evolution.py`
- `tests/test_attention.py`
- `audit_log.md`

Behavior changed:
- Evolution proposals may now carry `attention_terms`: vocabulary ADDITIONS to
  existing attention domains only. Terms are sanitized (lowercase single-token
  shape, built-ins deduped, unknown domains dropped, per-domain cap via new
  config `ATTENTION_MAX_LEARNED_TERMS_PER_DOMAIN`, default 12).
- The proposer's payload includes the router's domains, built-in terms, and
  currently learned terms, with prompt guidance to suggest terms only when the
  log/Wisdom shows missed focus — and never to invent domains or remove terms.
- attention scoring merges approved learned terms with built-ins at score time
  (`evolution.learned_attention_terms`), fetched once per scoring pass.
- Full lifecycle parity with other mutations: attention_terms-only proposals
  still require a complete preserve/improve canary review, are captured in
  before/after snapshots, restore with `restore_snapshot`, and count as runtime
  config for rollback's auto-revert guard.

Verification:
- Command: `.venv/bin/python -m pytest tests/test_evolution.py tests/test_attention.py -rN`
- Result: 28 passed (3 new: sanitize+deploy+restore, canary gate on
  terms-only proposals, end-to-end propose→approve→router-focus).
- Command: `.venv/bin/python -m pytest -rN`
- Result: `396 passed`.

Architectural drift check:
- Status: `none`
- Notes: Reuses the evolution proposal/approval/snapshot machinery unchanged;
  attention keeps its deterministic scoring, just over a mergeable vocabulary.
  Lazy import in evolution avoids an attention<->evolution module cycle.

Open follow-ups:
- The UI's Evolution panel shows proposals as JSON; consider rendering
  attention_terms diffs explicitly once the feature sees real use.

## 2026-07-08 - Consolidate State-File Locking Into One Helper

Phase: Post-audit hardening
Scope: Cross-process file locking shared by all persistent-state modules.
Intent: Keep the (deliberately robust) dual thread-lock + flock pattern, but
stop copy-pasting it — ten byte-identical lock classes invited drift.

Files changed:
- `brain/utils.py`
- `brain/attention.py`, `brain/awareness.py`, `brain/self_model.py`,
  `brain/emotion.py`, `brain/initiative.py`, `brain/will.py`,
  `brain/cognitive_state.py`, `brain/evolution.py`, `brain/sleep.py`,
  `brain/store.py`
- `tests/test_evolution.py`, `tests/test_sleep.py`
- `audit_log.md`

Behavior changed:
- New `brain.utils.file_lock(lock_file)` context manager holds the single
  fcntl.flock implementation; each module's `_state_lock()` / `_status_lock()` /
  `_revision_lock()` is now a two-line wrapper around it.
- No semantic change: identical lock-file names (a running server and background
  jobs keep locking the same files), lock path still resolved at acquisition
  time (tests that monkeypatch `_STATE_FILE` keep working), same
  unlock-then-close ordering, same pairing with each module's threading.Lock.
- The two lock-behavior tests now monkeypatch `brain.utils.fcntl` instead of
  the (removed) per-module fcntl imports.

Verification:
- Transformation script asserted each lock class matched byte-for-byte exactly
  once per file before rewriting; `grep fcntl brain/` now matches only utils.py.
- Command: `.venv/bin/python -m compileall -q brain/`
- Result: passed.
- Command: `.venv/bin/python -m pytest -rN`
- Result: `396 passed`.

Architectural drift check:
- Status: `none`
- Notes: This is anti-drift: one loader/path for a pattern that was duplicated
  ten times. brain/utils keeps its "no brain-module dependencies" charter.

Open follow-ups:
- The atomic-write helper (tmp + os.replace) is still duplicated across the
  same modules; a follow-up could consolidate it the same way.

## 2026-07-08 - Awareness Open-Loop TTL

Phase: Post-audit hardening
Scope: Awareness layer open-loop carry-forward.
Intent: Stop orphaned open loops from recycling indefinitely ("loop
stagnation") and, at the list cap, blocking newer loops.

Files changed:
- `brain/awareness.py`
- `config.py`
- `tests/test_awareness.py`
- `audit_log.md`

Behavior changed:
- `_open_loops` now distinguishes confirmed loops (produced this turn by a live
  reminder/initiative/goal source, which are re-fetched every turn) from
  orphans (carried only from the previous state). Confirmed loops lead the list
  with carry 0; orphans age one turn each update and retire past
  `AWARENESS_LOOP_TTL_TURNS` (new config, env
  `NEURAL_MAP_AWARENESS_LOOP_TTL_TURNS`, default 8).
- New `open_loop_meta` state field (loop text -> unconfirmed-turn count),
  repaired/bounded on read like the rest of the state; `open_loops` stays a
  list of strings so the prompt snapshot, /api/awareness, and the UI chip are
  unchanged.

Verification:
- Command: `.venv/bin/python -m pytest tests/test_awareness.py -rN`
- Result: 6 passed (3 new: TTL expiry, live-source indefinite confirmation,
  fresh-before-carried ordering).
- Command: `.venv/bin/python -m pytest -rN`
- Result: `399 passed`.

Architectural drift check:
- Status: `none`
- Notes: Same state file, locks, and repair pattern; one config-backed knob;
  external state shape preserved (strings plus one additive metadata field).

Open follow-ups:
- None.

## 2026-07-08 - Self-Model Lifecycle: TTL Archive For Active Work

Phase: Post-audit hardening
Scope: Self-model staleness and prompt-slot churn.
Intent: Give the additive self-model a retirement path — the lists were already
capped (SELF_MODEL_MAX_ITEMS), but nothing ever LEFT: stale projects/promises
squatted in the few prompt slots and eviction ignored which entries were still
being confirmed.

Files changed:
- `brain/self_model.py`
- `config.py`
- `tests/test_self_model.py`
- `audit_log.md`

Behavior changed:
- current_projects and open_promises entries whose updated_at ages past
  `SELF_MODEL_ACTIVE_TTL_DAYS` (new config, default 45; 0 disables) move to a
  new `archive` list during the state repair pass. The archive is capped
  (`SELF_MODEL_MAX_ARCHIVE`, default 24), visible in /api/self_model, and
  excluded from the prompt snapshot.
- lessons_learned and relationship_notes are exempt: they are identity, not
  workload, and stay until displaced by the item cap.
- `_append_unique` now moves a re-confirmed entry to the end of its list, so
  the item cap evicts the least-recently-confirmed row instead of the oldest
  insertion, and re-confirmation refreshes the TTL clock.

Verification:
- Command: `.venv/bin/python -m pytest tests/test_self_model.py -rN`
- Result: 7 passed (3 new: TTL archive + lesson exemption + snapshot shape,
  TTL=0 disable, LRU re-confirmation ordering).
- Command: `.venv/bin/python -m pytest -rN`
- Result: `402 passed`.

Architectural drift check:
- Status: `none`
- Notes: Same state file/lock/repair pattern; archival happens inside the
  existing repair pass (self-healing on read, persisted on next write);
  follows the cognitive-state precedent of filtering the prompt snapshot while
  preserving the full state view.

Open follow-ups:
- Promises are only ever archived by TTL; detecting fulfilled promises (e.g.
  matching a later "done/fixed" turn) could retire them sooner.

## 2026-07-08 - Awareness Energy Momentum

Phase: Post-audit hardening
Scope: Awareness energy dynamics.
Intent: Stop energy from snapping instantly between activity levels (0.46 ->
0.71 in one turn) — ease it toward the per-turn target instead, matching the
attention router's momentum approach.

Files changed:
- `brain/awareness.py`
- `config.py`
- `tests/test_awareness.py`
- `audit_log.md`

Behavior changed:
- New `_smoothed_energy`: the bool-flag formula now produces the per-turn
  TARGET; the carried value moves a fraction of the remaining distance per turn
  (`energy = old + (target - old) * AWARENESS_ENERGY_SMOOTHING`).
- New config `AWARENESS_ENERGY_SMOOTHING` (env
  `NEURAL_MAP_AWARENESS_ENERGY_SMOOTHING`, default 0.3): one busy turn nudges
  0.5 -> 0.563 rather than jumping to 0.71; sustained activity converges to the
  target within a few turns; 1.0 restores the old instantaneous behavior.
- Also removed a duplicated `last_user_seen` key introduced and caught during
  this edit.

Verification:
- Command: `.venv/bin/python -m pytest tests/test_awareness.py -rN`
- Result: 9 passed (3 new: fractional single-turn move, convergence over
  sustained turns, smoothing=1.0 parity with old behavior).
- Command: `.venv/bin/python -m pytest -rN`
- Result: `405 passed`.

Architectural drift check:
- Status: `none`
- Notes: Same helper/knob pattern as the attention momentum blend; one
  config-backed setting with a documented escape hatch.

Open follow-ups:
- None.

## 2026-07-08 - Approval-Gated Lessons Learned

Phase: Post-audit hardening
Scope: Self-model lesson authority.
Intent: A lesson is a standing behavioral rule injected into every prompt, so a
turn must not be able to self-commit one ("hallucinated lessons"). Lessons now
get action-package rigor: propose -> review -> approve/reject, with undo.

Files changed:
- `brain/toolkit/lessons.py` (new)
- `brain/self_model.py`
- `brain/toolkit/registry.py`
- `brain/toolkit/governance.py`
- `brain/pipeline.py`
- `tests/test_self_model.py`
- `audit_log.md`

Behavior changed:
- New WRITE_SAFE tool `record_lesson`: dispatch queues a pending action with a
  lesson_record preview; approval commits via the new `self_model.add_lesson`;
  undo removes via `self_model.remove_lesson`. Callable by the model in the
  tool loop too, where it inherits the same approval gate.
- `self_model.update_from_turn` no longer writes lessons_learned directly: its
  deterministic lesson candidates are dispatched through record_lesson (tainted
  "turn_interpretation" so the review shows the origin). Projects, promises,
  and relationship notes remain direct — they are continuity state, not
  standing rules.
- Dedup treats committed, pending, AND rejected proposals as tracked, so a
  rejected lesson is not re-proposed every subsequent matching turn.

Verification:
- Command: `.venv/bin/python -m pytest tests/test_self_model.py tests/test_action_tools.py -rN`
- Result: 55 passed (3 new: approve-commits-and-undo-removes, rejected-not-
  reproposed, committed-not-reproposed; 1 updated to assert the proposal shape).
- Command: `.venv/bin/python -m pytest -rN`
- Result: `408 passed`.

Architectural drift check:
- Status: `none`
- Notes: Reuses the existing pending/preview/audit/undo machinery and the same
  taint convention; the tool registry, permissions panel, and prompt tool note
  pick the new tool up through the standard paths.

Open follow-ups:
- The Wisdom notes written by metacognition are LLM-generated self-corrections
  stored as ordinary memories; they feed evolution proposals (already human-
  gated) but are not themselves approval-gated. Consider whether they deserve
  the same treatment.

## 2026-07-08 - Word-Boundary Truncation And Identity Lock

Phase: Post-audit hardening
Scope: Prompt-facing text truncation (awareness, self-model) and self-model
identity integrity.
Intent: Close the last two audit-review items — silent mid-sentence truncation
handing the model fragments that read as complete statements, and the absence
of any lock on the identity block injected into every prompt.

Files changed:
- `brain/utils.py`
- `brain/awareness.py`
- `brain/self_model.py`
- `tests/test_self_model.py`
- `audit_log.md`

Behavior changed:
- New shared `utils.clean_text`: whitespace-normalized, capped text that cuts
  at a word boundary (when one is reasonably close) and appends a visible
  ellipsis, so both the model and a human reviewer can SEE that text was
  truncated. awareness and self-model `_clean_text` now delegate to it,
  keeping their per-module default limits.
- `recent_turns` summaries (API/UI only — they never enter the prompt
  snapshot) get a generous 400-char cap instead of the prompt-tier 180, so
  semantic meaning is preserved where tokens don't cost anything.
- Self-model identity (name, role, principles) is LOCKED to the code
  constants: `_repair_state` ignores whatever the state file claims and heals
  it on the next write. No tool, API, or file edit can rewrite the persona;
  changing identity is a reviewed source-code change.

Verification:
- Command: `.venv/bin/python -m pytest tests/test_self_model.py tests/test_awareness.py -rN`
- Result: 22 passed (3 new: state-file identity tampering is ignored,
  word-boundary cut with visible marker, recent_turns keep detail past the old
  cap).
- Command: `.venv/bin/python -m pytest -rN`
- Result: `411 passed`.

Architectural drift check:
- Status: `none`
- Notes: clean_text joins file_lock in brain/utils (still no brain-module
  dependencies); the identity lock strengthens the existing repair pass rather
  than adding a new mechanism.

Open follow-ups:
- None.

## 2026-07-09 - Homeostasis Coupling Layer ("Synthesis Engine")

Phase: Brain layer — derived operating posture.
Scope: New `brain/homeostasis.py`; wiring in `brain/pipeline.py`, `server.py`,
`config.py`; test harness hook in `tests/conftest.py`.
Intent: Couple isolated state into interdependent variables: a sustained rise
in tracked life load (clinical hours in the cognitive-state world model,
corroborated by attention salience) automatically re-weights derived behavior
dials (`portfolio_intensity`, `proactivity`) through explicit rules — the
"homeostasis" model where the workload change IS the posture shift, not a fact
retrieved and then deliberated. Deliberately NOT a retrieval replacement:
memory stays the evidence layer; this is derived policy on top.

Files changed:
- `brain/homeostasis.py` (new)
- `brain/pipeline.py`
- `server.py`
- `config.py`
- `tests/conftest.py`
- `tests/test_homeostasis.py` (new)

Behavior changed:
- New per-turn update after attention: sensed `user_load` level-tracks its
  signal (world-model hours/qualitative load x confidence; attention salience
  scaled to 0.45 so it can NEVER fire a rule without world-model evidence);
  derived dials smooth toward baseline + active-rule effects.
- Rules are declarative data with activation threshold + lower release point
  (hysteresis), so posture cannot oscillate. Defaults ship in code; replacement
  only via `POST /api/homeostasis/rules` (the approval gate), never inference.
- Every rule fire/release is appended to a bounded adjustments log
  (source value, target before/after) and emitted as an observability event —
  no silent behavior drift.
- Prompt gains a "Homeostasis Posture JSON" block (guidance-only; never
  overrides Source of Truth / instructions / evidence).
- Conservative proactivity posture trims low-priority initiative candidates
  (reminders and arbiter mediation items always pass; drops are logged).
- `POST /api/cognitive_state/world` propagates through the coupling
  immediately. New endpoints: GET/POST `/api/homeostasis`, `/reset`, `/rules`;
  health reports the feature.

Verification:
- Command: `.venv/bin/python -m pytest tests/test_homeostasis.py -rN`
- Result: 12 passed (coupling fires on sustained load, single mention and
  attention-only signals cannot fire, hysteresis release, attributable audit
  entries, rule validation/replacement, initiative gating, prompt block, API
  endpoints, world-upsert propagation).
- Command: `.venv/bin/python -m pytest -rN`
- Result: 423 passed.
- End-to-end scenario (isolated store): 20 h/week settles load=0.42, posture
  expansive, no rules active; raising the world item to 40 h/week converges
  load=0.86, fires both rules, portfolio_intensity 0.70 -> 0.21 (conservative),
  proactivity 0.60 -> 0.36 (balanced), with both transitions in the
  adjustments log.

Architectural drift check:
- Status: `none`
- Notes: follows the existing state-layer pattern (locked JSON state file,
  clamped floats, repair pass, prompt_snapshot); truth-layers precedence is
  unchanged — homeostasis is derived policy, not a truth source, and reads
  only from layers that already own their facts.

Open follow-ups:
- Web UI panel for posture + rules (API is available; UI not yet wired).
- Consider a `market_stress` sensed variable once a deterministic source is
  chosen.

## 2026-07-10 - Recurrent Shared Workspace

Phase: Consciousness-adjacent engineering, phase 1.
Scope: Cross-layer recurrent coordination, prompt broadcast, partition metrics,
observability, API/UI inspection, and research protocol.
Intent: Move Neural Map from parallel prompt snapshots toward measurable causal
interdependence without presenting a software proxy as IIT Phi or consciousness.

Behavior changed:
- Every chat turn collects bounded, authority-preserving contents from the
  current request and existing cognitive layers, settles related contents for
  three recurrent rounds, and broadcasts a diverse set into both response paths.
- Traces record content/source counts, recurrent edges, rounds, and the
  integration proxy. The persisted workspace records coalition membership and
  source-removal impacts.
- The Systems panel and `/api/workspace` expose the state; reset and feature-off
  controls are available.
- Untrusted memory remains explicitly untrusted inside the workspace. Prompt
  guidance states that the workspace is coordination, not evidence, IIT Phi, or
  evidence of consciousness.

Verification:
- Focused workspace, cognitive-layer, frontend, and grounding suites passed.
- Full suite: 482 passed, 1 skipped.

Architectural drift check:
- Status: none.
- Notes: the workspace coordinates existing layers but does not become a truth
  source, bypass approval gates, add an LLM call, or expose private reasoning.

Open follow-ups:
- Phase 2 predictive self-model and calibration scoring.
- Phase 3 structured counterfactual interventions.

## 2026-07-10 - Semantic Grounding / Hard-Problem Protocol

Phase: Hard-problem route, tractable foundation.
Scope: Semantic evidence channels, qualia-agnostic reporting, prompt integration,
trace metrics, API/UI inspection, and phenomenology protocol.
Intent: Push concepts beyond language-only association where evidence exists,
without treating multimodality, control state, behavior, or self-report as proof
of subjective experience.

Behavior changed:
- Current concepts are classified across linguistic, sensory, action,
  regulatory, relational, and external channels.
- Pre-answer grounding informs both reasoning paths; executed tool consequences
  and current external data enrich the map after the first pass.
- Every concept and the system-wide report keep `phenomenal_status=unknown`;
  self-report is excluded as phenomenal evidence, and regulatory evidence is
  explicitly described as control state rather than feeling.
- Prompt-facing evidence descriptions and authority are reconstructed from the
  channel, preventing persisted state from escalating authority or injecting
  instructions.
- `/api/grounding`, the Systems panel, health diagnostics, and cognitive traces
  expose the functional measurements.

Verification:
- Semantic-grounding tests: 17 passed.
- Full suite: 500 passed, 1 skipped.

Architectural drift check:
- Status: none.
- Notes: grounding is advisory metadata, not a truth layer, consciousness score,
  new LLM call, or approval bypass.

Open follow-ups:
- Cross-modal invariance and novel-combination transfer corpus.
- Authorized action-closure experiments with predicted consequences.

### Grounding-gate enforcement follow-up

- Added deterministic claim classes for conceptual, sensory-observation,
  current-external, action-result, verification, and phenomenal requests.
- Textual Source of Truth is now visible to grounding as trusted linguistic
  provenance but cannot satisfy a non-linguistic requirement.
- Missing required evidence forces System 2 after the ordinary Fast-mode gate;
  the critique prompt receives explicit claim constraints.
- Prefetched/current external data enriches grounding before generation, and
  executed tool consequences update the gate before final verification.
- Requirements inspect substantive concepts rather than global channel presence,
  preventing unrelated evidence from satisfying a claim through trigger-word
  overlap.

## 2026-07-10 - Grounding-Aware Proactive Arbiter

Phase: Hard-problem route, proactive admissibility.
Scope: Arbiter mediation, Will provenance, initiative storage/surfacing,
briefings, prompt metadata, UI packaging/dismissal, and observability.
Intent: Make semantic grounding independent of directional conflict so an
unopposed generated instruction cannot become actionable merely because nothing
argues against it.

Behavior changed:
- The Arbiter now evaluates two axes: direction (`advance`/`caution`) and
  grounding (`grounded`/`symbolic_only`/`unassessed`).
- Generated initiatives without non-linguistic support become synthetic
  `arbiter_grounding` evidence-review questions with `admissibility=needs_evidence`.
- Direction conflicts still collapse into one `arbiter_conflict`, now annotated
  `review_conflict` or `review_conflict_and_evidence`.
- Will records lexical supporting memory IDs and honestly labels its current
  language-corpus evidence `symbolic_only`; it is not promoted to world evidence.
- User reminders remain authoritative commitments. The UI blocks Action Package
  creation for evidence-pending reviews and dismisses their underlying records.
- `NEURAL_MAP_ARBITER_GROUNDING=0` restores conflict-only arbitration.

Verification:
- Targeted initiative/Will/homeostasis/frontend suite: 49 passed.
- Full suite at this stage: 505 passed, 1 skipped.

Architectural drift check:
- Status: none.
- Notes: no new LLM call, truth source, or action authority was added; the gate
  only reduces generated proactive authority until evidence or confirmation.

## 2026-07-10 - Per-Claim Grounding Receipts and Final Guard

Phase: Hard-problem route, fail-closed enforcement.
Scope: Final response claims, tool-outcome binding, post-Council validation,
chat API telemetry, observability, and regression coverage.
Intent: Turn request-level grounding guidance into a last-boundary structural
invariant that cannot be bypassed by a later answer-producing layer.

Behavior changed:
- Final sentences receive receipts with claim class, concepts, required/present
  channels, evidence references, and `assert`/`qualify`/`withhold` disposition.
- Unsupported sensory, completed-action, verification, current-external, and
  first-person phenomenal assertions fail closed: the sentence is removed and a
  deterministic evidence limitation is substituted without another LLM call.
- The guard runs after System 2 and the Inner Council.
- Executed tool identities are stored as action outcomes, allowing exact tool
  claims to bind to consequences. Attributed memory text is recognized as
  provenance rather than a fresh current-world assertion.
- Receipts, violations, and modification status are returned by chat and emitted
  in the `final_claim_guard` trace event.

Verification:
- Focused grounding/Council/intuition/context suite: 42 passed.
- Full suite: 510 passed, 1 skipped.

Architectural drift check:
- Status: none.
- Notes: enforcement is deterministic, bounded, post-Council, and adds no model
  call or action authority. Conceptual analysis remains available with a
  `qualify` receipt instead of being blocked.
## 2026-07-10 - Functional/Social Presence Demonstration Route

- Added a deterministic, bounded evidence ledger for continuity, relational
  adaptation, commitments, stable principles, self-correction, self-modeling,
  and executed tool consequences.
- Evaluations are evidence-backed and fail closed. Current-turn state cannot be
  used to prove that past context shaped the same answer.
- Operational social presence is explicitly separated from phenomenal
  consciousness. Human-like experience remains `unknown`; complexity and
  feeling-language remain insufficient evidence.
- Exposed the ledger through chat results, health, `GET /api/presence`, reset,
  turn observability, and the Systems panel.
- Focused verification: 36 passed.
- Full verification: 516 passed, 1 skipped.

## 2026-07-10 - Approval Resolver and Stale Execution Recovery

- Diagnosed the reported HTTP 500 from two live `run_tests` records stranded in
  `approving`: timeout output could contain bytes, causing JSON persistence to
  fail after the single-shot claim was acquired.
- Normalized subprocess output and made the pending store defensively
  JSON-safe.
- Added a five-minute approval execution lease; abandoned claims are recovered
  as terminal failures rather than silently remaining wedged.
- Plain “yes”/“continue” now resolves exactly one pending action. Multiple
  pending actions return an ambiguity response and require selection.
- Correction language such as “no, I actually approved it” is treated as an
  approval rather than a rejection.
- Tool timeouts and nonzero command results are reported as “approval accepted,
  execution failed,” removed from the pending queue, and not presented as
  successfully completed work.
- Existing tainted-external-send and unverified-initiative protections remain
  in force.
- Focused verification: 63 passed.
- Full verification: 521 passed, 1 skipped.

## 2026-07-11 - Repeated Test-Approval Loop Follow-up

- Correlated live timestamps and partial pytest progress: a second full-suite
  action started three seconds before the first timed out. The identical suite
  completed in roughly six seconds when run without overlap.
- Added a non-blocking cross-process single-flight lock around `run_tests`; a
  concurrent request now terminates immediately with “another test run is
  already in progress” instead of launching a competing pytest process.
- Stopped the Actions UI from sending failed execution results back through the
  model. The failure stays visible, but the model receives no automatic turn in
  which it can queue the identical approval again.
- Confirmed the last stranded action `act_adad8a17b26b` is terminal `failed` and
  no longer pending.
- Focused verification: 66 passed.
- Full verification: 524 passed, 1 skipped.

## 2026-07-11 - Live Context Utilization Meter

- Added request-local instrumentation at the LLM boundary, where the complete
  system prompt, history, user message, tool schemas, and iterative tool-result
  messages are observable.
- Uses Ollama `prompt_eval_count` when returned; otherwise falls back to a
  deliberately labeled character-based estimate.
- Aggregates multiple foreground calls and reports peak input-prompt occupancy,
  configured context, remaining tokens, percentage, measurement method, and
  call counts through the chat API.
- Added a header readout and progress meter with warning thresholds at 70% and
  90%, plus context telemetry in the Observe summary.
- Focused verification: 18 passed.
- Full verification: 527 passed, 1 skipped.

## 2026-07-11 - Context Meter Reliability Follow-up

- Verified the real UI's cancellable streaming path preserves Ollama's final
  `prompt_eval_count` as an exact measurement.
- Persisted the last valid measurement per model across page reloads; changing
  model/context or issuing `/clear` removes the stale measurement.
- The local in-app browser blocked direct `127.0.0.1` inspection and no Neural
  Map server was listening during diagnosis, so live-page verification fell
  back to API/static contracts and the streaming-client integration test.
- Focused verification: 20 passed.
- Full verification: 529 passed, 1 skipped.

## 2026-07-11 - Clear/Purge Separation and Telemetry Failure Signal

> Superseded later the same day by the Fresh-Chat Purge Semantics change below;
> no destructive purge route remains.

- Kept `/clear` non-destructive: it resets browser chat history, transcript,
  activated memories, and the displayed context measurement.
- Added confirmed `/purge` for irreversible server-side memory deletion across
  vector/text stores, graph, sensory anchors, cold archives, and persistent
  derived cognitive/identity layers.
- Preserved original source files, Source of Truth, backups, and audit logs.
- Added a capture-generation barrier so a pre-purge background episode cannot
  race in and repopulate an emptied memory store.
- A normal chat response missing `context_usage` now displays “usage unavailable
  · restart server,” identifying a stale Python backend instead of leaving the
  meter silently unchanged.
- Focused verification: 25 passed.
- Full verification: 534 passed, 1 skipped.

## 2026-07-11 - Stale Backend Launcher Repair

- Diagnosed the 405 as a split frontend/backend deployment: `app.js` was loaded
  from disk, while PID 713 continued serving an old in-memory FastAPI route
  table.
- Added a deterministic fingerprint over `server.py`, `config.py`, and all
  Python modules under `brain/`, exposed through `GET /api/build`.
- Updated the macOS launcher to compare the running fingerprint with disk and
  `launchctl kickstart -k` only when stale or when the old server lacks the
  build route.
- Corrected launcher authentication to use `X-Auth-Token`; query-token bootstrap
  intentionally redirects and cannot be parsed as JSON.
- Recompiled and ad-hoc re-signed `Neural Map.app`, then force-restarted the
  launchd agent.
- Live verification: running fingerprint matched disk and OpenAPI contained
  `POST /api/memory/purge`.
- Focused verification: 28 passed; launcher-specific follow-up: 3 passed.
- Full verification: 537 passed, 1 skipped.

## 2026-07-11 - Fresh-Chat Purge Semantics

- Corrected `/purge` after user clarification: it now behaves like starting a
  new chat and never deletes or resets server-side memory.
- Removed the destructive memory-purge and model-state-reset API routes.
- `/purge` clears the browser transcript and activated context, disables
  retrieval, and suppresses persistent cognitive, awareness, attention,
  workspace, self-model, homeostasis, emotion, reminder, initiative, and
  preference prompt blocks for that browser session.
- Durable memories, graph, sensory records, cold archives, Source of Truth,
  source files, backups, and global derived state remain unchanged.
- Added `/memory on` and `/memory off`; the mode is session-scoped and included
  explicitly in every chat request.
- Focused verification: 29 passed.
- Full verification: 538 passed, 1 skipped.

## 2026-07-14 - Will Runtime Receipt Visibility

- Diagnosed a false “no record” chat answer: the scheduled review had updated
  Will's protected runtime state and log, while chat searched the architectural
  `audit_log.md` because its file tools cannot read `memory_store`.
- Will now persists a bounded last-result receipt for every non-dry attempt,
  including completion status, reason, creation count, notification id, and
  timestamp. Failed and skipped attempts are recorded too.
- Added the read-only `will_status` tool so chat can inspect authoritative Will
  runtime state without exposing the protected runtime directory.
- Added explicit tool routing away from audit/git/workspace history for Will-run
  questions, and exposed the latest result in the Diagnostics summary.
- Backfilled the July 14 08:00 review receipt (`topic cooling down`) and restarted
  the server LaunchAgent successfully.
- Focused verification passed.
- Full verification passed with 1 skipped test.

## 2026-07-17 - Whole-Code Audit Privacy and Consistency Fixes

Phase: Post-audit hardening
Scope: Local data permissions, trusted prompt updates, media ingestion, and memory-tool model routing.
Intent: Fix all four findings from the root-to-runtime code audit.

Files changed:
- `config.py`
- `brain/utils.py`
- `brain/portable.py`
- `brain/toolkit/source_truth.py`
- `brain/toolkit/perspective_update.py`
- `brain/toolkit/memory_tools.py`
- `brain/toolkit/governance.py`
- `brain/documents.py`
- `tests/test_audit_round_two.py`
- `tests/test_image_ingest.py`

Behavior changed:
- The primary data directory is created and re-hardened to owner-only `0700`
  before Chroma opens it; filesystem-root configurations are refused.
- Automatic portable safety backups use a `0700` directory and explicit `0600`
  file creation instead of inheriting a permissive process umask.
- Source-of-Truth and Will-perspective edits hold one per-target thread/process
  lock across their complete read, validation, and atomic replacement, so two
  successful approvals cannot silently discard one another.
- CLI image/audio ingestion now keeps prior text chunks until the sensory write
  succeeds and removes only newly committed chunks when that write fails.
- An omitted memory-tool model remains Python `None` instead of becoming the
  literal model name `"None"` under selected-model embedding mode.
- Existing `memory_store` directories were repaired to `0700` and its 69 current
  files to `0600`; no stored data was deleted or rewritten.

Verification:
- Focused permission, concurrency, ingestion, perspective, source-of-truth,
  operability, and action-tool suites passed.
- Command: `./.venv/bin/python -m pytest`
- Result: `730 passed, 1 skipped`, with the existing TestClient deprecation warning.
- Command: `./.venv/bin/python -m compileall -q config.py brain`
- Result: passed.
- Command: `git diff --check`
- Result: passed.

Architectural drift check:
- Status: `none`
- Notes: Permission enforcement occurs at the existing config/store boundary;
  path locking is a shared utility; ingestion uses the web endpoint's established
  transaction ordering; model routing remains in the existing helpers.

Open follow-ups:
- Run the opt-in browser E2E suite when its environment is enabled.
- Replace the deprecated Starlette TestClient/httpx compatibility path when the
  dependency stack makes the migration available.
