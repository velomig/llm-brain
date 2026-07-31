# LLM Brain — Full Code Audit Report

**Date:** 2026-07-30
**Scope:** `server.py`, `config.py`, root-level scripts, `web/`, the full `brain/` package (core modules + `brain/toolkit/`), and the `tests/` suite.
**Method:** Three parallel deep-read audits (server/root, brain core, toolkit+tests) plus a full test-suite run.

---

## Executive Summary

| Severity | Count |
|---|---|
| CRITICAL | 0 |
| HIGH | 2 |
| MEDIUM | 12 |
| LOW | ~30 |

**Test suite:** 758 collected → **757 passed, 1 skipped** (Playwright e2e, skipped by design), **0 failed**. No vacuous tests found.

**Bottom line:** This is an unusually well-hardened personal project. There are no hardcoded secrets, no `eval`/`exec`/`shell=True`/unsafe deserialization anywhere, and the approval/taint/sandbox machinery is real and tested — not security theater. The residual risk is concentrated in two places:

1. **Self-modification:** one user approval on `write_file` can rewrite the enforcement code itself.
2. **Data loss, not compromise:** the top correctness bugs are state-corruption paths (restore rollback, unlocked concurrent reflect runs) involving the memory store itself.

Reliability gaps mostly fail *silently* (hangs, silently truncated prompts, silently deleted reply sentences) rather than loudly.

---

## HIGH Severity

### H1. `brain/toolkit/files.py:79` + `brain/toolkit/paths.py:49` — `write_file` can overwrite the sandbox's own enforcement code

- **What:** `resolve_tool_path` allows any file under `TOOL_ALLOWED_ROOTS` (default: entire `BASE_DIR`), and `write_file` only reserves the trusted-prompt markdown files. Nothing stops an approved `write_file` (mode `overwrite`) from replacing `brain/toolkit/registry.py`, `governance.py`, `config.py`, or `server.py`.
- **Impact:** The capability manifest, approval gate, and path policy are all self-hosting. A prompt-injection payload (from an ingested doc or web result) can propose a plausible-looking "bugfix" diff; a single user approval turns it into persistent arbitrary code execution at next import/restart — bypassing every other control.
- **Fix:** Extend the reserved-path set beyond prompt files — refuse `write_file` to `*.py` under `BASE_DIR`, `config.py`, `server.py`, and `.claude/`. If assistant-edits-code is a wanted workflow, gate it behind a separate config flag that defaults **off**.

### H2. `brain/portable.py:218–238` — Restore's destructive clears sit outside the rollback `try` block

- **What:** In `import_data`, the replace path runs `store.clear_all(model)` and `graph.clear_collections(...)`, and the merge path runs `store.delete_all(...)` / `graph.delete_fact_ids(...)`, *before* the `try:` that guards rollback. If a destructive call raises after `clear_all` succeeded, the exception propagates without entering the `except:` handler, so the pre-import snapshot is never restored.
- **Impact:** A failed restore (e.g., SQLite lock error from a concurrent cron job) can permanently wipe the entire memory store for the model, with the rollback data sitting unused in memory.
- **Fix:** Move the destructive `clear_all`/`clear_collections`/`delete_all`/`delete_fact_ids` calls **inside** the `try:` block so any failure after the snapshot triggers `_restore_snapshot` + edge re-import.

---

## MEDIUM Severity

### Security / privacy

**M1. `server.py:324–334` — Auth token accepted via `?token=` query param is written to access logs.**
The token middleware accepts the shared secret as a query parameter for browser bootstrap, but uvicorn runs with `access_log=True`, so `GET /?token=<secret>` is logged in plaintext. On a LAN-exposed instance, the only auth secret leaks into logs. **Fix:** prefer a bootstrap POST or one-time code exchange; if the query-param flow stays, run uvicorn with `access_log=False` or strip query strings via a log filter.

**M2. `brain/toolkit/paths.py:28` + `files.py:32,63` — File tools read/list dotfiles; only search excludes them.**
`read_file`/`list_files` apply only `TOOL_BLOCKED_PATH_PARTS` (`.git`, `.venv`, `__pycache__`, `memory_store`). `.claude/settings.local.json` is LLM-readable today, and any future `.env` at the project root would be too. Combined with any exfil channel, this leaks credentials. **Fix:** apply the same dot-part exclusion to read/list, or deny-list `.env*`, `*.pem`, `*secret*`, `*token*`, `.claude`.

**M3. `brain/toolkit/google_calendar.py:416–420` — Calendar attendees are an un-allowlisted outbound email channel.**
`create_google_calendar_event` with `attendees` calls `sendUpdates="all"`, emailing invitations to any syntactically valid address — no `EMAIL_ALLOWED_RECIPIENTS`-style allowlist like the email tool has. One approval becomes an injection-driven exfil path. **Fix:** reuse `EMAIL_ALLOWED_RECIPIENTS` (or a calendar-specific list) for attendee validation.

**M4. `brain/metacognition.py:86–101` — Self-authored "wisdom" notes commit to memory without approval.**
`record_lesson` is approval-gated because lessons inject into every prompt, but `metacognition.supervise` stores LLM-generated self-correction rules directly into memory with no approval, and `_recent_wisdom` feeds them back into later prompts. A single injection-influenced turn can persist a false behavioral rule. **Fix:** route `_store_note` through the pending/approval flow, or cap stored notes and mark them unreviewed.

### Reliability

**M5. `brain/llm.py:18,264,324` + `brain/embeddings.py:14` + `brain/vision.py:18` — No timeout on any Ollama call.**
All `ollama.Client(host=...)` instances are created without a timeout; httpx treats `None` as "no timeout". A wedged Ollama process freezes a chat turn or background layer (consolidate, will, reflect) indefinitely. **Fix:** pass a finite timeout (e.g., `timeout=config.OLLAMA_TIMEOUT`, several minutes) and log a fallback on `TimeoutError`.

**M6. `brain/pipeline.py:1513–1550` — System prompt has no total-size enforcement; silent context overflow.**
`build_system` concatenates ~15 blocks; only an observability event records the total. `llm.normalize_num_ctx` clamps the window, not the prompt, so Ollama silently truncates an oversized system prompt — potentially cutting out the authority-order preamble or untrusted-content guardrails. **Fix:** add a hard budget, shrink lowest-priority blocks in a defined order, and emit a trace event when trimming occurs.

**M7. `brain/grounding.py:67–68,529–544,663–668` — Final-claim guard silently deletes legitimate sentences.**
Regexes like `I (can )?(see|hear|...)` and `I (have )?(ran|sent|created|updated|...)` match figurative phrasing ("I can see how that feels", "I updated the plan earlier"). Classified sentences with no matching evidence get disposition `withhold` and are **deleted from the user-visible reply** without notice. **Fix:** downgrade first-pass regex hits to `qualify` (hedge) instead of `withhold`, or require stronger syntactic evidence before deletion.

**M8. `brain/reflect.py:184–298` — `reflect()` has no cross-process lock; concurrent runs can delete each other's fresh output.**
Run B's `_generated_ids()` snapshot (taken after run A stored) includes run A's just-written records, which run B then deletes. Generated insights/reminders can be silently wiped when web UI and cron overlap. **Fix:** wrap `reflect()` in a dedicated `file_lock` + `threading.Lock`, as other layers do.

**M9. `brain/evolution.py:549–553` — `proposals` list in `evolution_state.json` grows unboundedly.**
Snapshots are trimmed to 50 but proposals never are; every `state()` call (per retrieval hyperparam lookup) reads/parses the whole file under a flock — an O(file size) tax on every chat turn. **Fix:** cap proposals (last 50–100) on write, same pattern as snapshots.

**M10. `brain/toolkit/governance.py:198` — Unbounded file read + difflib in `write_file` preview.**
`_file_text()` reads with no size cap and `_preview_action` runs a worst-case quadratic `difflib.unified_diff` over full content. A `write_file` against a 100 MB file spikes memory/CPU at approval-queueing time. **Fix:** cap preview reads and mark large files "too large to diff" (reuse the existing `preview_truncated` refusal path).

**M11. `brain/toolkit/governance.py:133–150` — Audit log stores full args and rewrites the whole file per event.**
Complete args (email bodies to 10 KB, write_file content to 50 KB) are persisted, up to 500 events, each append a full read-modify-write. The audit file can reach tens of MB with sensitive content accumulating indefinitely. **Fix:** truncate stored args/results in audit entries, or rotate as append-only JSONL.

**M12. `server.py:79–122` — Body-size middleware buffers the full request body before auth.**
The outermost middleware buffers up to `MAX_REQUEST_BODY_BYTES` (~33 MB) per request before host/origin/token gates run. An unauthenticated LAN client can hold ~33 MB of RAM per concurrent connection. **Fix:** discard instead of retain chunks once over the limit, or move the token check ahead of body buffering.

---

## LOW Severity (grouped)

**Auth/session**
- `server.py:330–344` — `nm_token` cookie: no `Secure` flag, 1-year lifetime, re-set on every response. Shorten `max_age`; consider local TLS so `Secure` can be set.
- `server.py:704–711` — Colloquial affirmations starting with "no" ("no worries, go ahead") reject pending actions. Fails closed, but add an exception list.
- `config.py:560` — `MAX_REQUEST_BODY_BYTES` uses raw `int()` instead of the `_env_int` guard; a malformed env var crashes the app at import.

**Correctness**
- `server.py:2269–2277` — Media-ingest rollback skips graph-edge and partial-sensory cleanup that the CLI path performs (self-healing today, brittle if ingest changes).
- `ingest_md.py:62` — Watch-loop TOCTOU crash when a file vanishes between `exists()` and `stat()`; kills the daemon. Wrap in `try/except OSError`.
- `chat.py:37,92–93` — CLI conversation history grows unboundedly per session (only the slice passed to the pipeline is bounded).
- `brain/utils.py:115–116` — `safe_json` fence stripping via `strip("`")` is lossy on edge backticks; use a regex to strip a single fence.
- `brain/utils.py:16` — `_PATH_LOCKS` dict grows unboundedly (one lock per path, never evicted).
- `brain/memory.py:129` — `recency_score` divides by `halflife_days` with no zero guard.
- `brain/toolkit/local_state.py:52` — Uncaught `ValueError` on non-numeric `importance` surfaces as an unhandled exception.
- `brain/execution.py:54–55` — Post-yield cancel check discards completed side effects; the turn reports cancellation after the effect happened.
- `brain/observability.py:79–82` — Trace ContextVars set without tokens; nested `start()` clobbers the outer trace id.
- `brain/tools.py:30` — `TOOL_SPECS` frozen at import; later `set_permission` changes not reflected.
- `brain/will.py:415–419` — No shape check on `embed_documents` result; `zip` silently truncates and inflates novelty scores (arbiter.py checks, will.py doesn't).
- `brain/toolkit/pending.py:69–86` — Stale-claim recovery can transiently mislabel a still-running approval as `failed`.
- `brain/toolkit/email.py:16` — Email subject has no length cap (body is capped).

**Performance / scale**
- `server.py:977–1021` — Several endpoints do repeated full-store scans per request (3–5× `list_records_all()` per call). Hoist one snapshot per request.
- `brain/will.py:795–911` — Embedding network calls made while holding the cross-process flock; a slow embedding blocks all Will state reads.
- `brain/websearch.py:388+` and `brain/market.py:219` — Unbounded `response.read()`; cap at a few MB.
- `brain/documents.py:69–78` — `extract_pdf_text` has no page/size/chunk cap; a huge PDF pins RAM and the embedding backend.
- `brain/cognitive_state.py` — goals/world/beliefs lists uncapped on disk (prompt injection is capped).
- `brain/store.py:24` — Chroma `PersistentClient` created at import time; an unavailable external volume crashes the whole package import. Make it lazy like `graph._db()`.
- `brain/graph.py:30,336–359` — Single shared SQLite connection, no busy-timeout/WAL; multi-process writers can hit "database is locked" and silently drop edges. Use `timeout=15` + `PRAGMA journal_mode=WAL`.

**Retrieval / pipeline behavior**
- `brain/pipeline.py:1669` — A failed `web_search` still burns the one-per-turn allowance (incremented before dispatch).
- `brain/retriever.py:346–366` — `_categories()` reads only the vector store; text-only records written during embedding outages never get balanced-recall pulls.
- `brain/self_model.py:250–262` — Open promises are regex-scraped from the assistant's own reply into persistent state — a minor prompt-injection amplifier. Filter high-risk verbs.
- `brain/summarize.py:51` — Generated summaries stored at importance=5 (max) and can dominate retrieval. Use 3–4 with a provenance flag.
- `brain/chunking.py:14` — Frontmatter regex requires LF-only; CRLF frontmatter isn't stripped.

**Web**
- `web/index.html` — No Content-Security-Policy or `X-Content-Type-Options` headers. No XSS sink found today (all dynamic rendering uses `textContent`), so this is defense-in-depth only.

---

## Test Suite

- **Result:** 757 passed / 1 skipped / 0 failed (exit 0, ~1 min, no hangs).
- **Skipped:** `test_browser_e2e.py` — requires Playwright + `NEURAL_MAP_RUN_E2E=1` (by design).
- **Strong coverage:** toolkit security is genuinely well tested — sandbox escape/timeout/memory probes, path confinement incl. symlink escape, approval race single-shot, stale-preview, taint flow, truncation refusal (`test_action_tools.py`, `test_hardening_fixes.py`, `test_tool_policy_contract.py`). The sandbox itself survived 10 live escape probes during the audit (AST whitelist holds; `10**10**8` killed by wall-clock timeout).
- **Coverage gaps:**
  - No dedicated `test_pipeline.py` — the biggest module (~2400 lines) is only exercised indirectly.
  - `embeddings.py` / `vision.py` have no unit tests with a mocked Ollama client.
  - Many server routes are covered only via frontend-literal contract checks, not behavior tests.
  - `memory.py` decay math, `chunking.py` CRLF/nested-fence edge cases, `observability.py` nested traces, and `execution.py` cancel-after-completion are untested.

---

## Architecture Notes (non-scored)

- `pipeline.answer()` (~450 lines) and `_single_pass_answer()` (~280 lines) are god functions.
- The threading.Lock + flock + tmp-replace state-file scaffold is duplicated nearly verbatim across ~12 modules — deserves one shared helper.
- Several modules contain near-identical `_clamp`/`_tokens`/`_repair_state` implementations.
- ~26 cognitive layers each maintain per-turn file-backed state and feed one giant system prompt; the pipeline is at its complexity ceiling — further layers should consolidate rather than add.

## Positive Notes (verified, not assumed)

- Layered Host-allowlist (anti-DNS-rebinding), Origin same-origin checks, and Fetch-Metadata gates on the server; import-time refusal to bind non-loopback without a token.
- No dangerous primitives anywhere: no `eval`/`exec`/`subprocess(shell=True)`/`pickle`/`yaml.load`.
- Strict Pydantic models with `extra="forbid"` and tight length caps across all endpoints.
- Outbound exfiltration deliberately guarded: web_search user-derivation allowlist, ticker allowlist, taint tracking into approval-gated tools.
- Email defaults to draft-only; real sends require env opt-in + recipient allowlist. SMTP password from env only.
- Trusted prompt files are reserved from generic `write_file` and editable only via diff-previewed, approval-gated updaters.
- Private files written atomically with 0600; credentials stored in 0700 directories.
- Approval flow is solid: atomic `claim()` (flock + thread lock), stale-preview re-approval, preview-truncation refusal, taint gating of blind chat approval.

---

## Recommended Fix Order

1. **H1** — Block `write_file` from touching `*.py`/`config.py`/enforcement code (small change, closes the biggest hole).
2. **H2** — Move portable-restore destructive clears inside the rollback `try` (prevents catastrophic data loss).
3. **M5** — Add Ollama client timeouts (one-line-per-call-site, kills the worst hang class).
4. **M8** — Add a file lock around `reflect()` (prevents silent loss of generated insights).
5. **M1** — Stop logging the auth token (uvicorn `access_log=False` or query stripping).
6. **M6/M7** — Prompt-size budget + soften the claim guard from `withhold` to `qualify` (fix silent answer degradation).
7. Then the consistency batch: M2 (dotfiles), M3 (calendar attendees), M4 (metacognition approval), M9–M11 (unbounded growth), M12 (body buffering).
