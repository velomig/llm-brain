# Neural Map Development Roadmap

Development history and drift checks live in `audit_log.md`. Use `todo.md` for future work, not completed-change narratives.

## 🛠️ Phase 1: Robustness Sprint (Quick Wins)

## 🛡️ Phase 2: Hardening Sprint (Security & Privacy)

## 🚀 Phase 3: Optimization Sprint (Performance & UX)
- [ ] **File I/O Efficiency:** Implement `mtime`-cached reads for `evolution_state.json` to reduce lock contention.
- [ ] **Alert Fatigue Reduction:** Tune `_monitor_unsourced_context` and integrity check warnings to minimize noise.

## ✅ Done
- [x] **Consistent Env Parsing:** `config._env_int`/`_env_float` now parse every numeric env var. (ec0c5ec)
- [x] **Code Refactor:** `_safe_json` consolidated into `brain/utils.py`. (ec0c5ec)
- [x] **Council Latency Tuning:** study-context suppressor + tighter decision phrasing; convene reasons on the trace. (ba41082)
- [x] **Prompt Budget:** per-block size accounting on every turn trace; self-audit loop and agency docs no longer injected where unused (injection-mode prompts −46%). (ff84729)
- [x] **Taint Guard:** pending actions record untrusted context; tainted external sends require Actions-panel approval. (20f916d)
- [x] **Web UI Split:** inline script extracted to `web/app.js`, served no-cache. 
- [x] **Backup Privacy:** Added private backup permissions in `backup_job.sh` and locked down the existing `/Users/Shared/neuralmap-backups` mirror.
- [x] **Stop Race:** Added pending cancellation registration so Stop can abort requests that have not reached `/api/chat` yet.
- [x] **Symlink/Stat Resilience:** Made `list_files` tolerate bad or disappearing child entries.
- [x] **SMTP Security:** Updated `send_email` to use a default SSL context for STARTTLS.
- [x] **Auth Hardening:** Limited query-token auth to one-time GET bootstrap with redirect to a clean URL.
- [x] **AGI-Like Cognitive State:** Added a persistent goal stack, world model, and belief ledger with prompt injection and API endpoints.
- [x] **Phase 3 Audit Log:** Added `audit_log.md` with a standard Audit Summary format and Architectural Drift criteria.
- [x] **Normalize `num_ctx`** in `llm.py` to prevent KV cache exhaustion while preserving reasoning capacity. (db357cb)
- [x] **Dynamic `_TOOLS_NOTE`** in `pipeline.py` to sync tool availability with `guard_web` state. (db357cb)
- [x] **Sandbox Resource Limits:** memory cap + write-site stdout cap for the Python sandbox — rlimits where honored, parent-side RSS watch on macOS (RLIMIT_AS/DATA rejected there).
