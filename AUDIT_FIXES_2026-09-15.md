**Neural Map audit remediation — September 15, 2026**

All 13 findings from the [original audit](<AUDIT_REPORT_2026-09-14.md>) are addressed in the working tree. The project virtual environment now has pypdf 6.18.1 installed. Changes have not been committed, and the running application has not been restarted.

| Finding | Implemented fix |
|---|---|
| A01 — executable writes | The generic file tool blocks project scripts, launchers, app bundles, deployment/configuration inputs, executable files, and files with a shebang. Explicit code-write opt-in remains available. |
| A02 — contact/calendar undo | Snapshots contain the affected record. Undo preserves other records, rejects newer changes to the target, and uses the same thread/process lock and atomic writer as normal updates. |
| A03 — stale file undo | Approval records a SHA-256 fingerprint after writing. Undo verifies it under the shared target lock, restores atomically with private permissions, and serializes duplicate undo requests. |
| A04 — memory metadata undo | Undo restores pin state, provenance, and other metadata, removes fields introduced by the action, reuses saved embeddings, and updates an existing record without deleting it first. |
| A05 — concurrent replacement | Upload replacement holds a lock keyed by destination collection and filename through insertion, backup, and retirement. Direct document and CLI replacement share source locks, including nested media ingestion. |
| A06 — task retention | Retention prunes completed history first, preserves unfinished work even when capacity is reduced, and explicitly rejects new work when active tasks fill the queue. |
| A07 — token logs | Access-log redaction decodes parameter names and handles duplicates, percent encoding, and mapping-style logging arguments. |
| A08 — embedding outages | Missing models, connection failures, transport timeouts, and temporary upstream unavailability enter the existing text-only fallback. Validation errors still propagate. |
| A09 — unsafe restore validation | Complete archive validation runs before embedding or mutation: versions, count, every record, unique IDs, nonempty text, metadata types, and graph references. Malformed archives cannot become empty replacement imports. |
| A10 — foreign graph writes | Every imported edge must reference an imported memory and is assigned to its actual destination collection. Foreign or dangling references are rejected before writes. |
| A11 — PDF dependency/resources | Requirements enforce pypdf >=6.15.0; installed version is 6.18.1. Parsing runs in a separate worker with input, wall-time, CPU, memory, page, and output limits. One parser runs per database, and failed workers are terminated and reaped. |
| A12 — unbounded snapshots | File and legacy JSON snapshot helpers check size before reading and read at most the cap plus one byte, including growth-race protection. Preview reads are bounded too. |
| A13 — zero dream limit | Zero or negative dream limits select no pairs in either preview or execution. |

The PDF minimum follows the upstream fixes for [parser runtime](https://github.com/py-pdf/pypdf/security/advisories/GHSA-fc8x-2rww-xw9m), [CID font width ranges](https://github.com/py-pdf/pypdf/security/advisories/GHSA-fwg2-594c-jp42), and [ToUnicode memory usage](https://github.com/py-pdf/pypdf/security/advisories/GHSA-fp3f-mc75-235c). PDF budgets and their environment settings are documented in the [README](<README.md:417>).

| Validation | Result |
|---|---|
| Full test suite | 866 passed, 1 skipped, 23.51 seconds |
| Separate Chromium browser smoke test | 1 passed, 1.75 seconds; covers the default-skipped browser test |
| Added regression cases | 73, including the original reproductions, real PDFs, worker termination, concurrent undo/replacement, and process-lock exclusion |
| Dependency consistency | `pip check`: no broken requirements |
| Python syntax | 156 files parsed successfully |
| JavaScript syntax and patch whitespace | `node --check web/app.js` and `git diff --check` passed |

Regression coverage is in [test_audit_round_five.py](<tests/test_audit_round_five.py>) and [test_pdf_limits.py](<tests/test_pdf_limits.py>). The portable-import test formerly requiring foreign graph writes now requires rejection. Markdown link validation also recognizes angle-bracket paths and Codex line-number suffixes used in these reports.

Reproduce the successful checks:

```sh
OLLAMA_HOST=http://127.0.0.1:9 ./.venv/bin/python -m pytest -ra --tb=short
NEURAL_MAP_RUN_E2E=1 OLLAMA_HOST=http://127.0.0.1:9 ./.venv/bin/python -m pytest tests/test_browser_e2e.py -ra --tb=short
./.venv/bin/python -m pip check
```

Tests use temporary databases and synthetic records. The final runs explicitly disable access to the live Ollama endpoint. The desktop sandbox blocks PDF process monitoring and browser port binding; those checks passed with the required execution access. Dependency deprecation warnings remain for Starlette's test client and Uvicorn's WebSocket integration. The browser test uses Chromium; it does not validate iOS Safari or live model generation.

Reopen Neural Map using its macOS launcher to activate the backend changes; its fingerprint check detects the updated Python code and restarts a stale backend. Legacy file/contact/calendar actions lacking a trustworthy post-action snapshot now refuse automatic undo. Archives without a version field remain accepted as the early version-1 format when their contents validate. These fixes prevent the audited failures going forward; they do not reconstruct any records lost before remediation.
