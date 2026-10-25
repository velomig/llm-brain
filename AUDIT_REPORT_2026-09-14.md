**Neural Map code audit — September 14, 2026**

Remediation update, September 15: all 13 findings below have been addressed. See the [fixes and validation report](<AUDIT_FIXES_2026-09-15.md>). The following findings and line references describe the original audited revision.

**Result: 13 actionable findings: 5 high priority (P1), 8 medium priority (P2).** Twelve findings have isolated, executable reproductions. One is supported by the installed dependency version and three upstream security advisories. No production code, dependency versions, or service configuration was changed.

Audited revision: `23442b742b00b938055dceda2ed5c44b66fd0718`. The working tree was clean before this audit. The first-party scope inventory contains 87 production/launcher files and 28,423 lines, including Python, browser JavaScript/HTML, shell scripts, AppleScript, and launchd plists. Tests and dependency declarations were also examined.

This is a risk-oriented source and behavior audit, with particular attention to authentication, approval enforcement, undo, storage replacement, retrieval degradation, background work, and startup. It is not a claim that every possible defect has been eliminated.

**Findings at a glance**

| ID | Priority | Finding | Evidence |
|---|---|---|---|
| A01 | P1 | Scheduled shell scripts remain writable through the generic file tool | Reproduced |
| A02 | P1 | Undoing a contact operation deletes later, unrelated contacts | Reproduced |
| A03 | P1 | File undo silently overwrites later edits | Reproduced |
| A05 | P1 | Concurrent uploads of the same filename can delete both versions | Reproduced |
| A09 | P1 | Malformed replacement imports silently clear valid memories | Reproduced |
| A04 | P2 | Memory undo loses pin status and provenance | Reproduced |
| A06 | P2 | Queue retention silently discards unfinished tasks | Reproduced |
| A07 | P2 | URL-encoded token parameter names evade access-log redaction | Reproduced |
| A08 | P2 | A missing embedding model breaks the advertised degraded storage path | Reproduced |
| A10 | P2 | An import can modify graph collections outside its destination | Reproduced |
| A11 | P2 | Installed pypdf has known resource-exhaustion vulnerabilities | Installed version + upstream advisories |
| A12 | P2 | Snapshot size checks happen after an unbounded file read | Reproduced |
| A13 | P2 | A zero dream limit still generates and stores a dream | Reproduced |

P1 means a defect should be addressed promptly because it can lose user data or defeat an explicit execution boundary. P2 means a material correctness, availability, or confidentiality issue with a more specific trigger. These are review priorities, not CVSS scores.

**A01 — P1: Protect the scheduled shell scripts from generic writes**

Location: [brain/toolkit/paths.py](<brain/toolkit/paths.py:82>); execution context: [backup_job.sh](<backup_job.sh:1>).

`reserved_code_path()` protects Python extensions and the web directory, but permits `.sh`, AppleScript, launchd plists, and other executable/configuration inputs. In particular, the repository documents `backup_job.sh` as a cron-scheduled script. With code writes disabled, a proposed overwrite of that script still enters the normal file-approval queue.

The reproduction uses a harmless temporary script and confirms the proposal is accepted. The existing `write_file` handler uses the same incomplete guard. One approved overwrite can therefore change what the scheduled job executes as the account owner. This still requires file-action approval; it is not an unauthenticated remote execution finding. The audit did not run a modified script or verify the machine's current cron schedule.

Fix: protect all executable and deployment/configuration inputs, preferably by allowing generic writes only in explicit data/output directories. Keep any code-edit capability separate from ordinary file approval.

**A02 — P1: Undo contact changes at the record level**

Location: [brain/toolkit/governance.py](<brain/toolkit/governance.py:667>).

Contact approvals snapshot the entire contacts JSON file. Undo restores that whole file, even when later approved operations have added unrelated records. No concurrency is required: approve creation of Alpha, approve creation of Beta, then undo Alpha. Beta disappears. If Alpha created the file, undo can unlink the entire file.

The same whole-file mechanism is used for private local calendar events. It also bypasses the normal state writer's lock and atomic replacement.

Fix: snapshot and undo the affected record only, preserving unrelated records under the appropriate thread/process locks. Check that the target record still matches the action's post-change state before reversing it.

**A03 — P1: Check freshness before restoring a file snapshot**

Location: [brain/toolkit/governance.py](<brain/toolkit/governance.py:260>).

Approval checks whether a preview is stale, but undo has no equivalent protection. After an approved write, a user can edit the same file again; undoing the earlier action restores its original snapshot and silently overwrites the newer edit. Undo of a newly created file similarly unlinks it even if its contents have changed. This also affects dedicated trusted-prompt updates, which share this restoration path.

The reproduction approves a temporary text-file change, writes a later user edit, and confirms undo reports success while restoring the old file.

Fix: retain a post-action fingerprint and compare it under a path lock before undo. Refuse stale reversals or present a new reviewed merge. Restore atomically with the appropriate private-file permissions.

**A05 — P1: Serialize replacement uploads by filename and destination**

Location: [server.py](<server.py:2329>).

Each upload first stores its new content-hashed source, then treats all other sources with the same filename as stale. Two overlapping uploads can both finish insertion, each identify the other as stale, and each delete the other's records. Both requests report successful ingestion, but the final store contains neither version.

A deterministic two-thread reproduction synchronizes these existing phases. Both handlers return one ingested chunk, while the final memory count is zero. The reproduction substitutes the backup writer only; backup creation does not serialize the replacement operation.

Fix: hold a shared lock keyed by destination collection and normalized filename across insertion, backup, and retirement, or use a versioned atomic replacement protocol. Apply equivalent serialization to CLI replacement paths.

**A09 — P1: Validate the entire import before clearing memory**

Location: [brain/portable.py](<brain/portable.py:100>) and [brain/portable.py](<brain/portable.py:203>).

The import validator checks only that `neural_map_export.memories` is a list. The next step silently discards entries that are not dictionaries or lack an ID. With `replace=True`, a malformed export containing a record without an ID is consequently treated as an empty replacement and clears the existing store successfully. No exception triggers rollback. Export version and declared count are also not enforced.

The reproduction starts with a valid memory, imports a one-record malformed payload, and observes that the original memory is gone. The CLI creates a pre-replacement backup, so recovery may be possible, but the restore operation still reports success after losing the active records.

Fix: validate supported format version, every record, ID uniqueness, text, metadata types, declared count when present, and graph references before any destructive operation. Distinguish a valid empty export from invalid records.

**A04 — P2: Preserve pin status and provenance during memory undo**

Location: [brain/toolkit/governance.py](<brain/toolkit/governance.py:296>).

`_restore_memory_records()` removes `pinned` and `source_type` from `extra` using `BASE_META_KEYS`, then never passes them back into the reconstructed memory. `MemoryUnit.metadata()` supplies `pinned=False` and `source_type="user_entered"`.

The reproduction starts with a pinned uploaded-document memory, approves unpinning, and undoes it. The result remains unpinned and is relabeled as user-entered. This weakens forgetting protection and makes provenance inaccurate.

Fix: round-trip all persisted metadata explicitly. For metadata-only undo, avoid deleting and re-embedding the memory altogether.

**A06 — P2: Keep unfinished tasks when enforcing queue capacity**

Location: [brain/tasks.py](<brain/tasks.py:67>).

Every queue write keeps only the newest `BACKGROUND_TASK_MAX` entries, regardless of status. At the default cap of 100, adding the 101st task can silently remove the oldest queued, running, or approval-waiting task. A removed running task can still perform work, while its completion record can no longer be saved.

The reproduction reduces the cap to two, enqueues three unfinished tasks, and finds that the first has disappeared even though all enqueue operations succeeded.

Fix: prune terminal tasks first. Preserve active work, and explicitly reject or backpressure new submissions if the active queue alone reaches capacity.

**A07 — P2: Redact decoded token parameter names**

Location: [server.py](<server.py:56>).

The token middleware uses parsed query parameters, so `?%74oken=...` is accepted as `?token=...`. The access-log filter recognizes only the literal substring `token=`, leaving the encoded spelling and its secret untouched.

The reproduction confirms successful bootstrap with a synthetic token and shows that the filter retains that token in a Uvicorn-shaped log record. Impact requires a token-protected deployment and a request using the alternate spelling. No real credential was used or exposed during testing.

Fix: avoid logging authentication query strings, or parse and redact query parameters after decoding names. Add cases for encoded names, duplicate parameters, and encoded values.

**A08 — P2: Degrade safely when the embedding model is missing**

Location: [brain/embeddings.py](<brain/embeddings.py:21>); retrieval caller: [brain/retriever.py](<brain/retriever.py:509>).

`EmbeddingUnavailable` is raised only when an exception message says embeddings are unsupported. A normal Ollama “model not found” response bypasses that classification, as do connection failures. Callers therefore never enter their text-only storage or keyword-retrieval fallback.

The reproduction makes the embedding client return a synthetic 404. Adding a manual memory returns HTTP 503 and stores nothing. In ordinary chat, failure of the embedding step can likewise prevent using an otherwise available chat model.

Fix: classify missing embedding models and selected transport/availability failures explicitly, log degraded operation, and retain actionable diagnostics. Keep malformed embedding responses and database failures distinct.

**A10 — P2: Reject graph rows outside the import's memory set**

Location: [brain/portable.py](<brain/portable.py:255>).

Imported edges whose fact IDs are absent from the restored memories retain their supplied collection. `graph.import_edges()` then upserts them into that collection. A payload with zero memories can overwrite a relation in an unrelated model's graph.

The reproduction creates an edge in a synthetic foreign collection and imports an edge with the same triple but another fact ID. The foreign edge is replaced. Those foreign writes are also outside the normal destination-scoped rollback snapshots.

Fix: require every imported edge to reference an imported memory, and rewrite every accepted edge into its validated destination. Reject unsupported or inconsistent references before mutation.

**A11 — P2: Update pypdf and bound PDF extraction**

Locations: [requirements.txt](<requirements.txt:10>) and [brain/documents.py](<brain/documents.py:69>).

The installed version is `pypdf 6.14.2`, which is also allowed by the dependency lower bound. Upstream reports three resource-exhaustion advisories affecting versions below 6.15.0, with fixes in 6.15.0:

- CVE-2026-82398: excessive runtime processing long non-whitespace input. [Upstream advisory](https://github.com/py-pdf/pypdf/security/advisories/GHSA-fc8x-2rww-xw9m).
- CVE-2026-71852: excessive runtime/memory for large CID font-width ranges during operations including text extraction. [Upstream advisory](https://github.com/py-pdf/pypdf/security/advisories/GHSA-fwg2-594c-jp42).
- CVE-2026-71870: excessive memory for large ToUnicode font streams during operations including text extraction. [Upstream advisory](https://github.com/py-pdf/pypdf/security/advisories/GHSA-fp3f-mc75-235c).

Neural Map extracts uploaded PDFs inside the server process, without a per-extraction timeout or memory budget. The upload byte limit does not bound parser expansion. This makes the vulnerable functionality reachable when a user imports an affected PDF; the authentication gates still apply.

Fix: raise the minimum to at least 6.15.0, select and verify a maintained compatible release, and isolate extraction with resource limits. Vulnerability payloads were not executed against the running application.

**A12 — P2: Limit snapshot reads before allocating the file**

Location: [brain/toolkit/governance.py](<brain/toolkit/governance.py:252>).

The preview path caps some diff inputs, but action snapshots still call `read_bytes()` before checking `MAX_TOOL_UNDO_BYTES`. Queuing a write to a large allowed file therefore reads the entire file into the server process before deciding it is too large to snapshot. `_json_file_snapshot()` has the same ordering.

The reproduction uses a small synthetic file above a reduced limit and records the read occurring before refusal. It does not exhaust memory.

Fix: check file size first and then read at most the limit plus one byte to handle growth races. Apply the same rule to JSON snapshots.

**A13 — P2: Honor a zero dream-generation limit**

Location: [brain/sleep.py](<brain/sleep.py:118>).

`_dream_pairs()` adds a pair before checking whether the pair count has reached `max_dreams`. With a zero limit, one pair is still selected. `sleep.run(max_dreams=0)` consequently makes a generation call and stores a speculative memory when two eligible memories exist.

The reproduction supplies two synthetic eligible memories and a fake generation function. The result reports one dream and confirms that generation was invoked.

Fix: return an empty pair list immediately when the requested limit is zero or negative. Verify both preview and execution paths.

**Validation performed**

| Check | Result |
|---|---|
| Existing full pytest suite | 793 passed, 1 skipped; 11.87 seconds |
| Explicit Chromium browser smoke test | 1 passed; 2.55 seconds |
| New audit reproductions | 12 strict expected failures confirming the defects |
| Reproductions with xfail disabled | Each failed at its intended safety assertion |
| Installed dependency consistency (`pip check`) | No broken requirements |
| Python syntax | 152 tracked Python files parsed successfully |
| Browser JavaScript syntax | `node --check` passed |
| Shell syntax | Checked with the scripts' shell interpreters |

The browser test initially could not bind its test port inside the sandbox. It subsequently passed with execution approval, using the test fixture's temporary database, port 8765, and an intentionally unavailable Ollama address. This verifies panel opening when Ollama is offline; it does not verify live model generation or iOS Safari.

The original reproductions have since been converted into normally collected regression tests: [tests/test_audit_round_five.py](<tests/test_audit_round_five.py>). The validation table above records the original audit, before remediation. Run the current regression tests with:

```sh
./.venv/bin/python -m pytest tests/test_audit_round_five.py
```

The current tests require the safe behavior to pass. All writes use the temporary database or temporary files; no live email/calendar writes or scheduled script execution are involved.

**What the existing protections cover**

Source review and the passing regression suite support the presence of Host/Origin/Fetch-Metadata checks, token enforcement beyond loopback, approval claiming, stale approval previews, hidden-path and symlink checks, tool capability checks, outbound query controls, bounded model output, and rollback attempts during portable restore. Browser dynamic content is generally rendered through `textContent`; the inspected `innerHTML` uses clear containers.

Several July audit fixes are present, including explicit Ollama timeouts, restore cleanup inside the rollback boundary, authentication before request-body buffering, hidden-file restrictions, calendar attendee allowlisting, and reflection locking. A01, A07, and A12 identify incomplete coverage in related protections rather than assuming those earlier fixes were absent.

**Coverage and remaining limits**

| Area | Review coverage and limits |
|---|---|
| API/authentication | Middleware order, inputs, bootstrap, mutations, cancellation, and error handling; exercised through the suite and targeted reproductions |
| File/tool execution | Path restrictions, Python sandbox construction, approval/undo flows, email/calendar integration code; no live external writes |
| Memory/graph | Add/update/delete, provenance, replacement, import/export, rollback, retrieval fallback, pruning and sleep |
| Cognitive/background layers | State persistence and lock patterns, pipeline integration, prompt construction, task retention, and existing regression behavior; not a measurement of model reasoning quality |
| Browser UI | Storage/history, request flow, rendering sinks, syntax, and Chromium panel smoke test; no physical iPhone/iPad or Safari run |
| Launchers/maintenance | Source and shell syntax review; no service restart, migration, cron changes, or destructive maintenance |
| Dependencies | Installed consistency and targeted upstream advisory review; not an exhaustive transitive vulnerability or supply-chain scan |

Additional concerns remain worth follow-up but are not included in the 13 confirmed findings: prompt budgeting is character-based and does not account for the full model-specific conversation/tool context; streaming cancellation checks occur between received chunks; some network responses and PDF extraction remain unbounded; pending-action history grows without a retention policy; and “memory disconnected” still allows background capture and durable state updates. The last item needs an explicit product decision about whether the mode is only retrieval-disabled or is intended to prevent persistence.

Live memory contents, production credential validity, backup recoverability on another machine, filesystem power-loss behavior, and real OAuth/SMTP/Calendar transactions were not tested. No connection between these findings and the earlier transient OS-upgrade issue has been established.

**Recommended implementation order**

1. Close the executable-file gap and fix undo behavior, including metadata preservation: A01–A04.
2. Serialize replacements and validate imports before mutation: A05, A09, A10.
3. Update the vulnerable PDF dependency and bound extraction/snapshot work: A11, A12.
4. Fix queue retention, token redaction, embedding degradation, and the zero-limit behavior: A06–A08, A13.

After each fix, convert its reproduction to an ordinary passing regression test. Run the existing suite and the explicit browser test before applying the changes to the running service.
