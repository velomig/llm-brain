# LLM Brain startup diagnosis — 2026-09-15

The remaining delay correlates with APFS/TRIM maintenance on ExternalDrive (Samsung SSD 990 EVO Plus 4TB, firmware 2B2QKXG7), not the desktop launcher or launchd scheduling.

## Observed startup timing

Latest boot: 20:54:31. External volume mounted: 20:54:45. Server process launched: 20:54:56. External SSD TRIM completed: 20:56:27.088 after 101.694 seconds. Server readiness output last modified: 20:56:27.484, less than 0.4 seconds later. Server is confirmed running with Interactive process classification, so the prior scheduling change was active but did not eliminate this delay.

Earlier external TRIM durations from the same evening: 105.821, 105.852, 105.183, and 103.828 seconds. Internal SSD maintenance on the latest boot took 0.587 seconds. The external Python libraries are loaded across the long interval, with SQLite loading immediately after external TRIM completes.

Ollama also uses the external disk. Its latest server process began at 20:55:00 and began listening at 20:55:59 after scanning 5,073 blobs. Moving just the LLM Brain runtime would therefore not eliminate every delay in first-chat readiness.

## Prepared migration scope (not applied)

- Internal LLM Brain code, Python runtime/dependencies, memory database, and supporting state: approximately 1.5 GiB.
- Active local models: gemma4:26b, gemma4:e4b, gemma4:latest (shared helper weights), and nomic-embed-text:latest: 25.96 GiB of unique blobs, 10 blobs total.
- Approximately 637 GiB currently available on the internal SSD.
- Retain external originals. Do not disable TRIM or alter filesystem maintenance.
- Stop server and vault writer before final database copy; verify counts and state before switching startup paths. Avoid two writable copies diverging.
- Keep remaining model library on the external drive. Decide how to expose those models before changing Ollama storage; copying only active models must not silently hide other models.
- Keep project and launch configuration paths consistent, including the desktop launcher, document watcher, tool roots, and user-maintained state files.
- Verify readiness from a real reboot after cutover.

Migration awaits the user's storage preference. No files were moved during this diagnosis. Existing desktop-launcher correction and backed-up scheduling adjustments remain installed.

Samsung's firmware page currently lists the same installed 2B2QKXG7 version: https://semiconductor.samsung.com/consumer-storage/support/tools/
