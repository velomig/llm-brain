# Personal calibration files — setup for new users

Neural Map is calibrated by a set of markdown files that describe *you*: who you
are, what domains you work in, and how the assistant should behave. These files
are personal, so they are **gitignored** — the repository ships templates
instead. Copy each template you want into the repository root (dropping the
`.example` part of the name), then edit it to describe yourself.

| Template | Copy to | What it does |
| --- | --- | --- |
| `aboutme.example.md` | `aboutme.md` | Your profile: goals, context, preferences. Retrieved as memory context. |
| `source-of-truth.example.md` | `source-of-truth.md` | Every-turn calibration: identity, style, authority rules. |
| `domain.example.md` | e.g. `nursing.md`, `finance.md` | Per-domain calibration, injected only for requests in that domain. |
| `agency_protocol.example.md` | `agency_protocol.md` | How much initiative the assistant may take; your trigger tables. |
| `vault_location.example.md` | `LLM Brain/vault_location.md` | Points the assistant at your Obsidian vault (optional). |
| `launch.json.example` | `.claude/launch.json` | Claude Code dev-server launch config with your local paths (optional). |

All files are optional — the system runs without them, it just knows nothing
about you. File paths can be overridden with environment variables (see
`config.py`: `NEURAL_MAP_SOURCE_OF_TRUTH`, `NEURAL_MAP_NURSING`,
`NEURAL_MAP_FINANCE`, …).

## Keep your name out of web searches

Set your name(s) so the system refuses to include them in outbound web-search
queries and keyword extraction:

```bash
export NEURAL_MAP_OWNER_NAMES="yourfirstname,yourlastname"
```

Or, easier when running under launchd: put the same comma-separated list in a
file named `owner_names.txt` in the repository root (gitignored) — it is read
automatically whenever the env var is unset.

## What never belongs in git

Keep these local-only (already gitignored): `aboutme.md`, `finance.md`,
`nursing.md`, `source-of-truth.md`, `portfolio_scenarios.md`, `prioreval.md`,
`will_perspective_journal.md`, `LLM Brain/vault_location.md`,
`.claude/launch.json`, and the `memory_store/` database. If you fork this
repository, audit `git grep -i "<your name>"` before pushing.
