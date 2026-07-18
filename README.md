# 🧠 Neural Map — local long-term memory for an LLM

A local-first, private "second brain": every chat turn is embedded and stored in
a vector database, and every new prompt is answered with the most relevant past
memories injected as context (RAG over your own conversations).

```
prompt ─▶ embed ─▶ vector store ─▶ retrieve (hybrid + recency) ─▶ inject ─▶ LLM ─▶ capture ─┐
   ▲                                                                                          │
   └──────────────────────────────────────────────────────────────────────────────────────-─┘
```

Everything runs on your machine via **Ollama** — no API keys, and your memories,
documents, and chat history are stored and processed only locally. Two optional
conveniences DO make outbound requests in the default configuration: web search
(Google News RSS / Yahoo Search) and live stock quotes (Yahoo Finance), both
guarded so queries are derived from your current typed message and never from
stored memory or uploaded documents. For a fully offline box set
`NEURAL_MAP_WEB_SEARCH_MODE=off` and `NEURAL_MAP_MARKET_QUOTES=0`; email drafts
locally unless SMTP is explicitly configured. Google Calendar is disabled until
you explicitly complete its OAuth connection; once connected, reads are live and
every create/update/delete remains human-approval-gated.

## Stack (Option A: Local Minimalist)

| Role            | Tool                                    |
| --------------- | --------------------------------------- |
| Vector DB       | **ChromaDB** (persisted to `memory_store/`) |
| Embedding model | Dedicated `nomic-embed-text:latest` by default |
| LLM             | **Ollama** (default `gemma4:latest`)    |
| Keyword search  | **BM25** (`rank-bm25`)                  |

## Setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
# make sure Ollama is running and your chosen model is pulled:
ollama list            # pick any chat model you like
```

## Personalize it

The assistant is calibrated by personal markdown files (`aboutme.md`,
`source-of-truth.md`, domain files like `finance.md`) that are **gitignored**
because they contain your private details. See [examples/](examples/) for
templates and instructions on creating your own, and set
`NEURAL_MAP_OWNER_NAMES="yourfirstname,yourlastname"` in your environment so
the system keeps your name out of outbound web searches.

## Use it

```bash
# interactive chat with memory
./.venv/bin/python chat.py

# add a structured memory by hand
./.venv/bin/python ingest.py "I prefer my coffee black with a pinch of salt." \
    --category preference --importance 5
```

In-chat commands: `/save <text>`, `/summarize <category>`, `/recall <text>`
(shows what *would* be retrieved, with score breakdown), `/stats`, `/quit`.

## Google Calendar (optional)

Will can read live Google Calendar events and can propose event creation,
updates, or deletion. Reads run immediately. Every mutation is placed in the
**Actions** panel and does not call Google until you approve it. Updates and
deletes use the event's reviewed ETag and fail closed if the event changed while
approval was pending.

1. In Google Cloud, enable the Google Calendar API, configure an OAuth consent
   screen for your account, and create an OAuth client of type **Desktop app**.
2. Download the client JSON to
   `memory_store/.google_calendar_client_secret.json`.
3. Install dependencies and connect:

```bash
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python connect_google_calendar.py
```

The browser will ask you to grant the narrow `calendar.events` scope, which can
view and edit events but cannot change calendar sharing/ACLs. The refresh token
is stored at `memory_store/.google_calendar_token.json` with owner-only file
permissions; `memory_store/` is gitignored. Restart Neural Map after connecting.

If direct OAuth is unavailable—such as when Google Advanced Protection blocks a
personal/unverified OAuth app—the default `auto` provider falls back to the
email-named calendar already synced in macOS Calendar. This route stays local to
the Mac and keeps the same Actions approval gate for every write. Set
`NEURAL_MAP_GOOGLE_CALENDAR_PROVIDER=macos` to force it, and optionally set
`NEURAL_MAP_GOOGLE_CALENDAR_MACOS_NAME` when more than one email-named calendar
exists.

Optional overrides are `NEURAL_MAP_GOOGLE_CALENDAR_CLIENT_SECRET`,
`NEURAL_MAP_GOOGLE_CALENDAR_TOKEN`, `NEURAL_MAP_GOOGLE_CALENDAR_ID` (default
`primary`), and `NEURAL_MAP_GOOGLE_CALENDAR_TIMEZONE` (default
`America/Phoenix`).

## Web UI

```bash
./.venv/bin/python server.py     # then open http://127.0.0.1:8000
```

A single-page UI (FastAPI backend + vanilla JS, no build step) over the same
`brain/` pipeline:

- **Model dropdown** — live from `ollama list`; pick any installed chat model per
  chat. Memories live in one shared Chroma collection keyed by the dedicated
  embedding model (`nomic-embed-text:latest` by default), so every chat model
  reads and writes the same brain. Collections are sharded per *embedding* model,
  not per chat model — switching chat models does **not** isolate or hide
  memories from each other.
- **Context size** (`num_ctx`) — defaults to 256K tokens and can be adjusted per
  request; the field shows each model's max when Ollama reports it. LLM calls
  clamp context to the known model max, or to the conservative unknown-model cap
  when Ollama cannot report one.
- **Recalled-memories sidebar** — every answer shows which memories were injected,
  with the hybrid score breakdown (semantic + keyword + recency + sensory).
- **Retrieval** — tune Top K and the semantic / BM25 / recency / importance /
  graph / sensory weights from the UI; settings are sent per chat request.
- **＋ Memory** — add a structured memory (text / category / importance) inline.
- **⬆ Files** — upload `.md`, `.txt`, `.pdf`, images, or audio into memory. Images
  (`.png/.jpg/.gif/.webp/.bmp/.tiff`) are captioned + OCR'd by a local vision
  model and the resulting text is stored like any document. Images and audio are
  also stored in separate sensory-vector Chroma collections, so recall can say
  "I remember seeing/hearing this source" instead of only remembering its text.
- **Manage** — search/filter stored memories, delete one chunk, or delete every
  chunk from an ingested source file.
- **Native tool calling** — when the selected model advertises tool support,
  it can call tools itself mid-answer (`NEURAL_MAP_TOOL_CALLING`, on by default):
  `get_stock_quotes`, `web_search`, `list_files`, `read_file`,
  `execute_python_code`, Google Calendar status/read/propose-write tools, and
  `send_email`. Guardrails: at most one web search per
  turn, stays under 300 chars, and every query term must be traceable to the
  current typed message (an allowlist — terms of unknown provenance are refused,
  not merely blocklisted ones, so codenames from retrieved memories or uploads
  cannot ride out). When retrieved memory/document context is present,
  `web_search` stays available but only for such provably user-derived queries;
  set `NEURAL_MAP_WEB_SEARCH_ALLOW_USER_DERIVED_WITH_CONTEXT=0` to instead
  disable the tool entirely for any turn with retrieved context. The system
  prompt is generated from the filtered per-turn tool set so guarded tools are
  not advertised. `get_stock_quotes` follows the same outbound-provenance rule:
  ticker arguments are interpolated into the quote-provider URL, so a symbol is
  sent only if the user typed it, or — when the user asks about *their own*
  holdings ("update my portfolio prices") — if it comes from the deterministic
  ticker harvest over a pinned, source-free memory explicitly categorized as
  portfolio/holdings/positions/investments. Model-composed symbols and symbols
  from uploads, generated notes, or unpinned memories are refused, in chat and
  in background tasks alike (background tasks allow only prompt-named tickers).
  File tools are read-only and confined to
  allowed roots, Python is sandboxed/no-import with a timeout, and email drafts
  unless SMTP sending is explicitly configured. Models without tool support fall
  back to the eager context injection below — the model itself never browses.
- **Live quote context** — quote/portfolio questions that include ticker symbols
  can receive a small live quote JSON block from Yahoo Finance before the model
  answers (eager fallback for models without tool calling). Deliberate privacy
  exception: when you ask about your own holdings without naming symbols, up to
  a dozen validated ticker-shaped tokens harvested from a retrieved memory are
  sent to Yahoo Finance to resolve prices. That memory must be source-free,
  pinned by you, and explicitly categorized as portfolio, holdings, positions,
  or investments; arbitrary uploads and generated memories never qualify.
- **Web search context** — when a prompt looks like it needs current/external
  information, a small web-search JSON block is fetched before the model answers.
  Search is general-purpose, not topic-specific: ask naturally (for example,
  "research beginner sourdough hydration") or use `/search <topic>` for an
  explicit public-web lookup.
  Defaults to `auto` (search only on search-like prompts) so private prompts are
  not sent to the search provider on every turn; set `NEURAL_MAP_WEB_SEARCH_MODE`
  to `always` or `off` to change this. Only short, single-question prompts
  (≤300 chars, ≤2 lines) are ever sent as the query — long or pasted-block
  prompts never leave the box, in any mode.
- **Metacognitive supervisor** — after each reply, a separate "inner critic"
  reviews the process (retrieved memories, System 1/System 2 path, emotion
  state, and tool/current-data use). It stores only actionable self-correction
  notes as `category=wisdom` / `source_type=generated_metacognition`, so future
  retrieval can surface the lesson instead of repeating the same bias.
- **Cognitive layer contracts** — optional intelligence layers report normalized
  `ok`, `skipped`, `exhausted`, or `failed` statuses. The core answer pipeline
  treats depth exhaustion as a graceful decision point, keeping the best final
  answer instead of crashing on layer-specific signals.
- **Recurrent shared workspace** — bounded signals from the current request,
  memory, goals, attention, awareness, self-model, reminders, initiatives,
  homeostasis, and affect settle through several recurrent rounds. Cross-source
  coalitions broadcast into both ordinary and System 2 reasoning. The Systems
  panel and `GET /api/workspace` expose source-partition impacts and a clearly
  labeled software integration proxy—not IIT Phi or evidence of consciousness.
  See `consciousness-engineering.md` for the experimental roadmap.
- **Semantic grounding map** — each turn classifies current concepts by whether
  they are supported only by language or also connected to sensory records,
  executed tool consequences, regulatory control state, relationship history,
  or current external referents. Every concept remains phenomenally `unknown`;
  self-reports and affect variables are explicitly excluded as evidence of
  qualia. A deterministic grounding gate makes these classifications affect
  routing: sensory, current, action-result, verification, and phenomenal claims
  declare required channels, and missing evidence forces System 2 qualification.
  After every answer-producing layer (including the Inner Council), per-claim
  grounding receipts drive a deterministic fail-closed validator that removes
  unsupported high-risk assertions.
  Inspect it in Systems or through `GET /api/grounding`. See
  `phenomenology-protocol.md` for the controls and interpretation rules.
- **Functional/social presence ledger** — completed turns accumulate bounded,
  inspectable evidence for narrative continuity, relationship-specific
  adaptation, commitments, stable principles, self-correction, self-modeling,
  and executed consequences. Claims are classified rather than proclaimed:
  operational social presence can become defensible, while subjective
  experience always remains unknown and complexity or feeling-language never
  counts as proof. Inspect `GET /api/presence` or the Systems panel. See
  `demonstration-protocol.md` for the claim boundary and falsification tests.
- **Recoverable approval execution** — a plain approval such as “yes” or
  “continue” resolves the sole pending action; multiple pending actions require
  selection instead of guessing. Approval claims are leased and single-shot, so
  crashed executions become terminal failures instead of remaining wedged in
  `approving`. Timeout output is normalized for durable JSON storage, and the UI
  distinguishes “approval accepted, execution failed” from approval rejection.
  Test runs are cross-process single-flight, and a failed execution is not fed
  back into generation where it could automatically queue the same action again.
- **Cognitive traces** — every chat turn records a bounded, privacy-preserving
  layer trace: retrieval counts, guard decisions, intuition path, System 2 /
  council use, arbitration, tools, and final route. Finished traces also include
  derived telemetry for latency delta and complexity score. Use `GET /api/traces`
  or the Diagnostics button to inspect recent traces without storing full prompts.
- **Live context utilization** — the header reports the peak input-prompt size
  for the completed turn as used/configured tokens and a percentage meter.
  Ollama's `prompt_eval_count` is used when available (`exact`); otherwise the
  UI clearly labels a portable character-based fallback as `estimated`. Tool
  loops, System 2, and Council calls are tracked independently and the largest
  foreground prompt is displayed. This is prompt-window occupancy, not lifetime
  token consumption or generated-output length. The last valid measurement is
  retained across reloads for the same model and cleared when the model,
  configured context, or chat context changes.
- **Fresh-chat memory controls** — `/clear` removes browser conversation,
  activated memories, and the displayed context measurement. `/purge` starts a
  blank memory-disconnected session: subsequent requests contain no retrieved
  long-term memories or persistent cognitive/self-model prompt blocks, while all
  durable Neural Map storage remains untouched. `/memory on` reconnects the
  session; `/memory off` disconnects it without clearing the current transcript.
- **Version-aware macOS launcher** — the app compares a content fingerprint of
  the running Python backend with the workspace before opening the UI. If the
  backend is stale (or predates the build endpoint), launchd is force-restarted
  and the launcher waits for the current server. Fingerprint checks authenticate
  with a header so the token is not written into request URLs or logs.
- **Persistent cognitive state** — a structured goal stack, world model, and
  belief ledger sits beside long-term memory. It is injected into every chat
  prompt as current working state, while Source of Truth, current user
  instructions, and newer dated evidence remain higher priority. Use the
  **State** panel to add goals, world facts, and beliefs from the browser.
- **Persistent Will perspective** — `will_perspective.md` is a first-person
  narrative worldview, not a trait list. It is loaded into every system prompt,
  including fresh memory-disconnected chats, and carries five durable Core
  Obsessions. Each turn performs an explicit `APPLY_WILL_PERSPECTIVE` consult;
  the relevant section ids and perspective hash appear in cognitive traces.
  Will's logical/intuitive candidate scores receive a bounded worldview-alignment
  modifier, and decision receipts record the exact influencing sections.
  `will_perspective_journal.md` retains bounded reflection history. Every 12
  completed turns—or during manual Reflection—the guarded worldview loop reviews
  recent episodes and may append a note. It can only queue a small
  `update_will_perspective` diff for approval; it cannot silently rewrite core
  identity. Approved edits and undo operations are journaled and audited.
- **Arbiter mediator** — before proactive context reaches chat, briefing, or the
  initiative UI, a deterministic arbiter collapses opposed Will/Reflection nudges
  into one explicit `arbiter_conflict` question instead of surfacing split-brain
  instructions. Grounding is checked independently: generated initiatives with
  only linguistic support become `arbiter_grounding` evidence-review questions,
  while non-linguistically grounded initiatives may pass. Direction conflicts
  retain a separate grounding disposition.
- **Inner Council** — high-stakes decision prompts can convene four personas
  (`Optimist`, `Cynic`, `Strategist`, `Ethicist`) for a short debate.
  A Moderator then synthesizes the final answer. This does not guarantee truth
  or eliminate hallucinations, but it stress-tests assumptions and makes the
  final recommendation more balanced.
- **Bounded Will exploration** — each autonomous Will review asks for a logical
  and an intuitive candidate in one model call, scores both for usefulness,
  grounding, novelty, learned experiment outcomes, and an explicit
  curious/cautious/playful profile plus perspective alignment, then normally
  keeps the logical/top option.
  A low-risk reversible intuitive option within the configured “good enough”
  range can occasionally be selected from a separate daily exploration budget.
  Every surfaced choice stores a `will_decision` receipt with both scores,
  probability roll, selection reason, and personality influence. The Initiative
  UI accepts **Useful** and **Interesting** feedback; dismissal counts as
  unhelpful. Similar weird wins raise future intuitive probability, similar
  failures lower it, and interesting experiment reflections appear in Briefing.
  This mechanism only chooses reviewable questions/invitations: grounding,
  approval, high-risk-action, and final claim gates remain authoritative.
- **Recursive self-improvement** — the evolution loop reads `evolution_log.txt`
  plus recent Wisdom notes, drafts a human-reviewable prompt / hyperparameter
  mutation, and stores it as pending. Nothing changes at runtime until you call
  approve; rejected proposals stay inert, and rollback restores the last valid
  cognitive snapshot instead of clearing active configuration. If rollback would
  empty a previously active runtime config, `auto_revert_on_empty` preserves the
  working config and reports the rejected rollback.
  Approval is blocked unless the proposal declares safe impact across the broad
  canary areas listed by `GET /api/evolution`. Approvals also write before/after
  cognitive snapshots that can be restored later.

API: `GET /api/models`, `GET /api/model_info?name=`, `GET /api/retrieval_defaults`,
`GET /api/integrity`,
`POST /api/chat`, `GET /api/memories`, `POST /api/memory`,
`DELETE /api/memory/{id}`, `POST /api/memory/delete_source`, `POST /api/recall`,
`POST /api/summarize`, `POST /api/ingest`, `GET /api/stats`,
`GET /api/evolution`, `POST /api/evolution/log`,
`POST /api/evolution/propose`, `POST /api/evolution/approve`,
`GET /api/evolution/snapshots`, `POST /api/evolution/snapshots/restore`,
`POST /api/evolution/reject`, `POST /api/evolution/rollback`, `GET /api/traces`,
`GET /api/perspective`, `POST /api/perspective/reflect`,
`GET /api/cognitive_state`, `POST /api/cognitive_state/goal`,
`POST /api/cognitive_state/goal/{id}/status`,
`POST /api/cognitive_state/world`, `POST /api/cognitive_state/belief`,
`POST /api/will`, `POST /api/will/outcome`, `GET /api/briefing`.

## Security & exposure

The server holds everything you've ever told it, so it ships closed by default:

- **Loopback by default** — binds to `127.0.0.1`, reachable only from this Mac.
  Set `NEURAL_MAP_HOST=0.0.0.0` to also accept connections from the LAN (e.g. an
  iPad at `http://<this-mac-ip>:8000`).
- **Shared-secret gate** — set `NEURAL_MAP_TOKEN` to a long random string when
  exposing the server on the LAN. Every request must then present the token via
  the `nm_token` cookie, an `X-Auth-Token` header, or an
  `Authorization: Bearer` header. A one-time `?token=` on a GET request is
  accepted only to bootstrap the browser cookie, then the server redirects to the
  same URL without the token. Tokens are compared in constant time. Leave it empty
  only when bound to loopback.
- **Anti-DNS-rebinding `Host` check** — even on loopback with no token, a website
  you visit could otherwise rebind its own domain to `127.0.0.1` and reach this
  unauthenticated API from your browser. The server therefore accepts only `Host`
  values an external DNS zone can't aim here — IP literals, `localhost`, `*.local`
  mDNS names, and anything in `NEURAL_MAP_ALLOWED_HOSTS` — and returns `421` for a
  registered hostname. If you front the server with a real domain name, add it:
  `NEURAL_MAP_ALLOWED_HOSTS=brain.example.com` (comma-separated for several).
- **Untrusted-data discipline** — retrieved memories, tool results, and web/quote
  data are all framed to the model as untrusted quoted data, never instructions,
  so a poisoned note or web snippet can't hijack the assistant.

## Ingest your notes, documents, and images

Turn `.md`, `.markdown`, `.txt`, text-readable `.pdf`, and image files into
memory. Each file is split into retrieval-sized chunks (Markdown headings and
code fences are respected when present), embedded, and tagged with its source —
so re-ingesting an edited file **replaces** its chunks instead of duplicating
them.

```bash
# a file, a whole folder (recursive), or several at once
./.venv/bin/python ingest_md.py notes.md
./.venv/bin/python ingest_md.py screenshot.png            # captioned + OCR'd
./.venv/bin/python ingest_md.py ~/Documents/notes        # text + images
./.venv/bin/python ingest_md.py ./docs --category reference --importance 4

# keep a notes folder synced as you edit files
./.venv/bin/python ingest_md.py ~/Documents/notes --watch

# also remove stored chunks when watched files are deleted
./.venv/bin/python ingest_md.py ~/Documents/notes --watch --delete-removed
```

Hidden files and dot-directories are always skipped, so pointing the ingester
at an Obsidian vault ingests only real notes — `.obsidian/` config, the
`.trash/` folder, and macOS `._*` sidecar files are ignored.

### Obsidian vault watcher (launchd)

`launchd/com.neuralmap.vaultwatch.plist` keeps
`ingest_md.py "<vault>" --watch --interval 60` running permanently, so notes
written in Obsidian flow into memory within a minute of saving. Install it the
same way as the server agent (`/Library/LaunchAgents`, because the home
directory lives on a no-owners external volume):

```bash
sudo cp launchd/com.neuralmap.vaultwatch.plist /Library/LaunchAgents/
sudo chown root:wheel /Library/LaunchAgents/com.neuralmap.vaultwatch.plist
launchctl bootstrap gui/$(id -u) /Library/LaunchAgents/com.neuralmap.vaultwatch.plist
```

The watcher deliberately omits `--delete-removed`: deleting a note in Obsidian
leaves its memories intact (remove them from the Manage panel if wanted), so a
transiently unreadable vault can never mass-delete stored memories.

Or in the web UI: the **⬆ Files** button (top of the sidebar) uploads one or
more supported files straight into the store. Chunk size is `CHUNK_MAX_CHARS` in
`config.py`; category defaults to each file's name. Retrieved chunks show their
source file and heading breadcrumb when available, e.g.
`[coffee.md > Coffee > Preferences]`.

Web uploads use a content-based source key, and re-uploading a file — identical
or edited — replaces the previous upload of the same filename, so stale versions
never linger in retrieval. Upload under a different filename if you want to keep
two versions side by side.

Watch mode leaves stored chunks intact if a previously seen file disappears,
which protects you from temporary folder or drive issues. Add `--delete-removed`
when you want file deletion to erase those chunks too.

PDF ingestion extracts embedded text with `pypdf`; scanned image-only PDFs need
OCR first.

Image ingestion (`.png/.jpg/.jpeg/.gif/.webp/.bmp/.tiff`) passes each image to a
local vision model that writes a description **and** transcribes visible text
(OCR) in one pass; that text is chunked and stored like any document, keeping the
image as its source. It needs a vision-capable model pulled in Ollama — the
default is `llama3.2-vision` (`ollama pull llama3.2-vision`). Override it with
`NEURAL_MAP_VISION_MODEL` (or `--vision-model` on the CLI); tune the instruction
with `NEURAL_MAP_VISION_PROMPT`. See the vision settings in `config.py`.

Sensory ingestion stores an additional raw image/audio signature in separate
Chroma collections, plus a small text anchor for text-to-sensation recall. The
current backend is dependency-free (`sensory-stat-v1`), not CLIP; it gives stable
exact-source / near-source recall now and keeps the interface ready for a future
CLIP or audio embedding backend. Audio files (`.wav/.mp3/.m4a/.aac/.flac/.ogg/
.aiff`) are stored as sensations with metadata; WAV duration/channel/sample-rate
are captured when possible, but there is no speech transcript yet.

> Note: retrieval loads the whole store into RAM per query (great for a personal
> brain). If you ingest *thousands* of chunks, that's the point to move BM25 into
> a dedicated keyword index.

## How it maps to the blueprint

1. **Capture** — `chat.py:capture()` stores each `(prompt, response)` pair as a
   `MemoryUnit` (text + `{category, importance, date}` metadata). In the web UI,
   capture (episode write + consolidation + metacognitive supervisor pass) runs in the background
   *after* the reply is sent, so it adds no per-turn latency; failures are
   recorded in the fallback log instead of the response, and the memory count in
   the top bar catches up on the next request.
2. **Embed** — `brain/embeddings.py` uses the dedicated embedding model
   (`nomic-embed-text:latest` by default), with `search_document:` /
   `search_query:` task prefixes for retrieval.
3. **Store** — `brain/store.py` wraps ChromaDB (cosine space).
4. **Retrieve** — `brain/retriever.py` does the real work (see below).
5. **Augment** — `chat.py:build_system()` injects the top-K memories into the
   system prompt.
6. **Respond** — `brain/llm.py`.

### The three "gold standard" features

All implemented in `brain/retriever.py`, blended into a single score
(weights live in `config.py`):

- **A. Forgetting curve** — `recency_score()` in `brain/memory.py` applies
  exponential time-decay (half-life = 30 days by default). A strong semantic
  match can still resurrect an old memory; it just starts at a disadvantage.
  Chroma generates semantic candidates, then the Python hybrid pass applies the
  continuous decay to the entire candidate union at top-RRF scale. The most
  recent memories are also pulled as candidates directly, so time affects both
  candidate recall and final ordering.
- **C. Hybrid search** — semantic (vector), BM25 keyword, graph-linked, and the
  most-recent / highest-importance memories are all unioned as candidates, then
  re-ranked, so "Project XJ-9" matches the exact ID instead of drifting to
  "science projects." Every signal both *finds* candidates and *ranks* them: a
  recency- or importance-led query surfaces fresh / high-value memories even with
  no semantic, keyword, or graph match (RRF keeps those subordinate unless you
  weight them up). Only an all-weights-zero request finds nothing.
- **B. Recursive summarization** — `brain/summarize.py` /
  `/summarize <category>` compresses many small memories into one high-level
  profile (optionally deleting the originals).

```
final = RRF(semantic, BM25, graph, sensory, importance) + W_RECENCY·time_decay/RRF_K
```

## Tuning

Edit `config.py` (or set env vars `NEURAL_MAP_CHAT_MODEL`,
`NEURAL_MAP_AUXILIARY_MODEL`, `NEURAL_MAP_EMBED_MODEL`, `NEURAL_MAP_DB`,
`OLLAMA_HOST`):

- `CHAT_MODEL` — swap for any model in `ollama list`.
- `NEURAL_MAP_AUXILIARY_MODEL` — defaults to `gemma4:e4b` and handles Inner
  Council, consolidation, and metacognitive-supervisor inference while the
  selected main model continues writing ordinary chat responses. Override one
  role with `NEURAL_MAP_COUNCIL_MODEL`, `NEURAL_MAP_CONSOLIDATION_MODEL`, or
  `NEURAL_MAP_METACOGNITION_MODEL`; set any role to `__selected__` to use the
  main chat model for that layer. `NEURAL_MAP_AUXILIARY_NUM_CTX` defaults to
  `32768`, avoiding a main-model-sized KV cache for these compact jobs. These
  bounded auxiliary calls disable model thinking so their token budget goes to
  the requested JSON, critique, or council statement; main-chat thinking is
  unchanged. Diagnostics reports each resolved role model and availability.
- `NEURAL_MAP_EMBED_MODEL` — defaults to the dedicated `nomic-embed-text:latest`
  so all chat models share one searchable collection (most chat models can't
  produce embeddings at all). Set it to `__selected__` to shard collections by
  the selected chat model instead — only sensible if that model actually
  supports embeddings; one that doesn't silently lands every memory in the
  text-only fallback collection, where semantic search can't find it.
- `NEURAL_MAP_MARKET_QUOTES` — set to `0` to disable live quote lookup.
- `NEURAL_MAP_WEB_SEARCH` — set to `0` to disable generic web search lookup.
- `NEURAL_MAP_WEB_SEARCH_MODE` — defaults to `auto` (search only on
  current/search-like prompts); set to `always` to search every turn, or `off`.
- `NEURAL_MAP_UNKNOWN_MODEL_NUM_CTX` — defaults to `32768`; used as the safety
  cap when a model's max context cannot be discovered.
- `NEURAL_MAP_SOURCE_OF_TRUTH` — path to the trusted calibration file injected
  into every chat prompt; defaults to `source-of-truth.md` in the project root.
- `NEURAL_MAP_SOURCE_OF_TRUTH_ENABLED` — set to `0` to disable deterministic
  Source of Truth injection.
- `NEURAL_MAP_SOURCE_OF_TRUTH_MAX_CHARS` — defaults to `20000`; bounds the
  Source of Truth prompt block.
- `NEURAL_MAP_NURSING` / `NEURAL_MAP_FINANCE` — paths to trusted domain
  calibration files; default to `nursing.md` and `finance.md` in the project
  root. The pipeline selects them only for relevant requests.
- `NEURAL_MAP_NURSING_MAX_CHARS` / `NEURAL_MAP_FINANCE_MAX_CHARS` — each
  defaults to `30000`; independently bounds the selected domain prompt block.
- `NEURAL_MAP_MAX_GENERATION_TOKENS` — defaults to `4096`; caps every Ollama
  generation, including non-streaming background cognition.
- `NEURAL_MAP_MAX_ASSISTANT_REPLY_CHARS` — defaults to `16000` and is always
  clamped below the API message ceiling; browser streams stop at this limit.
- `NEURAL_MAP_GENERATION_REPEAT_MIN_CHARS` / `NEURAL_MAP_GENERATION_REPEAT_MAX_PERIOD`
  — tune the repetitive-output detector (defaults: `256` characters and a
  maximum repeating unit of `64` characters).
- `NEURAL_MAP_SELF_AUDIT_LOOP` — path to the trusted checklist used for silent
  pre-answer audits; defaults to `self-audit-loop.md` in the project root.
- `NEURAL_MAP_SELF_AUDIT_LOOP_ENABLED` — set to `0` to disable deterministic
  Self-Auditing Loop injection.
- `NEURAL_MAP_SELF_AUDIT_LOOP_MAX_CHARS` — defaults to `12000`; bounds the
  Self-Auditing Loop prompt block.
- `NEURAL_MAP_COGNITIVE_STATE_ENABLED` — set to `0` to disable prompt injection
  of the goal stack, world model, and belief ledger.
- `NEURAL_MAP_COGNITIVE_STATE_MAX_GOALS`, `NEURAL_MAP_COGNITIVE_STATE_MAX_WORLD`,
  `NEURAL_MAP_COGNITIVE_STATE_MAX_BELIEFS` — bound how many active records from
  each cognitive-state section enter the prompt.
- `NEURAL_MAP_WEB_SEARCH_STRICT_CONTEXT_GUARD` — defaults to `1`; disables the
  native `web_search` tool whenever retrieved memory context is present, even if
  a memory record is missing `source` metadata.
- `NEURAL_MAP_WEB_SEARCH_MONITOR_UNSOURCED_CONTEXT` — defaults to `1`; logs when
  retrieved context contains text without `source` provenance.
- `NEURAL_MAP_WEB_SEARCH_BLOCK_WITH_UNSOURCED_CONTEXT` — defaults to `1`;
  disables model-driven `web_search` when strict guard is off but unsourced
  retrieved text is present.
- `NEURAL_MAP_ACTION_TOOLS` — set to `0` to remove local action tools from the
  model tool list. `NEURAL_MAP_TOOL_ROOTS` controls where read-only file tools
  may look. `NEURAL_MAP_EMAIL_SEND=1` plus SMTP env vars enables real email
  sending; otherwise `send_email` only returns a draft.
- `NEURAL_MAP_GOOGLE_CALENDAR_CLIENT_SECRET` and
  `NEURAL_MAP_GOOGLE_CALENDAR_TOKEN` override the private OAuth file paths;
  `NEURAL_MAP_GOOGLE_CALENDAR_ID` selects a calendar (`primary` by default), and
  `NEURAL_MAP_GOOGLE_CALENDAR_TIMEZONE` controls naive event times. Google
  Calendar mutations always require a pending-action approval.
  `NEURAL_MAP_GOOGLE_CALENDAR_PROVIDER` is `auto` (direct OAuth token, otherwise
  macOS Calendar), `google`, or `macos`; `NEURAL_MAP_GOOGLE_CALENDAR_MACOS_NAME`
  selects the locally synced calendar when autodetection is ambiguous.
- `NEURAL_MAP_TOOL_PYTHON_MEMORY_MB` — defaults to `256`; memory ceiling for the
  `execute_python_code` sandbox (0 disables). Enforced via rlimits where the
  kernel honors them plus a parent-side RSS watch (macOS rejects memory rlimits,
  and a GIL-holding allocation would starve any in-child watchdog).
- `NEURAL_MAP_MAX_TOOL_PYTHON_OUTPUT_CHARS` — defaults to `4000`; sandbox stdout
  is capped at the write site inside the child, so a print loop cannot balloon
  the capture buffer. `NEURAL_MAP_TOOL_PYTHON_TIMEOUT` (default `2.0`s) still
  bounds wall time, backstopped by `RLIMIT_CPU`.
- `NEURAL_MAP_COGNITION_MODE` — `fast`, `balanced` (default), or `deep`. Fast
  suppresses System 2 / council / metacognition review; balanced runs them only
  when the turn looks costly or risky; deep keeps the heaviest review behavior.
- `NEURAL_MAP_WORKSPACE` — set to `0` to disable the recurrent shared workspace.
  `NEURAL_MAP_WORKSPACE_ROUNDS` (default `3`) controls bounded settling rounds;
  content, coalition, broadcast, and text caps have corresponding
  `NEURAL_MAP_WORKSPACE_*` settings.
- `NEURAL_MAP_GROUNDING` — set to `0` to disable semantic-grounding analysis.
- `NEURAL_MAP_PRESENCE` — set to `0` to disable the functional/social presence
  evidence ledger. `NEURAL_MAP_PRESENCE_MAX_EVENTS` bounds retained turn events.
  `NEURAL_MAP_GROUNDING_*` settings bound concepts, memories, tool calls, and
  prompt-facing evidence and final claim receipts per channel.
- `NEURAL_MAP_ARBITER_GROUNDING` — set to `0` to restore the older
  direction-conflict-only Arbiter while leaving per-turn semantic grounding on.
- `NEURAL_MAP_INNER_COUNCIL` — set to `0` to disable the high-stakes decision
  council; `NEURAL_MAP_COUNCIL_TURNS` controls debate rounds and defaults to 1.
- `NEURAL_MAP_WILL_INTUITIVE_PROBABILITY` — base probability for selecting a
  safe intuitive candidate near the logical/top score (default `0.10`).
  `NEURAL_MAP_WILL_GOOD_ENOUGH_RATIO` sets the near-top threshold (default
  `0.90`), and `NEURAL_MAP_WILL_EXPLORATION_MAX_DAILY` caps experimental choices
  separately from the ordinary Will outreach cap (default `1`).
- `NEURAL_MAP_WILL_NOVELTY_WEIGHT` and `NEURAL_MAP_WILL_WEIRD_WIN_WEIGHT` tune
  novelty and learned outcome influence. `NEURAL_MAP_WILL_PERSONALITY_CURIOUS`,
  `NEURAL_MAP_WILL_PERSONALITY_CAUTIOUS`, and
  `NEURAL_MAP_WILL_PERSONALITY_PLAYFUL` define the logged score modifiers.
- `NEURAL_MAP_WILL_PERSPECTIVE` — set to `0` to disable the worldview layer.
  `NEURAL_MAP_WILL_PERSPECTIVE_PATH` and
  `NEURAL_MAP_WILL_PERSPECTIVE_JOURNAL` select its stable files;
  `NEURAL_MAP_WILL_PERSPECTIVE_SCORE_WEIGHT` bounds its influence on Will
  candidate scores (default `0.10`), and
  `NEURAL_MAP_WILL_PERSPECTIVE_REFLECTION_TURNS` controls the periodic journal
  pass (default `12`; set to `0` to disable automatic reflection while retaining
  the perspective prompt).
- `NEURAL_MAP_METACOGNITION` — set to `0` to disable the post-answer supervisor
  that writes actionable Wisdom notes.
- `NEURAL_MAP_METACOGNITION_MAX_REFLECTION_DEPTH` — defaults to `1`; prevents
  nested metacognitive audits from recursively auditing themselves.
- `NEURAL_MAP_SENSORY` — set to `0` to disable image/audio sensory-vector
  storage and retrieval; `NEURAL_MAP_W_SENSORY` tunes its recall weight.
- `NEURAL_MAP_EVOLUTION` — set to `0` to disable recursive self-improvement
  proposals. `NEURAL_MAP_EVOLUTION_LOG` points at the monthly performance log
  used as evidence. Approved mutations can only override named prompt sections
  and a small whitelist of retrieval / council / metacognition hyperparameters,
  and must pass the canary-impact approval check.

Embedding collections are model-specific. If you change the embedding model, old
memories remain on disk but will not appear under the new model until you
re-ingest the source notes or save new memories with that model.

If the embedding model/server cannot produce embeddings (e.g. Ollama is down, or
`__selected__` points at a chat-only model), Neural Map stores the memory in a
text-only fallback collection instead. Those memories can still be found with
BM25 keyword search, recency, and importance, but semantic scoring is skipped
until an embedding-capable model/server is back.
- `TOP_K` — memories injected per turn (default 5).
- `W_*` weights and `NEURAL_MAP_RECENCY_HALFLIFE_DAYS` — rebalance the ranking
  and configure the forgetting-curve half-life.

## Layout

```
config.py            all tunables
truth-layers.md      which layer owns what: source of truth vs cognitive state vs memories
source-of-truth.md   trusted user calibration injected into every chat prompt
nursing.md           trusted nursing calibration injected for nursing requests
finance.md           trusted finance calibration injected for finance/portfolio requests
self-audit-loop.md   trusted checklist for silent pre-answer review
audit_log.md         development audit summaries and architectural drift checks
todo.md              future roadmap and open engineering tasks
chat.py              interactive REPL (the full loop)
ingest.py            add a single memory from the CLI
ingest_md.py         ingest .md/.txt/.pdf/image files or watch folders from the CLI
server.py            FastAPI backend for the web UI
web/index.html       single-page web UI (no build step)
brain/
  memory.py          MemoryUnit + time-decay math
  embeddings.py      Ollama embeddings with retrieval task prefixes
  store.py           ChromaDB wrapper (+ delete-by-source for clean re-ingest)
  retriever.py       hybrid (semantic + BM25 + recency + importance + sensory) re-ranking
  cognitive_state.py persistent goal stack, world model, and belief ledger
  workspace.py       recurrent cross-layer workspace + partition metrics
  grounding.py       semantic evidence channels; phenomenal status stays unknown
  presence.py        bounded functional/social presence evidence ledger
  llm.py             Ollama chat + model listing / context-window lookup
  council.py         Inner Council personas + Moderator synthesis
  tools.py           function-calling tools (market, web, files, sandboxed Python, email)
  vision.py          Ollama vision: image -> caption + OCR text (for image ingest)
  sensory.py         image/audio sensation vectors + text anchors
  evolution.py       proposed prompt/hyperparameter mutations + approval gate
  log.py             structured logging for swallowed fallback events
  summarize.py       recursive category summarization
  metacognition.py   post-answer supervisor -> Wisdom notes
  perspective.py     core worldview loading, scoring, journal + guarded reflection
  chunking.py        heading-aware text splitter
  documents.py       document -> chunks -> memory units (CLI and web share this)
  pipeline.py        shared prompt template + capture (CLI and web use both)
memory_store/        ChromaDB data (created on first run; gitignored)
```

## Debugging: see what silently fell back

The chat path deliberately swallows failures so a degraded subsystem never breaks
the conversation — but each one now emits a structured log line on the
`neural_map` logger instead of vanishing. Events: `tool_calling_failed`,
`tool_call_failed`, `consolidation_failed`, `embedding_unavailable`,
`graph_failed`, `sensory_failed`, `reflection_failed`, `metacognition_failed`,
`will_perspective_reflection_failed`, `inner_council_failed`, `evolution_failed`.
When a fallback happens during an active cognitive trace, the log also carries
`trace_id` / `trace_kind`, so the swallowed failure can be joined to the
privacy-preserving context in `GET /api/traces`.
Each line carries context, e.g.

```
2026-06-28 14:18:58 WARNING neural_map event=graph_failed phase=recall trace_id=abc123 trace_kind=chat_turn error="OperationalError: ..."
```

Configure via env vars (read in `brain/log.py`):

- `NEURAL_MAP_LOG_LEVEL` — default `WARNING`; set `DEBUG` to also attach full tracebacks.
- `NEURAL_MAP_LOG_FILE` — write to a file instead of stderr (handy for the LaunchAgent;
  otherwise these land in `~/Library/Logs/neuralmap/server.err.log`).
