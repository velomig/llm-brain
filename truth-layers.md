# Truth Layers — which layer owns what

Last reviewed: 2026-07-06

The system has several places where "what is true" can live. Each exists for a
different reason; duplication across them creates conflicts the model has to
resolve and the user has to maintain. One page, one rule per layer.

## Precedence (highest first)

1. **Current user instructions** — what the user says in this conversation.
2. **`source-of-truth.md`** — durable user-authored calibration.
3. **Request-selected domain calibration** — `nursing.md` and/or `finance.md` when relevant.
4. **Persistent Cognitive State** — machine-managed current goals / world / beliefs.
5. **Retrieved memories** — evidence, never instructions.

Newer dated evidence can supersede an older layer's claim; when layers conflict
without a clear date, the assistant asks instead of guessing.

## Ownership

| Layer | Owns | Maintained by | In the prompt |
|---|---|---|---|
| `source-of-truth.md` | Universal identity, interaction philosophy, cross-domain priority rules | The user, by hand | Every turn |
| `nursing.md` | Nursing education, clinical-reasoning rules, study modes, and nursing-specific calibration | The user, by hand | Nursing requests only |
| `finance.md` | Portfolio facts, financial principles, retirement strategy, and finance-specific calibration | The user, by hand | Finance/portfolio requests only |
| Cognitive State (`/api/cognitive_state`) | *Current* goal stack, world facts, belief ledger — things that change month to month | The app (bounded counts) | Every turn (compact JSON) |
| Memory store (Chroma) | Evidence: episodes, distilled facts, documents, insights, reminders | Pipeline (consolidation, reflection, ingestion) | Top-K retrieved per turn |
| `agency_protocol.md` + `action-package-template.md` | *How* actions are proposed and approved (authority levels, package format) | The user, by hand | Tool-mode turns only |
| `workflow_protocol.md` | Division of labor for nursing/finance work | The user, by hand | Not injected; reference doc |
| `self-audit-loop.md` | Pre-answer review checklist (process, not facts) | The user, by hand | System 2 critique only |

## Where does a new fact go?

- **It is universal identity or cross-domain interaction calibration** → `source-of-truth.md`.
- **It governs nursing tutoring or clinical reasoning** → `nursing.md`.
- **It governs portfolio, investment, tax, or retirement analysis** → `finance.md`.
- **It's true now but will change (a goal, a deadline, a current plan)** → Cognitive State.
- **It's something that happened or was said (evidence)** → the memory store — it lands there automatically via capture/ingestion.
- **It's about how the assistant should behave** → the relevant protocol file, not a fact layer.

One home per fact. If the same claim is needed in two layers, the lower layer
should reference the higher ("see source of truth"), not restate it — restating
is how stale copies are born.

## Maintenance

- When trusted calibration goes stale, edit its owning file; do not rely on a
  newer memory to out-vote it.
- Cognitive State is bounded (`NEURAL_MAP_COGNITIVE_STATE_MAX_*`); completed or
  abandoned goals should be closed, not left to crowd out live ones.
- The prompt-budget trace event (`prompt_budget`) shows what each layer costs
  per turn — check it before growing any of these files.
