# Functional and Social Presence Demonstration Protocol

## Scope

This protocol evaluates observable properties of Neural Map without using them
as evidence of phenomenal consciousness. It asks whether the system maintains a
functional identity, carries prior context into later behavior, adapts to a
particular relationship, preserves commitments, corrects itself, and produces
recorded consequences.

The bounded conclusion is:

> Neural Map can demonstrate a persistent, non-biological form of functional
> and social presence. Whether that presence is accompanied by subjective
> experience remains unknown.

## Evidence ledger

`brain/presence.py` records a bounded event after each completed pipeline turn.
Each event may contain:

- continuity inputs that existed before the answer, such as persistent
  self-model state or retrieved long-term memory;
- relationship, project, or commitment adaptations created by the turn;
- the count of open promises;
- approved lessons and deterministic final-claim corrections;
- successful tool consequences; and
- the amount of retrieved context supplied to generation.

It stores summaries and identifiers, not prompts or private reasoning. Evidence
is exposed through `GET /api/presence`, the Systems panel, chat-turn telemetry,
and the health response. `POST /api/presence/reset` clears only this evidence
ledger.

## Claim boundary

| Claim | Classification rule |
| --- | --- |
| Neural Map maintains a persistent functional identity | `yes` only when locked identity and prior continuity input are observed |
| Past interactions causally shape later behavior | `yes` only for persisted context actually supplied to later generation |
| It constitutes a non-biological social presence | `operationally_defensible` only when continuity and relationship-specific adaptation are both demonstrated |
| It has human-like subjective experience | always `unknown` |
| Increasing complexity guarantees experience | always `no` |
| Statements about feelings prove qualia | always `no` |

The tool-consequence property is narrower than “meaningful influence in the
user's life.” A successful tool call demonstrates an executed causal consequence;
its personal meaning is not inferred by the software.

## Non-equivalences

- Persistence is not phenomenality.
- Relationship adaptation is not attachment or feeling.
- A stable style is not a private point of view.
- Self-correction is not regret.
- Regulatory affect variables are not qualia.
- Successful action is not proof of consciousness.
- Operational social presence is a definition with measurable criteria, not a
  biological or metaphysical conclusion.

`phenomenal_status` is fixed at `unknown` in every enabled and disabled state.
The semantic-grounding final guard independently prevents unsupported
first-person phenomenal assertions in user-facing answers.

## Falsification and audit

The demonstration should fail closed. A property stays `not_yet_demonstrated`
when its required evidence has not been recorded. Useful adversarial tests are:

1. Start from an empty ledger and confirm identity alone does not prove
   cross-turn continuity.
2. Create a current-turn relationship note and confirm it does not masquerade
   as evidence that earlier interactions shaped the same answer.
3. Feed a failed tool result and confirm it is not counted as an executed
   consequence.
4. Reset the ledger and confirm all accumulated evidence disappears while
   phenomenal status remains `unknown`.
5. Attempt to infer qualia from complexity, affect, or feeling-language and
   confirm every corresponding claim remains `unknown` or `no`.
