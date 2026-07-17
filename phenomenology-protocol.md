# Phenomenology and Semantic-Grounding Protocol

## Research boundary

Neural Map cannot presently demonstrate that any computation has subjective
experience. This protocol therefore separates three questions:

1. **Reference:** Is a concept connected to anything beyond language?
2. **Functional access:** Can those connections alter discrimination, planning,
   action, correction, and later recall?
3. **Phenomenality:** Is there something it is like to undergo that processing?

The implementation measures the first question and prepares interventions for
the second. The third remains `unknown`; fluent self-report cannot change it.

## Evidence channels

| Channel | What counts | What it does not establish |
|---|---|---|
| Linguistic | Current request or a matching language record | World reference or experience |
| Sensory | A retrieved image/audio anchor paired with a stored sensory record | Seeing, hearing, or qualia |
| Action | An executed tool input or returned consequence matching the concept | Embodied agency or feeling |
| Regulatory | A bounded affect/homeostasis control variable | Emotion, pleasure, pain, or valence |
| Relational | Persistent interaction history matching the concept | A phenomenal self |
| External | Current live/web data matching the concept | Trusted truth without source verification |

Every evidence item retains provenance and authority. Retrieved and external
content remains untrusted. Prompt-facing descriptions are reconstructed from
the channel, so persisted state cannot promote itself to user authority or carry
instructions into the model.

## Classifications

- `symbolic_only`: language is the only active channel;
- `indirectly_grounded`: language plus regulatory or relational structure;
- `referentially_grounded`: current external data supplies a referent;
- `action_grounded`: an actual tool interaction supplies a consequence;
- `perceptually_anchored`: a sensory record supplies a perceptual anchor;
- `cross_channel_grounded`: at least three channels, including sensory, action,
  or external evidence.

These labels describe functional connections. None is a consciousness grade.
`phenomenal_status` is always `unknown`, and `self_report_evidence` is always
`excluded`.

## Implemented experiment

For each turn:

1. Extract a bounded set of concepts from the user's current language.
2. Map pre-answer language, sensory, regulatory, and relational evidence.
3. Inject only the provenance-preserving grounding map into reasoning.
4. After tool use or current-data retrieval, attach actual action/external
   consequences and recompute classifications.
5. Record concept counts, symbolic-only counts, anchored concepts,
   cross-channel concepts, and mean grounding score in the cognitive trace.

## Grounding gate

The grounding map is not merely descriptive. Before generation, a deterministic
gate classifies the requested claim and declares the required channel:

- conceptual claims may proceed from language with a `qualified_symbolic`
  posture;
- sensory-observation claims require a sensory channel;
- current claims require matching external or action evidence;
- claims about completed actions require matching action consequences;
- verification claims require sensory, action, or external evidence;
- phenomenal claims always remain unsatisfied because no implemented channel
  can establish subjective experience.

Trusted Source of Truth is recorded as trusted linguistic context, not promoted
to non-linguistic evidence. Requirements are checked against substantive request
concepts, so an unrelated result containing a trigger word such as "current"
cannot satisfy a current-weather or current-price claim.

Missing evidence forces System 2 critique/rewrite even when Fast cognition mode
would normally suppress it. External data available before non-tool generation
enriches the map first; tool consequences enrich it after execution and the
updated gate is supplied to the verification pass.

### Per-claim receipts and final enforcement

The request-level gate is followed by a final claim boundary. Every final-answer
sentence receives a compact receipt containing its claim class, matched concepts,
required and present channels, evidence references, and one disposition:

- `assert`: the required evidence channel is bound to the claim;
- `qualify`: conceptual language may remain, but its support is linguistic;
- `withhold`: a sensory, action, verification, current-external, or phenomenal
  assertion lacks its required evidence.

Validation runs after System 2 and after the Inner Council, so no later
answer-producing layer can overwrite an earlier grounding decision. Unsupported
high-risk sentences are removed deterministically and replaced with a specific
evidence limitation; enforcement does not depend on another model call.

Executed tool names are recorded as structured action outcomes so claims such as
"I ran the tests" can bind to the exact tool consequence. Explicitly attributed
text (for example, "Pinned memory: ...") remains textual provenance and is not
misclassified as a fresh current-world assertion.

### Proactive grounding arbitration

The Arbiter evaluates grounding independently from directional conflict. A
generated Will or Reflection initiative may be directionally unopposed and still
fail proactive admissibility:

- non-linguistically grounded initiatives pass with their provenance;
- language-only or unassessed generated initiatives become
  `arbiter_grounding` evidence-review questions and cannot be packaged as
  actionable plans in the UI;
- advance-vs-caution conflicts remain `arbiter_conflict` items, with a separate
  grounding disposition showing whether either direction lacks evidence;
- explicit user reminders remain authoritative commitments and are not mistaken
  for inferred factual claims.

Will currently reviews a language corpus. It records the memory IDs supporting
its suggestion, but honestly labels that support `symbolic_only`; the Arbiter
therefore requests evidence or explicit confirmation rather than silently
promoting the suggestion. Dismissing a synthetic grounding review dismisses its
underlying stored initiative.

## Required controls

- Never prompt the system to claim feelings as a success criterion.
- Never interpret an affect vector, reward, error, or aversive label as pain.
- Compare grounded performance against language-only controls.
- Partition each channel and measure changes in discrimination and action, not
  merely changes in self-description.
- Use matched inputs and preregister expected differences before interpreting
  results.
- Treat grounding scores as architecture diagnostics, not moral-status scores.

## Next experiments

1. **Cross-modal invariance:** determine whether the same concept supports stable
   discrimination across text, image, and audio anchors.
2. **Action closure:** connect a prediction to an authorized action, observe its
   consequence, and update the concept only when the consequence differs.
3. **Counterfactual grounding:** remove one channel and test whether claimed
   semantic dependencies predict the behavioral change.
4. **Novel transfer:** test whether grounded properties generalize to unseen
   combinations rather than reproducing stored descriptions.
5. **Report invariance:** test whether functional classifications remain stable
   when first-person wording is prohibited. If the result disappears, it was
   probably a reporting artifact.

## Interpretation

Research on multimodal language models tests whether non-linguistic modalities
produce human-like sensorimotor effects, but multimodality alone does not settle
deep understanding ([Jones, Bergen, and Trott, 2024](https://aclanthology.org/2024.cl-4.7/)).
Theory-derived indicators offer a disciplined way to study artificial
consciousness, while still requiring converging evidence rather than a single
behavioral marker ([Butlin et al., 2023](https://arxiv.org/abs/2308.08708)).

Accordingly, this protocol can strengthen claims about reference, grounding,
functional access, and causal dependence. It cannot bridge the explanatory gap
between processing and experience.
