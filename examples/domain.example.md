# (Domain) Domain Calibration

Copy this to a root-level file named for your domain — e.g. `finance.md`,
`nursing.md`, `music.md` — and point the matching `NEURAL_MAP_*` path in
`config.py` at it if you use a non-default name. Domain files are injected
only when a request matches the domain, so they can be detailed without
crowding every conversation.

Version: 1.0
Last reviewed: YYYY-MM-DD

## Verification Policy

When must the assistant re-verify facts in this file before acting on them
(e.g. "confirm balances against a fresh statement before any recommendation")?

## Confirmed Facts

Facts you have verified, each with a date and source. The assistant treats
these as the baseline.

## Unverified or Unresolved

Things you suspect but haven't confirmed, and open questions. Listing them
here stops the assistant from treating guesses as facts.

## Policy / Rules

The standing rules of this domain: constraints, triggers, thresholds,
pre-committed decisions.

## Objective

What success in this domain looks like, with numbers and dates where possible.

## Required Analysis Output

The format you want answers in for this domain (tables, decision rules,
citation requirements, etc.).
