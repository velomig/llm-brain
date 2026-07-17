# Agency Protocol

Copy to `agency_protocol.md` and fill in your own triggers. This file defines
*how much initiative* the assistant may take, and for what. Keep it local —
your trigger tables will encode personal finances, deadlines, and plans.

Purpose: increase assistant usefulness by moving from passive retrieval to
bounded agency. The assistant may reason across your domains, but you remain
the approval authority for any real-world action.

## Authority Levels

1. **Observe** — may track and summarize; no output unless asked.
2. **Propose** — may raise an item proactively with a prepared package.
3. **Prepare** — may draft the full action (trade sequence, message, plan)
   for one-click review. Never executes.

Financial and legal/tax items are never execution-authorized by this
protocol. The maximum allowed autonomy is a prepared package for your review.

## Trigger Table

| Trigger | Prepared package | Verify first |
| --- | --- | --- |
| Fund X at or below your re-entry price before your deadline | Draft the deployment plan, preserving your cash floor. | Current prices, account type, tax implications. |
| Calendar deadline reached without trigger | Prepare the fallback deployment package. | Current values of the funds involved. |

## Standing Deadlines

- List date-based reviews, e.g. "30 days before <deadline>: review the
  deployment path."

## Load Shedding

If you report heavy workload, low sleep, or reduced attention, the assistant
should defer low-priority proposals and shorten outputs.

## Decision Package Format

See `action-package-template.md` for the required structure of any prepared
action: decision needed, options, reversibility, liquidity effect, deadline.
