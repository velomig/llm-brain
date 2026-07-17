"""Shared nursing/finance domain routing for prompts and retrieval.

One decision, used everywhere it matters: which trusted calibration files a
turn loads (pipeline) and which protected domain memories retrieval may
surface (retriever). Keyword routing against only the latest message misfired
in both directions — "write a patient onboarding email for my dental clinic"
loaded nursing, "explain covalent bonds in chemistry" loaded finance, and a
follow-up like "what if I increase it by ten percent?" after a portfolio
question loaded nothing. Two changes address that:

  * Terms are tiered. A STRONG term is unambiguous on its own ("nclex",
    "portfolio"); WEAK terms are generic words that only signal the domain in
    combination ("patient", "bonds"), and "bonds" never counts when it follows
    a chemistry adjective.
  * The decision looks at a small window of recent user messages, not only
    the current one, so topic-anchored follow-ups keep their domain. Only
    user-authored text is scanned — retrieved memories and model output must
    not be able to steer which trusted files load.
"""
from __future__ import annotations

import re

DOMAINS = ("nursing", "finance")

# Prior user messages considered besides the current one. Two turns keeps
# "what if I increase it by ten percent?" anchored to the portfolio question
# that preceded it without dragging in a stale topic from much earlier.
HISTORY_WINDOW = 2
_MAX_SCAN_CHARS = 4000

_STRONG = {
    "nursing": re.compile(
        r"\b(?:nurs(?:e|es|ing)|rn|nclex|clinicals|pathophysiology|pharmacology|"
        r"dosage|vitals|abg|oxygenation|perfusion|sepsis|sata)\b",
        re.IGNORECASE,
    ),
    "finance": re.compile(
        r"\b(?:financ(?:e|es|ial)|portfolio|retirement|invest(?:ing|ment|ments|or|ors)?|"
        r"dividends?|brokerage|fidelity|etfs?|tickers?|401k|ira|roth|"
        r"fndx|fndy|fndz|acme|exco|coin|bitcoin|treasur(?:y|ies)|capital gains?)\b",
        re.IGNORECASE,
    ),
}

# "bonds" straight after a chemistry adjective is not a fixed-income signal.
# Fixed-width lookbehinds, one per adjective, because re requires it.
_NON_FINANCE_BOND = (
    r"(?<!covalent )(?<!ionic )(?<!hydrogen )(?<!chemical )(?<!peptide )"
    r"(?<!metallic )(?<!molecular )(?<!double )(?<!triple )"
)

_WEAK = {
    "nursing": re.compile(
        r"\b(?:clinical|patients?|medications?|drugs?|labs?|laboratory|"
        r"delegation|exams?|semester)\b",
        re.IGNORECASE,
    ),
    "finance": re.compile(
        r"\b(?:stocks?|" + _NON_FINANCE_BOND + r"bonds?|funds?|shares?|"
        r"tax(?:es|able|ation)?|markets?|money|prices?|quotes?|trades?|trading|"
        r"buy|holdings?|allocation)\b",
        re.IGNORECASE,
    ),
}


def _weak_stems(domain: str, text: str) -> set[str]:
    """Distinct weak-term stems, so "stocks and more stocks" counts once."""
    return {
        re.sub(r"(?:es|s)$", "", match.group(0).lower())
        for match in _WEAK[domain].finditer(text)
    }


def detect(text: str, strict: bool = True) -> set[str]:
    """Domains a single text opts into.

    strict=True (loading trusted calibration prompts) needs a strong term or
    two distinct weak ones. strict=False (retrieval visibility) accepts a
    single weak term: wrongly hiding the user's own domain memories costs far
    more than letting them compete in ranking, so that side stays permissive.
    """
    text = (text or "")[:_MAX_SCAN_CHARS]
    found: set[str] = set()
    threshold = 2 if strict else 1
    for domain in DOMAINS:
        if _STRONG[domain].search(text):
            found.add(domain)
        elif len(_weak_stems(domain, text)) >= threshold:
            found.add(domain)
    return found


def recent_user_texts(
    user_text: str, history: list[dict] | None = None, window: int = HISTORY_WINDOW
) -> list[str]:
    """The current message plus up to `window` most recent prior user messages."""
    texts = [user_text or ""]
    taken = 0
    for msg in reversed(history or []):
        if taken >= window:
            break
        if (msg or {}).get("role") != "user":
            continue
        content = str(msg.get("content") or "").strip()
        if not content or content == (user_text or "").strip():
            continue
        texts.append(content)
        taken += 1
    return texts


def active_domains(
    user_text: str, history: list[dict] | None = None, strict: bool = True
) -> set[str]:
    """The turn's domain decision over the bounded recent-history window."""
    found: set[str] = set()
    for text in recent_user_texts(user_text, history):
        found |= detect(text, strict=strict)
    return found
