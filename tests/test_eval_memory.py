"""Behavioral eval set for the memory system.

Compile and type checks don't catch the failures that actually matter for a
second brain: a ranking tweak that buries a stated preference, a consolidation
change that drops a decision, a reminder that quietly stops surfacing, or an
uploaded PDF that can't be recalled. These are end-to-end retrieval/recall
assertions over a small SEEDED corpus (target memory + distractors), so a failure
points at a behavior, not a stack trace.

Run after any change to retrieval, ranking, consolidation, reminders, or ingest:

    ./.venv/bin/python -m pytest tests/test_eval_memory.py -v

Hermetic: conftest swaps in a deterministic bag-of-words embedder, so this needs
no Ollama. Each query shares real content words with its target (which is how both
the fake embedder and real semantic search find it). Add a prompt + expectation
here as the system grows; keep the corpus small so each eval isolates one behavior.
"""
from __future__ import annotations

from datetime import date, timedelta

from brain.documents import ingest_document_text
from brain.reflect import active_reminders
from brain.retriever import retrieve_weighted
from util import ids, store_memory

# The canonical prompts this set covers (one eval each, below).
EVAL_PROMPTS = (
    "What does Alex prefer?",
    "What reminders are due?",
    "What did I decide about X?",
    "Recall from uploaded PDF.",
)

# Enough unrelated memories that top_k=5 has to actually choose — membership in
# the top results is then a meaningful signal, not a free pass. Distinct vocab so
# none of them accidentally share words with a target query.
_DISTRACTORS = (
    "The weather in Lisbon is mild through spring.",
    "Python's GIL serializes bytecode execution on one core.",
    "Groceries to buy: oat milk, almonds, coffee beans.",
    "The gym closes at ten on weeknights.",
    "A sourdough starter wants feeding twice a day.",
    "Rebasing rewrites commit hashes on the branch.",
    "Saturn has the most extensive ring system.",
    "The orchestra tunes to an oboe playing A440.",
)


def _seed_distractors() -> None:
    for note in _DISTRACTORS:
        store_memory(note, category="note")


# --- "What does Alex prefer?" -----------------------------------------------

def test_eval_what_does_alex_prefer():
    pref = store_memory(
        "Alex prefers local and private AI, using Ollama instead of cloud APIs.",
        category="fact",
        importance=5,
    )
    _seed_distractors()

    got = ids(retrieve_weighted("What does Alex prefer, local AI or cloud APIs?", top_k=5))
    assert pref.id in got, "stated preference must be recalled into the prompt context"


# --- "What reminders are due?" ------------------------------------------------

def test_eval_what_reminders_are_due():
    today = date.today()

    def _reminder(text: str, days: int) -> None:
        store_memory(
            text,
            category="reminder",
            extra={"due": (today + timedelta(days=days)).isoformat(), "status": "open"},
        )

    _reminder("Execute the $250 SPCX limit sell.", days=3)        # due soon
    _reminder("Renew the domain registration.", days=-2)          # expired
    _reminder("File the annual tax return.", days=120)            # beyond the horizon

    due = active_reminders()
    by_text = {r["text"]: r for r in due}

    assert "Execute the $250 SPCX limit sell." in by_text         # within horizon
    assert "Renew the domain registration." not in by_text        # passed date is suppressed
    assert "File the annual tax return." not in by_text           # too far out, suppressed


# --- "What did I decide about X?" ---------------------------------------------

def test_eval_what_did_i_decide_about_x():
    decision = store_memory(
        "Decided the database choice: SQLite, kept local and file-based, not Postgres.",
        category="fact",
        importance=4,
    )
    _seed_distractors()
    store_memory("Briefly considered MongoDB but never adopted it.", category="note")

    got = ids(retrieve_weighted("What did I decide about the database choice?", top_k=5))
    assert decision.id in got, "a recorded decision must resurface when asked about it"


# --- "Recall from uploaded PDF." ----------------------------------------------

def test_eval_recall_from_uploaded_pdf():
    source = "upload:deadbeef:lighthouse.pdf"
    n, _category = ingest_document_text(
        "Project Lighthouse targets a Q4 launch with a twelve-week runway and a small crew.",
        filename="lighthouse.pdf",
        source=source,
        importance=3,
    )
    assert n >= 1, "the PDF should produce at least one ingestible chunk"
    _seed_distractors()

    rows = retrieve_weighted("Recall the Lighthouse launch and runway from the uploaded PDF.", top_k=5)
    assert any(r.get("source") == source for r in rows), "uploaded-PDF content must be recallable by source"


# --- meta: keep the documented prompt list and the evals in sync --------------

def test_eval_set_covers_all_documented_prompts():
    evals = [name for name in globals() if name.startswith("test_eval_") and name != "test_eval_set_covers_all_documented_prompts"]
    assert len(evals) == len(EVAL_PROMPTS)
