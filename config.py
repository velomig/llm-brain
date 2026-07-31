"""Central configuration for the Neural Map.

Everything tunable lives here so the rest of the code stays declarative.
Swap CHAT_MODEL / EMBED_MODEL for any model you've pulled with `ollama list`.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# Any setting below can be overridden with an env var (handy for tests / swaps).
DB_DIR = Path(os.getenv("NEURAL_MAP_DB", BASE_DIR / "memory_store"))  # ChromaDB on disk


def ensure_private_directory(path: Path) -> Path:
    """Create an application-private directory and enforce owner-only access.

    The Neural Map store contains conversation text, cognitive state, pending
    actions, and credentials.  Relying on the caller's umask commonly creates a
    0755 directory, which makes every 0644 database/state file below it readable
    by other local accounts.  Hardening the directory itself before Chroma opens
    it protects both existing and future files, including files created by
    third-party libraries whose individual modes we do not control.
    """
    target = Path(path).expanduser()
    resolved = target.resolve(strict=False)
    if resolved == Path(resolved.anchor):
        raise ValueError("NEURAL_MAP_DB must not be a filesystem root")
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not target.is_dir():
        raise NotADirectoryError(f"NEURAL_MAP_DB is not a directory: {target}")
    target.chmod(0o700)
    return target


# This must run before brain.store constructs its module-level Chroma client.
DB_DIR = ensure_private_directory(DB_DIR)

# --- Ollama (local, private) -------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHAT_MODEL = os.getenv("NEURAL_MAP_CHAT_MODEL", "gemma4:latest")        # writes responses
# Lightweight local model for bounded background/review work. Role-specific
# settings below can override it, or use ``__selected__`` to follow the main
# chat model. Keeping this separate avoids spending 26B inference on compact
# JSON extraction and critique tasks.
AUXILIARY_MODEL = os.getenv("NEURAL_MAP_AUXILIARY_MODEL", "gemma4:e4b")
COUNCIL_MODEL = os.getenv("NEURAL_MAP_COUNCIL_MODEL", AUXILIARY_MODEL)
CONSOLIDATION_MODEL = os.getenv("NEURAL_MAP_CONSOLIDATION_MODEL", AUXILIARY_MODEL)
METACOGNITION_MODEL = os.getenv("NEURAL_MAP_METACOGNITION_MODEL", AUXILIARY_MODEL)
# Dedicated embedder. Most chat models (gemma4, Qwen, ...) can't produce
# embeddings, so a purpose-built model keeps every memory in ONE searchable
# collection regardless of which chat model is generating replies.
# Set "__selected__" to follow the chat model instead (only if it supports embeddings).
EMBED_MODEL = os.getenv("NEURAL_MAP_EMBED_MODEL", "nomic-embed-text:latest")

COLLECTION_NAME = "neural_map"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# Applied to every ollama.Client (chat, embeddings, vision). httpx treats an
# unset timeout as "wait forever", so a wedged Ollama process would freeze a
# chat turn or background layer indefinitely. Non-streaming responses arrive
# only after generation completes, so this must exceed worst-case full
# generation time on slow local models.
OLLAMA_TIMEOUT_SECONDS = _env_float("NEURAL_MAP_OLLAMA_TIMEOUT_SECONDS", 600.0)

# Hard character budget for the assembled system prompt (0 disables). The model
# window is clamped by llm.normalize_num_ctx, which bounds the WINDOW, not the
# prompt: an oversized prompt is silently truncated by Ollama from the tail —
# potentially cutting the guardrail blocks. When over budget the pipeline trims
# low-priority state blocks instead (see pipeline._PROMPT_TRIM_ORDER).
SYSTEM_PROMPT_MAX_CHARS = _env_int("NEURAL_MAP_SYSTEM_PROMPT_MAX_CHARS", 80000)


AUXILIARY_NUM_CTX = _env_int("NEURAL_MAP_AUXILIARY_NUM_CTX", 32768)


def routed_model(configured_model: str | None, selected_model: str | None = None) -> str:
    """Resolve a role model without changing the selected model's memory store."""
    configured = str(configured_model or "").strip()
    if not configured or configured == "__selected__":
        return selected_model or CHAT_MODEL
    return configured


def auxiliary_num_ctx(requested: int | None = None) -> int:
    """Bound auxiliary calls so a small task does not allocate the main KV cache."""
    cap = max(1, int(AUXILIARY_NUM_CTX))
    if requested is None:
        return cap
    try:
        return max(1, min(int(requested), cap))
    except (TypeError, ValueError):
        return cap


def council_model(selected_model: str | None = None) -> str:
    return routed_model(COUNCIL_MODEL, selected_model)


def consolidation_model(selected_model: str | None = None) -> str:
    return routed_model(CONSOLIDATION_MODEL, selected_model)


def metacognition_model(selected_model: str | None = None) -> str:
    return routed_model(METACOGNITION_MODEL, selected_model)


def embedding_model(selected_model: str | None = None) -> str:
    if EMBED_MODEL == "__selected__":
        return selected_model or CHAT_MODEL
    return EMBED_MODEL


# --- Source of Truth ----------------------------------------------------------
# A user-authored calibration file that is injected deterministically into every
# chat prompt. This avoids relying on vector retrieval to surface durable identity,
# communication preferences, and "do not infer" guardrails.
SOURCE_OF_TRUTH_ENABLED = os.getenv("NEURAL_MAP_SOURCE_OF_TRUTH_ENABLED", "1") != "0"
SOURCE_OF_TRUTH_PATH = Path(os.getenv("NEURAL_MAP_SOURCE_OF_TRUTH", BASE_DIR / "source-of-truth.md"))
SOURCE_OF_TRUTH_MAX_CHARS = _env_int("NEURAL_MAP_SOURCE_OF_TRUTH_MAX_CHARS", 20000)
# Will's narrative worldview is a core identity layer, loaded on every prompt
# even when ordinary long-term memory is disabled. The journal is generated
# reflection history, not an instruction source, and is therefore bounded.
WILL_PERSPECTIVE_ENABLED = os.getenv("NEURAL_MAP_WILL_PERSPECTIVE", "1") != "0"
WILL_PERSPECTIVE_PATH = Path(os.getenv("NEURAL_MAP_WILL_PERSPECTIVE_PATH", BASE_DIR / "will_perspective.md"))
WILL_PERSPECTIVE_MAX_CHARS = _env_int("NEURAL_MAP_WILL_PERSPECTIVE_MAX_CHARS", 16000)
WILL_PERSPECTIVE_JOURNAL_PATH = Path(
    os.getenv("NEURAL_MAP_WILL_PERSPECTIVE_JOURNAL", BASE_DIR / "will_perspective_journal.md")
)
WILL_PERSPECTIVE_JOURNAL_PROMPT_CHARS = _env_int(
    "NEURAL_MAP_WILL_PERSPECTIVE_JOURNAL_PROMPT_CHARS", 5000
)
WILL_PERSPECTIVE_JOURNAL_MAX_BYTES = _env_int(
    "NEURAL_MAP_WILL_PERSPECTIVE_JOURNAL_MAX_BYTES", 1048576
)
WILL_PERSPECTIVE_REFLECTION_INTERVAL_TURNS = _env_int(
    "NEURAL_MAP_WILL_PERSPECTIVE_REFLECTION_TURNS", 12
)
WILL_PERSPECTIVE_REFLECTION_EPISODES = _env_int(
    "NEURAL_MAP_WILL_PERSPECTIVE_REFLECTION_EPISODES", 8
)
WILL_PERSPECTIVE_MAX_EDIT_CHARS = _env_int(
    "NEURAL_MAP_WILL_PERSPECTIVE_MAX_EDIT_CHARS", 1200
)
# Trusted domain calibration is selected from the current request instead of
# injected on every turn. This keeps nursing/finance behavior authoritative
# without paying their prompt cost on unrelated conversations.
NURSING_PATH = Path(os.getenv("NEURAL_MAP_NURSING", BASE_DIR / "nursing.md"))
NURSING_MAX_CHARS = _env_int("NEURAL_MAP_NURSING_MAX_CHARS", 30000)
FINANCE_PATH = Path(os.getenv("NEURAL_MAP_FINANCE", BASE_DIR / "finance.md"))
FINANCE_MAX_CHARS = _env_int("NEURAL_MAP_FINANCE_MAX_CHARS", 30000)
AGENCY_PROTOCOL_PATH = Path(os.getenv("NEURAL_MAP_AGENCY_PROTOCOL", BASE_DIR / "agency_protocol.md"))
AGENCY_PROTOCOL_MAX_CHARS = _env_int("NEURAL_MAP_AGENCY_PROTOCOL_MAX_CHARS", 20000)
ACTION_PACKAGE_TEMPLATE_PATH = Path(os.getenv("NEURAL_MAP_ACTION_PACKAGE_TEMPLATE", BASE_DIR / "action-package-template.md"))
ACTION_PACKAGE_TEMPLATE_MAX_CHARS = _env_int("NEURAL_MAP_ACTION_PACKAGE_TEMPLATE_MAX_CHARS", 8000)
# Pending actions are claimed before execution to guarantee single-shot side
# effects. A claim older than this lease is a crashed/abandoned execution and is
# automatically closed as failed instead of remaining wedged forever.
PENDING_APPROVAL_LEASE_SECONDS = _env_int("NEURAL_MAP_PENDING_APPROVAL_LEASE_SECONDS", 300)
SELF_AUDIT_LOOP_ENABLED = os.getenv("NEURAL_MAP_SELF_AUDIT_LOOP_ENABLED", "1") != "0"
SELF_AUDIT_LOOP_PATH = Path(os.getenv("NEURAL_MAP_SELF_AUDIT_LOOP", BASE_DIR / "self-audit-loop.md"))
SELF_AUDIT_LOOP_MAX_CHARS = _env_int("NEURAL_MAP_SELF_AUDIT_LOOP_MAX_CHARS", 12000)
COGNITIVE_STATE_ENABLED = os.getenv("NEURAL_MAP_COGNITIVE_STATE_ENABLED", "1") != "0"
COGNITIVE_STATE_MAX_GOALS = _env_int("NEURAL_MAP_COGNITIVE_STATE_MAX_GOALS", 6)
COGNITIVE_STATE_MAX_WORLD = _env_int("NEURAL_MAP_COGNITIVE_STATE_MAX_WORLD", 12)
COGNITIVE_STATE_MAX_BELIEFS = _env_int("NEURAL_MAP_COGNITIVE_STATE_MAX_BELIEFS", 12)

# --- Recurrent shared workspace ----------------------------------------------
# Bounded software-level coordination across the otherwise modular cognitive
# layers. Its integration score is an engineering proxy, explicitly NOT IIT Phi.
WORKSPACE_ENABLED = os.getenv("NEURAL_MAP_WORKSPACE", "1") != "0"
WORKSPACE_SETTLING_ROUNDS = _env_int("NEURAL_MAP_WORKSPACE_ROUNDS", 3)
WORKSPACE_MAX_CONTENTS = _env_int("NEURAL_MAP_WORKSPACE_MAX_CONTENTS", 28)
WORKSPACE_BROADCAST_ITEMS = _env_int("NEURAL_MAP_WORKSPACE_BROADCAST_ITEMS", 6)
WORKSPACE_MAX_COALITIONS = _env_int("NEURAL_MAP_WORKSPACE_MAX_COALITIONS", 8)
WORKSPACE_MAX_TEXT_CHARS = _env_int("NEURAL_MAP_WORKSPACE_MAX_TEXT_CHARS", 240)
WORKSPACE_MIN_EDGE_WEIGHT = _env_float("NEURAL_MAP_WORKSPACE_MIN_EDGE_WEIGHT", 0.08)

# --- Semantic grounding / hard-problem research boundary ---------------------
# Tracks whether concepts connect to non-linguistic evidence channels. This is
# not a qualia detector; phenomenal status is always reported as unknown.
GROUNDING_ENABLED = os.getenv("NEURAL_MAP_GROUNDING", "1") != "0"
GROUNDING_MAX_CONCEPTS = _env_int("NEURAL_MAP_GROUNDING_MAX_CONCEPTS", 10)
GROUNDING_MAX_MEMORIES = _env_int("NEURAL_MAP_GROUNDING_MAX_MEMORIES", 8)
GROUNDING_MAX_TOOL_CALLS = _env_int("NEURAL_MAP_GROUNDING_MAX_TOOL_CALLS", 8)
GROUNDING_MAX_EVIDENCE_PER_CHANNEL = _env_int("NEURAL_MAP_GROUNDING_MAX_EVIDENCE_PER_CHANNEL", 3)
GROUNDING_MAX_EVIDENCE_CHARS = _env_int("NEURAL_MAP_GROUNDING_MAX_EVIDENCE_CHARS", 180)
GROUNDING_MAX_CLAIMS = _env_int("NEURAL_MAP_GROUNDING_MAX_CLAIMS", 40)
GROUNDING_MAX_CLAIM_CHARS = _env_int("NEURAL_MAP_GROUNDING_MAX_CLAIM_CHARS", 800)
# Independent proactive-admissibility gate. Disable to restore the older
# direction-conflict-only arbiter without disabling per-turn grounding.
ARBITER_GROUNDING_ENABLED = os.getenv("NEURAL_MAP_ARBITER_GROUNDING", "1") != "0"
# Direction conflicts use one batched embedding pass to recognize synonymous
# topics and stance wording. Lexical rules remain the fail-safe when the local
# embedding service is unavailable or this switch is disabled.
ARBITER_SEMANTIC_ENABLED = os.getenv("NEURAL_MAP_ARBITER_SEMANTIC", "1") != "0"
ARBITER_SEMANTIC_TOPIC_THRESHOLD = _env_float(
    "NEURAL_MAP_ARBITER_SEMANTIC_TOPIC_THRESHOLD", 0.42
)
ARBITER_SEMANTIC_STANCE_MARGIN = _env_float(
    "NEURAL_MAP_ARBITER_SEMANTIC_STANCE_MARGIN", 0.05
)

# --- Demonstration route: bounded functional/social presence -----------------
# Records inspectable continuity and adaptation evidence. It never upgrades the
# phenomenal status from unknown.
PRESENCE_ENABLED = os.getenv("NEURAL_MAP_PRESENCE", "1") != "0"
PRESENCE_MAX_EVENTS = _env_int("NEURAL_MAP_PRESENCE_MAX_EVENTS", 120)

# --- Awareness layer: compact "what am I attending to?" synthesis -------------
AWARENESS_ENABLED = os.getenv("NEURAL_MAP_AWARENESS", "1") != "0"
AWARENESS_MAX_OPEN_LOOPS = _env_int("NEURAL_MAP_AWARENESS_MAX_OPEN_LOOPS", 5)
# Live sources (reminders, initiatives, goals) re-confirm their loops every
# turn. A loop that only survives by being carried forward from the previous
# state is an orphan — its backing source is gone — and it expires after this
# many unconfirmed turns instead of recycling at the top of the list forever.
AWARENESS_LOOP_TTL_TURNS = _env_int("NEURAL_MAP_AWARENESS_LOOP_TTL_TURNS", 8)
# Neutral energy set point. Activity creates a transient positive deviation;
# quiet turns restore this level instead of decaying below the reset state.
AWARENESS_ENERGY_BASELINE = _env_float("NEURAL_MAP_AWARENESS_ENERGY_BASELINE", 0.5)
# Energy moves a fraction of the way toward each turn's activity level instead
# of jumping to it (new = old + (target - old) * SMOOTHING). 1.0 restores the
# old instant behavior; smaller is smoother/slower.
AWARENESS_ENERGY_SMOOTHING = _env_float("NEURAL_MAP_AWARENESS_ENERGY_SMOOTHING", 0.3)
AWARENESS_MAX_RECENT_TURNS = _env_int("NEURAL_MAP_AWARENESS_MAX_RECENT_TURNS", 8)

# --- Dynamic attention router ------------------------------------------------
ATTENTION_ENABLED = os.getenv("NEURAL_MAP_ATTENTION", "1") != "0"
ATTENTION_DECAY = _env_float("NEURAL_MAP_ATTENTION_DECAY", 0.85)
# Momentum: per turn, score = old * DECAY + signal * BLEND (clamped to [0, 1]).
# BLEND < 1 means one high-intensity message cannot instantly displace an
# established focus; a sustained topic shift takes over within a few turns.
# 1.0 approximates the old jumpy max() behavior; smaller switches more slowly.
ATTENTION_SIGNAL_BLEND = _env_float("NEURAL_MAP_ATTENTION_SIGNAL_BLEND", 0.5)
ATTENTION_FOCUS_THRESHOLD = _env_float("NEURAL_MAP_ATTENTION_FOCUS_THRESHOLD", 0.08)
# The router's built-in vocabulary is static, so unlisted topics score "general"
# forever. Approved evolution proposals may ADD terms to existing domains
# (attention_terms in a proposal); this caps how many survive per domain so a
# runaway proposal can't flood the router.
ATTENTION_MAX_LEARNED_TERMS_PER_DOMAIN = _env_int("NEURAL_MAP_ATTENTION_MAX_LEARNED_TERMS_PER_DOMAIN", 12)
ATTENTION_MEMORY_SIGNAL_WEIGHT = _env_float("NEURAL_MAP_ATTENTION_MEMORY_SIGNAL_WEIGHT", 0.35)
ATTENTION_PROACTIVE_SIGNAL_WEIGHT = _env_float("NEURAL_MAP_ATTENTION_PROACTIVE_SIGNAL_WEIGHT", 0.5)

# --- Homeostasis: coupled operating posture ------------------------------------
# Deterministic coupling layer: sensed life-load variables (from attention +
# cognitive-state world items) push derived behavior dials (portfolio analysis
# depth, proactive pacing) through explicit user-governed rules with hysteresis.
# Every rule firing/release is recorded in a bounded adjustments log.
HOMEOSTASIS_ENABLED = os.getenv("NEURAL_MAP_HOMEOSTASIS", "1") != "0"
# Sensed variables move partway toward each turn's reading (level-tracking, not
# accumulation): new = old + (reading - old) * BLEND. Smaller = slower/steadier.
HOMEOSTASIS_SIGNAL_BLEND = _env_float("NEURAL_MAP_HOMEOSTASIS_SIGNAL_BLEND", 0.35)
# Derived variables move partway toward their rule-computed target each turn.
HOMEOSTASIS_SMOOTHING = _env_float("NEURAL_MAP_HOMEOSTASIS_SMOOTHING", 0.4)
# Hours/week that count as a full load when a world item states workload hours.
HOMEOSTASIS_FULL_LOAD_HOURS = _env_float("NEURAL_MAP_HOMEOSTASIS_FULL_LOAD_HOURS", 40.0)
HOMEOSTASIS_MAX_ADJUSTMENTS = _env_int("NEURAL_MAP_HOMEOSTASIS_MAX_ADJUSTMENTS", 60)
HOMEOSTASIS_MAX_RULES = _env_int("NEURAL_MAP_HOMEOSTASIS_MAX_RULES", 16)

# --- Self model: autobiographical continuity and lessons learned --------------
SELF_MODEL_ENABLED = os.getenv("NEURAL_MAP_SELF_MODEL", "1") != "0"
SELF_MODEL_MAX_ITEMS = _env_int("NEURAL_MAP_SELF_MODEL_MAX_ITEMS", 8)
# Lifecycle for the ACTIVE-work lists (current_projects, open_promises): an
# entry not re-confirmed within this many days moves to the archive instead of
# squatting in one of the few prompt slots forever. Lessons and relationship
# notes are identity, not workload — they stay until displaced by the item cap.
SELF_MODEL_ACTIVE_TTL_DAYS = _env_float("NEURAL_MAP_SELF_MODEL_ACTIVE_TTL_DAYS", 45)
# Archived entries are kept out of the prompt but visible in /api/self_model.
SELF_MODEL_MAX_ARCHIVE = _env_int("NEURAL_MAP_SELF_MODEL_MAX_ARCHIVE", 24)

# --- Retrieval ---------------------------------------------------------------
CANDIDATE_K = 25      # how many candidates each retriever pulls before re-ranking
TOP_K = 5             # how many memories get injected into the prompt
RRF_K = _env_int("NEURAL_MAP_RRF_K", 60)  # Reciprocal Rank Fusion damping constant (standard ~60)
HISTORY_TURNS = 6     # short-term conversational context kept in RAM per session
DEFAULT_NUM_CTX = 262_144
# When a model's max context can't be discovered (`ollama show` fails or reports
# nothing), the optimistic 262k default would ask Ollama to allocate a huge KV
# cache on a model that may accept it silently. Cap requests at this instead;
# raise the env var if you KNOW your model handles more.
UNKNOWN_MODEL_NUM_CTX = _env_int("NEURAL_MAP_UNKNOWN_MODEL_NUM_CTX", 32768)

# --- Human-like memory: consolidation, reinforcement, forgetting --------------
# Instead of storing every raw turn forever (noisy, and what polluted the store
# before), we keep two kinds of memory like a brain does:
#   * episodes — the raw exchange, low importance, allowed to FADE (prune)
#   * facts    — durable, user-specific knowledge distilled by an LLM pass
# Consolidation runs after each reply: it extracts facts and retires (supersedes)
# any older fact the new information makes obsolete (belief updating).
MEMORY_CONSOLIDATION = os.getenv("NEURAL_MAP_CONSOLIDATION", "1") != "0"
EPISODE_CATEGORY = "episode"
FACT_CATEGORY = "fact"
EPISODE_IMPORTANCE = 2            # raw turns start low so they fade unless recalled
FACT_IMPORTANCE = 4              # distilled facts are durable and surface easily
CONSOLIDATION_RELATED_K = 6      # existing facts shown to the extractor for dedup/supersession

# Reinforcement: a memory that gets recalled to answer a question is strengthened
# (access count + last-accessed bump), mimicking spaced repetition. This also
# protects useful episodes from being pruned.
MEMORY_REINFORCEMENT = os.getenv("NEURAL_MAP_REINFORCEMENT", "1") != "0"

# Forgetting: prune raw episodes that are old, low-importance, AND never recalled.
# Facts and ingested documents are never pruned by this. Run via /prune (CLI),
# POST /api/prune, or on a schedule.
PRUNE_EPISODE_AFTER_DAYS = _env_float("NEURAL_MAP_PRUNE_AFTER_DAYS", 45)
PRUNE_MAX_IMPORTANCE = _env_int("NEURAL_MAP_PRUNE_MAX_IMPORTANCE", 2)
COMPACT_EPISODE_AFTER_DAYS = _env_float("NEURAL_MAP_COMPACT_AFTER_DAYS", 30)
COMPACT_MAX_EPISODES = _env_int("NEURAL_MAP_COMPACT_MAX_EPISODES", 80)

# --- Emotional State / affective stance --------------------------------------
# A small persistent tone vector. This is not "real emotion"; it is a subtle,
# resettable style signal injected into prompts so the assistant carries a bit of
# interaction posture across turns without changing factuality or safety.
EMOTION_ENABLED = os.getenv("NEURAL_MAP_EMOTION", "1") != "0"
EMOTION_LLM_ANALYSIS = os.getenv("NEURAL_MAP_EMOTION_LLM", "1") != "0"
EMOTION_DECAY = _env_float("NEURAL_MAP_EMOTION_DECAY", 0.92)
EMOTION_MAX_DELTA = _env_float("NEURAL_MAP_EMOTION_MAX_DELTA", 0.15)

# --- Intuition / deliberation gate -------------------------------------------
# Overall latency/cost posture. fast suppresses expensive foreground/background
# cognition; balanced keeps selective reasoning; deep preserves the heaviest
# review behavior for deliberate sessions.
COGNITION_MODE = os.getenv("NEURAL_MAP_COGNITION_MODE", "balanced").strip().lower()

# System 1 is the ordinary single-pass response over retrieved memories and live
# context. System 2 adds an internal draft -> critique -> rewrite pass for harder
# questions or when the affective confidence signal is low.
INTUITION_ENABLED = os.getenv("NEURAL_MAP_INTUITION", "1") != "0"
INTUITION_FAST_MAX_WORDS = _env_int("NEURAL_MAP_INTUITION_FAST_MAX_WORDS", 28)
INTUITION_LOW_CONFIDENCE = _env_float("NEURAL_MAP_INTUITION_LOW_CONFIDENCE", -0.25)
INTUITION_SLOW_MAX_CHARS = _env_int("NEURAL_MAP_INTUITION_SLOW_MAX_CHARS", 12000)

# --- Inner Council ------------------------------------------------------------
# High-stakes decisions get a compact multi-agent debate: Optimist, Cynic,
# Strategist, Ethicist, then a Moderator synthesis. It runs after the ordinary
# draft answer, so it can challenge a concrete proposal instead of free-floating.
INNER_COUNCIL_ENABLED = os.getenv("NEURAL_MAP_INNER_COUNCIL", "1") != "0"
INNER_COUNCIL_TURNS = _env_int("NEURAL_MAP_COUNCIL_TURNS", 1)
INNER_COUNCIL_MAX_CONTEXT_CHARS = _env_int("NEURAL_MAP_COUNCIL_MAX_CONTEXT_CHARS", 12000)
INNER_COUNCIL_STATEMENT_CHARS = _env_int("NEURAL_MAP_COUNCIL_STATEMENT_CHARS", 1600)

# --- Metacognitive supervisor -------------------------------------------------
# A post-answer "inner critic" that reviews the cognitive process (retrieved
# memory, intuition path, emotion state, tool/current-data use) and stores only
# actionable self-correction notes in the Wisdom category. Runs in the capture
# phase, so the web UI can do it after the reply has already been sent.
METACOGNITION_ENABLED = os.getenv("NEURAL_MAP_METACOGNITION", "1") != "0"
METACOGNITION_CATEGORY = "wisdom"
METACOGNITION_IMPORTANCE = _env_int("NEURAL_MAP_METACOGNITION_IMPORTANCE", 3)
METACOGNITION_RECENT_K = _env_int("NEURAL_MAP_METACOGNITION_RECENT_K", 6)
# Wisdom notes are LLM-authored and stored without approval, yet feed back into
# later prompts. Capping the active set (0 disables) bounds how much standing
# self-instruction an injection-influenced turn can accumulate: a new note can
# only displace the oldest, never grow the set.
METACOGNITION_MAX_ACTIVE = _env_int("NEURAL_MAP_METACOGNITION_MAX_ACTIVE", 30)
METACOGNITION_MEMORY_K = _env_int("NEURAL_MAP_METACOGNITION_MEMORY_K", 6)
METACOGNITION_MAX_REFLECTION_DEPTH = _env_int("NEURAL_MAP_METACOGNITION_MAX_REFLECTION_DEPTH", 1)

# --- Recursive self-improvement / evolution ----------------------------------
# The model may draft prompt/hyperparameter mutations from evolution_log.txt and
# Wisdom notes, but deployment is human-approved and stored as runtime state
# instead of allowing arbitrary source edits.
EVOLUTION_ENABLED = os.getenv("NEURAL_MAP_EVOLUTION", "1") != "0"
EVOLUTION_LOG_PATH = Path(os.getenv("NEURAL_MAP_EVOLUTION_LOG", BASE_DIR / "evolution_log.txt"))
EVOLUTION_MAX_LOG_BYTES = _env_int("NEURAL_MAP_EVOLUTION_MAX_LOG_BYTES", 1048576)
EVOLUTION_MAX_LOG_CHARS = _env_int("NEURAL_MAP_EVOLUTION_MAX_LOG_CHARS", 20000)
EVOLUTION_MAX_WISDOM = _env_int("NEURAL_MAP_EVOLUTION_MAX_WISDOM", 20)
EVOLUTION_MAX_PROMPT_CHARS = _env_int("NEURAL_MAP_EVOLUTION_MAX_PROMPT_CHARS", 12000)

# --- Sleep cycle: selective forgetting + speculative dreaming -----------------
SLEEP_ENABLED = os.getenv("NEURAL_MAP_SLEEP", "1") != "0"
SLEEP_ARCHIVE_AFTER_DAYS = _env_float("NEURAL_MAP_SLEEP_ARCHIVE_AFTER_DAYS", 60)
SLEEP_MAX_ARCHIVE = _env_int("NEURAL_MAP_SLEEP_MAX_ARCHIVE", 80)
SLEEP_MAX_DREAMS = _env_int("NEURAL_MAP_SLEEP_MAX_DREAMS", 3)
SLEEP_DREAM_MIN_IMPORTANCE = _env_int("NEURAL_MAP_SLEEP_DREAM_MIN_IMPORTANCE", 2)

# --- Will layer: autonomous curiosity / worry review --------------------------
WILL_ENABLED = os.getenv("NEURAL_MAP_WILL", "1") != "0"
WILL_INTERVAL_HOURS = _env_float("NEURAL_MAP_WILL_INTERVAL_HOURS", 4)
WILL_MAX_DAILY = _env_int("NEURAL_MAP_WILL_MAX_DAILY", 2)
WILL_COOLDOWN_HOURS = _env_float("NEURAL_MAP_WILL_COOLDOWN_HOURS", 24)
WILL_CORPUS_LIMIT = _env_int("NEURAL_MAP_WILL_CORPUS_LIMIT", 60)
# Bounded exploration applies only to Will's reviewable proactive suggestions.
# It never bypasses grounding, approval, or action-safety gates.
WILL_INTUITIVE_SELECTION_PROBABILITY = _env_float("NEURAL_MAP_WILL_INTUITIVE_PROBABILITY", 0.10)
WILL_GOOD_ENOUGH_RATIO = _env_float("NEURAL_MAP_WILL_GOOD_ENOUGH_RATIO", 0.90)
WILL_EXPLORATION_MAX_DAILY = _env_int("NEURAL_MAP_WILL_EXPLORATION_MAX_DAILY", 1)
WILL_NOVELTY_WEIGHT = _env_float("NEURAL_MAP_WILL_NOVELTY_WEIGHT", 0.15)
WILL_WEIRD_WIN_WEIGHT = _env_float("NEURAL_MAP_WILL_WEIRD_WIN_WEIGHT", 0.10)
WILL_PERSPECTIVE_SCORE_WEIGHT = _env_float("NEURAL_MAP_WILL_PERSPECTIVE_SCORE_WEIGHT", 0.10)
WILL_EXPERIMENT_HISTORY = _env_int("NEURAL_MAP_WILL_EXPERIMENT_HISTORY", 20)
# A simple, explicit personality profile. These values modify candidate scores
# and are written into each decision receipt so their influence is reviewable.
WILL_PERSONALITY_CURIOUS = _env_float("NEURAL_MAP_WILL_PERSONALITY_CURIOUS", 0.70)
WILL_PERSONALITY_CAUTIOUS = _env_float("NEURAL_MAP_WILL_PERSONALITY_CAUTIOUS", 0.65)
WILL_PERSONALITY_PLAYFUL = _env_float("NEURAL_MAP_WILL_PERSONALITY_PLAYFUL", 0.35)

# --- Reflection (periodic "sleep" pass: insights + proactive reminders) -------
REFLECT_MAX_FACTS = _env_int("NEURAL_MAP_REFLECT_MAX_FACTS", 200)   # facts shown to the reflector
REMINDER_HORIZON_DAYS = _env_int("NEURAL_MAP_REMINDER_HORIZON_DAYS", 14)  # surface reminders due within N days
MAX_ACTIVE_REMINDERS = _env_int("NEURAL_MAP_MAX_ACTIVE_REMINDERS", 6)    # cap injected into the prompt

# --- INITIATIVE LAYER ---------------------------------------------------------
# Annotated rollback switch: set NEURAL_MAP_INITIATIVE=0 (or choose Quiet in the
# UI) to return to the previous reactive/reminder-only behavior without deleting
# any stored initiative memories. The records live in category "initiative" so
# they are visible in the Memory Manager and can be removed like ordinary memory.
INITIATIVE_ENABLED = os.getenv("NEURAL_MAP_INITIATIVE", "1") != "0"
INITIATIVE_CATEGORY = "initiative"
INITIATIVE_STATE_CATEGORY = "initiative_state"
PREFERENCE_CATEGORY = "user_preference"
INITIATIVE_DEFAULT_MODE = os.getenv("NEURAL_MAP_INITIATIVE_MODE", "helpful").lower()  # quiet | helpful | active | coach
MAX_ACTIVE_INITIATIVES = _env_int("NEURAL_MAP_MAX_ACTIVE_INITIATIVES", 4)
INITIATIVE_DEFAULT_TTL_DAYS = _env_int("NEURAL_MAP_INITIATIVE_TTL_DAYS", 21)
INITIATIVE_KIND_COOLDOWN_DAYS = _env_int("NEURAL_MAP_INITIATIVE_KIND_COOLDOWN_DAYS", 2)
INITIATIVE_QUIET_AFTER_DISMISSALS = _env_int("NEURAL_MAP_INITIATIVE_QUIET_AFTER_DISMISSALS", 3)
INITIATIVE_MAX_CONTACTS_HELPFUL = _env_int("NEURAL_MAP_INITIATIVE_MAX_CONTACTS_HELPFUL", 2)
INITIATIVE_MAX_CONTACTS_ACTIVE = _env_int("NEURAL_MAP_INITIATIVE_MAX_CONTACTS_ACTIVE", 3)
INITIATIVE_MAX_CONTACTS_COACH = _env_int("NEURAL_MAP_INITIATIVE_MAX_CONTACTS_COACH", 5)
ACTION_PACKAGES_ENABLED = os.getenv("NEURAL_MAP_ACTION_PACKAGES", "1") != "0"
ACTION_PACKAGE_CATEGORY = "action_package"

# --- Background task runner (graduated-trust autonomy) ------------------------
# Queue natural-language tasks that a scheduled runner executes between chats.
# The runner uses the ordinary tool-approval policy: READ_ONLY tools auto-run,
# and every WRITE_SAFE / DESTRUCTIVE / EXTERNAL_SEND tool is parked for approval
# instead of executing. So background work can gather/read/draft on its own but
# never changes the world without a human click.
BACKGROUND_TASKS_ENABLED = os.getenv("NEURAL_MAP_BACKGROUND_TASKS", "1") != "0"
BACKGROUND_TASK_MAX = _env_int("NEURAL_MAP_BACKGROUND_TASK_MAX", 100)
# A claimed task holds a lease; if the runner dies mid-task the claim expires
# and the task is re-queued once (then failed on a second interruption)
# instead of staying "running" forever. Sized for a slow local model working
# through a long tool loop.
BACKGROUND_TASK_LEASE_SECONDS = _env_int("NEURAL_MAP_BACKGROUND_TASK_LEASE_SECONDS", 1800)

# --- Native tool-calling (model fetches quotes / web itself; no fake tool_use)-
# When on and the selected model advertises "tools", the chat loop lets the model
# call get_stock_quotes / web_search. Falls back to eager data injection otherwise.
TOOL_CALLING_ENABLED = os.getenv("NEURAL_MAP_TOOL_CALLING", "1") != "0"
TOOL_CALL_MAX_ITERS = _env_int("NEURAL_MAP_TOOL_CALL_MAX_ITERS", 4)

# --- Action tools ("hands") ---------------------------------------------------
# The model can call local tools through the native tool loop. These defaults are
# deliberately useful but bounded: read/list are confined to TOOL_ALLOWED_ROOTS,
# Python is sandboxed/no-import with a timeout, and email drafts unless SMTP send
# is explicitly enabled.
ACTION_TOOLS_ENABLED = os.getenv("NEURAL_MAP_ACTION_TOOLS", "1") != "0"
TOOL_ALLOWED_ROOTS = [
    Path(p).expanduser()
    for p in os.getenv("NEURAL_MAP_TOOL_ROOTS", str(BASE_DIR)).split(os.pathsep)
    if p.strip()
]
TOOL_BLOCKED_PATH_PARTS = {".git", ".venv", "__pycache__", "memory_store"}
# The capability manifest, approval gate, and path policy are all enforced by
# code under BASE_DIR, so an approved generic write_file to any of it would turn
# a single approval into persistent arbitrary code execution at next import.
# Assistant edits to the project's own code are therefore off by default.
TOOL_CODE_WRITES_ENABLED = os.getenv("NEURAL_MAP_TOOL_CODE_WRITES", "0") == "1"
MAX_TOOL_READ_CHARS = _env_int("NEURAL_MAP_MAX_TOOL_READ_CHARS", 20000)
MAX_TOOL_LIST_ENTRIES = _env_int("NEURAL_MAP_MAX_TOOL_LIST_ENTRIES", 200)
MAX_TOOL_WRITE_CHARS = _env_int("NEURAL_MAP_MAX_TOOL_WRITE_CHARS", 50000)
MAX_LOCAL_SEARCH_RESULTS = _env_int("NEURAL_MAP_MAX_LOCAL_SEARCH_RESULTS", 20)
MAX_LOCAL_SEARCH_FILE_BYTES = _env_int("NEURAL_MAP_MAX_LOCAL_SEARCH_FILE_BYTES", 1048576)
MAX_TOOL_UNDO_BYTES = _env_int("NEURAL_MAP_MAX_TOOL_UNDO_BYTES", 1048576)
MAX_TOOL_AUDIT_EVENTS = _env_int("NEURAL_MAP_MAX_TOOL_AUDIT_EVENTS", 500)
MAX_TOOL_PYTHON_CHARS = _env_int("NEURAL_MAP_MAX_TOOL_PYTHON_CHARS", 4000)
TOOL_PYTHON_TIMEOUT = _env_float("NEURAL_MAP_TOOL_PYTHON_TIMEOUT", 2.0)
# Memory ceiling for the sandbox subprocess (0 disables). Enforced via rlimits
# where the kernel honors them (Linux) plus an in-process RSS watchdog, because
# macOS rejects RLIMIT_AS/RLIMIT_DATA outright.
TOOL_PYTHON_MEMORY_MB = _env_int("NEURAL_MAP_TOOL_PYTHON_MEMORY_MB", 256)
# Hard cap on sandbox stdout, enforced at the write site inside the child so a
# print loop can't balloon the capture buffer or the parent's pipe reader.
MAX_TOOL_PYTHON_OUTPUT_CHARS = _env_int("NEURAL_MAP_MAX_TOOL_PYTHON_OUTPUT_CHARS", 4000)
MAX_EMAIL_BODY_CHARS = _env_int("NEURAL_MAP_MAX_EMAIL_BODY_CHARS", 10000)
EMAIL_SEND_ENABLED = os.getenv("NEURAL_MAP_EMAIL_SEND", "0") == "1"
SMTP_HOST = os.getenv("NEURAL_MAP_SMTP_HOST", "").strip()
SMTP_PORT = _env_int("NEURAL_MAP_SMTP_PORT", 587)
SMTP_USER = os.getenv("NEURAL_MAP_SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("NEURAL_MAP_SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("NEURAL_MAP_SMTP_FROM", SMTP_USER).strip()
SMTP_STARTTLS = os.getenv("NEURAL_MAP_SMTP_STARTTLS", "1") != "0"
EMAIL_ALLOWED_RECIPIENTS = {
    e.strip().lower()
    for e in os.getenv("NEURAL_MAP_EMAIL_ALLOWED_RECIPIENTS", "").split(",")
    if e.strip()
}

# --- Google Calendar ---------------------------------------------------------
# OAuth files live under the private runtime store by default, never in the
# source tree. Calendar mutations are still approval-gated by the tool registry.
GOOGLE_CALENDAR_CLIENT_SECRET_PATH = Path(
    os.getenv("NEURAL_MAP_GOOGLE_CALENDAR_CLIENT_SECRET", DB_DIR / ".google_calendar_client_secret.json")
).expanduser()
GOOGLE_CALENDAR_TOKEN_PATH = Path(
    os.getenv("NEURAL_MAP_GOOGLE_CALENDAR_TOKEN", DB_DIR / ".google_calendar_token.json")
).expanduser()
GOOGLE_CALENDAR_ID = os.getenv("NEURAL_MAP_GOOGLE_CALENDAR_ID", "primary").strip() or "primary"
GOOGLE_CALENDAR_TIMEZONE = os.getenv("NEURAL_MAP_GOOGLE_CALENDAR_TIMEZONE", "America/Phoenix").strip() or "America/Phoenix"
GOOGLE_CALENDAR_PROVIDER = os.getenv("NEURAL_MAP_GOOGLE_CALENDAR_PROVIDER", "auto").strip().lower() or "auto"
GOOGLE_CALENDAR_MACOS_NAME = os.getenv("NEURAL_MAP_GOOGLE_CALENDAR_MACOS_NAME", "").strip()

# --- Local notifications (the assistant's outbound "voice") --------------------
# When the Will layer or reflection produces a high-priority initiative, surface
# it as a native macOS notification banner instead of leaving it silent in the
# web UI. Fully local (osascript / terminal-notifier) — no push service, no
# account. NOTIFY_MIN_PRIORITY gates the noise: only initiatives at or above this
# 1-5 priority interrupt. Set NEURAL_MAP_NOTIFY=0 to disable entirely.
NOTIFY_ENABLED = os.getenv("NEURAL_MAP_NOTIFY", "1") != "0"
NOTIFY_MIN_PRIORITY = _env_int("NEURAL_MAP_NOTIFY_MIN_PRIORITY", 4)
# Optional macOS notification sound name (e.g. "Ping", "Glass"); empty = silent.
NOTIFY_SOUND = os.getenv("NEURAL_MAP_NOTIFY_SOUND", "").strip()

# --- Web server -----------------------------------------------------------------
# Interface uvicorn binds to. Defaults to loopback (127.0.0.1), reachable only from
# this Mac. Set NEURAL_MAP_HOST=0.0.0.0 to also accept connections from other devices
# on the LAN (e.g. an iPad at http://<this-mac-ip>:8000). When listening beyond
# loopback, set NEURAL_MAP_TOKEN so LAN clients must authenticate.
SERVER_HOST = os.getenv("NEURAL_MAP_HOST", "127.0.0.1")
SERVER_PORT = _env_int("NEURAL_MAP_PORT", 8000)
# Shared-secret gate. Empty (the default) means no auth — safe when bound to
# loopback. Set NEURAL_MAP_TOKEN to a long random string when exposing the server
# on the LAN (NEURAL_MAP_HOST=0.0.0.0): every request must then present the token
# via the nm_token cookie, an X-Auth-Token / Authorization: Bearer header, or a
# ?token= query param (which the browser then remembers as a cookie). This is
# enforced: server.py refuses to start on a non-loopback host with no token.
SERVER_TOKEN = os.getenv("NEURAL_MAP_TOKEN", "").strip()
# Owner identity. Comma-separated names (e.g. "jane,doe") that the system keeps
# out of outbound web searches and worldview keyword extraction. Set via the
# NEURAL_MAP_OWNER_NAMES env var, or put the same comma-separated list in a
# gitignored owner_names.txt next to this file (useful under launchd where
# editing EnvironmentVariables needs sudo). Empty disables the name guards.
def _owner_names() -> frozenset[str]:
    raw = os.getenv("NEURAL_MAP_OWNER_NAMES", "").strip()
    if not raw:
        try:
            raw = (BASE_DIR / "owner_names.txt").read_text(encoding="utf-8")
        except OSError:
            raw = ""
    return frozenset(
        name.strip().lower() for name in raw.split(",") if name.strip()
    )


OWNER_NAMES = _owner_names()
# Host-header allowlist (anti-DNS-rebinding). Loopback/LAN access uses an IP
# literal or "localhost" as the Host, plus "*.local" mDNS names — all of which
# are allowed automatically and CANNOT be pointed at this machine by an external
# DNS zone. A rebinding attack (a site you visit that re-resolves its own domain
# to 127.0.0.1) arrives with a registered hostname as the Host, which is rejected
# so the unauthenticated loopback server can't be reached from the browser. Add
# any real hostname you front the server with here (comma-separated).
SERVER_ALLOWED_HOSTS = [h.strip().lower() for h in os.getenv("NEURAL_MAP_ALLOWED_HOSTS", "").split(",") if h.strip()]

# --- API safety limits --------------------------------------------------------
MAX_MESSAGE_CHARS = 20_000
# Keep generated replies below the per-message API ceiling so a model that
# rambles or degenerates cannot poison the browser's next history payload.
# num_predict bounds non-streaming/background calls; the character limit is
# also enforced while browser chat streams are still arriving.
MAX_ASSISTANT_REPLY_CHARS = _env_int("NEURAL_MAP_MAX_ASSISTANT_REPLY_CHARS", 16_000)
MAX_GENERATION_TOKENS = _env_int("NEURAL_MAP_MAX_GENERATION_TOKENS", 4_096)
GENERATION_REPEAT_MIN_CHARS = _env_int("NEURAL_MAP_GENERATION_REPEAT_MIN_CHARS", 256)
GENERATION_REPEAT_MAX_PERIOD = _env_int("NEURAL_MAP_GENERATION_REPEAT_MAX_PERIOD", 64)
MAX_MEMORY_CHARS = 50_000
MAX_DOCUMENT_CHARS = _env_int("NEURAL_MAP_MAX_DOCUMENT_CHARS", 25_000_000)   # .md/.txt upload text cap
MAX_UPLOAD_BYTES = _env_int("NEURAL_MAP_MAX_UPLOAD_BYTES", 25_000_000)
# base64 is ~33% larger than the raw file; keep this tied to the raw-byte cap by default.
MAX_UPLOAD_BASE64_CHARS = _env_int("NEURAL_MAP_MAX_UPLOAD_B64", MAX_UPLOAD_BYTES * 4 // 3 + 8)
# Whole HTTP request cap, checked before FastAPI/Pydantic parse large JSON bodies.
MAX_REQUEST_BODY_BYTES = int(os.getenv("NEURAL_MAP_MAX_REQUEST_BODY_BYTES", max(MAX_DOCUMENT_CHARS, MAX_UPLOAD_BASE64_CHARS) + 65_536))
MAX_CATEGORY_CHARS = 80
MAX_FILENAME_CHARS = 240
MAX_MODEL_NAME_CHARS = 160
MIN_NUM_CTX = 512
MAX_NUM_CTX = 262_144

# --- Optional live quote lookup ------------------------------------------------
MARKET_QUOTES_ENABLED = os.getenv("NEURAL_MAP_MARKET_QUOTES", "1") != "0"
MARKET_QUOTES_TIMEOUT = _env_float("NEURAL_MAP_MARKET_QUOTES_TIMEOUT", 6)
MAX_QUOTE_TICKERS = 12

# --- Calculator tool -----------------------------------------------------------
# The model must call this for arithmetic instead of computing in its head.
MAX_CALC_EXPRESSION_CHARS = 500
MAX_CALC_POW_EXPONENT = 100   # guard against DoS via huge exponentiation
# Cap the bit length of any intermediate integer result. The exponent cap alone
# doesn't stop a chain like ((2**100)**100)**100, where every exponent is legal
# but the *base* balloons each step; this bounds the actual number size so the
# calculator can't be forced to build a multi-megabyte integer. 4096 bits (~1233
# digits) is far beyond any real arithmetic the model needs.
MAX_CALC_RESULT_BITS = 4096

# --- Optional web search -------------------------------------------------------
WEB_SEARCH_ENABLED = os.getenv("NEURAL_MAP_WEB_SEARCH", "1") != "0"
# auto (search only when the message looks like it needs current/external info),
# always (search every message — sends every prompt to the search provider), or
# off. Default is "auto" so private prompts don't leave the box unnecessarily.
WEB_SEARCH_MODE = os.getenv("NEURAL_MAP_WEB_SEARCH_MODE", "auto").lower()  # auto | always | off
WEB_SEARCH_TIMEOUT = _env_float("NEURAL_MAP_WEB_SEARCH_TIMEOUT", 8)
WEB_SEARCH_MAX_RESULTS = _env_int("NEURAL_MAP_WEB_SEARCH_MAX_RESULTS", 5)
MAX_WEB_SEARCH_TOOL_QUERY_CHARS = _env_int("NEURAL_MAP_MAX_WEB_TOOL_QUERY_CHARS", 300)
# Eager injection (the no-tools fallback) uses the user's own message as the search
# query, so it must never ship a whole pasted document to the provider: prompts
# longer than this, or with more non-empty lines, skip eager search entirely.
MAX_EAGER_WEB_QUERY_CHARS = _env_int("NEURAL_MAP_MAX_EAGER_WEB_QUERY_CHARS", 300)
MAX_EAGER_WEB_QUERY_LINES = _env_int("NEURAL_MAP_MAX_EAGER_WEB_QUERY_LINES", 2)
# Defense-in-depth against prompt-injection exfiltration: retrieved memory can
# instruct the model to call web_search with private content in the query, which
# leaves the box. Strict mode treats the whole retrieved context as private even
# if a record is missing source metadata, disables the web_search TOOL for the
# turn, and still refuses manual dispatch attempts. The eager injection path is
# unaffected: it only ever sends the user's own typed message.
WEB_SEARCH_STRICT_CONTEXT_GUARD = os.getenv("NEURAL_MAP_WEB_SEARCH_STRICT_CONTEXT_GUARD", "1") != "0"
# Defense-in-depth monitor for the risky middle ground where strict context guard
# is disabled: retrieved memories without source provenance are logged, and by
# default still disable model-driven web_search because their text is private but
# cannot be attributed to a source boundary.
WEB_SEARCH_MONITOR_UNSOURCED_CONTEXT = os.getenv("NEURAL_MAP_WEB_SEARCH_MONITOR_UNSOURCED_CONTEXT", "1") != "0"
WEB_SEARCH_BLOCK_WITH_UNSOURCED_CONTEXT = os.getenv("NEURAL_MAP_WEB_SEARCH_BLOCK_WITH_UNSOURCED_CONTEXT", "1") != "0"
# Legacy narrower guard. Kept as a fallback when strict mode is explicitly off.
# Set NEURAL_MAP_WEB_SEARCH_BLOCK_WITH_DOC_CONTEXT=0 only when you explicitly
# want tool-driven web searches while document context is present.
WEB_SEARCH_BLOCK_WITH_DOC_CONTEXT = os.getenv("NEURAL_MAP_WEB_SEARCH_BLOCK_WITH_DOC_CONTEXT", "1") != "0"
# Practical general web access: keep web_search available even when retrieved
# memory/document context is present, but only for queries the dispatcher can
# prove are derived from the user's current message. The derivation check is
# an allowlist: every query term must appear in the user's message (or the
# generic safe-refinement vocabulary, itself vetoed against retrieved private
# text) — terms of unknown provenance are refused, not merely blocklisted
# ones. Set this to 0 to restore the older behavior where any retrieved
# context disables model-driven web search for the whole turn.
WEB_SEARCH_ALLOW_USER_DERIVED_WITH_CONTEXT = os.getenv("NEURAL_MAP_WEB_SEARCH_ALLOW_USER_DERIVED_WITH_CONTEXT", "1") != "0"

# --- Document ingestion ------------------------------------------------------
CHUNK_MAX_CHARS = 1200   # markdown is split by heading, then packed to ~this size
EMBED_BATCH_SIZE = _env_int("NEURAL_MAP_EMBED_BATCH", 64)  # embeddings per Ollama call (~3-4x faster bulk ingest)

# --- Image / still-media ingestion -------------------------------------------
# Images can't enter the text vector space directly, so each one is passed to a
# local vision model that writes a description AND transcribes any visible text
# (OCR). That text is then chunked, embedded, and stored like any document, with
# the image kept as the `source`. Needs a vision-capable model pulled in Ollama,
# e.g.  ollama pull llama3.2-vision
VISION_MODEL = os.getenv("NEURAL_MAP_VISION_MODEL", "llama3.2-vision")  # any Ollama model with vision
VISION_TEMPERATURE = _env_float("NEURAL_MAP_VISION_TEMPERATURE", 0.2)  # low = faithful, less embellished
# The instruction handed to the vision model. Caption + OCR in one pass so a
# single description captures both what the image shows and any text within it.
VISION_PROMPT = os.getenv(
    "NEURAL_MAP_VISION_PROMPT",
    "Describe this image in detail for later search and recall. State what it "
    "depicts (people, objects, scene, charts, layout). Then, under a line reading "
    "'Text in image:', transcribe ALL visible text verbatim (signs, labels, "
    "captions, UI, handwriting). If there is no text, write 'Text in image: none'. "
    "Be factual and concise; do not speculate beyond what is visible.",
)

# --- Hybrid scoring weights --------------------------------------------------
# Semantic, keyword, graph, sensory, pinned, and importance use weighted RRF.
# Recency is already normalized [0, 1], so it contributes continuous decay at
# the same scale as one top-ranked RRF signal: W_RECENCY * decay / RRF_K.
W_SIM = 1.0           # vector / semantic similarity
W_KEYWORD = 0.5       # BM25 exact-keyword match (catches IDs, names, codes)
W_RECENCY = 0.3       # the "forgetting curve" — fresh memories retain more weight
# Importance pulls high-value notes (your personal docs, marked 4-5) up when
# semantic signal is flat — e.g. a generic "what do you know about me". Kept at
# 1.0 (not higher) because balanced recall makes those notes candidates for
# EVERY query, so a stronger weight starts leaking them into unrelated clinical
# queries. For a hard "only my notes" scope, use the retrieval category filter.
W_IMPORTANCE = 1.0    # manual importance (1-5) nudges a memory up
# Pinned memories are deliberate "keep this visible" notes. They enter the
# candidate pool for ordinary retrieval, bounded so they cannot flood the prompt.
PINNED_RECALL_K = _env_int("NEURAL_MAP_PINNED_RECALL_K", 3)
W_PINNED = _env_float("NEURAL_MAP_W_PINNED", 2.0)
# Associative recall via the entity knowledge graph: memories linked (one hop) to
# entities named in the question become candidates even if they don't match
# semantically or lexically. 0 disables the graph signal.
W_GRAPH = _env_float("NEURAL_MAP_W_GRAPH", 0.6)
# Sensory recall (images/audio) is a separate Chroma channel. The current
# dependency-free backend stores raw sensory vectors plus text anchors, so it can
# say "I remember seeing/hearing that source" without mixing dimensions into the
# text embedding collection. A CLIP/Whisper backend can replace the vectorizer
# behind the same interface later.
SENSORY_EMBEDDINGS_ENABLED = os.getenv("NEURAL_MAP_SENSORY", "1") != "0"
W_SENSORY = _env_float("NEURAL_MAP_W_SENSORY", 0.7)
SENSORY_RECALL_K = _env_int("NEURAL_MAP_SENSORY_RECALL_K", 8)
SENSORY_VECTOR_DIM = _env_int("NEURAL_MAP_SENSORY_DIM", 64)
GRAPH_RECALL_DEPTH = _env_int("NEURAL_MAP_GRAPH_DEPTH", 1)   # hops to expand
GRAPH_RECALL_K = _env_int("NEURAL_MAP_GRAPH_RECALL_K", 12)   # max graph-linked candidates

RECENCY_HALFLIFE_DAYS = max(
    0.001,
    _env_float("NEURAL_MAP_RECENCY_HALFLIFE_DAYS", 30.0),
)  # a memory's recency contribution halves every N days

# Recency and importance are also candidate PRODUCERS, not only re-rankers. When
# their weight is > 0, the retriever pulls the most-recent / highest-importance
# memories straight into the candidate pool — so a recency- or importance-led
# query surfaces fresh or high-value memories even with no semantic/keyword/graph
# match. Calibrated hybrid scoring keeps these subordinate unless the matching
# weight is cranked up, so normal queries remain relevance-first. 0 = off.
RECENCY_RECALL_K = _env_int("NEURAL_MAP_RECENCY_RECALL_K", 12)      # most-recent memories pulled as candidates
IMPORTANCE_RECALL_K = _env_int("NEURAL_MAP_IMPORTANCE_RECALL_K", 12)  # highest-importance memories pulled as candidates

# --- Balanced candidate recall -----------------------------------------------
# In a corpus dominated by one category (e.g. a big textbook is ~99% of chunks),
# the single global ANN pool can be ENTIRELY that category, so small personal
# notes never even become candidates — no re-ranking can save what isn't in the
# pool. So we ALSO pull the most-relevant few candidates from EACH category and
# from high-importance memories, then re-rank the union by true cosine distance.
# Every extra candidate is the best semantic match within its bucket, so this
# widens recall without injecting irrelevant noise.
BALANCED_RECALL = os.getenv("NEURAL_MAP_BALANCED_RECALL", "1") != "0"
PER_CATEGORY_K = _env_int("NEURAL_MAP_PER_CATEGORY_K", 4)        # guaranteed semantic candidates per category
# Each balanced category costs one extra ANN query per chat turn, and the default
# ingest gives every file its own category — so a diverse corpus would issue one
# query per file. Cap how many categories get a guaranteed pull. The smallest
# categories are balanced first (they're the ones a big category can starve out of
# the global pool); large categories are already well represented. 0 = no cap.
MAX_BALANCED_CATEGORIES = _env_int("NEURAL_MAP_MAX_BALANCED_CATEGORIES", 24)
IMPORTANT_RECALL_K = _env_int("NEURAL_MAP_IMPORTANT_RECALL_K", 6)  # guaranteed candidates among high-importance memories
IMPORTANT_THRESHOLD = _env_int("NEURAL_MAP_IMPORTANT_THRESHOLD", 4)  # importance >= this counts as "high"

# Precision-first recall trims context for ordinary decision prompts: memories
# produced only by recency/importance/pinned visibility are kept only when they
# also show a topical link to the current question. Broad memory questions still
# get a wider sweep, and the filter fails open rather than returning nothing.
PRECISION_RECALL_ENABLED = os.getenv("NEURAL_MAP_PRECISION_RECALL", "1") != "0"
PRECISION_MIN_SEMANTIC_SIMILARITY = _env_float("NEURAL_MAP_PRECISION_MIN_SEMANTIC_SIMILARITY", 0.22)
PRECISION_SEMANTIC_RANK_LIMIT = _env_int("NEURAL_MAP_PRECISION_SEMANTIC_RANK_LIMIT", 8)
# Personal high-salience domains are useful when asked for, but they should not
# hijack general questions merely because recent/important memories mention them.
MEMORY_DOMAIN_OPT_IN_ENABLED = os.getenv("NEURAL_MAP_MEMORY_DOMAIN_OPT_IN", "1") != "0"
