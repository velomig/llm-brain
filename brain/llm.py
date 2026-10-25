"""LLM wrapper (Ollama). The model that actually writes the responses."""
from __future__ import annotations

import contextvars
import json
import logging
import math
import re
import threading
from typing import Callable

import ollama

import config
from brain import log
from brain.execution import context_aware_execution

_client = ollama.Client(host=config.OLLAMA_HOST, timeout=config.OLLAMA_TIMEOUT_SECONDS)


class GenerationCancelled(BaseException):
    """The active cancel event was set mid-generation.

    Deliberately a BaseException: the pipeline swallows `Exception` in many
    places to stay conversational under partial failure, and a user-requested
    cancel must unwind all the way to the API handler instead of being logged
    as a fallback and papered over.
    """


# Set per-request by the API layer (contextvars, so the event follows the
# request's thread through every nested llm call — System 2, council, tools).
# When an event is present, generations run as Ollama streams and check it per
# chunk; closing the stream is what actually makes Ollama stop generating.
_CANCEL_EVENT: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar(
    "neural_map_cancel_event", default=None
)
_CONTEXT_USAGE: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "neural_map_context_usage", default=None
)


def begin_context_tracking(num_ctx: int | None, model: str | None = None) -> contextvars.Token:
    """Start request-local prompt occupancy tracking across nested LLM calls."""
    limit = normalize_num_ctx(num_ctx, model)
    return _CONTEXT_USAGE.set(
        {
            "model": model or config.CHAT_MODEL,
            "configured_tokens": limit,
            "peak_prompt_tokens": 0,
            "last_prompt_tokens": 0,
            "total_prompt_tokens": 0,
            "calls": 0,
            "exact_calls": 0,
            "peak_measurement": "estimated",
            # Context copies share this ledger across thread handoffs. Mutate
            # it under one lock so child LLM usage remains visible to the
            # parent execution scope instead of forking silently.
            "_lock": threading.Lock(),
        }
    )


def reset_context_tracking(token: contextvars.Token) -> None:
    _CONTEXT_USAGE.reset(token)


def context_usage() -> dict | None:
    current = _CONTEXT_USAGE.get()
    if current is None:
        return None
    lock = current.get("_lock")
    if lock is not None:
        with lock:
            snapshot = dict(current)
    else:
        snapshot = dict(current)
    limit = max(1, int(snapshot.get("configured_tokens") or 1))
    used = max(0, int(snapshot.get("peak_prompt_tokens") or 0))
    return {
        "model": snapshot.get("model"),
        "used_tokens": used,
        "configured_tokens": limit,
        "remaining_tokens": max(0, limit - used),
        "percent": round((used / limit) * 100.0, 2),
        "measurement": snapshot.get("peak_measurement") or "estimated",
        "calls": max(0, int(snapshot.get("calls") or 0)),
        "exact_calls": max(0, int(snapshot.get("exact_calls") or 0)),
        "turn_total_tokens": max(0, int(snapshot.get("total_prompt_tokens") or 0)),
        "scope": "peak input prompt across foreground model calls in this turn",
    }


def _estimated_prompt_tokens(messages: list[dict], tools: list[dict] | None = None) -> int:
    """Portable fallback when Ollama does not return prompt_eval_count.

    Four UTF-8-ish characters per token is deliberately labeled an estimate;
    small per-message overhead approximates role/control tokens.
    """
    try:
        payload = json.dumps({"messages": messages, "tools": tools or []}, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        payload = str(messages) + str(tools or [])
    return max(1, math.ceil(len(payload) / 4) + len(messages) * 4)


def _record_context_usage(
    messages: list[dict],
    options: dict,
    *,
    model: str | None = None,
    tools: list[dict] | None = None,
    exact_tokens=None,
) -> None:
    current = _CONTEXT_USAGE.get()
    if current is None:
        return
    # The UI meter describes the selected foreground model. Auxiliary council,
    # consolidation, and metacognition calls have their own smaller context
    # window and must not overwrite that meter or inflate its call count.
    if model and model != current.get("model"):
        return
    try:
        exact = int(exact_tokens)
    except (TypeError, ValueError):
        exact = 0
    used = exact if exact > 0 else _estimated_prompt_tokens(messages, tools)
    measurement = "exact" if exact > 0 else "estimated"
    lock = current.get("_lock")
    if lock is None:
        lock = threading.Lock()
        current["_lock"] = lock
    with lock:
        current["configured_tokens"] = int(options.get("num_ctx") or current["configured_tokens"])
        current["last_prompt_tokens"] = used
        current["total_prompt_tokens"] = int(current.get("total_prompt_tokens") or 0) + used
        current["calls"] = int(current.get("calls") or 0) + 1
        if exact > 0:
            current["exact_calls"] = int(current.get("exact_calls") or 0) + 1
        if used >= int(current.get("peak_prompt_tokens") or 0):
            current["peak_prompt_tokens"] = used
            current["peak_measurement"] = measurement


def set_cancel_event(event: threading.Event) -> contextvars.Token:
    return _CANCEL_EVENT.set(event)


def reset_cancel_event(token: contextvars.Token) -> None:
    _CANCEL_EVENT.reset(token)


def active_cancel_event() -> threading.Event | None:
    return _CANCEL_EVENT.get()


def raise_if_cancelled() -> None:
    event = _CANCEL_EVENT.get()
    if event is not None and event.is_set():
        raise GenerationCancelled()


_REPETITION_GUARD_REPLY = (
    "I stopped this response because the model entered a repetitive-output loop. "
    "Please retry the request."
)
_OUTPUT_LIMIT_MARKER = "\n\n[Response stopped at the configured output safety limit.]"


def _reply_char_limit() -> int:
    """Effective reply ceiling, always no larger than the API message limit."""
    try:
        configured = int(config.MAX_ASSISTANT_REPLY_CHARS)
    except (TypeError, ValueError):
        configured = 16_000
    return max(1, min(configured, int(config.MAX_MESSAGE_CHARS)))


def _has_repetitive_tail(text: str) -> bool:
    """Detect a short exact cycle repeated across a substantial output tail.

    This targets model degeneration such as ``Hh4eHh4e...`` without flagging
    ordinary repeated prose. At least GENERATION_REPEAT_MIN_CHARS must be an
    exact periodic suffix, and the period must repeat at least four times.
    """
    try:
        minimum = max(32, int(config.GENERATION_REPEAT_MIN_CHARS))
    except (TypeError, ValueError):
        minimum = 256
    try:
        max_period = max(1, int(config.GENERATION_REPEAT_MAX_PERIOD))
    except (TypeError, ValueError):
        max_period = 64
    if len(text) < minimum:
        return False
    max_period = min(max_period, len(text) // 4)
    for period in range(1, max_period + 1):
        repeats = max(4, math.ceil(minimum / period))
        span = period * repeats
        if len(text) < span:
            continue
        sample = text[-span:]
        unit = sample[-period:]
        if unit.strip() and sample == unit * repeats:
            return True
    return False


def _truncate_output(text: str) -> str:
    limit = _reply_char_limit()
    if len(text) <= limit:
        return text
    if limit <= len(_OUTPUT_LIMIT_MARKER):
        return _OUTPUT_LIMIT_MARKER[:limit]
    body = text[: limit - len(_OUTPUT_LIMIT_MARKER)].rstrip()
    return body + _OUTPUT_LIMIT_MARKER


def _guard_output(
    text: str,
    model: str | None = None,
    *,
    forced_reason: str | None = None,
) -> tuple[str, bool]:
    """Return bounded model output plus whether a safety guard intervened."""
    raw = text or ""
    reason = forced_reason
    if reason is None and _has_repetitive_tail(raw):
        reason = "repetitive_output"
    if reason is None and len(raw) > _reply_char_limit():
        reason = "character_limit"
    if reason is None:
        return raw, False
    log.fallback(
        "generation_guard_triggered",
        model=model,
        reason=reason,
        chars=len(raw),
    )
    if reason == "repetitive_output":
        return _REPETITION_GUARD_REPLY, True
    return _truncate_output(raw), True


def _streamed_message(
    name: str,
    messages: list[dict],
    options: dict,
    event: threading.Event,
    tools: list[dict] | None = None,
    think: bool | str | None = None,
) -> tuple[str, list]:
    """One generation as a stream, aborted between chunks if `event` is set.

    Returns (content, tool_calls). Ollama emits each tool call whole within a
    chunk, so extending the list chunk-by-chunk reassembles the same shape the
    non-streaming response carries.
    """
    kwargs: dict = {"model": name, "messages": messages, "options": options, "stream": True}
    if tools is not None:
        kwargs["tools"] = tools
    if think is not None:
        kwargs["think"] = think
    stream = _client.chat(**kwargs)
    parts: list[str] = []
    tool_calls: list = []
    prompt_eval_count = None
    guard_reason = None
    try:
        for chunk in stream:
            if event.is_set():
                raise GenerationCancelled()
            msg = _msg_get(chunk, "message") or {}
            parts.append(_msg_get(msg, "content", "") or "")
            tool_calls.extend(_msg_get(msg, "tool_calls") or [])
            prompt_eval_count = _msg_get(chunk, "prompt_eval_count", prompt_eval_count)
            assembled = "".join(parts)
            if _has_repetitive_tail(assembled):
                guard_reason = "repetitive_output"
                break
            if len(assembled) > _reply_char_limit():
                guard_reason = "character_limit"
                break
    finally:
        # On cancel/error, closing the generator closes the HTTP stream, which
        # is the signal that makes Ollama abort the generation server-side.
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    _record_context_usage(
        messages, options, model=name, tools=tools, exact_tokens=prompt_eval_count
    )
    content, guarded = _guard_output(
        "".join(parts), name, forced_reason=guard_reason
    )
    return content, ([] if guarded else tool_calls)


def _chat_message(
    name: str,
    messages: list[dict],
    options: dict,
    tools: list[dict] | None = None,
    think: bool | str | None = None,
) -> tuple[str, list]:
    """One generation, cancellable when a cancel event is active.

    No event (CLI, cron jobs, background capture) → plain non-streaming call,
    byte-for-byte the old behavior.
    """
    event = _CANCEL_EVENT.get()
    if event is not None:
        if event.is_set():
            raise GenerationCancelled()
        return _streamed_message(name, messages, options, event, tools=tools, think=think)
    kwargs: dict = {"model": name, "messages": messages, "options": options}
    if tools is not None:
        kwargs["tools"] = tools
    if think is not None:
        kwargs["think"] = think
    resp = _client.chat(**kwargs)
    msg = resp["message"]
    _record_context_usage(
        messages,
        options,
        model=name,
        tools=tools,
        exact_tokens=_msg_get(resp, "prompt_eval_count"),
    )
    content, guarded = _guard_output(_msg_get(msg, "content", "") or "", name)
    return content, ([] if guarded else (_msg_get(msg, "tool_calls") or []))

# Some models (esp. abliterated ones) emit fake tool-call syntax as plain text
# instead of using the real tool channel. Strip it so it never reaches the user
# or gets captured into memory as junk.
_FAKE_TOOL = re.compile(
    r"<tool_use\b[^>]*>.*?</tool_use>|<tool_use\b[^>]*/?>|\{\{?\s*\"tool_(?:use|call)\".*?\}\}?",
    re.DOTALL | re.IGNORECASE,
)

_TOOL_FINAL_ANSWER_PROMPT = (
    "Using the tool results above, answer the user's last request now. "
    "Do not call any more tools. If web_search returned results, base current/news claims only on those "
    "result titles, snippets, sources, and URLs; cite at least one URL. Do not fill gaps from memory or "
    "prior conversation. If the tool results are incomplete, generic, or off-topic, say exactly what is missing."
)


def normalize_num_ctx(num_ctx: int | None = None, model: str | None = None) -> int:
    """Clamp a requested num_ctx to [MIN, MAX] and the model's real ceiling.

    When the model's max context is undiscoverable, cap at
    UNKNOWN_MODEL_NUM_CTX instead of passing the optimistic default through.
    Single source of truth — the API layer delegates here too.
    """
    requested = num_ctx if num_ctx is not None else config.DEFAULT_NUM_CTX
    try:
        requested = int(requested)
    except (TypeError, ValueError):
        requested = config.DEFAULT_NUM_CTX
    requested = max(config.MIN_NUM_CTX, min(requested, config.MAX_NUM_CTX))

    max_context = model_max_context(model or config.CHAT_MODEL)
    if max_context:
        ceiling = max(1, min(int(max_context), config.MAX_NUM_CTX))
    else:
        ceiling = max(1, min(config.UNKNOWN_MODEL_NUM_CTX, config.MAX_NUM_CTX))
    return max(1, min(requested, ceiling))


def _chat_options(num_ctx: int | None = None, model: str | None = None) -> dict:
    """Default every Ollama call to a normalized, model-safe context window."""
    options = {
        "temperature": 0.7,
        "num_predict": max(1, int(config.MAX_GENERATION_TOKENS)),
    }
    options["num_ctx"] = normalize_num_ctx(num_ctx, model)
    return options


def _sanitize(text: str) -> str:
    return _FAKE_TOOL.sub("", text or "").strip()


def sanitize_reply(text: str, model: str | None = None) -> str:
    """Final API-boundary cleanup for replies from any pipeline path."""
    cleaned = _sanitize(text)
    guarded, _ = _guard_output(cleaned, model)
    return guarded


@context_aware_execution
def chat(
    system: str,
    user: str,
    history: list[dict] | None = None,
    model: str | None = None,
    num_ctx: int | None = None,
    think: bool | str | None = None,
) -> str:
    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})
    name = model or config.CHAT_MODEL
    options = _chat_options(num_ctx, name)
    content, _ = _chat_message(name, messages, options, think=think)
    return sanitize_reply(content, name)


# Capability answers are cached only on a SUCCESSFUL probe. A failed probe (e.g.
# Ollama briefly down at first call) must not be cached: with lru_cache it pinned
# `False` for the process lifetime, silently disabling tool-calling for that model
# until restart. Now the next turn simply re-probes.
_TOOLS_CAPABILITY: dict[str, bool] = {}


def supports_tools(name: str) -> bool:
    """Whether a model advertises native function-calling (from `ollama show`)."""
    cached = _TOOLS_CAPABILITY.get(name)
    if cached is not None:
        return cached
    try:
        info = _client.show(name)
    except Exception as exc:
        # Probe failed — treat as no tool support and fall back to injection for
        # THIS turn only (not cached). DEBUG because it's a routine capability
        # check, not a per-turn failure.
        log.fallback("tool_calling_failed", exc, level=logging.DEBUG, model=name, phase="capability_probe")
        return False
    caps = getattr(info, "capabilities", None)
    if caps is None and isinstance(info, dict):
        caps = info.get("capabilities")
    result = "tools" in (caps or [])
    if len(_TOOLS_CAPABILITY) < 256:   # bound the cache like lru_cache did
        _TOOLS_CAPABILITY[name] = result
    return result


def _msg_get(msg, key, default=None):
    """Read a field off an Ollama message whether it's a dict or an object."""
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)


def _tool_call_fields(tc) -> tuple[str, dict]:
    fn = _msg_get(tc, "function")
    name = _msg_get(fn, "name", "")
    raw_args = _msg_get(fn, "arguments", {})
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            raw_args = {}
    return name, (raw_args if isinstance(raw_args, dict) else {})


@context_aware_execution
def chat_with_tools(
    system: str,
    user: str,
    tool_specs: list[dict],
    dispatch: Callable[[str, dict], dict],
    history: list[dict] | None = None,
    model: str | None = None,
    num_ctx: int | None = None,
    max_iters: int = 4,
    outcome: dict | None = None,
    stop_on_pending: bool = False,
) -> tuple[str, list[dict]]:
    """Return (reply, tools_used), optionally recording the loop's stop reason.

    Background runs stop at an approval boundary; interactive callers retain
    the existing behavior unless stop_on_pending is explicitly enabled.
    """
    if outcome is not None:
        outcome["stop_reason"] = "completed"
    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})
    name = model or config.CHAT_MODEL
    options = _chat_options(num_ctx, name)
    used: list[dict] = []

    for _ in range(max_iters):
        content, tool_calls = _chat_message(name, messages, options, tools=tool_specs)
        if not tool_calls:
            reply = sanitize_reply(content, name)
            if reply or not used:
                return reply, used
            messages.append({"role": "user", "content": _TOOL_FINAL_ANSWER_PROMPT})
            content, _ = _chat_message(name, messages, options)
            return sanitize_reply(content, name), used
        # Echo the assistant's tool request, then answer each call.
        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        for tc in tool_calls:
            raise_if_cancelled()
            tname, targs = _tool_call_fields(tc)
            try:
                result = dispatch(tname, targs)
            except Exception as exc:
                log.fallback("tool_call_failed", exc, tool=tname)
                result = {"error": str(exc)}
            used.append({"tool": tname, "args": targs, "result": result})
            messages.append({"role": "tool", "name": tname, "content": json.dumps(result, ensure_ascii=True)})
            if stop_on_pending and isinstance(result, dict) and result.get("pending"):
                if outcome is not None:
                    outcome["stop_reason"] = "awaiting_approval"
                return "Waiting for approval before continuing.", used

    if outcome is not None:
        outcome["stop_reason"] = "budget_exhausted"
    # Out of iterations — ask once more without tools to force a written answer.
    messages.append({"role": "user", "content": _TOOL_FINAL_ANSWER_PROMPT})
    content, _ = _chat_message(name, messages, options)
    return sanitize_reply(content, name), used


def list_models() -> list[str]:
    """Names of every model available to Ollama (for the UI dropdown)."""
    resp = _client.list()
    models = getattr(resp, "models", None)
    if models is None and isinstance(resp, dict):
        models = resp.get("models", [])
    names = []
    for m in models or []:
        name = getattr(m, "model", None)
        if name is None and isinstance(m, dict):
            name = m.get("model") or m.get("name")
        if name:
            names.append(name)
    return sorted(names)


# Max-context answers are static per model, but normalize_num_ctx asks on EVERY
# generation — uncached, a System 2 turn paid ~4 blocking `ollama show` round
# trips. Cache successful probes only (a failed probe must re-ask next turn, like
# _TOOLS_CAPABILITY above). Keyed per client instance so tests that swap _client
# for a fake with a different context length never see a stale answer.
_MAX_CONTEXT_CACHE: dict[str, int | None] = {}
_MAX_CONTEXT_CLIENT: object | None = None


def model_max_context(name: str) -> int | None:
    """The model's max context window (from `ollama show`), if discoverable."""
    global _MAX_CONTEXT_CLIENT
    if _MAX_CONTEXT_CLIENT is not _client:
        _MAX_CONTEXT_CACHE.clear()
        _MAX_CONTEXT_CLIENT = _client
    if name in _MAX_CONTEXT_CACHE:
        return _MAX_CONTEXT_CACHE[name]
    try:
        info = _client.show(name)
    except Exception:
        return None
    data = getattr(info, "modelinfo", None)
    if data is None and isinstance(info, dict):
        data = info.get("model_info") or info.get("modelinfo")
    result: int | None = None
    for key, val in (data or {}).items():
        if key.endswith("context_length"):
            try:
                result = int(val)
            except (TypeError, ValueError):
                result = None
            break
    if len(_MAX_CONTEXT_CACHE) < 256:
        _MAX_CONTEXT_CACHE[name] = result
    return result
