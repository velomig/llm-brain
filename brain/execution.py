"""Context-aware execution boundaries for pipelines, generations, and tools.

The decorator in this module is the single place that installs request-local
context tracking and cancellation state. Nested decorated calls reuse the
active scope, so a pipeline, its LLM calls, and its tools share one usage ledger
and one Stop signal without repeatedly resetting either ContextVar.
"""
from __future__ import annotations

import contextvars
import functools
import inspect
import threading
from contextlib import contextmanager
from typing import Callable

_EXECUTION_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "neural_map_execution_depth", default=0
)


def _arguments(func: Callable, args: tuple, kwargs: dict) -> dict:
    try:
        return dict(inspect.signature(func).bind_partial(*args, **kwargs).arguments)
    except (TypeError, ValueError):
        return dict(kwargs)


@contextmanager
def execution_scope(
    *,
    model: str | None = None,
    num_ctx: int | None = None,
    cancel_event: threading.Event | None = None,
    track_context: bool = True,
):
    """Install context/cancellation state once and restore it reliably."""
    # Lazy import avoids a module cycle: llm functions themselves use this
    # decorator, while the state primitives remain backward-compatible there.
    from brain import llm

    cancel_token = None
    context_token = None
    depth_token = None
    try:
        if cancel_event is not None and llm.active_cancel_event() is not cancel_event:
            cancel_token = llm.set_cancel_event(cancel_event)
        if track_context and llm.context_usage() is None:
            normalized = llm.normalize_num_ctx(num_ctx, model)
            context_token = llm.begin_context_tracking(normalized, model)
        depth_token = _EXECUTION_DEPTH.set(_EXECUTION_DEPTH.get() + 1)
        llm.raise_if_cancelled()
        yield
        # Covers tools or non-streaming work that completed just as Stop fired.
        llm.raise_if_cancelled()
    finally:
        if depth_token is not None:
            _EXECUTION_DEPTH.reset(depth_token)
        if context_token is not None:
            llm.reset_context_tracking(context_token)
        if cancel_token is not None:
            llm.reset_cancel_event(cancel_token)


def _attach_usage(result):
    if isinstance(result, dict):
        from brain import llm

        result["context_usage"] = llm.context_usage()
    return result


def context_aware_execution(
    func: Callable | None = None,
    *,
    track_context: bool = True,
    attach_usage: bool = False,
    model_arg: str = "model",
    num_ctx_arg: str = "num_ctx",
    cancel_event_arg: str = "cancel_event",
):
    """Decorate a sync or async execution boundary.

    Arguments are discovered by name from the wrapped function. A supplied
    cancellation event is installed as the listener for nested LLM/tool calls;
    otherwise any active event is inherited. Context tracking is initialized
    only when no outer scope already owns it.
    """

    def decorate(target: Callable):
        def scope_values(args: tuple, kwargs: dict):
            bound = _arguments(target, args, kwargs)
            return (
                bound.get(model_arg),
                bound.get(num_ctx_arg),
                bound.get(cancel_event_arg),
            )

        if inspect.iscoroutinefunction(target):
            @functools.wraps(target)
            async def async_wrapper(*args, **kwargs):
                model, num_ctx, cancel_event = scope_values(args, kwargs)
                with execution_scope(
                    model=model,
                    num_ctx=num_ctx,
                    cancel_event=cancel_event,
                    track_context=track_context,
                ):
                    result = await target(*args, **kwargs)
                    return _attach_usage(result) if attach_usage else result

            async_wrapper.__context_aware_execution__ = {  # type: ignore[attr-defined]
                "track_context": track_context,
                "attach_usage": attach_usage,
            }
            return async_wrapper

        @functools.wraps(target)
        def sync_wrapper(*args, **kwargs):
            model, num_ctx, cancel_event = scope_values(args, kwargs)
            with execution_scope(
                model=model,
                num_ctx=num_ctx,
                cancel_event=cancel_event,
                track_context=track_context,
            ):
                result = target(*args, **kwargs)
                return _attach_usage(result) if attach_usage else result

        sync_wrapper.__context_aware_execution__ = {  # type: ignore[attr-defined]
            "track_context": track_context,
            "attach_usage": attach_usage,
        }
        return sync_wrapper

    return decorate(func) if func is not None else decorate


def bind_context(func: Callable, /, *args, **kwargs) -> Callable[[], object]:
    """Capture the current ContextVars for work scheduled on another thread."""
    captured = contextvars.copy_context()

    @functools.wraps(func)
    def bound():
        # Context objects cannot be entered concurrently; a copy makes the
        # returned callable safe if a scheduler invokes it more than once.
        return captured.copy().run(func, *args, **kwargs)

    return bound


def execution_depth() -> int:
    return _EXECUTION_DEPTH.get()
