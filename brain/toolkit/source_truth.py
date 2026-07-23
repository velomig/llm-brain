"""Source of Truth maintenance tool."""
from __future__ import annotations

import re
from pathlib import Path

import config
from brain.utils import path_lock, replace_private


def target_path(args: dict | None = None) -> Path:
    """Resolve an allow-listed trusted calibration target.

    The generic file writer deliberately cannot edit files injected into the
    system prompt. Keep this updater equally narrow: callers choose a logical
    target, never an arbitrary path. The default preserves the original
    update_source_of_truth API.
    """
    target = str((args or {}).get("target") or "source_of_truth").strip().lower()
    paths = {
        "source_of_truth": config.SOURCE_OF_TRUTH_PATH,
        "nursing": config.NURSING_PATH,
        "finance": config.FINANCE_PATH,
    }
    if target not in paths:
        raise ValueError("target must be source_of_truth, nursing, or finance")
    return paths[target].resolve(strict=False)


def plan_replacement(text: str, old_text: str, new_text: str) -> dict:
    """Plan one replacement, tolerating line wrapping but not ambiguity."""
    exact = [match.span() for match in re.finditer(re.escape(old_text), text)]
    if exact:
        spans = exact
        match_kind = "exact"
    else:
        pieces = [re.escape(piece) for piece in re.split(r"\s+", old_text.strip()) if piece]
        pattern = r"\s+".join(pieces)
        spans = [match.span() for match in re.finditer(pattern, text)] if pattern else []
        match_kind = "whitespace_normalized"
    if len(spans) != 1:
        return {"matches": len(spans), "match_kind": match_kind, "updated": text}
    start, end = spans[0]
    return {
        "matches": 1,
        "match_kind": match_kind,
        "updated": text[:start] + new_text + text[end:],
    }


def update_source_of_truth(args: dict) -> dict:
    old_text = str((args or {}).get("old_text") or "")
    new_text = str((args or {}).get("new_text") or "")
    target = str((args or {}).get("target") or "source_of_truth").strip().lower()
    if not old_text or not new_text:
        return {"error": "old_text and new_text are required"}
    if len(new_text) > config.MAX_TOOL_WRITE_CHARS:
        return {"error": "new_text too long"}

    try:
        path = target_path(args)
    except ValueError as exc:
        return {"error": str(exc)}
    with path_lock(path, config.DB_DIR, prefix="trusted_prompt"):
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {"path": str(path), "error": "source of truth file not found"}
        except OSError as exc:
            return {"path": str(path), "error": str(exc)}

        plan = plan_replacement(text, old_text, new_text)
        matches = plan["matches"]
        if matches <= 0:
            return {"path": str(path), "error": "old_text not found"}
        if matches > 1:
            return {"path": str(path), "error": "old_text is ambiguous; it appears more than once"}

        updated = plan["updated"]
        try:
            replace_private(path, updated)
        except OSError as exc:
            return {"path": str(path), "error": str(exc)}

    return {
        "ok": True,
        "target": target,
        "path": str(path),
        "old_chars": len(old_text),
        "new_chars": len(new_text),
        "match_kind": plan["match_kind"],
    }
