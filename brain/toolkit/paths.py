"""Shared path policy for local tools."""
from __future__ import annotations

from pathlib import Path

import config


def is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def within_allowed_roots(resolved: Path) -> bool:
    """True if an already-resolved path lies inside an allowed root and touches
    no blocked part. Used to re-validate paths reached by traversal (e.g. rglob
    results), where a symlink child can point at its resolved target outside the
    roots even though the search root itself validated."""
    if set(resolved.parts) & config.TOOL_BLOCKED_PATH_PARTS:
        return False
    roots = [root.resolve(strict=False) for root in config.TOOL_ALLOWED_ROOTS]
    return any(is_under(resolved, root) or resolved == root for root in roots)


def resolve_tool_path(raw: str) -> Path:
    value = (raw or "").strip()
    if not value:
        raise ValueError("path is required")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config.BASE_DIR / candidate
    resolved = candidate.resolve(strict=False)
    blocked = set(resolved.parts) & config.TOOL_BLOCKED_PATH_PARTS
    if blocked:
        raise PermissionError(f"path is blocked: {sorted(blocked)[0]}")
    roots = [root.resolve(strict=False) for root in config.TOOL_ALLOWED_ROOTS]
    if not any(is_under(resolved, root) or resolved == root for root in roots):
        raise PermissionError("path is outside allowed tool roots")
    return resolved


def skip_search_path(path: Path) -> bool:
    return any(part in config.TOOL_BLOCKED_PATH_PARTS or part.startswith(".") for part in path.parts)


def trusted_prompt_path(resolved: Path) -> bool:
    """True when a path is one of the markdown files injected into the system
    prompt as trusted (source of truth, domain calibrations, agency docs).

    These files instruct the model on every future turn, so a generic
    write_file to one of them would let untrusted content (a poisoned memory
    or document) propose a persistent trusted-prompt change behind a one-line
    approval. They are reserved for their dedicated, preview-visible update
    paths. Read from config at call time so tests and env overrides apply."""
    trusted = {
        config.SOURCE_OF_TRUTH_PATH,
        config.NURSING_PATH,
        config.FINANCE_PATH,
        config.AGENCY_PROTOCOL_PATH,
        config.ACTION_PACKAGE_TEMPLATE_PATH,
        config.SELF_AUDIT_LOOP_PATH,
        config.WILL_PERSPECTIVE_PATH,
        config.WILL_PERSPECTIVE_JOURNAL_PATH,
    }
    return any(resolved == Path(p).resolve(strict=False) for p in trusted)
