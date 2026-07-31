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


def _hidden_part_below(resolved: Path, roots: list[Path]) -> str | None:
    """First dot-prefixed component of the path BELOW its allowed root, if any.

    Dotfiles hold exactly the things the file tools must not surface —
    .claude/settings.local.json, a future .env, .ssh keys reached via symlink.
    Search already excludes them (skip_search_path); read/list/write apply the
    same rule here so no tool becomes the one exfil channel. The roots' own
    components are exempt: an allowed root may itself live under a dot dir."""
    for root in roots:
        if is_under(resolved, root) or resolved == root:
            rel_parts = resolved.relative_to(root).parts if resolved != root else ()
            return next((part for part in rel_parts if part.startswith(".")), None)
    return None


def within_allowed_roots(resolved: Path) -> bool:
    """True if an already-resolved path lies inside an allowed root and touches
    no blocked or hidden part. Used to re-validate paths reached by traversal
    (e.g. rglob results), where a symlink child can point at its resolved target
    outside the roots even though the search root itself validated."""
    if set(resolved.parts) & config.TOOL_BLOCKED_PATH_PARTS:
        return False
    roots = [root.resolve(strict=False) for root in config.TOOL_ALLOWED_ROOTS]
    if not any(is_under(resolved, root) or resolved == root for root in roots):
        return False
    return _hidden_part_below(resolved, roots) is None


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
    hidden = _hidden_part_below(resolved, roots)
    if hidden:
        raise PermissionError(f"hidden path is blocked: {hidden}")
    return resolved


def skip_search_path(path: Path) -> bool:
    return any(part in config.TOOL_BLOCKED_PATH_PARTS or part.startswith(".") for part in path.parts)


def reserved_code_path(resolved: Path) -> bool:
    """True when a path is part of the project's own code or harness config.

    The sandbox, approval gate, and this path policy itself are enforced by
    files under BASE_DIR, so a generic write_file to any of them would let one
    approval become persistent arbitrary code execution at the next import or
    restart. Reserved unless TOOL_CODE_WRITES_ENABLED is explicitly on.
    Read from config at call time so tests and env overrides apply."""
    if config.TOOL_CODE_WRITES_ENABLED:
        return False
    base = config.BASE_DIR.resolve(strict=False)
    if not (is_under(resolved, base) or resolved == base):
        return False
    if resolved.suffix.lower() in {".py", ".pyc", ".pth", ".so"}:
        return True
    rel_parts = resolved.relative_to(base).parts if resolved != base else ()
    return bool(rel_parts) and rel_parts[0] in {".claude", "web"}


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


def trusted_prompt_update_target(resolved: Path) -> str | None:
    """Return the dedicated updater target for editable trusted calibrations.

    Other trusted prompt files have specialized semantics or no model-facing
    updater and must remain direct-edit only.
    """
    editable = {
        config.SOURCE_OF_TRUTH_PATH.resolve(strict=False): "source_of_truth",
        config.NURSING_PATH.resolve(strict=False): "nursing",
        config.FINANCE_PATH.resolve(strict=False): "finance",
    }
    return editable.get(resolved)
