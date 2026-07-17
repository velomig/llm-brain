"""Approval-gated updates to Will's trusted core perspective."""
from __future__ import annotations

import os
import threading
import uuid

import config
from brain import perspective
from brain.toolkit.source_truth import plan_replacement


def update_will_perspective(args: dict) -> dict:
    old_text = str((args or {}).get("old_text") or "")
    new_text = str((args or {}).get("new_text") or "")
    reason = str((args or {}).get("reason") or "approved perspective refinement")[:600]
    if not old_text or not new_text:
        return {"error": "old_text and new_text are required"}
    limit = max(100, int(config.WILL_PERSPECTIVE_MAX_EDIT_CHARS))
    if len(old_text) > limit or len(new_text) > limit:
        return {"error": "perspective edit is larger than the configured identity-edit limit"}

    path = config.WILL_PERSPECTIVE_PATH.resolve(strict=False)
    try:
        current = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"path": str(path), "error": "Will perspective file not found"}
    except OSError as exc:
        return {"path": str(path), "error": str(exc)}

    plan = plan_replacement(current, old_text, new_text)
    if plan["matches"] <= 0:
        return {"path": str(path), "error": "old_text not found"}
    if plan["matches"] > 1:
        return {"path": str(path), "error": "old_text is ambiguous; it appears more than once"}

    updated = plan["updated"]
    if len(updated) > config.WILL_PERSPECTIVE_MAX_CHARS:
        return {"path": str(path), "error": "updated perspective exceeds its configured prompt limit"}
    old_hash = perspective.perspective_hash(current)
    new_hash = perspective.perspective_hash(updated)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(updated, encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        return {"path": str(path), "error": str(exc)}

    journal = perspective.append_identity_change(old_hash, new_hash, reason)
    return {
        "ok": True,
        "path": str(path),
        "old_chars": len(old_text),
        "new_chars": len(new_text),
        "match_kind": plan["match_kind"],
        "old_hash": old_hash,
        "new_hash": new_hash,
        "journal_logged": bool(journal.get("written")),
    }
