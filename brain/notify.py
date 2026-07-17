"""Local macOS notifications — the assistant's outbound "voice".

The Will layer and reflection produce initiatives that used to sit silently in
the web UI until the user happened to open it. This module lets a high-priority
initiative surface as a native macOS notification banner instead, closing the
proactive loop without any third-party push service or account (fully local,
matching the no-API-key stack).

Delivery is best-effort and never raises into the caller: a notification is a
courtesy, not a transaction, and a background job must not crash because the GUI
session is unavailable. On non-macOS hosts, or when disabled, it is a no-op.
"""
from __future__ import annotations

import shutil
import subprocess
import sys

import config
from brain import log


def _escape_applescript(value: str) -> str:
    """Escape a string for embedding inside an AppleScript double-quoted literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def available() -> bool:
    """True when local notifications can plausibly be delivered on this host."""
    return sys.platform == "darwin" and bool(
        shutil.which("terminal-notifier") or shutil.which("osascript")
    )


def notify(title: str, message: str, *, subtitle: str | None = None) -> dict:
    """Fire a local notification banner. Returns a small status dict and never
    raises. ``title``/``message`` are truncated defensively so a runaway initiative
    text can't build a multi-kilobyte AppleScript command."""
    if not config.NOTIFY_ENABLED:
        return {"delivered": False, "channel": None, "reason": "notifications disabled"}
    if sys.platform != "darwin":
        return {"delivered": False, "channel": None, "reason": "not macOS"}

    title = (title or "Neural Map").strip()[:120] or "Neural Map"
    message = (message or "").strip()[:300]
    subtitle = (subtitle or "").strip()[:120] or None
    if not message:
        return {"delivered": False, "channel": None, "reason": "empty message"}
    sound = config.NOTIFY_SOUND or None

    # Prefer terminal-notifier when installed: it delivers reliably from launchd
    # and cron contexts where a bare `osascript display notification` sometimes
    # can't reach the Aqua session. Fall back to osascript (always present on macOS).
    tn = shutil.which("terminal-notifier")
    try:
        if tn:
            args = [tn, "-title", title, "-message", message]
            if subtitle:
                args += ["-subtitle", subtitle]
            if sound:
                args += ["-sound", sound]
            subprocess.run(args, capture_output=True, timeout=10, check=False)
            return {"delivered": True, "channel": "terminal-notifier"}

        script = f'display notification "{_escape_applescript(message)}" with title "{_escape_applescript(title)}"'
        if subtitle:
            script += f' subtitle "{_escape_applescript(subtitle)}"'
        if sound:
            script += f' sound name "{_escape_applescript(sound)}"'
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10, check=False)
        return {"delivered": True, "channel": "osascript"}
    except (subprocess.SubprocessError, OSError) as exc:
        log.fallback("notify_failed", exc, phase="deliver")
        return {"delivered": False, "channel": None, "reason": str(exc)}
