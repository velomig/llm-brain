"""Vision model wrapper (Ollama): turn a still image into searchable text.

The Neural Map is a text index, so an image becomes memory by first being
*described*. A local vision-capable model writes what the image shows and
transcribes any visible text (OCR) in one pass; that text is then chunked,
embedded, and stored like any other document (see brain.documents).

Raw image bytes are passed straight to Ollama — no Pillow / decoding step — so
this stays dependency-free. Needs a vision model pulled, e.g.
    ollama pull llama3.2-vision
"""
from __future__ import annotations

import ollama

import config

_client = ollama.Client(host=config.OLLAMA_HOST, timeout=config.OLLAMA_TIMEOUT_SECONDS)


class VisionUnavailable(RuntimeError):
    """The configured model can't see images, or isn't pulled in Ollama."""


def _is_vision_unavailable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "not found" in msg                     # model isn't pulled
        or "does not support" in msg           # model has no vision capability
        or "no vision" in msg
        or "image" in msg and "support" in msg
    )


def describe_image(
    data: bytes,
    model: str | None = None,
    prompt: str | None = None,
) -> str:
    """Caption + OCR an image. Returns the description text.

    Raises VisionUnavailable if the model can't see / isn't installed, so callers
    can surface a clear "pull a vision model" message instead of a raw error.
    """
    if not data:
        raise ValueError("image has no data")
    name = model or config.VISION_MODEL
    try:
        resp = _client.chat(
            model=name,
            messages=[{"role": "user", "content": prompt or config.VISION_PROMPT, "images": [data]}],
            options={"temperature": config.VISION_TEMPERATURE},
        )
    except Exception as exc:
        if _is_vision_unavailable(exc):
            raise VisionUnavailable(
                f"vision model {name!r} is unavailable. Pull one (e.g. "
                f"`ollama pull llama3.2-vision`) and set NEURAL_MAP_VISION_MODEL "
                f"if it differs. Original error: {exc}"
            ) from exc
        raise
    msg = resp["message"]
    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
    return (content or "").strip()
