"""Local file, document, and media tools."""
from __future__ import annotations

from pathlib import Path

import config
from brain import sensory
from brain.utils import path_lock
from brain.toolkit.paths import (
    reserved_code_path,
    resolve_tool_path,
    skip_search_path,
    trusted_prompt_path,
    trusted_prompt_update_target,
    within_allowed_roots,
)


def _read_text_prefix(path: Path, max_chars: int) -> tuple[str | None, bool, int | None, str | None]:
    try:
        size = path.stat().st_size
        byte_limit = max(4096, max_chars * 4 + 4)
        with path.open("rb") as fh:
            raw = fh.read(byte_limit)
    except OSError as exc:
        return None, False, None, str(exc)
    if b"\x00" in raw[:4096]:
        return None, False, size, "file appears to be binary"
    text = raw.decode("utf-8", errors="replace")
    truncated = size > len(raw) or len(text) > max_chars
    return text[:max_chars], truncated, size, None


def list_files(args: dict) -> dict:
    path = str((args or {}).get("path", "."))
    try:
        directory = resolve_tool_path(path)
        if not directory.exists():
            return {"path": str(directory), "error": "path does not exist"}
        if not directory.is_dir():
            return {"path": str(directory), "error": "path is not a directory"}
        rows = []
        for child in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
            if child.name in config.TOOL_BLOCKED_PATH_PARTS or child.name.startswith("."):
                continue
            row = {"name": child.name, "path": str(child), "type": "unknown", "size": None}
            try:
                if child.is_dir():
                    row["type"] = "directory"
                elif child.is_file():
                    row["type"] = "file"
                    row["size"] = child.stat().st_size
                else:
                    row["type"] = "other"
            except OSError as exc:
                row["error"] = str(exc)
            rows.append(row)
            if len(rows) >= config.MAX_TOOL_LIST_ENTRIES:
                break
        return {"path": str(directory), "entries": rows, "truncated": len(rows) >= config.MAX_TOOL_LIST_ENTRIES}
    except Exception as exc:
        return {"path": path, "error": str(exc)}


def read_file(args: dict) -> dict:
    filename = str((args or {}).get("filename", ""))
    try:
        path = resolve_tool_path(filename)
        if not path.exists():
            return {"path": str(path), "error": "file does not exist"}
        if not path.is_file():
            return {"path": str(path), "error": "path is not a file"}
        text, truncated, size, error = _read_text_prefix(path, config.MAX_TOOL_READ_CHARS)
        if error:
            return {"path": str(path), "error": error}
        return {"path": str(path), "content": text or "", "truncated": truncated, "bytes": size}
    except Exception as exc:
        return {"path": filename, "error": str(exc)}


def write_file(args: dict) -> dict:
    try:
        path = resolve_tool_path(str((args or {}).get("filename", "")))
        with path_lock(path, config.DB_DIR, prefix="trusted_prompt"):
            return _write_file_locked(args)
    except Exception as exc:
        return {"error": str(exc)}


def _write_file_locked(args: dict) -> dict:
    filename = str((args or {}).get("filename", ""))
    content = "" if (args or {}).get("content") is None else str((args or {}).get("content", ""))
    mode = str((args or {}).get("mode", "create")).strip().lower()
    try:
        path = resolve_tool_path(filename)
        if trusted_prompt_path(path):
            target = trusted_prompt_update_target(path)
            hint = (
                f"use update_source_of_truth with target={target}"
                if target
                else "use its dedicated update tool"
            )
            return {
                "path": str(path),
                "error": "this file is injected into the system prompt as trusted and is "
                f"reserved from write_file; {hint} or edit it directly",
            }
        if reserved_code_path(path):
            return {
                "path": str(path),
                "error": "this path is part of the project's own code and is reserved "
                "from write_file; edit it directly (or set NEURAL_MAP_TOOL_CODE_WRITES=1)",
            }
        if mode not in {"create", "overwrite", "append"}:
            return {"path": str(path), "error": "mode must be create, overwrite, or append"}
        if len(content) > config.MAX_TOOL_WRITE_CHARS:
            return {"path": str(path), "error": "content too long"}
        if path.exists() and not path.is_file():
            return {"path": str(path), "error": "path is not a file"}
        if mode == "create" and path.exists():
            return {"path": str(path), "error": "file already exists"}
        path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "append":
            with path.open("a", encoding="utf-8") as fh:
                fh.write(content)
        else:
            path.write_text(content, encoding="utf-8")
        return {"path": str(path), "mode": mode, "chars": len(content), "ok": True}
    except Exception as exc:
        return {"path": filename, "error": str(exc)}


def search_local_files(args: dict) -> dict:
    query = str((args or {}).get("query", "")).strip()
    path = str((args or {}).get("path", "."))
    if not query:
        return {"error": "query is required"}
    try:
        root = resolve_tool_path(path or ".")
        if not root.exists():
            return {"path": str(root), "error": "path does not exist"}
        limit = min(max(1, int((args or {}).get("max_results") or config.MAX_LOCAL_SEARCH_RESULTS)), config.MAX_LOCAL_SEARCH_RESULTS)
        files = [root] if root.is_file() else root.rglob("*")
        rows = []
        needle = query.casefold()
        for item in files:
            if len(rows) >= limit:
                break
            if skip_search_path(item.relative_to(root) if item != root and root.is_dir() else item):
                continue
            if not item.is_file():
                continue
            # A file reached by traversal may be a symlink whose target escapes
            # the allowed roots; read_file() rejects such links via resolve, so
            # search must too or it becomes an approval-free read of any path.
            if not within_allowed_roots(item.resolve(strict=False)):
                continue
            try:
                if item.stat().st_size > config.MAX_LOCAL_SEARCH_FILE_BYTES:
                    continue
                raw = item.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:4096]:
                continue
            text = raw.decode("utf-8", errors="replace")
            pos = text.casefold().find(needle)
            if pos == -1:
                continue
            start = max(0, pos - 120)
            end = min(len(text), pos + len(query) + 120)
            rows.append({"path": str(item), "snippet": text[start:end].strip(), "chars": len(text)})
        return {"path": str(root), "query": query, "results": rows, "truncated": len(rows) >= limit}
    except Exception as exc:
        return {"path": path, "query": query, "error": str(exc)}


def inspect_document(args: dict) -> dict:
    filename = str((args or {}).get("filename", ""))
    query = (args or {}).get("query")
    try:
        path = resolve_tool_path(filename)
        if not path.exists() or not path.is_file():
            return {"path": str(path), "error": "file does not exist"}
        suffix = path.suffix.lower()
        truncated = False
        size = None
        if suffix == ".pdf":
            from brain.documents import extract_pdf_text

            text = extract_pdf_text(path)
        elif suffix in {".md", ".markdown", ".txt", ".py", ".json", ".csv", ".tsv", ".html", ".css", ".js"}:
            text, truncated, size, error = _read_text_prefix(path, config.MAX_TOOL_READ_CHARS)
            if error:
                return {"path": str(path), "error": error}
        else:
            return {"path": str(path), "error": "unsupported document type"}
        if query:
            needle = str(query).casefold()
            pos = text.casefold().find(needle)
            if pos == -1:
                return {"path": str(path), "query": query, "matches": [], "truncated": truncated}
            start = max(0, pos - 500)
            end = min(len(text), pos + len(str(query)) + 1500)
            return {"path": str(path), "query": query, "matches": [{"snippet": text[start:end].strip()}], "truncated": truncated, "bytes": size}
        truncated = truncated or len(text) > config.MAX_TOOL_READ_CHARS
        return {"path": str(path), "content": text[: config.MAX_TOOL_READ_CHARS], "truncated": truncated, "chars": len(text), "bytes": size}
    except Exception as exc:
        return {"path": filename, "error": str(exc)}


def transcribe_media(args: dict) -> dict:
    filename = str((args or {}).get("filename", ""))
    try:
        path = resolve_tool_path(filename)
        if not path.exists() or not path.is_file():
            return {"path": str(path), "error": "file does not exist"}
        if path.suffix.lower() in sensory.AUDIO_SUFFIXES:
            return {
                "path": str(path),
                "summary": sensory.audio_summary(path.read_bytes(), path.name),
                "transcript": None,
                "warning": "No speech-to-text backend is configured; returning available audio metadata.",
            }
        return {"path": str(path), "error": "unsupported media type"}
    except Exception as exc:
        return {"path": filename, "error": str(exc)}
