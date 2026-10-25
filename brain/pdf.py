"""Run PDF parsing outside the server with bounded input, time, memory and output."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import config
from brain.utils import file_lock


def _rss_bytes(pid: int) -> int | None:
    try:
        result = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(pid)],
                                capture_output=True, text=True, timeout=0.5)
        return int(result.stdout.strip()) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _wait_worker(proc, timeout: float, memory_bytes: int) -> None:
    deadline = time.monotonic() + timeout
    try:
        while proc.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("PDF extraction exceeded its time limit")
            rss = _rss_bytes(proc.pid)
            if rss is None and proc.poll() is None:
                raise ValueError("cannot monitor PDF parser memory usage")
            if rss is not None and rss > memory_bytes:
                raise ValueError("PDF extraction exceeded its memory limit")
            try:
                proc.wait(timeout=min(0.05, remaining))
            except subprocess.TimeoutExpired:
                pass
        if proc.returncode != 0:
            raise ValueError("PDF parser failed or exceeded a resource limit")
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def extract_text(data: bytes | str | Path) -> str:
    # Bound aggregate parser memory as well as each worker's memory budget.
    with file_lock(config.DB_DIR / ".neural_map_pdf_worker.lock"):
        return _extract_text_locked(data)


def _extract_text_locked(data: bytes | str | Path) -> str:
    if not isinstance(data, bytes):
        path = Path(data)
        if path.stat().st_size > config.MAX_UPLOAD_BYTES:
            raise ValueError("PDF exceeds the upload byte limit")
        with path.open("rb") as handle:
            data = handle.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise ValueError("PDF exceeds the upload byte limit")
    text_cap = min(config.PDF_MAX_TEXT_CHARS, config.MAX_DOCUMENT_CHARS)
    output_cap = text_cap * 4 + 65536
    with tempfile.TemporaryDirectory(prefix="neural-map-pdf-") as directory:
        source, output = Path(directory) / "input.pdf", Path(directory) / "result.json"
        source.write_bytes(data)
        proc = subprocess.Popen(
            [sys.executable, "-I", str(Path(__file__).with_name("pdf_worker.py")),
             str(source), str(output), str(config.PDF_TIMEOUT_SECONDS),
             str(config.PDF_MEMORY_MB), str(config.PDF_MAX_PAGES), str(text_cap)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _wait_worker(proc, config.PDF_TIMEOUT_SECONDS, config.PDF_MEMORY_MB * 1024 * 1024)
        if not output.is_file() or output.stat().st_size > output_cap:
            raise ValueError("PDF parser produced no valid bounded result")
        with output.open("rb") as handle:
            raw = handle.read(output_cap + 1)
        if len(raw) > output_cap:
            raise ValueError("PDF extraction exceeded its output limit")
        result = json.loads(raw)
        if result.get("error"):
            raise ValueError(result["error"])
        text = result.get("text")
        if not isinstance(text, str) or len(text) > text_cap:
            raise ValueError("PDF extraction exceeded its text limit")
        return text
