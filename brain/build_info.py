"""Stable fingerprint for Python code loaded by the Neural Map server."""
from __future__ import annotations

import hashlib
from pathlib import Path


def source_fingerprint(base_dir: Path | None = None) -> str:
    base = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    paths = [base / "server.py", base / "config.py"]
    paths.extend(sorted((base / "brain").rglob("*.py")))
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(base)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:20]
