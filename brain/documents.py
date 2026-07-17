"""Ingest documents as memory units.

Shared by the CLI (ingest_md.py) and the web upload endpoint so both extract, chunk,
tag, and store files identically. Each chunk is stored with a `source` so that
re-ingesting a file replaces its old chunks instead of duplicating them.
"""
from __future__ import annotations

from io import BytesIO
import re
from pathlib import Path

import config
from brain import graph, sensory, store
from brain.chunking import chunk_markdown
from brain.embeddings import EmbeddingUnavailable, embed_documents, text_only_embedding
from brain.memory import MemoryUnit

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}
# Still images: each is described + OCR'd by a vision model, then stored as text.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
# Audio files are stored as sensory vectors plus a small metadata note. A future
# speech/transcript backend can enrich the text side without changing callers.
AUDIO_SUFFIXES = sensory.AUDIO_SUFFIXES
SUPPORTED_MEDIA = SUPPORTED_SUFFIXES | IMAGE_SUFFIXES | AUDIO_SUFFIXES


def slugify(name: str) -> str:
    s = re.sub(r"[^\w]+", "-", name.lower()).strip("-")
    return s or "document"


def is_supported_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def is_supported_audio(path: str | Path) -> bool:
    return Path(path).suffix.lower() in AUDIO_SUFFIXES


def is_supported_document(path: str | Path) -> bool:
    """True for any ingestible file — text documents, PDFs, images, and audio."""
    return Path(path).suffix.lower() in SUPPORTED_MEDIA


def _require_pypdf():
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires pypdf. Run: ./.venv/bin/python -m pip install -r requirements.txt") from exc
    return PdfReader


_SOFT_HYPHEN = re.compile(r"([A-Za-z])-\n([a-z])")
_BLANK_PAGE = re.compile(r"this page (?:is )?intentionally left blank", re.IGNORECASE)
_STRUCT_HEADING = re.compile(
    r"^\s*((?:MODULE|PART|UNIT|CHAPTER|SECTION|EXEMPLAR|APPENDIX)\s+[\dIVXLC]+[^\n]{0,90})$",
    re.IGNORECASE | re.MULTILINE,
)


def _normalize_pdf_text(text: str) -> str:
    """Repair common PDF-extraction artifacts so chunks embed and search well."""
    text = _SOFT_HYPHEN.sub(r"\1\2", text)                                  # join line-wrap hyphenation
    text = _STRUCT_HEADING.sub(lambda m: f"## {m.group(1).strip()}", text)  # promote textbook structure to headings
    return text


def extract_pdf_text(data: bytes | str | Path) -> str:
    PdfReader = _require_pypdf()
    reader = PdfReader(BytesIO(data) if isinstance(data, bytes) else str(data))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if not text or _BLANK_PAGE.search(text):   # skip empty / "intentionally left blank" pages
            continue
        pages.append(f"Page {i}\n{text}")
    return _normalize_pdf_text("\n\n".join(pages).strip())


def describe_image_file(path: str | Path, model: str | None = None) -> str:
    """Caption + OCR an image file into text. Vision model is separate from the
    chat/embed model, so it defaults to config.VISION_MODEL unless overridden."""
    from brain.vision import describe_image

    return describe_image(Path(path).read_bytes(), model=model)


def read_document(path: str | Path, vision_model: str | None = None) -> tuple[str, str]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(p), p.name
    if suffix in IMAGE_SUFFIXES:
        return describe_image_file(p, model=vision_model), p.name
    if suffix in AUDIO_SUFFIXES:
        return sensory.audio_summary(p.read_bytes(), p.name), p.name
    if suffix in {".md", ".markdown", ".txt"}:
        return p.read_text(encoding="utf-8", errors="ignore"), p.name
    raise ValueError(f"unsupported document type: {p.suffix}")


def ingest_document_text(
    text: str,
    filename: str,
    source: str,
    category: str | None = None,
    importance: int = 3,
    max_chars: int | None = None,
    replace: bool = True,
    model: str | None = None,
) -> tuple[int, str]:
    """Chunk document text and store it. Returns (chunk_count, category)."""
    if not text.strip():
        raise ValueError("document has no extractable text")
    max_chars = max_chars or config.CHUNK_MAX_CHARS
    filename = Path(filename).name
    if not is_supported_document(filename):
        raise ValueError("filename must be a .md, .markdown, .txt, .pdf, image, or audio file")
    category = category or slugify(Path(filename).stem)
    suffix = Path(filename).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        source_type = "vision_ocr"
    elif suffix in AUDIO_SUFFIXES:
        source_type = "audio_sensation"
    else:
        source_type = "uploaded_doc"

    chunks = chunk_markdown(text, max_chars=max_chars)
    if not chunks:
        raise ValueError("document produced no ingestible chunks")
    units = []
    for chunk, heading in chunks:
        prefix = f"[{filename} > {heading}]" if heading else f"[{filename}]"
        units.append(
            MemoryUnit(
                text=f"{prefix}\n{chunk}",
                category=category,
                importance=importance,
                source=source,
                extra={"source_type": source_type},
            )
        )
    texts = [unit.text for unit in units]
    text_only = False
    try:
        embeddings = embed_documents(texts, model=model)
    except EmbeddingUnavailable:
        text_only = True
        embeddings = [text_only_embedding() for _ in units]

    old_ids: dict[bool, list[str]] = {False: [], True: []}
    if replace and source:
        for rec in store.list_records_all(model):
            if (rec.get("meta") or {}).get("source") == source:
                old_ids[rec.get("mode") == "text_only"].append(rec["id"])

    try:
        store.add_many(units, embeddings, model=model, text_only=text_only)
    except Exception:
        # add_many writes in batches, so a later-batch failure can leave earlier
        # units committed. Always remove this attempt's fresh UUIDs; ``replace``
        # controls retirement of OLD source records, not rollback of new writes.
        store.delete([unit.id for unit in units], model=model, text_only=text_only)
        raise

    if replace and source:
        for old_text_only, ids in old_ids.items():
            store.delete(ids, model=model, text_only=old_text_only)
        # Replaced chunks may carry graph edges (e.g. imported documents);
        # deleting the records without the edges leaves dangling_graph_fact
        # integrity errors behind.
        replaced = old_ids[False] + old_ids[True]
        if replaced:
            graph.delete_fact_ids(replaced, collections=store.graph_collection_names(model))
    return len(chunks), category


def ingest_markdown_text(*args, **kwargs) -> tuple[int, str]:
    """Backward-compatible alias for older markdown-only callers."""
    return ingest_document_text(*args, **kwargs)


def ingest_file(
    path: str | Path,
    category: str | None = None,
    importance: int = 3,
    max_chars: int | None = None,
    replace: bool = True,
    model: str | None = None,
    vision_model: str | None = None,
) -> tuple[int, str]:
    p = Path(path)
    text, filename = read_document(p, vision_model=vision_model)
    n, cat = ingest_document_text(
        text,
        filename=filename,
        source=str(p.resolve()),
        category=category,
        importance=importance,
        max_chars=max_chars,
        replace=replace,
        model=model,
    )
    suffix = p.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        sensory.add_media("image", p.read_bytes(), str(p.resolve()), filename, text, cat, importance)
    elif suffix in AUDIO_SUFFIXES:
        sensory.add_media("audio", p.read_bytes(), str(p.resolve()), filename, text, cat, importance)
    return n, cat


def find_documents(paths: list[str], recursive: bool = True) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            iterator = p.rglob("*") if recursive else p.glob("*")
            # skip hidden files/dirs (.obsidian, .trash, AppleDouble ._* files)
            found.extend(
                sorted(
                    f
                    for f in iterator
                    if f.is_file()
                    and is_supported_document(f)
                    and not any(part.startswith(".") for part in f.relative_to(p).parts)
                )
            )
        elif is_supported_document(p):
            found.append(p)
        else:
            print(f"  skip {p} (not a supported document)")
    # de-dup, preserve order
    return list(dict.fromkeys(found))


def find_markdown(paths: list[str], recursive: bool = True) -> list[Path]:
    """Backward-compatible alias for older markdown-only callers."""
    return find_documents(paths, recursive=recursive)
