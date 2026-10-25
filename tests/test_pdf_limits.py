"""Real extraction and deterministic enforcement of the PDF worker budgets."""
from io import BytesIO
import subprocess
import sys

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

import config
from brain import documents, pdf


def sample_pdf(pages=1, text="Hello PDF"):
    writer = PdfWriter()
    for _ in range(pages):
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)}),
        })
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_pdf_extraction_works_for_bytes_and_paths(tmp_path):
    data = sample_pdf()
    assert documents.extract_pdf_text(data) == "Page 1\nHello PDF"
    path = tmp_path / "document.pdf"
    path.write_bytes(data)
    assert documents.extract_pdf_text(path) == "Page 1\nHello PDF"


def test_pdf_page_limit(monkeypatch):
    monkeypatch.setattr(config, "PDF_MAX_PAGES", 1)
    with pytest.raises(ValueError, match="page limit"):
        documents.extract_pdf_text(sample_pdf(pages=2))


def test_pdf_text_limit(monkeypatch):
    monkeypatch.setattr(config, "PDF_MAX_TEXT_CHARS", 10)
    with pytest.raises(ValueError, match="text limit"):
        documents.extract_pdf_text(sample_pdf())


def test_pdf_byte_limit_is_checked_before_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 8)
    monkeypatch.setattr(pdf.subprocess, "Popen", lambda *a, **kw: pytest.fail("oversized PDF spawned a parser"))
    path = tmp_path / "large.pdf"
    path.write_bytes(b"x" * 9)
    for data in (path, b"x" * 9):
        with pytest.raises(ValueError, match="byte limit"):
            documents.extract_pdf_text(data)


def test_invalid_pdf_fails_cleanly():
    with pytest.raises(ValueError, match="PDF extraction failed"):
        documents.extract_pdf_text(b"this is not a PDF")


@pytest.mark.parametrize("breach", ["time", "memory", "unmonitored"])
def test_worker_breaches_kill_and_reap_process(monkeypatch, breach):
    proc = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(10)"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    monkeypatch.setattr(pdf, "_rss_bytes", lambda pid: {"time": 0, "memory": 2048, "unmonitored": None}[breach])
    with pytest.raises(ValueError, match={"time": "time limit", "memory": "memory limit", "unmonitored": "monitor"}[breach]):
        pdf._wait_worker(proc, timeout=0.03, memory_bytes=1024)
    assert proc.poll() is not None
