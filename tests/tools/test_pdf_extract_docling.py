from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "productivity"
    / "docling-kb-pdf-ingest"
    / "scripts"
)


def load_module(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    for name in ["pdf_ingest_config", "pdf_ingest_lib", "pdf_extract_docling"]:
        sys.modules.pop(name, None)
    return importlib.import_module("pdf_extract_docling")


def test_docling_uses_ocr_fallback_when_markdown_is_too_short(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)

    class _FakeDocument:
        def export_to_markdown(self):
            return "<!-- image -->"

        def export_to_dict(self):
            return {"kind": "fake"}

    class _FakeResult:
        document = _FakeDocument()

    class _FakeConverter:
        def convert(self, _src):
            return _FakeResult()

    monkeypatch.setattr(module, "DocumentConverter", lambda: _FakeConverter())
    monkeypatch.setattr(
        module,
        "_ocr_fallback_markdown",
        lambda _pdf_path, _output_dir: (
            "## OCR Fallback — Page 1\n\nVALID FOR WORK ONLY WITH DHS AUTHORIZATION",
            [
                {
                    "ordinal": 1,
                    "section_heading": "OCR Fallback — Page 1",
                    "page_start": 1,
                    "page_end": 1,
                    "parser_chunk_id": "ocr-fallback-page-1",
                    "provenance_refs": [1],
                    "text": "VALID FOR WORK ONLY WITH DHS AUTHORIZATION",
                }
            ],
        ),
    )

    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-scan")
    output_dir = tmp_path / "out"

    payload = module.extract_pdf(pdf_path, output_dir, 1)

    assert payload["ocr_fallback_used"] is True
    assert "VALID FOR WORK ONLY WITH DHS AUTHORIZATION" in payload["markdown_text"]
    saved_json = json.loads((output_dir / "docling.json").read_text(encoding="utf-8"))
    assert saved_json["ocr_fallback"]["used"] is True
