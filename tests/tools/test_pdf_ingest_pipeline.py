from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "productivity"
    / "docling-kb-pdf-ingest"
    / "scripts"
)


def load_pipeline_modules(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    for name in [
        "pdf_ingest_config",
        "pdf_ingest_lib",
        "pdf_extract_docling",
        "pdf_extract_pymupdf4llm",
        "pdf_promote_to_wiki",
        "pdf_ingest_pipeline",
        "telegram_pdf_drop_ingest",
        "pdf_pilot_report",
    ]:
        sys.modules.pop(name, None)
    config = importlib.import_module("pdf_ingest_config")
    lib = importlib.import_module("pdf_ingest_lib")
    pipeline = importlib.import_module("pdf_ingest_pipeline")
    telegram = importlib.import_module("telegram_pdf_drop_ingest")
    return config, lib, pipeline, telegram


@pytest.fixture()
def loaded(monkeypatch, tmp_path):
    return load_pipeline_modules(monkeypatch, tmp_path)


def fake_extractor_factory(parser_name: str):
    def _extract(pdf_path: Path, output_dir: Path, page_count: int):
        output_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = output_dir / f"{parser_name}.md"
        json_path = output_dir / f"{parser_name}.json"
        chunk_path = output_dir / "chunks.json"
        markdown_text = f"# {parser_name} extraction\n\nParsed {pdf_path.name}"
        chunks = [
            {
                "ordinal": 1,
                "section_heading": f"{parser_name} heading",
                "page_start": 1,
                "page_end": page_count,
                "parser_chunk_id": f"{parser_name}-1",
                "provenance_refs": [1],
                "text": markdown_text,
            }
        ]
        markdown_path.write_text(markdown_text, encoding="utf-8")
        json_path.write_text(json.dumps({"parser": parser_name}), encoding="utf-8")
        chunk_path.write_text(json.dumps(chunks), encoding="utf-8")
        return {
            "parser_name": parser_name,
            "parser_version": "test-1",
            "markdown_text": markdown_text,
            "markdown_path": markdown_path,
            "json_path": json_path,
            "chunk_path": chunk_path,
            "chunks": chunks,
        }

    return _extract


def test_routes_short_text_pdf_to_pymupdf4llm(loaded, monkeypatch, tmp_path):
    config, lib, pipeline, _telegram = loaded
    staged = tmp_path / "input.pdf"
    staged.write_bytes(b"%PDF-short-text")

    monkeypatch.setattr(
        pipeline,
        "inspect_pdf",
        lambda _path: {
            "page_count": 3,
            "text_layer_present": True,
            "ocr_needed": False,
            "text_pages": 3,
            "text_chars": 900,
            "file_size_bytes": staged.stat().st_size,
        },
    )
    pipeline.PARSER_MAP["pymupdf4llm"] = fake_extractor_factory("pymupdf4llm")
    pipeline.PARSER_MAP["docling"] = fake_extractor_factory("docling")

    result = pipeline.ingest_pdf(pdf_path=staged, ingress_channel="manual_pilot", source_context={"pilot_label": "simple"})

    assert result["status"] == "success"
    assert result["parser_selected"] == "pymupdf4llm"
    assert result["record_id"].startswith("src_pdf_")
    assert Path(result["raw_path"]).exists()


def test_routes_non_text_pdf_to_docling(loaded, monkeypatch, tmp_path):
    _config, _lib, pipeline, _telegram = loaded
    staged = tmp_path / "scan.pdf"
    staged.write_bytes(b"%PDF-scan")

    monkeypatch.setattr(
        pipeline,
        "inspect_pdf",
        lambda _path: {
            "page_count": 2,
            "text_layer_present": False,
            "ocr_needed": True,
            "text_pages": 0,
            "text_chars": 0,
            "file_size_bytes": staged.stat().st_size,
        },
    )
    pipeline.PARSER_MAP["pymupdf4llm"] = fake_extractor_factory("pymupdf4llm")
    pipeline.PARSER_MAP["docling"] = fake_extractor_factory("docling")

    result = pipeline.ingest_pdf(pdf_path=staged, ingress_channel="manual_pilot", source_context={"pilot_label": "scan"})

    assert result["parser_selected"] == "docling"
    assert "no text layer detected" in result["route_reason"]


def test_duplicate_pdf_reuses_existing_record(loaded, monkeypatch, tmp_path):
    _config, _lib, pipeline, _telegram = loaded
    staged = tmp_path / "dup.pdf"
    staged.write_bytes(b"%PDF-dup")

    monkeypatch.setattr(
        pipeline,
        "inspect_pdf",
        lambda _path: {
            "page_count": 1,
            "text_layer_present": True,
            "ocr_needed": False,
            "text_pages": 1,
            "text_chars": 100,
            "file_size_bytes": staged.stat().st_size,
        },
    )
    pipeline.PARSER_MAP["pymupdf4llm"] = fake_extractor_factory("pymupdf4llm")
    pipeline.PARSER_MAP["docling"] = fake_extractor_factory("docling")

    first = pipeline.ingest_pdf(pdf_path=staged, ingress_channel="manual_pilot", source_context={})
    second = pipeline.ingest_pdf(pdf_path=staged, ingress_channel="manual_pilot", source_context={})

    assert first["record_id"] == second["record_id"]
    assert second["status"] == "skipped_duplicate"


def test_pipeline_writes_evidence_records(loaded, monkeypatch, tmp_path):
    config, _lib, pipeline, _telegram = loaded
    staged = tmp_path / "evidence.pdf"
    staged.write_bytes(b"%PDF-evidence")

    monkeypatch.setattr(
        pipeline,
        "inspect_pdf",
        lambda _path: {
            "page_count": 4,
            "text_layer_present": True,
            "ocr_needed": False,
            "text_pages": 4,
            "text_chars": 400,
            "file_size_bytes": staged.stat().st_size,
        },
    )
    pipeline.PARSER_MAP["pymupdf4llm"] = fake_extractor_factory("pymupdf4llm")
    pipeline.PARSER_MAP["docling"] = fake_extractor_factory("docling")

    result = pipeline.ingest_pdf(pdf_path=staged, ingress_channel="manual_pilot", source_context={"pilot_label": "evidence"})

    sources_lines = [line for line in config.SOURCES_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    chunks_lines = [line for line in config.CHUNKS_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    transforms_lines = [line for line in config.TRANSFORMS_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(sources_lines) == 1
    assert len(chunks_lines) == 1
    assert len(transforms_lines) >= 1

    source_record = json.loads(sources_lines[0])
    assert source_record["record_id"] == result["record_id"]
    assert source_record["parser_selected"] == result["parser_selected"]
    assert Path(result["promotion_candidate_path"]).exists()


def test_telegram_ingest_copies_to_staging_and_runs_pipeline(loaded, monkeypatch, tmp_path):
    config, _lib, _pipeline, telegram = loaded
    cached = tmp_path / "doc_cached.pdf"
    cached.write_bytes(b"%PDF-telegram")

    def _fake_ingest(**kwargs):
        return {
            "status": "success",
            "record_id": "src_pdf_deadbeefcafe",
            "source_id": "src_pdf_deadbeefcafe",
            "parser_selected": "docling",
            "route_reason": "default Docling path",
            "staging_path": str(kwargs["pdf_path"]),
        }

    monkeypatch.setattr(telegram, "ingest_pdf", _fake_ingest)

    result = telegram.ingest_telegram_pdf(
        cached_path=cached,
        chat_id="1632061707",
        message_id="42",
        original_filename="drop.pdf",
        caption="",
    )

    assert result["record_id"] == "src_pdf_deadbeefcafe"
    assert Path(result["staged_path"]).exists()
    assert str(Path(result["staged_path"]).parent).startswith(str(config.TELEGRAM_STAGING_HOME))


def test_pipeline_can_promote_source_summary_page(loaded, monkeypatch, tmp_path):
    config, _lib, pipeline, _telegram = loaded
    staged = tmp_path / "promote.pdf"
    staged.write_bytes(b"%PDF-promote")

    monkeypatch.setattr(
        pipeline,
        "inspect_pdf",
        lambda _path: {
            "page_count": 2,
            "text_layer_present": True,
            "ocr_needed": False,
            "text_pages": 2,
            "text_chars": 300,
            "file_size_bytes": staged.stat().st_size,
        },
    )
    pipeline.PARSER_MAP["pymupdf4llm"] = fake_extractor_factory("pymupdf4llm")
    pipeline.PARSER_MAP["docling"] = fake_extractor_factory("docling")

    result = pipeline.ingest_pdf(
        pdf_path=staged,
        ingress_channel="manual_pilot",
        source_context={"pilot_label": "promote"},
        dry_run_promotion=True,
        promote_source_page=True,
    )

    assert result["source_page_path"] is not None
    source_page = Path(result["source_page_path"])
    assert source_page.exists()
    assert "type: source" in source_page.read_text(encoding="utf-8")
