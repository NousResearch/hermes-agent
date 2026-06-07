"""Tests for tools/document_tools.py — first-class document text extraction.

The ``read_document`` tool gives the agent a reflexive, in-process way to read
PDFs (and other pymupdf-supported formats) instead of hand-rolling
``python3 -c`` / ``pip install`` in code-exec — see decision
2026-06-04-opinionated-hsm-base-package.
"""

import pytest

# pymupdf is an optional ("documents") extra, not in the lean CI install
# (`.[all,dev]`). Skip cleanly there; runs in the image where it ships eager.
fitz = pytest.importorskip("fitz")

from tools.document_tools import (
    READ_DOCUMENT_SCHEMA,
    _handle_read_document,
    check_document_requirements,
    read_document_tool,
)


def _make_pdf(path, pages_text):
    """Write a PDF with one page per entry in ``pages_text`` (empty str = no text)."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


class TestReadDocument:
    @pytest.mark.asyncio
    async def test_reads_text_from_pdf(self, tmp_path):
        pdf = tmp_path / "hello.pdf"
        _make_pdf(pdf, ["Hello from page one"])
        result = await read_document_tool(str(pdf))
        assert "Hello from page one" in result

    @pytest.mark.asyncio
    async def test_missing_file_returns_clear_error(self, tmp_path):
        # Must not raise — the agent gets an actionable message instead.
        result = await read_document_tool(str(tmp_path / "nope.pdf"))
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_page_range_limits_pages(self, tmp_path):
        pdf = tmp_path / "two.pdf"
        _make_pdf(pdf, ["AlphaPageOne", "BetaPageTwo"])
        result = await read_document_tool(str(pdf), pages="1")
        assert "AlphaPageOne" in result
        assert "BetaPageTwo" not in result

    @pytest.mark.asyncio
    async def test_image_only_pdf_signals_needs_ocr(self, tmp_path):
        # A page with no text layer should not look like a successful empty
        # read — it should redirect the agent to the vision/OCR path.
        pdf = tmp_path / "blank.pdf"
        _make_pdf(pdf, [""])
        result = await read_document_tool(str(pdf))
        low = result.lower()
        assert "ocr" in low or "vision" in low

    @pytest.mark.asyncio
    async def test_handler_reads_path_arg(self, tmp_path):
        pdf = tmp_path / "h.pdf"
        _make_pdf(pdf, ["HandlerSees"])
        result = await _handle_read_document({"path": str(pdf)})
        assert "HandlerSees" in result


class TestSchemaAndRegistration:
    def test_schema_name_and_required_path(self):
        assert READ_DOCUMENT_SCHEMA["name"] == "read_document"
        props = READ_DOCUMENT_SCHEMA["parameters"]["properties"]
        assert "path" in props
        assert "path" in READ_DOCUMENT_SCHEMA["parameters"]["required"]

    def test_registered_in_registry(self):
        import tools.document_tools  # noqa: F401  (self-registers on import)
        from tools.registry import registry

        assert registry.get_entry("read_document") is not None


class TestRequirements:
    def test_check_requirements_true_when_pymupdf_present(self):
        # fitz imported at module top, so the dep is satisfied in this env.
        assert check_document_requirements() is True
