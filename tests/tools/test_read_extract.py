#!/usr/bin/env python3
"""
Tests for structured-document extraction in the read_file tool.

Covers .ipynb / .docx / .xlsx extraction (ported from Kilo-Org/kilocode
#10733, #10737, #10740) and the read_file_tool integration: pagination,
line-numbering, graceful fallback on malformed input, and hidden-sheet
omission.

Run with:  python -m pytest tests/tools/test_read_extract.py -v
"""

import base64
import json
import os
import tempfile
import unittest
import zipfile
from unittest import mock

from tools.read_extract import (
    ExtractionError,
    extract_document_text,
    is_extractable_document,
)
from tools.file_tools import read_file_tool


# ---------------------------------------------------------------------------
# Fixture builders — construct minimal valid OOXML / notebook files.
# ---------------------------------------------------------------------------

def _write_notebook(path, cells, nbformat=4):
    nb = {"cells": cells, "metadata": {}, "nbformat": nbformat, "nbformat_minor": 5}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh)


def _write_docx(path, document_xml):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", document_xml)


def _write_xlsx(path, *, workbook, rels, shared, sheets):
    """sheets: dict of part-name -> xml string."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", rels)
        if shared is not None:
            z.writestr("xl/sharedStrings.xml", shared)
        for part, xml in sheets.items():
            z.writestr(part, xml)


_NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_NS_S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


# ---------------------------------------------------------------------------
# is_extractable_document
# ---------------------------------------------------------------------------

class TestIsExtractable(unittest.TestCase):
    def test_recognized_extensions(self):
        self.assertTrue(is_extractable_document("a.ipynb"))
        self.assertTrue(is_extractable_document("/x/B.DOCX"))
        self.assertTrue(is_extractable_document("report.xlsx"))

    def test_unrecognized_extensions(self):
        self.assertFalse(is_extractable_document("a.py"))
        self.assertFalse(is_extractable_document("a.txt"))
        self.assertFalse(is_extractable_document("a.mp4"))

    def test_anydoc_extensions_track_availability(self):
        """PDF (and the other anydoc formats) are extractable exactly when
        the optional `anydoc` converter is importable."""
        from tools import read_extract

        available = read_extract._anydoc() is not None
        self.assertEqual(is_extractable_document("a.pdf"), available)
        self.assertEqual(is_extractable_document("a.odt"), available)
        self.assertEqual(is_extractable_document("a.epub"), available)


# ---------------------------------------------------------------------------
# Optional anydoc-backed formats (PDF, legacy Office, ODF, RTF, EPUB)
# ---------------------------------------------------------------------------

class TestAnydocExtraction(unittest.TestCase):
    """Real-binding tests — skipped when firecrawl-anydoc is not installed."""

    @classmethod
    def setUpClass(cls):
        from tools import read_extract

        cls.mod = read_extract._anydoc()
        if cls.mod is None:
            raise unittest.SkipTest("firecrawl-anydoc not installed")

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rex_anydoc_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_rtf_extracts_markdown(self):
        p = os.path.join(self.tmp, "doc.rtf")
        with open(p, "w", encoding="ascii") as fh:
            fh.write(r"{\rtf1\ansi {\b Bold title}\par plain body\par}")
        text = extract_document_text(p)
        self.assertIn("Bold title", text)
        self.assertIn("plain body", text)
        self.assertTrue(text.endswith("\n"))

    def test_malformed_file_raises_extraction_error(self):
        p = os.path.join(self.tmp, "junk.pdf")
        with open(p, "wb") as fh:
            fh.write(b"\x00\x01 not a pdf at all")
        with self.assertRaises(ExtractionError):
            extract_document_text(p)

    def test_stdlib_docx_path_still_authoritative(self):
        """A .docx keeps using the stdlib extractor even with anydoc
        installed — behavior must be identical either way."""
        p = os.path.join(self.tmp, "d.docx")
        _write_docx(
            p,
            f'<w:document xmlns:w="{_NS_W}"><w:body>'
            "<w:p><w:r><w:t>hello</w:t></w:r></w:p>"
            "</w:body></w:document>",
        )
        text = extract_document_text(p)
        self.assertEqual(text, "hello\n")


class TestAnydocSizeCap(unittest.TestCase):
    """Oversized inputs must be rejected before anydoc converts them.
    Uses a fake binding so it runs regardless of local install state."""

    def setUp(self):
        from tools import read_extract

        self.rex = read_extract
        self._saved_module = read_extract._anydoc_module
        self._saved_cap = read_extract.MAX_ANYDOC_BYTES
        self.tmp = tempfile.mkdtemp(prefix="rex_cap_")
        self.calls = []

        class _FakeAnydoc:
            def to_markdown(_self, path):
                self.calls.append(path)
                return "converted\n"

        read_extract._anydoc_module = _FakeAnydoc()

    def tearDown(self):
        import shutil

        self.rex._anydoc_module = self._saved_module
        self.rex.MAX_ANYDOC_BYTES = self._saved_cap
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, name, size):
        p = os.path.join(self.tmp, name)
        with open(p, "wb") as fh:
            fh.write(b"x" * size)
        return p

    def test_oversized_file_rejected_before_conversion(self):
        from tools.read_extract import _extract_anydoc

        self.rex.MAX_ANYDOC_BYTES = 10
        p = self._write("big.pdf", 11)
        with self.assertRaises(ExtractionError) as ctx:
            _extract_anydoc(p)
        self.assertIn("too large", str(ctx.exception))
        self.assertEqual(self.calls, [])

    def test_file_at_limit_converts(self):
        from tools.read_extract import _extract_anydoc

        self.rex.MAX_ANYDOC_BYTES = 10
        p = self._write("ok.pdf", 10)
        self.assertEqual(_extract_anydoc(p), "converted\n")
        self.assertEqual(self.calls, [p])

    def test_missing_file_raises_extraction_error(self):
        from tools.read_extract import _extract_anydoc

        with self.assertRaises(ExtractionError):
            _extract_anydoc(os.path.join(self.tmp, "gone.pdf"))
        self.assertEqual(self.calls, [])


class TestAnydocAbsent(unittest.TestCase):
    """The absent-dep contract, verified regardless of local install state
    by forcing the cached module handle to None."""

    def setUp(self):
        from tools import read_extract

        self._saved = read_extract._anydoc_module
        read_extract._anydoc_module = None

    def tearDown(self):
        from tools import read_extract

        read_extract._anydoc_module = self._saved

    def test_pdf_not_extractable_without_anydoc(self):
        self.assertFalse(is_extractable_document("a.pdf"))
        self.assertFalse(is_extractable_document("a.rtf"))

    def test_extract_raises_unsupported_without_anydoc(self):
        from tools.read_extract import _extract_anydoc

        with self.assertRaises(ExtractionError):
            _extract_anydoc("/tmp/whatever.pdf")

    def test_stdlib_formats_unaffected(self):
        self.assertTrue(is_extractable_document("a.ipynb"))
        self.assertTrue(is_extractable_document("a.docx"))
        self.assertTrue(is_extractable_document("a.xlsx"))


class TestAnydocInitLifecycle(unittest.TestCase):
    """First-load lifecycle: one failed load must not disable extraction
    for the rest of the process, and concurrent first use must not race."""

    def setUp(self):
        from tools import read_extract

        self.rex = read_extract
        self._saved_module = read_extract._anydoc_module
        self._saved_failed_at = read_extract._anydoc_failed_at
        self._saved_retry = read_extract.ANYDOC_RETRY_SECONDS
        read_extract._anydoc_module = read_extract._ANYDOC_UNSET
        read_extract._anydoc_failed_at = None
        self._ensure = mock.patch("tools.lazy_deps.ensure", return_value=None)
        self._ensure.start()

    def tearDown(self):
        self._ensure.stop()
        self.rex._anydoc_module = self._saved_module
        self.rex._anydoc_failed_at = self._saved_failed_at
        self.rex.ANYDOC_RETRY_SECONDS = self._saved_retry

    def test_successful_load_is_cached(self):
        fake = object()
        calls = []

        def fake_import(name):
            calls.append(name)
            return fake

        with mock.patch("importlib.import_module", side_effect=fake_import):
            self.assertIs(self.rex._anydoc(), fake)
            self.assertIs(self.rex._anydoc(), fake)
        self.assertEqual(calls, ["anydoc"])

    def test_failed_reconciliation_does_not_import_unverified_binding(self):
        with mock.patch(
            "tools.lazy_deps.ensure", side_effect=RuntimeError("wrong version")
        ), mock.patch("importlib.import_module") as import_module:
            self.assertIsNone(self.rex._anydoc())
        import_module.assert_not_called()

    def test_failed_load_is_retried_after_cooldown(self):
        fake = object()
        calls = []

        def fake_import(name):
            calls.append(name)
            if len(calls) == 1:
                raise ImportError("boom")
            return fake

        self.rex.ANYDOC_RETRY_SECONDS = 0.0
        with mock.patch("importlib.import_module", side_effect=fake_import):
            self.assertIsNone(self.rex._anydoc())
            self.assertIs(self.rex._anydoc(), fake)
        self.assertEqual(calls, ["anydoc", "anydoc"])

    def test_failed_load_not_retried_within_cooldown(self):
        calls = []

        def fake_import(name):
            calls.append(name)
            raise ImportError("boom")

        self.rex.ANYDOC_RETRY_SECONDS = 3600.0
        with mock.patch("importlib.import_module", side_effect=fake_import):
            self.assertIsNone(self.rex._anydoc())
            self.assertIsNone(self.rex._anydoc())
        # One import attempt total, and the handle stays UNSET so a retry
        # remains possible once the cooldown expires.
        self.assertEqual(calls, ["anydoc"])
        self.assertIs(self.rex._anydoc_module, self.rex._ANYDOC_UNSET)

    def test_concurrent_first_load_imports_once(self):
        import threading

        fake = object()
        calls = []
        barrier = threading.Barrier(4)

        def fake_import(name):
            calls.append(name)
            return fake

        def worker(out):
            barrier.wait(5)
            out.append(self.rex._anydoc())

        with mock.patch("importlib.import_module", side_effect=fake_import):
            results = []
            threads = [threading.Thread(target=worker, args=(results,)) for _ in range(3)]
            for t in threads:
                t.start()
            barrier.wait(5)
            for t in threads:
                t.join(5)
        self.assertEqual(calls, ["anydoc"])
        self.assertEqual(results, [fake, fake, fake])


# ---------------------------------------------------------------------------
# Notebooks (.ipynb) — #10733
# ---------------------------------------------------------------------------

class TestNotebookExtraction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rex_nb_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_markdown_and_code_in_order(self):
        p = os.path.join(self.tmp, "nb.ipynb")
        _write_notebook(p, [
            {"cell_type": "markdown", "source": ["# Title\n", "para"]},
            {"cell_type": "code", "source": "x = 1\nprint(x)",
             "outputs": [{"output_type": "stream", "text": ["1\n"]}],
             "execution_count": 1},
        ])
        text = extract_document_text(p)
        self.assertIn("# Title", text)
        self.assertIn("print(x)", text)
        # Output payloads must NOT leak into the extracted text.
        self.assertNotIn("output_type", text)
        self.assertNotIn("execution_count", text)
        # Order preserved: markdown before code.
        self.assertLess(text.index("Title"), text.index("print(x)"))


    def test_empty_cells_raises(self):
        p = os.path.join(self.tmp, "empty.ipynb")
        _write_notebook(p, [])
        with self.assertRaises(ExtractionError):
            extract_document_text(p)


# ---------------------------------------------------------------------------
# Word documents (.docx) — #10737
# ---------------------------------------------------------------------------

class TestDocxExtraction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rex_docx_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _doc(self, body):
        return (f'<?xml version="1.0"?><w:document xmlns:w="{_NS_W}">'
                f'<w:body>{body}</w:body></w:document>')

    def test_paragraphs_and_runs(self):
        p = os.path.join(self.tmp, "d.docx")
        _write_docx(p, self._doc(
            '<w:p><w:r><w:t>Hello </w:t></w:r><w:r><w:t>World</w:t></w:r></w:p>'
            '<w:p><w:r><w:t>Second</w:t></w:r></w:p>'))
        text = extract_document_text(p)
        self.assertIn("Hello World", text)
        self.assertIn("Second", text)


    def test_missing_document_xml_raises(self):
        p = os.path.join(self.tmp, "nodoc.docx")
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("other.xml", "<x/>")
        with self.assertRaises(ExtractionError):
            extract_document_text(p)


# ---------------------------------------------------------------------------
# Excel workbooks (.xlsx) — #10740
# ---------------------------------------------------------------------------

class TestXlsxExtraction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rex_xlsx_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build(self, path, *, include_hidden=True):
        r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        hidden_sheet = (f'<sheet name="Hidden" sheetId="2" state="hidden" '
                        f'xmlns:r="{r}" r:id="rId2"/>') if include_hidden else ""
        workbook = (
            f'<workbook xmlns="{_NS_S}" xmlns:r="{r}"><sheets>'
            f'<sheet name="Data" sheetId="1" r:id="rId1"/>{hidden_sheet}'
            f'</sheets></workbook>')
        rels = (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="x"/>'
            '<Relationship Id="rId2" Target="worksheets/sheet2.xml" Type="x"/>'
            '</Relationships>')
        shared = (f'<sst xmlns="{_NS_S}"><si><t>Name</t></si><si><t>Score</t></si>'
                  f'<si><t>Alice</t></si></sst>')
        sheet1 = (
            f'<worksheet xmlns="{_NS_S}"><sheetData>'
            '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
            '<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>95</v></c></row>'
            '</sheetData></worksheet>')
        sheet2 = (f'<worksheet xmlns="{_NS_S}"><sheetData>'
                  '<row r="1"><c r="A1" t="str"><v>SECRETDATA</v></c></row>'
                  '</sheetData></worksheet>')
        _write_xlsx(path, workbook=workbook, rels=rels, shared=shared,
                    sheets={"xl/worksheets/sheet1.xml": sheet1,
                            "xl/worksheets/sheet2.xml": sheet2})

    def test_visible_sheet_content(self):
        p = os.path.join(self.tmp, "wb.xlsx")
        self._build(p)
        text = extract_document_text(p)
        self.assertIn("Data", text)        # sheet label
        self.assertIn("Name\tScore", text)  # shared-string header row
        self.assertIn("Alice\t95", text)    # string + numeric cells


    def test_not_a_zip_raises(self):
        p = os.path.join(self.tmp, "bad.xlsx")
        with open(p, "wb") as fh:
            fh.write(b"nope")
        with self.assertRaises(ExtractionError):
            extract_document_text(p)


# ---------------------------------------------------------------------------
# read_file_tool integration
# ---------------------------------------------------------------------------

class TestReadFileToolIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rex_int_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_notebook_read_is_line_numbered(self):
        p = os.path.join(self.tmp, "nb.ipynb")
        _write_notebook(p, [
            {"cell_type": "markdown", "source": "# H"},
            {"cell_type": "code", "source": "print(1)"},
        ])
        res = json.loads(read_file_tool(p))
        self.assertTrue(res.get("extracted_document"))
        self.assertIn("1|", res["content"])  # line-number gutter
        self.assertIn("print(1)", res["content"])


    def test_corrupt_docx_surfaces_extraction_error(self):
        p = os.path.join(self.tmp, "bad.docx")
        with open(p, "wb") as fh:
            fh.write(b"not a zip")
        res = json.loads(read_file_tool(p))
        # Should NOT crash; the binary guard fires but surfaces the
        # specific extraction failure instead of the generic message.
        self.assertIn("error", res)
        self.assertIn("extraction failed", res["error"].lower())
        self.assertIn("docx", res["error"].lower())

    def test_oversized_anydoc_read_surfaces_size_error(self):
        import tools.read_extract as rex

        saved_cap = rex.MAX_ANYDOC_BYTES
        saved_module = rex._anydoc_module

        class _FakeAnydoc:
            def to_markdown(self, path):  # pragma: no cover - must not be called
                raise AssertionError("conversion should be rejected before call")

        rex._anydoc_module = _FakeAnydoc()
        rex.MAX_ANYDOC_BYTES = 10
        try:
            p = os.path.join(self.tmp, "big.pdf")
            with open(p, "wb") as fh:
                fh.write(b"x" * 11)
            res = json.loads(read_file_tool(p))
            self.assertIn("error", res)
            self.assertIn("too large", res["error"].lower())
            # The size hint reaches the agent instead of a generic binary error.
            self.assertNotIn("cannot read binary file", res["error"].lower())
        finally:
            rex.MAX_ANYDOC_BYTES = saved_cap
            rex._anydoc_module = saved_module

    def test_unavailable_converter_falls_back_to_raw_read(self):
        import time

        import tools.read_extract as rex

        saved_module = rex._anydoc_module
        saved_failed_at = rex._anydoc_failed_at
        # Simulate "converter unavailable and in cooldown": _anydoc() returns
        # None, the .pdf is not treated as extractable, and read_file keeps
        # its historical raw-read fallthrough (no extraction error surfaced).
        rex._anydoc_module = None
        rex._anydoc_failed_at = time.monotonic()
        try:
            p = os.path.join(self.tmp, "doc.pdf")
            with open(p, "wb") as fh:
                fh.write(b"%PDF-1.4 fake")
            res = json.loads(read_file_tool(p))
            self.assertNotIn("error", res)
            self.assertIn("%PDF-1.4 fake", res.get("content", ""))
        finally:
            rex._anydoc_module = saved_module
            rex._anydoc_failed_at = saved_failed_at

    def test_docx_read_extracts(self):
        p = os.path.join(self.tmp, "d.docx")
        _write_docx(p, (f'<?xml version="1.0"?><w:document xmlns:w="{_NS_W}">'
                        '<w:body><w:p><w:r><w:t>Report body</w:t></w:r></w:p>'
                        '</w:body></w:document>'))
        res = json.loads(read_file_tool(p))
        self.assertTrue(res.get("extracted_document"))
        self.assertIn("Report body", res["content"])

    def test_backend_only_anydoc_path_uses_transferred_bytes(self):
        from tools import file_tools, read_extract
        from tools.file_operations import ReadResult

        payload = br"{\rtf1\ansi Remote body\par}"

        class FakeAnydoc:
            def to_markdown_bytes(self, data):
                self.seen = data
                return "Remote body\n"

        class FakeFileOps:
            def read_file_bytes(self, path, max_bytes=None):
                self.path = path
                return ReadResult(
                    base64_content=base64.b64encode(payload).decode("ascii"),
                    file_size=len(payload),
                    is_binary=True,
                )

            @staticmethod
            def _add_line_numbers(content, start_line=1):
                return "\n".join(
                    f"{number}|{line}"
                    for number, line in enumerate(content.split("\n"), start_line)
                )

        fake_anydoc = FakeAnydoc()
        fake_ops = FakeFileOps()
        saved_module = read_extract._anydoc_module
        read_extract._anydoc_module = fake_anydoc
        try:
            with mock.patch.object(file_tools, "_get_file_ops", return_value=fake_ops), \
                    mock.patch.object(
                        file_tools,
                        "_resolve_path_for_task",
                        return_value=file_tools.PurePosixPath("/workspace/remote.rtf"),
                    ), mock.patch("os.path.getsize", side_effect=AssertionError("host read")):
                res = json.loads(read_file_tool("/workspace/remote.rtf", task_id="remote"))
        finally:
            read_extract._anydoc_module = saved_module

        self.assertTrue(res.get("extracted_document"))
        self.assertIn("Remote body", res["content"])
        self.assertEqual(fake_anydoc.seen, payload)
        self.assertEqual(fake_ops.path, "/workspace/remote.rtf")


# ---------------------------------------------------------------------------
# Scanned-PDF coverage warning
# ---------------------------------------------------------------------------

class TestPdfCoverageNote(unittest.TestCase):
    """The coverage footer flags PDFs whose pages yielded no text."""

    def _note_with_counts(self, counts):
        """Drive _pdf_coverage_note with synthetic per-page texts whose
        stripped lengths equal ``counts``."""
        from tools import read_extract
        texts = None if counts is None else ["x" * n for n in counts]
        with mock.patch.object(read_extract, "_pdf_page_texts",
                               return_value=texts):
            return read_extract._pdf_coverage_note("/x/doc.pdf")

    def test_mostly_scanned_pdf_warns_with_page_ranges(self):
        # 3 text pages then 6 empty ones (scanned) — well past the ratio.
        note = self._note_with_counts([900, 800, 700, 0, 0, 3, 0, 0, 0])
        self.assertIn("EXTRACTION COVERAGE WARNING", note)
        self.assertIn("6 of 9 pages", note)
        self.assertIn("pages 4-9", note)        # contiguous empty gap
        self.assertIn("(6 pages)", note)        # gap size stated
        self.assertIn("vision_analyze", note)   # recovery path is named
        self.assertIn("ocr-and-documents", note)
        self.assertIn("do NOT OCR or render everything", note)

    def test_gap_labels_carry_preceding_section_text(self):
        """Each gap is labeled with the last text page before it (usually
        a section divider), so the agent can pick which gaps to read."""
        from tools import read_extract
        texts = (
            ["Section One: Bylaws of the Corporation"] + [""] * 5
            + ["Section Two: Budget details here"] + [""] * 4
        )
        with mock.patch.object(read_extract, "_pdf_page_texts",
                               return_value=texts):
            note = read_extract._pdf_coverage_note("/x/doc.pdf")
        self.assertIn(
            'pages 2-6 (5 pages) — after "Section One: Bylaws of the Corporation" (p1)',
            note,
        )
        self.assertIn(
            'pages 8-11 (4 pages) — after "Section Two: Budget details here" (p7)',
            note,
        )

    def test_gap_map_caps_pathological_alternation(self):
        """Hundreds of alternating text/scan pages must not balloon the
        warning — gaps beyond the cap collapse to one summary line."""
        from tools import read_extract
        texts = []
        for i in range(60):  # 60 gaps of 1 page each
            texts.extend([f"Divider page number {i} with enough text", ""])
        with mock.patch.object(read_extract, "_pdf_page_texts",
                               return_value=texts):
            note = read_extract._pdf_coverage_note("/x/doc.pdf")
        gap_lines = [ln for ln in note.splitlines() if ln.startswith("  ")]
        self.assertEqual(
            len(gap_lines), read_extract.PDF_GAP_MAP_MAX_ENTRIES + 1
        )
        self.assertIn("more gaps", gap_lines[-1])
        self.assertIn("(40 pages)", gap_lines[-1])

    def test_full_text_pdf_is_silent(self):
        self.assertEqual(self._note_with_counts([500] * 20), "")

    def test_one_blank_page_is_tolerated(self):
        # A single separator/blank page in a text PDF should not warn.
        self.assertEqual(self._note_with_counts([500, 0, 500, 500]), "")

    def test_small_share_below_ratio_and_absolute_is_silent(self):
        # 3 empty of 40 (7.5% < 20%, and < absolute threshold of 10).
        counts = [400] * 37 + [0, 0, 0]
        self.assertEqual(self._note_with_counts(counts), "")

    def test_large_absolute_count_warns_even_below_ratio(self):
        # 12 empty of 100 (12% < 20% ratio) still warns: 12 lost pages
        # is real data loss regardless of document size.
        counts = [400] * 88 + [0] * 12
        note = self._note_with_counts(counts)
        self.assertIn("12 of 100 pages", note)

    def test_undeterminable_counts_are_silent(self):
        self.assertEqual(self._note_with_counts(None), "")
        self.assertEqual(self._note_with_counts([0]), "")  # single page

    def test_page_ranges_compact(self):
        from tools.read_extract import _page_ranges
        self.assertEqual(_page_ranges([2, 3, 4, 7, 9, 10]), "2-4, 7, 9-10")
        self.assertEqual(_page_ranges([5]), "5")

    def test_page_char_counts_missing_pdftotext(self):
        from tools import read_extract
        with mock.patch.object(read_extract.shutil, "which", return_value=None):
            self.assertIsNone(read_extract._pdf_page_char_counts("/x/doc.pdf"))

    def test_page_char_counts_parses_formfeeds(self):
        from tools import read_extract
        fake = mock.Mock(returncode=0, stdout=b"alpha beta\fgamma\f\f")
        with mock.patch.object(read_extract.shutil, "which",
                               return_value="/usr/bin/pdftotext"), \
             mock.patch.object(read_extract.subprocess, "run",
                               return_value=fake):
            counts = read_extract._pdf_page_char_counts("/x/doc.pdf")
        # Trailing empty segment after the final \f is dropped; the real
        # empty page between the two \f markers is preserved.
        self.assertEqual(counts, [len("alpha beta"), len("gamma"), 0])

    def test_extract_anydoc_prepends_note_for_pdf(self):
        """The warning leads the extracted text for .pdf inputs (a trailing
        footer would land on a page the model may never fetch)."""
        from tools import read_extract
        fake_mod = mock.Mock()
        fake_mod.to_markdown.return_value = "# Title\n\nBody"
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
            fh.write(b"%PDF-1.4 fake")
            p = fh.name
        try:
            with mock.patch.object(read_extract, "_anydoc",
                                   return_value=fake_mod), \
                 mock.patch.object(read_extract, "_pdf_coverage_note",
                                   return_value="[EXTRACTION COVERAGE WARNING: test]\n"):
                text = read_extract._extract_anydoc(p)
        finally:
            os.unlink(p)
        self.assertTrue(text.startswith("[EXTRACTION COVERAGE WARNING"))
        self.assertIn("# Title", text)

    def test_extract_anydoc_no_note_for_non_pdf(self):
        from tools import read_extract
        fake_mod = mock.Mock()
        fake_mod.to_markdown.return_value = "converted"
        with tempfile.NamedTemporaryFile(suffix=".rtf", delete=False) as fh:
            fh.write(b"{\\rtf1 fake}")
            p = fh.name
        try:
            with mock.patch.object(read_extract, "_anydoc",
                                   return_value=fake_mod), \
                 mock.patch.object(read_extract, "_pdf_coverage_note") as note:
                text = read_extract._extract_anydoc(p)
        finally:
            os.unlink(p)
        note.assert_not_called()
        self.assertEqual(text, "converted\n")

    def test_bytes_path_prepends_note_with_display_path(self):
        """Backend-transferred PDF bytes get the same warning, and the
        recovery command names the backend-visible path, not the host
        temp file the scan ran against."""
        from tools import read_extract
        fake_mod = mock.Mock()
        fake_mod.to_markdown_bytes.return_value = "# Title\n\nBody"
        seen = {}

        def fake_note(path, display_path=None):
            seen["scan_path"] = path
            seen["display_path"] = display_path
            return f"[EXTRACTION COVERAGE WARNING: test '{display_path}']\n"

        with mock.patch.object(read_extract, "_anydoc",
                               return_value=fake_mod), \
             mock.patch.object(read_extract, "_pdf_coverage_note",
                               side_effect=fake_note):
            text = read_extract._extract_anydoc_bytes(
                b"%PDF-1.4 fake", "/workspace/remote.pdf"
            )
        self.assertTrue(text.startswith("[EXTRACTION COVERAGE WARNING"))
        self.assertIn("/workspace/remote.pdf", text)
        self.assertEqual(seen["display_path"], "/workspace/remote.pdf")
        # The scanned file is a host temp materialization, already removed.
        self.assertNotEqual(seen["scan_path"], "/workspace/remote.pdf")
        self.assertFalse(os.path.exists(seen["scan_path"]))

    def test_bytes_path_no_note_for_non_pdf(self):
        from tools import read_extract
        fake_mod = mock.Mock()
        fake_mod.to_markdown_bytes.return_value = "converted"
        with mock.patch.object(read_extract, "_anydoc",
                               return_value=fake_mod), \
             mock.patch.object(read_extract,
                               "_pdf_coverage_note_from_bytes") as note:
            text = read_extract._extract_anydoc_bytes(
                b"{\\rtf1 fake}", "/workspace/remote.rtf"
            )
        note.assert_not_called()
        self.assertEqual(text, "converted\n")


if __name__ == "__main__":
    unittest.main()
