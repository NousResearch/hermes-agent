#!/usr/bin/env python3
"""Invariant: the bytes route must reach hosted OCR for scanned PDFs.

``read_extract`` has two anydoc routes:

* ``_extract_anydoc(path)``      — path-based callers
* ``_extract_anydoc_bytes(data, path)`` — ``read_file`` and every other
  caller that already holds the file's bytes

Only the PATH route caught ``anydoc.NeedsOcrError`` and retried through
``_ocr_scanned_pdf()`` (hosted OCR). The BYTES route caught every exception
and re-raised it as ``ExtractionError``, so a scanned PDF failed outright with
``NeedsOcrError: page 1 of 1 needs OCR`` — even with ``FIRECRAWL_API_KEY``
configured and hosted OCR available.

These tests pin the invariant: the bytes route must reach the same hosted-OCR
retry the path route already had.

Run with:  python -m pytest tests/tools/test_read_extract_hosted_ocr_bytes.py -v
"""

import os
import tempfile
import unittest
from unittest import mock

from tools import read_extract
from tools.read_extract import ExtractionError


class _FakeNeedsOcrError(Exception):
    """Stand-in for ``anydoc.NeedsOcrError`` (a real one carries ``pages``)."""

    def __init__(self, pages=(1,)):
        super().__init__("page 1 of 1 needs OCR")
        self.pages = list(pages)


class _FakeAnydoc:
    """Minimal anydoc stand-in: the bytes route always reports "needs OCR"."""

    NeedsOcrError = _FakeNeedsOcrError

    def to_markdown(self, path, **kwargs):
        raise _FakeNeedsOcrError()

    def to_markdown_bytes(self, data, **kwargs):
        raise _FakeNeedsOcrError()


class TestBytesRouteReachesHostedOcr(unittest.TestCase):
    """The bytes route must not lose the hosted-OCR retry."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rex_hosted_ocr_")
        self.pdf = os.path.join(self.tmp, "scan.pdf")
        with open(self.pdf, "wb") as fh:
            fh.write(b"%PDF-1.4\n% scanned image only\n")

        self._saved_module = read_extract._anydoc_module
        self._saved_failed_at = read_extract._anydoc_failed_at
        read_extract._anydoc_module = _FakeAnydoc()
        read_extract._anydoc_failed_at = None

    def tearDown(self):
        import shutil

        read_extract._anydoc_module = self._saved_module
        read_extract._anydoc_failed_at = self._saved_failed_at
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_bytes_route_retries_via_hosted_ocr(self):
        """A NeedsOcrError from the bytes route must trigger the hosted retry."""
        with mock.patch.object(
            read_extract, "_ocr_scanned_pdf", return_value="hosted ocr text\n"
        ) as hosted:
            text = read_extract._extract_anydoc_bytes(b"%PDF-1.4", self.pdf)

        hosted.assert_called_once()
        self.assertEqual(text, "hosted ocr text\n")

    def test_bytes_route_matches_path_route(self):
        """Both routes must reach hosted OCR for the same scanned document."""
        with mock.patch.object(
            read_extract, "_ocr_scanned_pdf", return_value="hosted ocr text\n"
        ) as hosted:
            read_extract._extract_anydoc_bytes(b"%PDF-1.4", self.pdf)
            bytes_calls = hosted.call_count
            read_extract._extract_anydoc(self.pdf)
            path_calls = hosted.call_count - bytes_calls

        self.assertEqual(bytes_calls, 1, "bytes route must attempt hosted OCR")
        self.assertEqual(path_calls, 1, "path route must attempt hosted OCR")

    def test_non_ocr_error_still_raises(self):
        """Only NeedsOcrError routes to hosted OCR; other errors still raise."""

        class _FakeOtherError(Exception):
            pass

        class _BrokenAnydoc(_FakeAnydoc):
            def to_markdown_bytes(self, data, **kwargs):
                raise _FakeOtherError("malformed document")

        read_extract._anydoc_module = _BrokenAnydoc()

        with mock.patch.object(read_extract, "_ocr_scanned_pdf") as hosted:
            with self.assertRaises(ExtractionError):
                read_extract._extract_anydoc_bytes(b"%PDF-1.4", self.pdf)

        hosted.assert_not_called()


if __name__ == "__main__":
    unittest.main()
