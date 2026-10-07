"""read_file scanned-PDF OCR: the BYTES extraction path (the one read_file
uses) must recover an image-only PDF by rasterizing the NeedsOcrError pages
with pypdfium2 and transcribing them with the configured auxiliary vision
model, instead of raising ExtractionError. Hosted OCR stays a fallback, and
with no renderer/vision backend the existing NEEDS-OCR warning is returned.

Hermetic: the rasterizer and the vision call are injected/monkeypatched, so
no pypdfium2 install and no network are touched.
"""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class _NeedsOcrError(Exception):
    def __init__(self, pages):
        super().__init__("needs ocr")
        self.pages = pages


def _scan_mod(hosted_result=None, hosted_exc=None):
    """Fake anydoc module: to_markdown_bytes raises NeedsOcrError on the
    no-OCR call and records any hosted call it receives."""
    calls = []

    def to_markdown_bytes(data, **kw):
        calls.append(kw)
        if not kw:
            raise _NeedsOcrError([1])
        if hosted_exc is not None:
            raise hosted_exc
        return hosted_result

    mod = SimpleNamespace(NeedsOcrError=_NeedsOcrError, to_markdown_bytes=to_markdown_bytes)
    return mod, calls


class _FakeImage:
    def save(self, buffer, format=None):
        buffer.write(b"\x89PNG\r\n\x1a\nfake")


class _FakeBitmap:
    def to_pil(self):
        return _FakeImage()


class _FakePage:
    def render(self, scale=1.0):
        assert scale >= 1.0, "OCR render scale must not drop below ~144 dpi"
        return _FakeBitmap()


class _FakeDoc:
    def __init__(self, data):
        assert data == b"%PDF-1.4 scan"

    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return _FakePage()


class _FakePdfium:
    PdfDocument = _FakeDoc


class _FakeCompletion:
    class _Msg:
        content = "INVOICE BS-2026-0042\nKode unik: QX-77-ALFA"

    choices = [type("_C", (), {"message": _Msg()})()]


class TestBytesPathScannedPdf(unittest.TestCase):
    def test_vision_recovers_and_returns_text(self):
        from tools import read_extract as rx

        mod, calls = _scan_mod()
        with patch.object(rx, "_anydoc", return_value=mod), \
                patch.object(rx, "_vision_ocr_scanned_pdf", return_value="VISION TEXT\n"):
            out = rx._extract_anydoc_bytes(b"%PDF-1.4 scan", "scan.pdf")
        self.assertEqual(out, "VISION TEXT\n")
        self.assertEqual(calls, [{}])  # only the initial no-OCR attempt; hosted untouched

    def test_vision_precedes_hosted(self):
        from tools import read_extract as rx

        mod, calls = _scan_mod(hosted_result="HOSTED TEXT")
        with patch.object(rx, "_anydoc", return_value=mod), \
                patch.object(rx, "_vision_ocr_scanned_pdf", return_value="VISION TEXT"), \
                patch.object(rx, "_hosted_ocr_config", return_value=(True, "key", None)):
            out = rx._extract_anydoc_bytes(b"%PDF-1.4 scan", "scan.pdf")
        self.assertIn("VISION TEXT", out)
        self.assertEqual(len(calls), 1)  # hosted OCR never invoked

    def test_hosted_remains_fallback_when_vision_unavailable(self):
        from tools import read_extract as rx

        mod, calls = _scan_mod(hosted_result="HOSTED TEXT")
        with patch.object(rx, "_anydoc", return_value=mod), \
                patch.object(rx, "_vision_ocr_scanned_pdf", return_value=None), \
                patch.object(rx, "_hosted_ocr_config", return_value=(True, "key", None)):
            out = rx._extract_anydoc_bytes(b"%PDF-1.4 scan", "scan.pdf")
        self.assertEqual(out, "HOSTED TEXT\n")
        self.assertEqual(calls[1].get("ocr"), "hosted")

    def test_warns_when_no_vision_and_hosted_off(self):
        from tools import read_extract as rx

        mod, _ = _scan_mod()
        with patch.object(rx, "_anydoc", return_value=mod), \
                patch.object(rx, "_vision_ocr_scanned_pdf", return_value=None), \
                patch.object(rx, "_hosted_ocr_config", return_value=(False, None, None)):
            out = rx._extract_anydoc_bytes(b"%PDF-1.4 scan", "scan.pdf")
        self.assertIn("[NEEDS OCR", out)
        self.assertIn("pages 1", out)
        self.assertNotIn("pdftoppm", out)  # unrunnable advice is gone

    def test_non_needsocr_error_still_raises(self):
        from tools import read_extract as rx

        class Mod:
            NeedsOcrError = _NeedsOcrError

            @staticmethod
            def to_markdown_bytes(data, **kw):
                raise ValueError("malformed")

        mod = Mod()
        with patch.object(rx, "_anydoc", return_value=mod):
            with self.assertRaises(rx.ExtractionError):
                rx._extract_anydoc_bytes(b"%PDF-1.4 scan", "scan.pdf")


class TestVisionOcrHelper(unittest.TestCase):
    def test_assembles_pages_with_markers_and_note(self):
        from tools import read_extract as rx

        with patch.object(rx, "_pdfium", return_value=_FakePdfium), \
                patch("agent.auxiliary_client.call_llm", return_value=_FakeCompletion()):
            out = rx._vision_ocr_scanned_pdf(b"%PDF-1.4 scan", [1], "scan.pdf")
        self.assertIsNotNone(out)
        assert out is not None
        self.assertIn("[OCR RECOVERY", out)
        self.assertIn("page 1", out)
        self.assertIn("── page 1 (OCR) ──", out)
        self.assertIn("INVOICE BS-2026-0042", out)
        self.assertIn("Kode unik: QX-77-ALFA", out)

    def test_returns_none_without_renderer(self):
        from tools import read_extract as rx

        with patch.object(rx, "_pdfium", return_value=None):
            self.assertIsNone(rx._vision_ocr_scanned_pdf(b"%PDF-1.4 scan", [1], "scan.pdf"))

    def test_uses_data_url(self):
        from tools import read_extract as rx

        seen = {}

        def _fake_call_llm(**kw):
            seen.update(kw)
            return _FakeCompletion()

        with patch.object(rx, "_pdfium", return_value=_FakePdfium), \
                patch("agent.auxiliary_client.call_llm", side_effect=_fake_call_llm):
            rx._vision_ocr_scanned_pdf(b"%PDF-1.4 scan", [1], "scan.pdf")
        self.assertEqual(seen.get("task"), "vision")
        url = seen["messages"][0]["content"][1]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
