"""read_file schema diet (#95681): static unconditional format list
(anydoc bundled in core) + PDF-coverage teaching moved to the
response-time warning.

Maintainer-directed: the schema advertised anydoc-gated formats
unconditionally ("convert too when the optional anydoc converter is
available") and pre-taught the EXTRACTION COVERAGE WARNING's own
instructions. Now the format list renders only when anydoc is importable,
and the warning (read_extract.py) is the single teacher — it fires exactly
when pages are missing, with the page map and recovery commands.
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))



class TestReadFileSchemaStatic(unittest.TestCase):
    """Gate DROPPED by maintainer decision: anydoc is a core dependency
    (bundled), so format support is stated unconditionally — a missing
    converter is a broken install handled by read_extract's teaching
    error, not a schema variant."""

    def test_formats_stated_unconditionally(self):
        from tools.file_tools import READ_FILE_SCHEMA

        desc = READ_FILE_SCHEMA["description"]
        for token in (".ipynb", ".docx", ".pptx", ".doc/.ppt/.xls",
                      "PDF (text layer)", "OpenDocument", "RTF", "EPUB"):
            self.assertIn(token, desc, token)
        # No availability hedging, no install mechanics, no gate.
        self.assertNotIn("when the optional", desc)
        self.assertNotIn("auto-installed", desc)
        self.assertNotIn("anydoc", desc)

    def test_pdf_wording_upgrades_with_hosted_ocr_route(self):
        """The ONE dynamic word: text-layer → scanned-or-text, keyed on
        hosted_ocr_available(). Nous gateway deliberately does not
        upgrade (Parse proxy live-probed broken 2026-08-28)."""
        import tools.file_tools as ft

        with patch("tools.read_extract.hosted_ocr_available",
                   return_value=True):
            d = ft._read_file_schema_overrides()["description"]
        self.assertIn("PDF (scanned or text)", d)
        self.assertNotIn("PDF (text layer)", d)
        with patch("tools.read_extract.hosted_ocr_available",
                   return_value=False):
            o = ft._read_file_schema_overrides()
        self.assertEqual(o, {})  # base wording stands

    def test_hosted_ocr_available_gate_states(self):
        """Gate states: key unlocks (consent signal for egress),
        config true unlocks keyless (opt-in free tier), false disables,
        default (nothing set) stays key-gated as shipped in #97195."""
        import tools.read_extract as rx

        # direct key → True
        with patch.dict(rx.os.environ, {"FIRECRAWL_API_KEY": "fc-x"}):
            with patch("hermes_cli.config.load_config_readonly",
                       return_value={}):
                self.assertTrue(rx.hosted_ocr_available())
        # config false beats key
        with patch.dict(rx.os.environ, {"FIRECRAWL_API_KEY": "fc-x"}):
            with patch("hermes_cli.config.load_config_readonly",
                       return_value={"file_tools": {"hosted_ocr": False}}):
                self.assertFalse(rx.hosted_ocr_available())
        # config true WITHOUT key → True (keyless opt-in)
        with patch.dict(rx.os.environ, {}, clear=False):
            rx.os.environ.pop("FIRECRAWL_API_KEY", None)
            with patch("hermes_cli.config.load_config_readonly",
                       return_value={"file_tools": {"hosted_ocr": True}}):
                self.assertTrue(rx.hosted_ocr_available())
        # nothing → False (default stays key-gated)
        with patch("hermes_cli.config.load_config_readonly",
                   return_value={}):
            rx.os.environ.pop("FIRECRAWL_API_KEY", None)
            self.assertFalse(rx.hosted_ocr_available())

    def test_hosted_ocr_config_values_survive_hand_edited_yaml(self):
        """Quoted strings from hand-edited YAML must not silently read as
        auto: a quoted 'false' failing to disable an outbound call is the
        expensive direction, and a quoted 'true' failing to opt in is a
        silent no-op."""
        import tools.read_extract as rx

        # quoted 'false' with a key disables (strict is-checks missed it)
        with patch.dict(rx.os.environ, {"FIRECRAWL_API_KEY": "fc-x"}):
            with patch("hermes_cli.config.load_config_readonly",
                       return_value={"file_tools": {"hosted_ocr": "false"}}):
                self.assertFalse(rx.hosted_ocr_available())
        # quoted 'true' without a key opts in (keyless)
        rx.os.environ.pop("FIRECRAWL_API_KEY", None)
        for raw in ("true", "yes", "1"):
            with patch("hermes_cli.config.load_config_readonly",
                       return_value={"file_tools": {"hosted_ocr": raw}}):
                self.assertTrue(
                    rx.hosted_ocr_available(), raw,
                )
        # anything unrecognized stays auto: no key means key-gated off
        for raw in ("banana", 1, ["true"]):
            with patch("hermes_cli.config.load_config_readonly",
                       return_value={"file_tools": {"hosted_ocr": raw}}):
                self.assertFalse(
                    rx.hosted_ocr_available(), raw,
                )

    def test_runtime_route_is_direct_key_only(self):
        """_hosted_ocr_config never resolves the Nous gateway: api_url is
        always None (anydoc defaults to api.firecrawl.dev) and enabled
        tracks the key by default."""
        import tools.read_extract as rx

        with patch.dict(rx.os.environ, {"FIRECRAWL_API_KEY": "fc-x"}):
            with patch("hermes_cli.config.load_config_readonly",
                       return_value={}):
                enabled, key, url = rx._hosted_ocr_config()
        self.assertTrue(enabled)
        self.assertEqual(key, "fc-x")
        self.assertIsNone(url)
        with patch("hermes_cli.config.load_config_readonly",
                   return_value={}):
            rx.os.environ.pop("FIRECRAWL_API_KEY", None)
            enabled, key, url = rx._hosted_ocr_config()
        self.assertFalse(enabled)
        self.assertIsNone(key)
        self.assertIsNone(url)

    def test_runtime_keyless_route_with_opt_in(self):
        """file_tools.hosted_ocr: true with no key enables the keyless
        route: the hosted attempt goes out with ocr='hosted' and no
        api_key, so anydoc falls back to its keyless free tier."""
        import tools.read_extract as rx

        rx.os.environ.pop("FIRECRAWL_API_KEY", None)
        with patch("hermes_cli.config.load_config_readonly",
                   return_value={"file_tools": {"hosted_ocr": True}}):
            enabled, key, url = rx._hosted_ocr_config()
        self.assertTrue(enabled)
        self.assertIsNone(key)
        self.assertIsNone(url)
        # And the extraction attempt omits api_key for keyless:
        mod, calls = self._fake_mod_for_gate(hosted_result="KEYLESS TEXT")
        with patch.object(rx, "_anydoc", return_value=mod),              patch.object(rx, "_hosted_ocr_config",
                          return_value=(True, None, None)),              patch.object(rx.os.path, "getsize", return_value=10):
            out = rx._extract_anydoc("scan.pdf")
        self.assertEqual(out, "KEYLESS TEXT\n")
        self.assertEqual(calls[1], {"ocr": "hosted"})

    @staticmethod
    def _fake_mod_for_gate(hosted_result=None):
        """Shared fake anydoc module for gate tests (raises NeedsOcrError
        on the plain call, returns hosted_result on the hosted call)."""
        class NeedsOcrError(Exception):
            def __init__(self, pages):
                super().__init__("needs ocr")
                self.pages = pages

        calls = []

        class Mod:
            pass

        mod = Mod()
        mod.NeedsOcrError = NeedsOcrError

        def to_markdown(path, **kw):
            calls.append(kw)
            if not kw:
                raise NeedsOcrError([2, 3])
            return hosted_result

        mod.to_markdown = to_markdown
        return mod, calls

    def test_coverage_warning_teaching_left_to_the_warning(self):
        """The response-time warning owns the recovery curriculum."""
        from tools.file_tools import READ_FILE_SCHEMA

        desc = READ_FILE_SCHEMA["description"]
        self.assertNotIn("EXTRACTION COVERAGE WARNING", desc)
        self.assertNotIn("NEEDS OCR", desc)
        self.assertNotIn("pdftoppm", desc)
        import inspect
        from tools import read_extract

        src = inspect.getsource(read_extract)
        self.assertIn("NEEDS OCR", src)
        self.assertIn("pdftoppm", src)
        self.assertIn("vision_analyze", src)

    def test_binary_note_stays_last(self):
        from tools.file_tools import READ_FILE_SCHEMA

        desc = READ_FILE_SCHEMA["description"]
        self.assertLess(desc.find("EPUB"), desc.find("Cannot read images/binary"))

    def test_missing_anydoc_error_teaches_install(self):
        from tools.read_extract import _anydoc_missing_error

        err = _anydoc_missing_error("x.epub")
        self.assertIn("firecrawl-anydoc", err)
        self.assertNotEqual(err, "Unsupported document type: 'x.epub'")


class TestNeedsOcrPath(unittest.TestCase):
    """anydoc>=0.2 NeedsOcrError wiring: hosted OCR attempt + typed warning
    (maintainer caveats: #1 nous-gateway Parse was live-probed HTTP 500 →
    attempt-and-fall-through; #2 warning recommends LOCAL OCR skills)."""

    def _fake_mod(self, hosted_result=None, hosted_exc=None):
        class NeedsOcrError(Exception):
            def __init__(self, pages):
                super().__init__("needs ocr")
                self.pages = pages

        calls = []

        class Mod:
            pass

        mod = Mod()
        mod.NeedsOcrError = NeedsOcrError

        def to_markdown(path, **kw):
            calls.append(kw)
            if not kw:
                raise NeedsOcrError([2, 3])
            if hosted_exc is not None:
                raise hosted_exc
            return hosted_result

        mod.to_markdown = to_markdown
        return mod, calls

    def test_hosted_success_returns_ocr_text(self):
        from tools import read_extract as rx

        mod, calls = self._fake_mod(hosted_result="OCR TEXT")
        with patch.object(rx, "_anydoc", return_value=mod),              patch.object(rx, "_hosted_ocr_config",
                          return_value=(True, "key", None)),              patch.object(rx.os.path, "getsize", return_value=10):
            out = rx._extract_anydoc("scan.pdf")
        self.assertEqual(out, "OCR TEXT\n")
        self.assertEqual(calls[1].get("ocr"), "hosted")

    def test_hosted_failure_warns_and_prefers_local_skills(self):
        from tools import read_extract as rx

        mod, _ = self._fake_mod(hosted_exc=RuntimeError("HTTP 500"))
        with patch.object(rx, "_anydoc", return_value=mod),              patch.object(rx, "_hosted_ocr_config",
                          return_value=(True, "key", "https://gw")),              patch.object(rx.os.path, "getsize", return_value=10):
            out = rx._extract_anydoc("scan.pdf")
        self.assertIn("[NEEDS OCR", out)
        self.assertIn("pages 2, 3", out)
        self.assertIn("attempted and failed", out)
        # Maintainer-directed: HINT at checking for an OCR skill; never
        # name one (none is guaranteed to exist), never sell config knobs.
        self.assertIn("check whether an OCR skill is available", out)
        self.assertIn("skills_list", out)
        self.assertNotIn("ocr-and-documents", out)
        self.assertNotIn("marker-pdf", out)
        self.assertNotIn("hosted_ocr", out)

    def test_disabled_warns_without_attempt(self):
        from tools import read_extract as rx

        mod, calls = self._fake_mod()
        with patch.object(rx, "_anydoc", return_value=mod),              patch.object(rx, "_hosted_ocr_config",
                          return_value=(False, None, None)),              patch.object(rx.os.path, "getsize", return_value=10):
            out = rx._extract_anydoc("scan.pdf")
        self.assertIn("[NEEDS OCR", out)
        self.assertEqual(len(calls), 1)  # no hosted attempt
        # Same shape when disabled: skill hint, no knob advertising.
        self.assertIn("check whether an OCR skill is available", out)
        self.assertNotIn("hosted_ocr", out)
        self.assertNotIn("ocr-and-documents", out)

    def test_pin_lockstep(self):
        """pyproject core pin and lazy_deps self-heal pin must match."""
        import re
        from pathlib import Path

        py = Path("pyproject.toml").read_text(encoding="utf-8")
        lz = Path("tools/lazy_deps.py").read_text(encoding="utf-8")
        m1 = re.search(r'"firecrawl-anydoc==([\d.]+)"', py)
        m2 = re.search(r'"firecrawl-anydoc==([\d.]+)"', lz)
        self.assertIsNotNone(m1)
        self.assertIsNotNone(m2)
        self.assertEqual(m1.group(1), m2.group(1))


if __name__ == "__main__":
    unittest.main()
