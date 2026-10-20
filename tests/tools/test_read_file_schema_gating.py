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



    def test_hosted_ocr_available_gate_states(self):
        """Maintainer decision: ONLY a direct FIRECRAWL_API_KEY unlocks —
        not config true, not the Nous gateway."""
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
        # config true WITHOUT key → False (key is the one gate)
        with patch.dict(rx.os.environ, {}, clear=False):
            rx.os.environ.pop("FIRECRAWL_API_KEY", None)
            with patch("hermes_cli.config.load_config_readonly",
                       return_value={"file_tools": {"hosted_ocr": True}}):
                self.assertFalse(rx.hosted_ocr_available())
        # nothing → False (Nous gateway alone must NOT unlock)
        with patch("hermes_cli.config.load_config_readonly",
                   return_value={}):
            rx.os.environ.pop("FIRECRAWL_API_KEY", None)
            self.assertFalse(rx.hosted_ocr_available())

    def test_runtime_route_is_direct_key_only(self):
        """_hosted_ocr_config never resolves the Nous gateway: api_url is
        always None (anydoc defaults to api.firecrawl.dev) and enabled
        tracks the key."""
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

    def test_disabled_warns_without_attempt(self):
        from tools import read_extract as rx

        mod, calls = self._fake_mod()
        with patch.object(rx, "_anydoc", return_value=mod),              patch.object(rx, "_hosted_ocr_config",
                          return_value=(False, None, None)),              patch.object(rx.os.path, "getsize", return_value=10):
            out = rx._extract_anydoc("scan.pdf")
        self.assertIn("[NEEDS OCR", out)
        self.assertEqual(len(calls), 1)  # no hosted attempt

    def test_pin_lockstep(self):
        """The doc-extract extra PM uses to self-heal must pin core's anydoc."""
        import tomllib
        from pathlib import Path

        from packaging.requirements import Requirement

        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]

        def anydoc_pins(reqs):
            return {str(r.specifier) for r in map(Requirement, reqs) if r.name == "firecrawl-anydoc"}

        core = anydoc_pins(project["dependencies"])
        extra = anydoc_pins(project["optional-dependencies"]["doc-extract"])
        self.assertEqual(len(core), 1)
        self.assertEqual(extra, core)

if __name__ == "__main__":
    unittest.main()
