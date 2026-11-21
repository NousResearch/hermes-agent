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
from unittest.mock import Mock, patch

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

    def test_transferred_scanned_pdf_returns_actionable_warning_without_hosted_ocr(self):
        from tools import read_extract as rx
        mod, calls = self._fake_mod()
        seen = []
        def convert(data, **kw):
            seen.append(data)
            return mod.to_markdown("unused", **kw)
        mod.to_markdown_bytes = convert
        with patch.object(rx, "_anydoc", return_value=mod), patch.object(
                rx, "_hosted_ocr_config", return_value=(False, None, None)):
            out = rx._extract_anydoc_bytes(b"%PDF fictional", "/workspace/scan with spaces.pdf")
        self.assertIn("[NEEDS OCR", out)
        self.assertIn("content is MISSING", out)
        self.assertIn("pages 2, 3", out)
        self.assertIn("'/workspace/scan with spaces.pdf'", out)
        self.assertEqual(seen, [b"%PDF fictional"])
        self.assertEqual(len(calls), 1)

    def test_transferred_scanned_pdf_uses_existing_configured_hosted_byte_route(self):
        from tools import read_extract as rx
        mod, calls = self._fake_mod(hosted_result="OCR TEXT")
        seen = []
        def convert(data, **kw):
            seen.append(data)
            return mod.to_markdown("unused", **kw)
        mod.to_markdown_bytes = convert
        with patch.object(rx, "_anydoc", return_value=mod), patch.object(
                rx, "_hosted_ocr_config", return_value=(True, "fictional", None)):
            out = rx._extract_anydoc_bytes(b"%PDF fictional", "/remote/scan.pdf")
        self.assertEqual(out, "OCR TEXT\n")
        self.assertEqual(seen, [b"%PDF fictional", b"%PDF fictional"])
        self.assertEqual(calls[1]["ocr"], "hosted")

    def test_transferred_scanned_pdf_failed_hosted_ocr_still_warns(self):
        from tools import read_extract as rx
        mod, _ = self._fake_mod(hosted_exc=RuntimeError("HTTP 500"))
        mod.to_markdown_bytes = lambda data, **kw: mod.to_markdown("unused", **kw)
        with patch.object(rx, "_anydoc", return_value=mod), patch.object(
                rx, "_hosted_ocr_config", return_value=(True, "fictional", None)):
            out = rx._extract_anydoc_bytes(b"%PDF fictional", "/remote/scan.pdf")
        self.assertIn("[NEEDS OCR", out)
        self.assertIn("Hosted OCR was attempted and failed", out)
        self.assertIn("/remote/scan.pdf", out)

    def test_read_file_scanned_bytes_are_recoverable_guidance_not_tool_error(self):
        import base64
        import json
        from pathlib import PurePosixPath
        from tools import read_extract as rx, file_tools
        from tools.file_operations import ReadResult
        from agent.display import _detect_tool_failure
        mod, _ = self._fake_mod()
        mod.to_markdown_bytes = lambda data, **kw: mod.to_markdown("unused", **kw)
        ops = Mock()
        ops.read_file_bytes.return_value = ReadResult(
            base64_content=base64.b64encode(b"%PDF fictional").decode(), file_size=14, is_binary=True)
        ops._add_line_numbers.side_effect = lambda text, start_line=1: text
        with patch.object(rx, "_anydoc", return_value=mod), patch.object(
                rx, "_hosted_ocr_config", return_value=(False, None, None)), patch.object(
                file_tools, "_get_file_ops", return_value=ops), patch.object(
                file_tools, "_resolve_path_for_task", return_value=PurePosixPath("/remote/scan.pdf")):
            result = file_tools.read_file_tool("/remote/scan.pdf", task_id="fictional")
        receipt = json.loads(result)
        self.assertNotIn("error", receipt)
        self.assertIn("[NEEDS OCR", receipt["content"])
        self.assertIn("MISSING", receipt["content"])
        self.assertFalse(_detect_tool_failure("read_file", result)[0])

    def test_empty_hosted_text_keeps_missing_content_warning_on_both_routes(self):
        from tools import read_extract as rx
        for raw in ("", " \t\n"):
            for byte_route in (False, True):
                with self.subTest(raw=repr(raw), byte_route=byte_route):
                    mod, _ = self._fake_mod(hosted_result=raw)
                    mod.to_markdown_bytes = lambda data, **kw: mod.to_markdown("unused", **kw)
                    with patch.object(rx, "_anydoc", return_value=mod), patch.object(
                            rx, "_hosted_ocr_config", return_value=(True, "fictional", None)), patch.object(
                            rx.os.path, "getsize", return_value=14):
                        out = (rx._extract_anydoc_bytes(b"%PDF fictional", "/remote/scan.pdf")
                               if byte_route else rx._extract_anydoc("/remote/scan.pdf"))
                    self.assertIn("[NEEDS OCR", out)
                    self.assertIn("MISSING", out)

    def test_hosted_exception_does_not_copy_response_secrets_into_tool_context(self):
        from tools import read_extract as rx
        for byte_route in (False, True):
            with self.subTest(byte_route=byte_route):
                mod, _ = self._fake_mod(hosted_exc=RuntimeError("request included fictional-private-secret"))
                mod.to_markdown_bytes = lambda data, **kw: mod.to_markdown("unused", **kw)
                with patch.object(rx, "_anydoc", return_value=mod), patch.object(
                        rx, "_hosted_ocr_config", return_value=(True, "fictional", None)), patch.object(
                        rx.os.path, "getsize", return_value=14):
                    out = (rx._extract_anydoc_bytes(b"%PDF fictional", "/remote/scan.pdf")
                           if byte_route else rx._extract_anydoc("/remote/scan.pdf"))
                self.assertIn("RuntimeError", out)
                self.assertIn("MISSING", out)
                self.assertNotIn("fictional-private-secret", out)

    def test_real_sdk_does_not_inherit_other_profile_credentials_or_endpoint(self):
        import io
        import itertools
        import json
        import tempfile
        from pathlib import Path

        import anydoc

        from tools import read_extract as rx

        def scanned(*args, **kwargs):
            raise anydoc.NeedsOcrError("fictional scanned page")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fictional.pdf"
            data = b"%PDF fictional fixture"
            path.write_bytes(data)
            cases = itertools.product((False, True), (None, "profile-fixture-key"),
                                      (None, "https://profile-explicit.invalid"))
            for byte_route, key, url in cases:
                with self.subTest(byte_route=byte_route, key=key, url=url):
                    captured = []

                    def transport(request, **kwargs):
                        captured.append(request)
                        payload = {"success": True, "data": {"markdown": "OCR FIXTURE"}}
                        response = io.BytesIO(json.dumps(payload).encode())
                        response.status = 200
                        return response

                    with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "other-profile-fixture-key",
                                                "FIRECRAWL_API_URL": "https://other-profile.invalid"}), \
                         patch.object(rx, "_anydoc", return_value=anydoc), \
                         patch.object(rx, "_hosted_ocr_config", return_value=(True, key, url)), \
                         patch.object(anydoc, "_to_markdown", side_effect=scanned), \
                         patch.object(anydoc, "_to_markdown_bytes", side_effect=scanned), \
                         patch.object(anydoc.urllib.request, "urlopen", side_effect=transport):
                        result = (rx.extract_document_bytes(data, str(path)) if byte_route
                                  else rx.extract_document_text(str(path)))
                    self.assertEqual(result, "OCR FIXTURE\n")
                    self.assertEqual(len(captured), 1)
                    self.assertEqual(captured[0].get_header("Authorization"),
                                     f"Bearer {key}" if key else None)
                    self.assertEqual(captured[0].full_url,
                                     (url or "https://api.firecrawl.dev") + "/v2/parse")

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
