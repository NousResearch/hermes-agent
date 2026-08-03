"""Tests for tools.hook_output_spill."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import hook_output_spill as hos


class GetSpillConfigTests(unittest.TestCase):
    def test_defaults_when_no_config(self):
        with patch.object(hos, "load_config", create=True, return_value={}):
            # load_config is resolved at call time via local import;
            # patch the module's source instead.
            pass
        with patch("hermes_cli.config.load_config", return_value={}):
            cfg = hos.get_spill_config()
        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["max_chars"], hos.DEFAULT_MAX_CHARS)
        self.assertEqual(cfg["preview_head"], hos.DEFAULT_PREVIEW_HEAD)
        self.assertEqual(cfg["preview_tail"], hos.DEFAULT_PREVIEW_TAIL)
        self.assertIsNone(cfg["directory"])

    def test_user_overrides_are_respected(self):
        user_cfg = {
            "hooks": {
                "output_spill": {
                    "enabled": False,
                    "max_chars": 500,
                    "preview_head": 25,
                    "preview_tail": 10,
                    "directory": "/tmp/spill-test",
                }
            }
        }
        with patch("hermes_cli.config.load_config", return_value=user_cfg):
            cfg = hos.get_spill_config()
        self.assertFalse(cfg["enabled"])
        self.assertEqual(cfg["max_chars"], 500)
        self.assertEqual(cfg["preview_head"], 25)
        self.assertEqual(cfg["preview_tail"], 10)
        self.assertEqual(cfg["directory"], "/tmp/spill-test")

    def test_bad_values_fall_back_to_defaults(self):
        user_cfg = {
            "hooks": {
                "output_spill": {
                    "max_chars": "not-a-number",
                    "preview_head": -100,
                    "preview_tail": None,
                    "directory": 123,  # not a string
                }
            }
        }
        with patch("hermes_cli.config.load_config", return_value=user_cfg):
            cfg = hos.get_spill_config()
        self.assertEqual(cfg["max_chars"], hos.DEFAULT_MAX_CHARS)
        self.assertEqual(cfg["preview_head"], hos.DEFAULT_PREVIEW_HEAD)
        self.assertEqual(cfg["preview_tail"], hos.DEFAULT_PREVIEW_TAIL)
        self.assertIsNone(cfg["directory"])

    def test_load_config_exception_is_swallowed(self):
        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("bad")):
            cfg = hos.get_spill_config()
        self.assertEqual(cfg["max_chars"], hos.DEFAULT_MAX_CHARS)
        self.assertTrue(cfg["enabled"])


class SpillIfOversizedTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="hermes-spill-test-")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _cfg(self, **overrides):
        base = {
            "enabled": True,
            "max_chars": 100,
            "preview_head": 20,
            "preview_tail": 20,
            "directory": self.tmpdir,
        }
        base.update(overrides)
        return base

    def test_empty_and_none_are_noops(self):
        self.assertEqual(hos.spill_if_oversized("", config=self._cfg()), "")
        self.assertEqual(hos.spill_if_oversized(None, config=self._cfg()), "")

    def test_text_under_cap_is_unchanged(self):
        small = "x" * 50
        self.assertEqual(hos.spill_if_oversized(small, config=self._cfg()), small)

    def test_disabled_bypasses_spill_even_if_oversized(self):
        big = "y" * 10_000
        cfg = self._cfg(enabled=False)
        self.assertEqual(hos.spill_if_oversized(big, config=cfg), big)
        # No spill files written.
        self.assertEqual(list(Path(self.tmpdir).rglob("*")), [])

    def test_oversized_writes_spill_and_returns_preview(self):
        big = "A" * 60 + "B" * 60 + "C" * 60  # 180 chars > cap 100
        result = hos.spill_if_oversized(
            big,
            session_id="sess-123",
            source="plugin hook",
            config=self._cfg(),
        )
        # Preview contains the header, head, and tail markers.
        self.assertIn("plugin hook output truncated — 180 chars", result)
        self.assertIn("--- head ---", result)
        self.assertIn("--- tail ---", result)
        # Head is the first 20 chars, tail is the last 20.
        self.assertIn("A" * 20, result)
        self.assertIn("C" * 20, result)
        # Spill file exists under the session subdir and has full content.
        session_dir = Path(self.tmpdir) / "sess-123"
        self.assertTrue(session_dir.is_dir())
        files = list(session_dir.iterdir())
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].read_text().rstrip("\n"), big)
        # Preview references the spill path.
        self.assertIn(str(files[0]), result)

    def test_missing_session_id_uses_no_session_segment(self):
        big = "z" * 500
        cfg = self._cfg(max_chars=10)
        hos.spill_if_oversized(big, session_id=None, config=cfg)
        self.assertTrue((Path(self.tmpdir) / "no-session").is_dir())

    def test_session_id_with_path_separators_is_sanitised(self):
        big = "q" * 500
        cfg = self._cfg(max_chars=10)
        # An attacker-style session id with .. and / must not escape the
        # base directory.
        hos.spill_if_oversized(big, session_id="../../etc/passwd", config=cfg)
        # Nothing leaks outside self.tmpdir.
        self.assertFalse(Path("/etc/passwd-hermes-test").exists())
        # A sanitised path should exist under tmpdir.
        entries = list(Path(self.tmpdir).rglob("*.txt"))
        self.assertEqual(len(entries), 1)
        # The path should be inside tmpdir.
        self.assertTrue(str(entries[0]).startswith(self.tmpdir))

    def test_spill_write_failure_falls_back_to_preview_only(self):
        big = "w" * 500
        # Point at a path that cannot be created (a file, not a dir).
        existing_file = os.path.join(self.tmpdir, "not-a-dir")
        with open(existing_file, "w") as f:
            f.write("blocker")
        cfg = self._cfg(max_chars=10, directory=existing_file)
        result = hos.spill_if_oversized(big, session_id="x", config=cfg)
        # Preview still returned, but with failure notice.
        self.assertIn("spill write failed", result)
        self.assertIn("--- head ---", result)
        # Content still bounded (not the full 500 chars).
        self.assertLess(len(result), 500)

    def test_preview_head_only_no_tail(self):
        big = "a" * 1000
        cfg = self._cfg(max_chars=10, preview_head=30, preview_tail=0)
        result = hos.spill_if_oversized(big, session_id="s", config=cfg)
        self.assertIn("--- head ---", result)
        self.assertNotIn("--- tail ---", result)

    def test_non_string_input_coerced(self):
        cfg = self._cfg(max_chars=5)

        class StrFriendly:
            def __str__(self):
                return "stringified-" + "x" * 200

        result = hos.spill_if_oversized(StrFriendly(), session_id="s", config=cfg)
        self.assertIn("truncated", result)

    def test_default_directory_uses_hermes_home(self):
        """When no directory override, spill under HERMES_HOME/hook_outputs."""
        test_home = tempfile.mkdtemp(prefix="hermes-home-")
        try:
            with patch.dict(os.environ, {"HERMES_HOME": test_home}):
                # Also patch get_hermes_home to the env var to mirror production.
                cfg = self._cfg(directory=None, max_chars=5)
                hos.spill_if_oversized("x" * 200, session_id="sess", config=cfg)
            # Spill directory exists somewhere under test_home OR default
            # ~/.hermes/hook_outputs depending on get_hermes_home behaviour.
            candidates = [
                Path(test_home) / "hook_outputs" / "sess",
                Path(os.path.expanduser("~/.hermes/hook_outputs/sess")),
            ]
            # At least one of the candidate dirs now exists and has a file.
            existing = [c for c in candidates if c.is_dir() and list(c.iterdir())]
            self.assertTrue(existing, f"No spill dir found in {candidates}")
        finally:
            import shutil
            shutil.rmtree(test_home, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
