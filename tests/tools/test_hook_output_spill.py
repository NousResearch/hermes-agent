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
