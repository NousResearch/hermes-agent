#!/usr/bin/env python3
"""Tests for the Devin CLI delegate backend (tools/devin_delegate.py).

Covers the check_fn gate, config resolution, prompt building, and the
subprocess handler's success/error/timeout paths — all with the `devin`
binary and its auth probe mocked so no real Devin invocation happens.

Run with:  python -m pytest tests/tools/test_devin_delegate.py -v
   or:     python tests/tools/test_devin_delegate.py
"""

import json
import os
import subprocess
import unittest
from unittest.mock import MagicMock, patch

import tools.devin_delegate as mod
from tools.devin_delegate import (
    DELEGATE_TO_DEVIN_SCHEMA,
    _build_prompt,
    _resolve_max_result_chars,
    _resolve_permission_mode,
    _resolve_timeout,
    check_devin_requirements,
    delegate_to_devin,
)


def _mock_popen(stdout="", stderr="", returncode=0):
    """A mock Popen for the handler's Popen+communicate pattern."""
    proc = MagicMock()
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = returncode
    proc.pid = 12345
    proc.communicate = MagicMock(return_value=(stdout, stderr))
    return proc


class TestCheckRequirements(unittest.TestCase):
    """check_devin_requirements is the schema gate — must be cheap and exact."""

    def test_disabled_when_not_enabled(self):
        with patch.object(mod, "_devin_config", lambda: {}), \
             patch.object(mod, "_devin_binary", lambda: "/usr/bin/devin"):
            self.assertFalse(check_devin_requirements())

    def test_disabled_when_enabled_but_no_binary(self):
        with patch.object(mod, "_devin_config", lambda: {"enabled": True}), \
             patch.object(mod, "_devin_binary", lambda: None):
            self.assertFalse(check_devin_requirements())

    def test_enabled_when_enabled_and_binary_present(self):
        with patch.object(mod, "_devin_config", lambda: {"enabled": "true"}), \
             patch.object(mod, "_devin_binary", lambda: "/usr/bin/devin"):
            self.assertTrue(check_devin_requirements())

    def test_falsy_enabled_value_rejected(self):
        with patch.object(mod, "_devin_config", lambda: {"enabled": "false"}), \
             patch.object(mod, "_devin_binary", lambda: "/usr/bin/devin"):
            self.assertFalse(check_devin_requirements())


class TestConfigResolution(unittest.TestCase):
    def test_permission_mode_default(self):
        self.assertEqual(_resolve_permission_mode({}), "accept-edits")

    def test_permission_mode_override(self):
        self.assertEqual(_resolve_permission_mode({"permission_mode": "auto"}), "auto")

    def test_permission_mode_invalid_falls_back(self):
        self.assertEqual(_resolve_permission_mode({"permission_mode": "yolo"}), "accept-edits")

    def test_permission_mode_dangerous_requires_explicit_config(self):
        self.assertEqual(_resolve_permission_mode({"permission_mode": "dangerous"}), "dangerous")

    def test_timeout_default(self):
        self.assertEqual(_resolve_timeout({}, None), 1800.0)

    def test_timeout_floored_at_60(self):
        self.assertEqual(_resolve_timeout({}, 5), 60.0)

    def test_timeout_override(self):
        self.assertEqual(_resolve_timeout({}, 120), 120.0)

    def test_timeout_config_used_when_no_override(self):
        self.assertEqual(_resolve_timeout({"timeout_seconds": 300}, None), 300.0)

    def test_timeout_invalid_falls_back(self):
        self.assertEqual(_resolve_timeout({"timeout_seconds": "abc"}, None), 1800.0)

    def test_timeout_model_override_clamped_to_config(self):
        """Model override cannot exceed config timeout (prompt-injection guard)."""
        self.assertEqual(_resolve_timeout({"timeout_seconds": 300}, 99999), 300.0)

    def test_timeout_model_override_clamped_to_hard_cap(self):
        """With a high config timeout, model override is still capped at
        _MAX_MODEL_TIMEOUT_SECONDS (the hard safety ceiling)."""
        from tools.devin_delegate import _MAX_MODEL_TIMEOUT_SECONDS
        self.assertEqual(
            _resolve_timeout({"timeout_seconds": 999999}, 999999),
            _MAX_MODEL_TIMEOUT_SECONDS,
        )

    def test_timeout_model_override_can_shorten(self):
        """Model can shorten the timeout for a quick task."""
        self.assertEqual(_resolve_timeout({"timeout_seconds": 1800}, 120), 120.0)

    def test_timeout_rejects_nan(self):
        """NaN must fall back to default, not bypass the floor check."""
        self.assertEqual(_resolve_timeout({}, float("nan")), 1800.0)
        self.assertEqual(_resolve_timeout({"timeout_seconds": float("nan")}, None), 1800.0)

    def test_timeout_rejects_infinity(self):
        """Infinity must fall back to default, not remove the timeout cap."""
        self.assertEqual(_resolve_timeout({}, float("inf")), 1800.0)
        self.assertEqual(_resolve_timeout({"timeout_seconds": float("inf")}, None), 1800.0)

    def test_max_result_chars_default(self):
        self.assertEqual(_resolve_max_result_chars({}), 20000)

    def test_max_result_chars_floor(self):
        self.assertEqual(_resolve_max_result_chars({"max_result_chars": 10}), 20000)


class TestBuildPrompt(unittest.TestCase):
    def test_goal_only(self):
        self.assertEqual(_build_prompt("fix the bug", None), "fix the bug")
        self.assertEqual(_build_prompt("fix the bug", ""), "fix the bug")

    def test_goal_with_context(self):
        out = _build_prompt("fix the bug", "see src/foo.py line 42")
        self.assertIn("fix the bug", out)
        self.assertIn("--- Context ---", out)
        self.assertIn("see src/foo.py line 42", out)


class TestDelegateToDevin(unittest.TestCase):
    """Handler paths with subprocess + auth fully mocked."""

    def _patch_env(self, cfg=None, binary="/usr/bin/devin", logged_in=True,
                   auth_detail=""):
        cfg = cfg if cfg is not None else {"enabled": True}
        return (
            patch.object(mod, "_devin_config", lambda: cfg),
            patch.object(mod, "_devin_binary", lambda: binary),
            patch.object(mod, "_is_logged_in", lambda: (logged_in, auth_detail)),
        )

    def test_missing_goal_returns_error(self):
        with patch.object(mod, "_devin_config", lambda: {"enabled": True}):
            result = json.loads(delegate_to_devin(goal="   "))
        self.assertIn("error", result)
        self.assertIn("goal", result["error"])

    def test_no_binary_returns_error(self):
        with patch.object(mod, "_devin_config", lambda: {"enabled": True}), \
             patch.object(mod, "_devin_binary", lambda: None):
            result = json.loads(delegate_to_devin(goal="do thing"))
        self.assertIn("error", result)
        self.assertIn("not on $PATH", result["error"])

    def test_handler_rejects_when_disabled_at_runtime(self):
        """Handler must re-check enabled gate — registry caches check_fn,
        so a runtime config change could leave the tool callable."""
        with patch.object(mod, "_devin_config", lambda: {"enabled": False}):
            result = json.loads(delegate_to_devin(goal="do thing"))
        self.assertIn("error", result)
        self.assertIn("disabled", result["error"])

    def test_not_logged_in_returns_error(self):
        p1, p2, p3 = self._patch_env(logged_in=False,
                                     auth_detail="Devin is not authenticated (logged out).")
        with p1, p2, p3:
            result = json.loads(delegate_to_devin(goal="do thing"))
        self.assertIn("error", result)
        self.assertIn("not authenticated", result["error"])

    def test_success_returns_completed_result(self):
        p1, p2, p3 = self._patch_env()
        with p1, p2, p3, \
             patch.object(mod.subprocess, "Popen",
                          return_value=_mock_popen(stdout="All done. Fixed the bug.")):
            result = json.loads(delegate_to_devin(goal="fix the bug", context="see foo.py"))
        self.assertEqual(result["results"][0]["status"], "completed")
        self.assertEqual(result["results"][0]["summary"], "All done. Fixed the bug.")
        self.assertEqual(result["results"][0]["exit_reason"], "completed")
        self.assertEqual(result["results"][0]["backend"], "devin")
        self.assertFalse(result["results"][0]["truncated"])
        self.assertIn("duration_seconds", result["results"][0])

    def test_nonzero_exit_returns_error_result(self):
        p1, p2, p3 = self._patch_env()
        with p1, p2, p3, \
             patch.object(mod.subprocess, "Popen",
                          return_value=_mock_popen(stdout="", stderr="boom",
                                                   returncode=2)):
            result = json.loads(delegate_to_devin(goal="do thing"))
        entry = result["results"][0]
        self.assertEqual(entry["status"], "error")
        self.assertEqual(entry["exit_reason"], "error")
        self.assertIn("code 2", entry["error"])
        self.assertIn("boom", entry["error"])

    def test_error_truncation_flag_when_stderr_exceeds_limit(self):
        """Error path must report truncated=True when error text is clipped."""
        p1, p2, p3 = self._patch_env(cfg={"enabled": True, "max_result_chars": 256})
        long_err = "e" * 1000
        with p1, p2, p3, \
             patch.object(mod.subprocess, "Popen",
                          return_value=_mock_popen(stdout="", stderr=long_err,
                                                   returncode=1)):
            result = json.loads(delegate_to_devin(goal="do thing"))
        entry = result["results"][0]
        self.assertEqual(entry["status"], "error")
        self.assertTrue(entry["truncated"])

    def test_timeout_returns_timeout_result(self):
        p1, p2, p3 = self._patch_env(cfg={"enabled": True, "timeout_seconds": 60})
        proc = _mock_popen()
        proc.communicate.side_effect = subprocess.TimeoutExpired(cmd=["devin"], timeout=60)
        with p1, p2, p3, \
             patch.object(mod.subprocess, "Popen", return_value=proc), \
             patch.object(mod, "_kill_process_group", lambda p: None):
            result = json.loads(delegate_to_devin(goal="do thing"))
        entry = result["results"][0]
        self.assertEqual(entry["status"], "timeout")
        self.assertEqual(entry["exit_reason"], "timeout")
        self.assertIn("60s", entry["error"])
        # No model override → hint points to config
        self.assertIn("timeout_seconds", entry["error"])

    def test_timeout_with_model_override_shows_override_hint(self):
        """When the model supplied a timeout override that actually shortens
        the effective timeout, the error hint should mention the override."""
        p1, p2, p3 = self._patch_env(cfg={"enabled": True, "timeout_seconds": 1800})
        proc = _mock_popen()
        proc.communicate.side_effect = subprocess.TimeoutExpired(cmd=["devin"], timeout=120)
        with p1, p2, p3, \
             patch.object(mod.subprocess, "Popen", return_value=proc), \
             patch.object(mod, "_kill_process_group", lambda p: None):
            result = json.loads(delegate_to_devin(goal="do thing", timeout=120))
        entry = result["results"][0]
        self.assertEqual(entry["status"], "timeout")
        self.assertIn("override", entry["error"].lower())

    def test_timeout_with_override_at_ceiling_shows_config_hint(self):
        """When the model override is at/above the config ceiling, the resolved
        timeout equals the config default — the hint should point to config,
        not mention the override."""
        p1, p2, p3 = self._patch_env(cfg={"enabled": True, "timeout_seconds": 300})
        proc = _mock_popen()
        proc.communicate.side_effect = subprocess.TimeoutExpired(cmd=["devin"], timeout=300)
        with p1, p2, p3, \
             patch.object(mod.subprocess, "Popen", return_value=proc), \
             patch.object(mod, "_kill_process_group", lambda p: None):
            # Override at ceiling → resolved = 300 = config default
            result = json.loads(delegate_to_devin(goal="do thing", timeout=99999))
        entry = result["results"][0]
        self.assertEqual(entry["status"], "timeout")
        # Should show config hint, not override hint
        self.assertIn("timeout_seconds", entry["error"])
        self.assertNotIn("override", entry["error"].lower())

    def test_truncation_flag_when_stdout_exceeds_limit(self):
        p1, p2, p3 = self._patch_env(cfg={"enabled": True, "max_result_chars": 256})
        long_out = "x" * 1000
        with p1, p2, p3, \
             patch.object(mod.subprocess, "Popen",
                          return_value=_mock_popen(stdout=long_out)):
            result = json.loads(delegate_to_devin(goal="do thing"))
        entry = result["results"][0]
        self.assertEqual(entry["status"], "completed")
        self.assertTrue(entry["truncated"])
        self.assertEqual(len(entry["summary"]), 256)

    def test_model_override_passed_to_argv(self):
        p1, p2, p3 = self._patch_env()
        captured = {}

        def fake_popen(argv, **kw):
            captured["argv"] = argv
            return _mock_popen(stdout="ok")

        with p1, p2, p3, patch.object(mod.subprocess, "Popen", side_effect=fake_popen):
            delegate_to_devin(goal="do thing", model="opus")
        argv = captured["argv"]
        self.assertIn("--model", argv)
        self.assertIn("opus", argv)
        # Print mode + unattended defaults always present.
        self.assertIn("-p", argv)
        self.assertIn("--permission-mode", argv)
        self.assertIn("accept-edits", argv)  # new safe default
        self.assertIn("--respect-workspace-trust", argv)
        self.assertIn("false", argv)
        # Prompt after the -- separator.
        self.assertEqual(argv[argv.index("--") + 1:], ["do thing"])

    def test_permission_mode_from_config_not_model_controllable(self):
        p1, p2, p3 = self._patch_env(cfg={"enabled": True, "permission_mode": "dangerous"})
        captured = {}

        def fake_popen(argv, **kw):
            captured["argv"] = argv
            return _mock_popen(stdout="ok")

        with p1, p2, p3, patch.object(mod.subprocess, "Popen", side_effect=fake_popen):
            delegate_to_devin(goal="do thing")
        argv = captured["argv"]
        idx = argv.index("--permission-mode")
        self.assertEqual(argv[idx + 1], "dangerous")

    def test_uses_terminal_cwd_when_set(self):
        """Devin should launch in TERMINAL_CWD, not os.getcwd()."""
        p1, p2, p3 = self._patch_env()
        captured = {}

        def fake_popen(argv, **kw):
            captured["cwd"] = kw.get("cwd")
            captured["stdin"] = kw.get("stdin")
            return _mock_popen(stdout="ok")

        with p1, p2, p3, \
             patch.dict(os.environ, {"TERMINAL_CWD": "/tmp"}), \
             patch.object(mod.subprocess, "Popen", side_effect=fake_popen):
            delegate_to_devin(goal="do thing")
        self.assertTrue(captured["cwd"].endswith("/tmp") or captured["cwd"] == "/tmp")
        # stdin must be DEVNULL to prevent interactive prompts from blocking
        self.assertEqual(captured["stdin"], subprocess.DEVNULL)

    def test_falls_back_to_cwd_when_terminal_cwd_invalid(self):
        """If TERMINAL_CWD points to a non-existent dir, fall back to os.getcwd()."""
        p1, p2, p3 = self._patch_env()
        captured = {}

        def fake_popen(argv, **kw):
            captured["cwd"] = kw.get("cwd")
            return _mock_popen(stdout="ok")

        with p1, p2, p3, \
             patch.dict(os.environ, {"TERMINAL_CWD": "/nonexistent/path/xyz"}), \
             patch.object(mod.subprocess, "Popen", side_effect=fake_popen):
            delegate_to_devin(goal="do thing")
        self.assertEqual(captured["cwd"], os.getcwd())


class TestSchema(unittest.TestCase):
    def test_name_and_required(self):
        self.assertEqual(DELEGATE_TO_DEVIN_SCHEMA["name"], "delegate_to_devin")
        self.assertEqual(DELEGATE_TO_DEVIN_SCHEMA["parameters"]["required"], ["goal"])

    def test_has_goal_context_model_timeout(self):
        props = DELEGATE_TO_DEVIN_SCHEMA["parameters"]["properties"]
        for key in ("goal", "context", "model", "timeout"):
            self.assertIn(key, props)

    def test_permission_mode_not_model_controllable(self):
        """permission_mode is a security-sensitive knob — config-only, never
        in the model-facing schema (a prompt-injected model must not be able
        to escalate Devin's autonomy)."""
        props = DELEGATE_TO_DEVIN_SCHEMA["parameters"]["properties"]
        self.assertNotIn("permission_mode", props)


class TestRegistryWiring(unittest.TestCase):
    """The tool self-registers and is gated by check_fn in the registry."""

    def test_registered_in_delegation_toolset(self):
        from tools.registry import registry
        entry = registry.get_entry("delegate_to_devin")
        self.assertIsNotNone(entry, "delegate_to_devin not registered")
        self.assertEqual(entry.toolset, "delegation")
        self.assertEqual(entry.check_fn, check_devin_requirements)

    def test_listed_in_delegation_toolset_not_core(self):
        """delegate_to_devin is in the delegation toolset but NOT in
        _HERMES_CORE_TOOLS — it's a third-party opt-in backend, not a
        core tool that ships on every API call."""
        import toolsets
        self.assertNotIn("delegate_to_devin", toolsets._HERMES_CORE_TOOLS)
        self.assertIn("delegate_to_devin", toolsets.TOOLSETS["delegation"]["tools"])


if __name__ == "__main__":
    unittest.main()
