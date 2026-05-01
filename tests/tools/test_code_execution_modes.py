#!/usr/bin/env python3
"""Tests for execute_code's strict / project execution modes.

The mode switch controls two things:
  - working directory: staging tmpdir (strict) vs session CWD (project)
  - interpreter:       sys.executable (strict) vs active venv's python (project)

Security-critical invariants — env scrubbing, tool whitelist, resource caps —
must apply identically in both modes. These tests guard all three layers.

Mode is sourced exclusively from ``code_execution.mode`` in config.yaml —
there is no env-var override. Tests patch ``_load_config`` directly.
"""

import json
import os
import sys
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import pytest

os.environ["TERMINAL_ENV"] = "local"


@pytest.fixture(autouse=True)
def _force_local_terminal(monkeypatch):
    """Mirror test_code_execution.py — guarantee local backend under xdist."""
    monkeypatch.setenv("TERMINAL_ENV", "local")


from tools.code_execution_tool import (
    SANDBOX_ALLOWED_TOOLS,
    DEFAULT_EXECUTION_MODE,
    EXECUTION_MODES,
    _get_execution_mode,
    _is_usable_python,
    _resolve_child_cwd,
    _resolve_child_python,
    build_execute_code_schema,
    execute_code,
)


@contextmanager
def _mock_mode(mode):
    """Context manager that pins code_execution.mode to the given value."""
    with patch("tools.code_execution_tool._load_config",
               return_value={"mode": mode}):
        yield


def _mock_handle_function_call(function_name, function_args, task_id=None, user_task=None):
    """Minimal mock dispatcher reused across tests."""
    if function_name == "terminal":
        return json.dumps({"output": "mock", "exit_code": 0})
    if function_name == "read_file":
        return json.dumps({"content": "line1\n", "total_lines": 1})
    return json.dumps({"error": f"Unknown tool: {function_name}"})


# ---------------------------------------------------------------------------
# Mode resolution
# ---------------------------------------------------------------------------

class TestGetExecutionMode(unittest.TestCase):
    """_get_execution_mode reads config.yaml only (no env var surface)."""

    def test_default_is_project(self):
        self.assertEqual(DEFAULT_EXECUTION_MODE, "project")

    def test_config_project(self):
        with patch("tools.code_execution_tool._load_config",
                   return_value={"mode": "project"}):
            self.assertEqual(_get_execution_mode(), "project")

    def test_config_strict(self):
        with patch("tools.code_execution_tool._load_config",
                   return_value={"mode": "strict"}):
            self.assertEqual(_get_execution_mode(), "strict")

    def test_config_case_insensitive(self):
        with patch("tools.code_execution_tool._load_config",
                   return_value={"mode": "STRICT"}):
            self.assertEqual(_get_execution_mode(), "strict")

    def test_config_strips_whitespace(self):
        with patch("tools.code_execution_tool._load_config",
                   return_value={"mode": "  project  "}):
            self.assertEqual(_get_execution_mode(), "project")

    def test_empty_config_falls_back_to_default(self):
        with patch("tools.code_execution_tool._load_config", return_value={}):
            self.assertEqual(_get_execution_mode(), DEFAULT_EXECUTION_MODE)

    def test_bogus_config_falls_back_to_default(self):
        with patch("tools.code_execution_tool._load_config",
                   return_value={"mode": "banana"}):
            self.assertEqual(_get_execution_mode(), DEFAULT_EXECUTION_MODE)

    def test_none_config_falls_back_to_default(self):
        with patch("tools.code_execution_tool._load_config",
                   return_value={"mode": None}):
            # str(None).lower() = "none" → not in EXECUTION_MODES → default
            self.assertEqual(_get_execution_mode(), DEFAULT_EXECUTION_MODE)

    def test_execution_modes_tuple(self):
        """Canonical set of modes — tests + config layer rely on this shape."""
        self.assertEqual(set(EXECUTION_MODES), {"project", "strict"})


# ---------------------------------------------------------------------------
# Interpreter resolver
# ---------------------------------------------------------------------------

class TestResolveChildPython(unittest.TestCase):
    """_resolve_child_python — picks the right interpreter per mode."""

    def test_strict_always_sys_executable(self):
        """Strict mode never leaves sys.executable, even if venv is set."""
        with patch.dict(os.environ, {"VIRTUAL_ENV": "/some/venv"}):
            self.assertEqual(_resolve_child_python("strict"), sys.executable)

    def test_project_with_no_venv_falls_back(self):
        """Project mode without VIRTUAL_ENV or CONDA_PREFIX → sys.executable."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("VIRTUAL_ENV", "CONDA_PREFIX")}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(_resolve_child_python("project"), sys.executable)

    def test_project_with_virtualenv_picks_venv_python(self):
        """Project mode + VIRTUAL_ENV pointing at a real venv → that python."""
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as td:
            fake_venv = pathlib.Path(td)
            (fake_venv / "bin").mkdir()
            # Symlink to real python so the version check actually passes
            (fake_venv / "bin" / "python").symlink_to(sys.executable)
            with patch.dict(os.environ, {"VIRTUAL_ENV": str(fake_venv)}):
                # Clear cache — _is_usable_python memoizes on path
                _is_usable_python.cache_clear()
                result = _resolve_child_python("project")
                self.assertEqual(result, str(fake_venv / "bin" / "python"))

    def test_project_with_broken_venv_falls_back(self):
        """VIRTUAL_ENV set but bin/python missing → sys.executable."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            # No bin/python inside — broken venv
            with patch.dict(os.environ, {"VIRTUAL_ENV": td}):
                _is_usable_python.cache_clear()
                self.assertEqual(_resolve_child_python("project"), sys.executable)

    def test_project_prefers_virtualenv_over_conda(self):
        """If both VIRTUAL_ENV and CONDA_PREFIX are set, VIRTUAL_ENV wins."""
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as ve_td, tempfile.TemporaryDirectory() as conda_td:
            ve = pathlib.Path(ve_td)
            (ve / "bin").mkdir()
            (ve / "bin" / "python").symlink_to(sys.executable)

            conda = pathlib.Path(conda_td)
            (conda / "bin").mkdir()
            (conda / "bin" / "python").symlink_to(sys.executable)

            with patch.dict(os.environ, {"VIRTUAL_ENV": str(ve), "CONDA_PREFIX": str(conda)}):
                _is_usable_python.cache_clear()
                result = _resolve_child_python("project")
                self.assertEqual(result, str(ve / "bin" / "python"))

    def test_is_usable_python_rejects_nonexistent(self):
        _is_usable_python.cache_clear()
        self.assertFalse(_is_usable_python("/does/not/exist/python"))

    def test_is_usable_python_accepts_real_python(self):
        _is_usable_python.cache_clear()
        self.assertTrue(_is_usable_python(sys.executable))


# ---------------------------------------------------------------------------
# CWD resolver
# ---------------------------------------------------------------------------

class TestResolveChildCwd(unittest.TestCase):

    def test_strict_uses_staging_dir(self):
        self.assertEqual(_resolve_child_cwd("strict", "/tmp/staging"), "/tmp/staging")

    def test_project_without_terminal_cwd_uses_getcwd(self):
        env = {k: v for k, v in os.environ.items() if k != "TERMINAL_CWD"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(_resolve_child_cwd("project", "/tmp/staging"), os.getcwd())

    def test_project_uses_terminal_cwd_when_set(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with patch.dict(os.environ, {"TERMINAL_CWD": td}):
                self.assertEqual(_resolve_child_cwd("project", "/tmp/staging"), td)

    def test_project_bogus_terminal_cwd_falls_back_to_getcwd(self):
        with patch.dict(os.environ, {"TERMINAL_CWD": "/does/not/exist/anywhere"}):
            self.assertEqual(_resolve_child_cwd("project", "/tmp/staging"), os.getcwd())

    def test_project_expands_tilde(self):
        import pathlib
        home = str(pathlib.Path.home())
        with patch.dict(os.environ, {"TERMINAL_CWD": "~"}):
            self.assertEqual(_resolve_child_cwd("project", "/tmp/staging"), home)


# ---------------------------------------------------------------------------
# Schema description
# ---------------------------------------------------------------------------

class TestModeAwareSchema(unittest.TestCase):

    def test_strict_description_mentions_temp_dir(self):
        desc = build_execute_code_schema(mode="strict")["description"]
        self.assertIn("temp dir", desc)

    def test_project_description_mentions_session_and_venv(self):
        desc = build_execute_code_schema(mode="project")["description"]
        self.assertIn("session", desc)
        self.assertIn("venv", desc)

    def test_neither_description_uses_sandbox_language(self):
        """REGRESSION GUARD for commit 39b83f34.

        Agents on local backends falsely believed they were sandboxed and
        refused networking tasks. Do not reintroduce any 'sandbox' /
        'isolated' / 'cloud' language in the tool description.
        """
        for mode in EXECUTION_MODES:
            desc = build_execute_code_schema(mode=mode)["description"].lower()
            for forbidden in ("sandbox", "isolated", "cloud"):
                self.assertNotIn(forbidden, desc,
                                 f"mode={mode}: '{forbidden}' leaked into description")

    def test_descriptions_are_similar_length(self):
        """Both modes should have roughly the same-size description."""
        strict = len(build_execute_code_schema(mode="strict")["description"])
        project = len(build_execute_code_schema(mode="project")["description"])
        self.assertLess(abs(strict - project), 200)

    def test_default_mode_reads_config(self):
        """build_execute_code_schema() with mode=None reads config.yaml."""
        with _mock_mode("strict"):
            desc = build_execute_code_schema()["description"]
            self.assertIn("temp dir", desc)
        with _mock_mode("project"):
            desc = build_execute_code_schema()["description"]
            self.assertIn("session", desc)


# ---------------------------------------------------------------------------
# Integration: what actually happens when execute_code runs per mode
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform == "win32", reason="execute_code is POSIX-only")
class TestExecuteCodeModeIntegration(unittest.TestCase):
    """End-to-end: verify the subprocess actually runs where we expect."""

    def _run(self, code, mode, enabled_tools=None, extra_env=None):
        env_overrides = extra_env or {}
        with _mock_mode(mode):
            with patch.dict(os.environ, env_overrides):
                with patch("model_tools.handle_function_call",
                           side_effect=_mock_handle_function_call):
                    raw = execute_code(
                        code=code,
                        task_id=f"test-{mode}",
                        enabled_tools=enabled_tools or list(SANDBOX_ALLOWED_TOOLS),
                    )
        return json.loads(raw)

    def test_strict_mode_runs_in_tmpdir(self):
        """Strict mode: script's os.getcwd() is the staging tmpdir."""
        result = self._run("import os; print(os.getcwd())", mode="strict")
        self.assertEqual(result["status"], "success")
        self.assertIn("hermes_sandbox_", result["output"])

    def test_project_mode_runs_in_session_cwd(self):
        """Project mode: script's os.getcwd() is the session's working dir."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            result = self._run(
                "import os; print(os.getcwd())",
                mode="project",
                extra_env={"TERMINAL_CWD": td},
            )
            self.assertEqual(result["status"], "success")
            # Resolve symlinks (macOS /tmp → /private/tmp) on both sides
            self.assertEqual(
                os.path.realpath(result["output"].strip()),
                os.path.realpath(td),
            )

    def test_project_mode_interpreter_is_venv_python(self):
        """Project mode: sys.executable inside the child is the venv's python
        when VIRTUAL_ENV is set to a real venv."""
        # The hermes-agent venv is always active during tests, so this also
        # happens to equal sys.executable of the parent. What we're asserting
        # is: resolver picked a venv-bin/python path, not that it differs
        # from sys.executable.
        result = self._run("import sys; print(sys.executable)", mode="project")
        self.assertEqual(result["status"], "success")
        # Either VIRTUAL_ENV-bin/python or sys.executable fallback, both OK.
        output = result["output"].strip()
        ve = os.environ.get("VIRTUAL_ENV", "").strip()
        if ve:
            self.assertTrue(
                output.startswith(ve) or output == sys.executable,
                f"project-mode python should be under VIRTUAL_ENV={ve} or sys.executable={sys.executable}, got {output}",
            )

    def test_project_mode_can_still_import_hermes_tools(self):
        """Regression: hermes_tools still importable from non-tmpdir CWD.

        This is the PYTHONPATH fix — without it, switching to session CWD
        breaks `from hermes_tools import terminal`.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            code = (
                "from hermes_tools import terminal\n"
                "r = terminal('echo x')\n"
                "print(r.get('output', 'MISSING'))\n"
            )
            result = self._run(code, mode="project", extra_env={"TERMINAL_CWD": td})
            self.assertEqual(result["status"], "success")
            self.assertIn("mock", result["output"])

    def test_strict_mode_can_still_import_hermes_tools(self):
        """Regression: strict mode's tmpdir CWD still works for imports."""
        code = (
            "from hermes_tools import terminal\n"
            "r = terminal('echo x')\n"
            "print(r.get('output', 'MISSING'))\n"
        )
        result = self._run(code, mode="strict")
        self.assertEqual(result["status"], "success")
        self.assertIn("mock", result["output"])


# ---------------------------------------------------------------------------
# SECURITY-CRITICAL regression guards
#
# These MUST pass in both strict and project mode. The whole tiered-mode
# proposition rests on the claim that switching from strict to project only
# changes CWD + interpreter, not the security posture.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform == "win32", reason="execute_code is POSIX-only")
class TestSecurityInvariantsAcrossModes(unittest.TestCase):

    def _run(self, code, mode):
        with _mock_mode(mode):
            with patch("model_tools.handle_function_call",
                       side_effect=_mock_handle_function_call):
                raw = execute_code(
                    code=code,
                    task_id=f"test-sec-{mode}",
                    enabled_tools=list(SANDBOX_ALLOWED_TOOLS),
                )
        return json.loads(raw)

    def test_api_keys_scrubbed_in_strict_mode(self):
        code = (
            "import os\n"
            "print('KEY=' + os.environ.get('OPENAI_API_KEY', 'MISSING'))\n"
            "print('TOK=' + os.environ.get('ANTHROPIC_API_KEY', 'MISSING'))\n"
        )
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-should-not-leak",
            "ANTHROPIC_API_KEY": "ant-should-not-leak",
        }):
            result = self._run(code, mode="strict")
        self.assertEqual(result["status"], "success")
        self.assertIn("KEY=MISSING", result["output"])
        self.assertIn("TOK=MISSING", result["output"])
        self.assertNotIn("sk-should-not-leak", result["output"])
        self.assertNotIn("ant-should-not-leak", result["output"])

    def test_api_keys_scrubbed_in_project_mode(self):
        """CRITICAL: the project-mode default does NOT leak user credentials."""
        code = (
            "import os\n"
            "print('KEY=' + os.environ.get('OPENAI_API_KEY', 'MISSING'))\n"
            "print('TOK=' + os.environ.get('ANTHROPIC_API_KEY', 'MISSING'))\n"
            "print('SEC=' + os.environ.get('GITHUB_TOKEN', 'MISSING'))\n"
        )
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-should-not-leak",
            "ANTHROPIC_API_KEY": "ant-should-not-leak",
            "GITHUB_TOKEN": "ghp-should-not-leak",
        }):
            result = self._run(code, mode="project")
        self.assertEqual(result["status"], "success")
        for needle in ("KEY=MISSING", "TOK=MISSING", "SEC=MISSING"):
            self.assertIn(needle, result["output"])
        for leaked in ("sk-should-not-leak", "ant-should-not-leak", "ghp-should-not-leak"):
            self.assertNotIn(leaked, result["output"])

    def test_secret_substrings_scrubbed_in_project_mode(self):
        """SECRET/PASSWORD/CREDENTIAL/PASSWD/AUTH filters still apply."""
        code = (
            "import os\n"
            "for k in ('MY_SECRET', 'DB_PASSWORD', 'VAULT_CREDENTIAL', "
            "'LDAP_PASSWD', 'AUTH_TOKEN'):\n"
            "    print(f'{k}=' + os.environ.get(k, 'MISSING'))\n"
        )
        with patch.dict(os.environ, {
            "MY_SECRET": "secret-should-not-leak",
            "DB_PASSWORD": "password-should-not-leak",
            "VAULT_CREDENTIAL": "cred-should-not-leak",
            "LDAP_PASSWD": "passwd-should-not-leak",
            "AUTH_TOKEN": "auth-should-not-leak",
        }):
            result = self._run(code, mode="project")
        self.assertEqual(result["status"], "success")
        for leaked in ("secret-should-not-leak", "password-should-not-leak",
                       "cred-should-not-leak", "passwd-should-not-leak",
                       "auth-should-not-leak"):
            self.assertNotIn(leaked, result["output"])

    def test_tool_whitelist_enforced_in_strict_mode(self):
        """A script cannot RPC-call tools outside SANDBOX_ALLOWED_TOOLS."""
        # execute_code is NOT in SANDBOX_ALLOWED_TOOLS (no recursion)
        self.assertNotIn("execute_code", SANDBOX_ALLOWED_TOOLS)
        code = (
            "import hermes_tools as ht\n"
            "print('execute_code_available:', hasattr(ht, 'execute_code'))\n"
            "print('delegate_task_available:', hasattr(ht, 'delegate_task'))\n"
        )
        result = self._run(code, mode="strict")
        self.assertEqual(result["status"], "success")
        self.assertIn("execute_code_available: False", result["output"])
        self.assertIn("delegate_task_available: False", result["output"])

    def test_tool_whitelist_enforced_in_project_mode(self):
        """CRITICAL: project mode does NOT widen the tool whitelist."""
        code = (
            "import hermes_tools as ht\n"
            "print('execute_code_available:', hasattr(ht, 'execute_code'))\n"
            "print('delegate_task_available:', hasattr(ht, 'delegate_task'))\n"
        )
        result = self._run(code, mode="project")
        self.assertEqual(result["status"], "success")
        self.assertIn("execute_code_available: False", result["output"])
        self.assertIn("delegate_task_available: False", result["output"])


if __name__ == "__main__":
    unittest.main()
