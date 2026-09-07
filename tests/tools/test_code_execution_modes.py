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
import subprocess
import sys
import unittest
import unittest.mock
from contextlib import contextmanager, ExitStack
from unittest.mock import patch

import pytest

os.environ["TERMINAL_ENV"] = "local"


@pytest.fixture(autouse=True)
def _force_local_terminal(monkeypatch):
    """Mirror test_code_execution.py — guarantee local backend under xdist."""
    monkeypatch.setenv("TERMINAL_ENV", "local")


@pytest.fixture(autouse=True)
def _fresh_kernel_registry():
    """Session kernels are always on: dispose them per-test so a lingering
    kernel child can't outlive the run (hangs pytest at exit) or leak one
    test's interpreter state into the next."""
    from tools.code_kernel import shutdown_all_kernels

    shutdown_all_kernels()
    yield
    shutdown_all_kernels()


from tools.code_execution_tool import (
    SANDBOX_ALLOWED_TOOLS,
    DEFAULT_EXECUTION_MODE,
    EXECUTION_MODES,
    _get_execution_mode,
    _is_usable_python,
    _python_environment_prefix,
    _python_prefix_cache,
    _usable_python_cache,
    _resolve_child_cwd,
    _resolve_child_python,
    _uses_hermes_python_environment,
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
               if k not in {"VIRTUAL_ENV", "CONDA_PREFIX"}}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(_resolve_child_python("project"), sys.executable)


    def test_is_usable_python_accepts_real_python(self):
        _usable_python_cache.clear()
        self.assertTrue(_is_usable_python(sys.executable))

    def test_is_usable_python_failure_is_not_cached(self):
        """A transient probe failure must not stick — the next call retries.

        A sticky cached False would silently pin project mode to
        sys.executable for the process lifetime.
        """
        _usable_python_cache.clear()
        try:
            with patch("subprocess.run",
                       side_effect=subprocess.TimeoutExpired(cmd=[], timeout=5)) as mock_run:
                self.assertFalse(_is_usable_python("/flaky/python"))
            self.assertEqual(mock_run.call_count, 1)
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = unittest.mock.MagicMock(returncode=0)
                self.assertTrue(_is_usable_python("/flaky/python"))
            self.assertEqual(mock_run.call_count, 1,
                             "probe must be retried after a failure")
        finally:
            _usable_python_cache.clear()


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


    def test_project_stale_record_falls_through_to_override(self):
        """A recorded directory that no longer exists is skipped; the
        registered override is the next rung."""
        import tempfile
        import tools.terminal_tool as terminal_tool

        with tempfile.TemporaryDirectory() as reg:
            task_id = "stale-record-test"
            with patch.dict(os.environ, {"TERMINAL_CWD": "/does/not/exist"}):
                with patch.object(terminal_tool, "_task_env_overrides", {}, create=False), \
                     patch.object(terminal_tool, "_session_cwd", {}, create=False):
                    terminal_tool.register_task_env_overrides(task_id, {"cwd": reg})
                    terminal_tool.record_session_cwd(task_id, "/deleted/dir/gone")
                    self.assertEqual(
                        _resolve_child_cwd("project", "/tmp/staging", task_id=task_id), reg
                    )


# ---------------------------------------------------------------------------
# Schema description
# ---------------------------------------------------------------------------

class TestModeAwareSchema(unittest.TestCase):

    def test_strict_description_mentions_temp_dir(self):
        desc = build_execute_code_schema(mode="strict")["description"]
        self.assertIn("temp dir", desc)


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

@pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "Assumes POSIX venv layout (bin/python) and symlink creation "
        "privileges.  execute_code itself works on Windows — these "
        "integration tests just haven't been ported to the Scripts/"
        "python.exe layout yet."
    ),
)
class TestExecuteCodeModeIntegration(unittest.TestCase):
    """End-to-end: verify the subprocess actually runs where we expect."""

    def _run(self, code, mode, enabled_tools=None, extra_env=None):
        env_overrides = extra_env or {}
        with _mock_mode(mode):
            with patch.dict(os.environ, env_overrides):
                with patch("model_tools.handle_function_call",
                           side_effect=_mock_handle_function_call):
                    # reset=True: kernel cwd/interpreter are frozen at spawn
                    # (like env), so mode-resolution rules are only
                    # observable on a fresh kernel.
                    raw = execute_code(
                        code=code,
                        task_id=f"test-{mode}",
                        enabled_tools=enabled_tools or list(SANDBOX_ALLOWED_TOOLS),
                        reset=True,
                    )
        return json.loads(raw)

    def test_strict_mode_runs_in_tmpdir(self):
        """Strict mode: script's os.getcwd() is a staging tmpdir, never the
        session cwd. Behavior contract, not a prefix snapshot: the per-call
        path stages in hermes_sandbox_*, the session kernel in
        hermes_kernel_* — either satisfies strict mode's isolation promise."""
        result = self._run("import os; print(os.getcwd())", mode="strict")
        self.assertEqual(result["status"], "success")
        cwd = result["output"].strip()
        self.assertTrue(
            "hermes_sandbox_" in cwd or "hermes_kernel_" in cwd,
            f"strict-mode cwd is not a staging tmpdir: {cwd!r}",
        )
        self.assertNotEqual(os.path.realpath(cwd), os.path.realpath(os.getcwd()))


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

@pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "Assumes POSIX venv layout (bin/python) and symlink creation "
        "privileges.  execute_code itself works on Windows — these "
        "integration tests just haven't been ported to the Scripts/"
        "python.exe layout yet."
    ),
)
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


# ---------------------------------------------------------------------------
# _python_environment_prefix / _uses_hermes_python_environment
# ---------------------------------------------------------------------------

class TestPythonEnvironmentPrefix(unittest.TestCase):
    """Unit tests for the helper that queries sys.prefix of an interpreter."""

    def setUp(self):
        _python_prefix_cache.clear()

    def tearDown(self):
        _python_prefix_cache.clear()

    def test_returns_realpath_of_current_interpreter_prefix(self):
        """Happy path: sys.executable reports its own prefix."""
        prefix = _python_environment_prefix(sys.executable)
        self.assertEqual(prefix, os.path.realpath(sys.prefix))

    def test_returns_empty_string_for_nonexistent_path(self):
        """A path that doesn't exist → OSError → empty string."""
        result = _python_environment_prefix("/nonexistent/python-does-not-exist")
        self.assertEqual(result, "")

    def test_returns_empty_string_when_subprocess_times_out(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=[], timeout=5)):
            result = _python_environment_prefix("/some/python")
        self.assertEqual(result, "")

    def test_returns_empty_string_on_nonzero_exit(self):
        mock_result = unittest.mock.MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            result = _python_environment_prefix("/bad/python")
        self.assertEqual(result, "")

    def test_returns_empty_string_when_stdout_is_blank(self):
        mock_result = unittest.mock.MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "   \n"
        with patch("subprocess.run", return_value=mock_result):
            result = _python_environment_prefix("/blank/python")
        self.assertEqual(result, "")

    def test_result_is_cached(self):
        """Second call returns cached value without spawning another process."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = unittest.mock.MagicMock(
                returncode=0, stdout="/fake/prefix\n"
            )
            _python_environment_prefix("/cached/python")
            _python_environment_prefix("/cached/python")
        self.assertEqual(mock_run.call_count, 1)

    def test_failure_is_not_cached(self):
        """A transient probe failure must not stick — the next call retries.

        A sticky cached failure would silently drop the hermes root from
        every subsequent execute_code call in the process.
        """
        with patch("subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd=[], timeout=5)) as mock_run:
            self.assertEqual(_python_environment_prefix("/flaky/python"), "")
        self.assertEqual(mock_run.call_count, 1)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = unittest.mock.MagicMock(
                returncode=0, stdout="/recovered/prefix\n"
            )
            result = _python_environment_prefix("/flaky/python")
        self.assertEqual(mock_run.call_count, 1, "probe must be retried after a failure")
        self.assertEqual(result, os.path.realpath("/recovered/prefix"))


class TestUsesHermesPythonEnvironment(unittest.TestCase):
    """Unit tests for _uses_hermes_python_environment."""

    def setUp(self):
        _python_prefix_cache.clear()

    def tearDown(self):
        _python_prefix_cache.clear()

    def test_true_for_current_interpreter(self):
        """sys.executable always belongs to the current environment."""
        self.assertTrue(_uses_hermes_python_environment(sys.executable))

    def test_true_for_current_interpreter_without_probe(self):
        """sys.executable short-circuits — no subprocess probe on the default path.

        Guards the strict-mode invariant: a flaky probe (timeout under load)
        must never drop the hermes root for the interpreter Hermes itself runs.
        """
        with patch("subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd=[], timeout=5)) as mock_run:
            self.assertTrue(_uses_hermes_python_environment(sys.executable))
        mock_run.assert_not_called()

    def test_false_for_different_prefix(self):
        """An interpreter reporting a different prefix is external."""
        with patch("tools.code_execution_tool._python_environment_prefix",
                   return_value="/some/other/venv"):
            self.assertFalse(_uses_hermes_python_environment("/other/python"))

    def test_false_when_prefix_is_empty(self):
        """If prefix cannot be determined (error path), treat as external."""
        with patch("tools.code_execution_tool._python_environment_prefix",
                   return_value=""):
            self.assertFalse(_uses_hermes_python_environment("/bad/python"))

    def test_true_when_prefix_matches_sys_prefix(self):
        hermes_prefix = os.path.realpath(sys.prefix)
        with patch("tools.code_execution_tool._python_environment_prefix",
                   return_value=hermes_prefix):
            self.assertTrue(_uses_hermes_python_environment("/same/env/python"))


# ---------------------------------------------------------------------------
# PYTHONPATH composition — hermes root included only for same-env interpreters
# ---------------------------------------------------------------------------

class TestPythonPathComposition(unittest.TestCase):
    """Verify hermes root inclusion in PYTHONPATH depends on env match.

    Patches ``_uses_hermes_python_environment`` directly so these tests are
    independent of subprocess availability — the unit tests above already
    cover the detection logic end-to-end.
    """

    def _capture_pythonpath(self, same_env: bool) -> tuple:
        """Return (PYTHONPATH, staging_dir) that execute_code passes to the child."""
        captured = {}

        class _Captured(RuntimeError):
            pass

        def _fake_popen(cmd, **kwargs):
            env = kwargs.get("env") or {}
            captured["PYTHONPATH"] = env.get("PYTHONPATH", "")
            # cmd is [python, <staging_dir>/script.py] (per-call) or
            # [python, <staging_dir>/hermes_kernel_runner.py] (session
            # kernel) — staging dir derivation is identical.
            captured["staging_dir"] = os.path.dirname(cmd[1])
            # Abort the spawn after capture: returning a MagicMock proc
            # would leave the kernel's reader threads spinning on mock
            # reads and hang the cell wait loop (always-on session
            # kernels; the pre-kernel version of this helper could get
            # away with a fake proc because the per-call path only
            # .wait()ed on it).
            raise _Captured()

        with patch("tools.code_execution_tool._load_config", return_value={"mode": "strict"}), \
             patch("model_tools.handle_function_call", side_effect=_mock_handle_function_call), \
             patch("tools.code_execution_tool._uses_hermes_python_environment",
                   return_value=same_env), \
             patch("subprocess.Popen", side_effect=_fake_popen):
            try:
                execute_code(code="pass", task_id="test-pp",
                             enabled_tools=[], reset=True)
            except _Captured:
                pass  # expected: spawn aborted right after env capture
            except Exception:
                pass  # kernel path wraps the abort; capture already happened

        # If execute_code never reached Popen, the capture is empty and any
        # "X not in PYTHONPATH" assertion downstream would pass vacuously.
        self.assertIn("PYTHONPATH", captured,
                      "execute_code never spawned the child process")
        return captured["PYTHONPATH"], captured["staging_dir"]

    def _hermes_root(self) -> str:
        import tools.code_execution_tool as _cet
        tools_dir = os.path.dirname(os.path.abspath(_cet.__file__))
        return os.path.dirname(tools_dir)

    def test_hermes_root_included_when_same_env(self):
        """When interpreter is in the Hermes env, hermes root is in PYTHONPATH."""
        pythonpath, _ = self._capture_pythonpath(same_env=True)
        parts = pythonpath.split(os.pathsep)
        self.assertIn(self._hermes_root(), parts,
                      "hermes root must be in PYTHONPATH for same-env interpreters")

    def test_hermes_root_excluded_when_external_env(self):
        """When interpreter is external, hermes root must NOT be in PYTHONPATH."""
        pythonpath, _ = self._capture_pythonpath(same_env=False)
        parts = pythonpath.split(os.pathsep)
        self.assertNotIn(self._hermes_root(), parts,
                         "hermes root must not leak into an external interpreter's PYTHONPATH")

    def test_staging_dir_always_first(self):
        """The staging tmpdir must always be the first PYTHONPATH entry."""
        for same_env in (True, False):
            with self.subTest(same_env=same_env):
                pythonpath, staging_dir = self._capture_pythonpath(same_env=same_env)
                parts = pythonpath.split(os.pathsep)
                self.assertEqual(parts[0], staging_dir,
                                 "PYTHONPATH must start with the staging tmpdir")


if __name__ == "__main__":
    unittest.main()
