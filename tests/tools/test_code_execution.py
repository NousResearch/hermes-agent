#!/usr/bin/env python3
"""

Tests for the code execution sandbox (programmatic tool calling).

These tests monkeypatch handle_function_call so they don't require API keys
or a running terminal backend. They verify the core sandbox mechanics:
UDS socket lifecycle, hermes_tools generation, timeout enforcement,
output capping, tool call counting, and error propagation.

Run with:  python -m pytest tests/test_code_execution.py -v
   or:     python tests/test_code_execution.py
"""

import pytest
# pytestmark removed — tests run fine (61 pass, ~99s)

import json
import os
import socket
import time

os.environ["TERMINAL_ENV"] = "local"


@pytest.fixture(autouse=True)
def _force_local_terminal(monkeypatch):
    """Re-set TERMINAL_ENV=local before every test.

    The module-level assignment above covers import time, but under xdist
    another worker can overwrite os.environ between tests.  monkeypatch
    ensures each test starts (and ends) with the correct value.
    """
    monkeypatch.setenv("TERMINAL_ENV", "local")
import sys
import threading
import unittest
from unittest.mock import patch, MagicMock

from tools.code_execution_tool import (
    SANDBOX_ALLOWED_TOOLS,
    execute_code,
    generate_hermes_tools_module,
    check_sandbox_requirements,
    build_execute_code_schema,
    EXECUTE_CODE_SCHEMA,
    _TOOL_DOC_LINES,
    _execute_remote,
)


def _mock_handle_function_call(function_name, function_args, task_id=None, user_task=None):
    """Mock dispatcher that returns canned responses for each tool."""
    if function_name == "terminal":
        cmd = function_args.get("command", "")
        return json.dumps({"output": f"mock output for: {cmd}", "exit_code": 0})
    if function_name == "web_search":
        return json.dumps({"results": [{"url": "https://example.com", "title": "Example", "description": "A test result"}]})
    if function_name == "read_file":
        return json.dumps({"content": "line 1\nline 2\nline 3\n", "total_lines": 3})
    if function_name == "write_file":
        return json.dumps({"status": "ok", "path": function_args.get("path", "")})
    if function_name == "search_files":
        return json.dumps({"matches": [{"file": "test.py", "line": 1, "text": "match"}]})
    if function_name == "patch":
        return json.dumps({"status": "ok", "replacements": 1})
    if function_name == "web_extract":
        return json.dumps("# Extracted content\nSome text from the page.")
    return json.dumps({"error": f"Unknown tool in mock: {function_name}"})


class TestSandboxRequirements(unittest.TestCase):
    def test_available_on_posix(self):
        if sys.platform != "win32":
            self.assertTrue(check_sandbox_requirements())

    def test_schema_is_valid(self):
        self.assertEqual(EXECUTE_CODE_SCHEMA["name"], "execute_code")
        self.assertIn("code", EXECUTE_CODE_SCHEMA["parameters"]["properties"])
        self.assertIn("code", EXECUTE_CODE_SCHEMA["parameters"]["required"])


class TestHermesToolsGeneration(unittest.TestCase):
    def test_generates_all_allowed_tools(self):
        src = generate_hermes_tools_module(list(SANDBOX_ALLOWED_TOOLS))
        for tool in SANDBOX_ALLOWED_TOOLS:
            self.assertIn(f"def {tool}(", src)


    def test_empty_list_generates_nothing(self):
        src = generate_hermes_tools_module([])
        self.assertNotIn("def terminal(", src)
        self.assertIn("def _call(", src)  # infrastructure still present


    def test_file_transport_uses_tempfile_fallback_for_rpc_dir(self):
        src = generate_hermes_tools_module(["terminal"], transport="file")
        self.assertIn("import json, os, shlex, tempfile, threading, time", src)
        self.assertIn("os.path.join(tempfile.gettempdir(), \"hermes_rpc\")", src)
        self.assertNotIn('os.environ.get("HERMES_RPC_DIR", "/tmp/hermes_rpc")', src)

    def test_uds_transport_serializes_concurrent_calls(self):
        """Regression: UDS _call() must hold a lock across send+recv so that
        concurrent tool calls from multiple threads don't interleave on the
        shared socket and receive each other's responses."""
        src = generate_hermes_tools_module(["terminal"], transport="uds")
        self.assertIn("_call_lock = threading.Lock()", src)
        self.assertIn("with _call_lock:", src)

    def test_file_transport_serializes_seq_allocation(self):
        """Regression: file transport _call() must allocate `_seq` under a
        lock, otherwise concurrent threads can pick the same seq and clobber
        each other's request files."""
        src = generate_hermes_tools_module(["terminal"], transport="file")
        self.assertIn("_seq_lock = threading.Lock()", src)
        self.assertIn("with _seq_lock:", src)


class TestExecuteCodeRemoteTempDir(unittest.TestCase):
    def test_execute_remote_uses_backend_temp_dir_for_sandbox(self):
        class FakeEnv:
            def __init__(self):
                self.commands = []

            def get_temp_dir(self):
                return "/data/data/com.termux/files/usr/tmp"

            def execute(self, command, cwd=None, timeout=None):
                self.commands.append((command, cwd, timeout))
                if "command -v python3" in command:
                    return {"output": "OK\n"}
                if "python3 script.py" in command:
                    return {"output": "hello\n", "returncode": 0}
                return {"output": ""}

        env = FakeEnv()
        fake_thread = MagicMock()

        with patch("tools.code_execution_tool._load_config", return_value={"timeout": 30, "max_tool_calls": 5}), \
             patch("tools.code_execution_tool._get_or_create_env", return_value=(env, "ssh")), \
             patch("tools.code_execution_tool._ship_file_to_remote"), \
             patch("tools.code_execution_tool.threading.Thread", return_value=fake_thread):
            result = json.loads(_execute_remote("print('hello')", "task-1", ["terminal"]))

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["stdout_truncated"])
        self.assertEqual(result["stdout_bytes_total"], len("hello\n".encode("utf-8")))
        mkdir_cmd = env.commands[1][0]
        run_cmd = next(cmd for cmd, _, _ in env.commands if "python3 script.py" in cmd)
        cleanup_cmd = env.commands[-1][0]
        self.assertIn("mkdir -p /data/data/com.termux/files/usr/tmp/hermes_exec_", mkdir_cmd)
        self.assertIn("HERMES_RPC_DIR=/data/data/com.termux/files/usr/tmp/hermes_exec_", run_cmd)
        self.assertIn("rm -rf /data/data/com.termux/files/usr/tmp/hermes_exec_", cleanup_cmd)
        self.assertNotIn("mkdir -p /tmp/hermes_exec_", mkdir_cmd)

    def test_timezone_shell_quoted_in_remote_execution(self):
        """HERMES_TIMEZONE must be shell-quoted in remote env_prefix to prevent injection."""
        class FakeEnv:
            def __init__(self):
                self.commands = []

            def get_temp_dir(self):
                return "/tmp"

            def execute(self, command, cwd=None, timeout=None):
                self.commands.append((command, cwd, timeout))
                if "command -v python3" in command:
                    return {"output": "OK\n"}
                if "python3 script.py" in command:
                    return {"output": "hello\n", "returncode": 0}
                return {"output": ""}

        env = FakeEnv()
        fake_thread = MagicMock()

        malicious_tz = "US/Eastern; echo PWNED"

        with patch("tools.code_execution_tool._load_config",
                   return_value={"timeout": 30, "max_tool_calls": 5}), \
             patch("tools.code_execution_tool._get_or_create_env",
                   return_value=(env, "ssh")), \
             patch("tools.code_execution_tool._ship_file_to_remote"), \
             patch("tools.code_execution_tool.threading.Thread",
                   return_value=fake_thread), \
             patch.dict(os.environ, {"HERMES_TIMEZONE": malicious_tz}):
            result = json.loads(_execute_remote("print('hello')", "task-1", ["terminal"]))

        self.assertEqual(result["status"], "success")
        run_cmd = next(cmd for cmd, _, _ in env.commands if "python3 script.py" in cmd)
        # The TZ value must be shell-quoted — it should NOT contain unescaped semicolons
        self.assertNotIn("TZ=US/Eastern; echo PWNED", run_cmd,
                         "TZ value with shell metacharacters must not appear unquoted")
        # shlex.quote wraps values containing special characters in single quotes
        self.assertIn("TZ='US/Eastern; echo PWNED'", run_cmd,
                      "TZ value must be wrapped in single quotes by shlex.quote()")


@unittest.skipIf(sys.platform == "win32", "UDS not available on Windows")
class TestExecuteCode(unittest.TestCase):
    """Integration tests using the mock dispatcher."""

    def _run(self, code, enabled_tools=None):
        """Helper: run code with mocked handle_function_call."""
        with patch("tools.code_execution_tool._rpc_server_loop") as mock_rpc:
            # Use real execution but mock the tool dispatcher
            pass
        # Actually run with full integration, mocking at the model_tools level
        with patch("model_tools.handle_function_call", side_effect=_mock_handle_function_call):
            result = execute_code(
                code=code,
                task_id="test-task",
                enabled_tools=enabled_tools or list(SANDBOX_ALLOWED_TOOLS),
            )
        return json.loads(result)

    def test_basic_print(self):
        """Script that just prints -- no tool calls."""
        result = self._run('print("hello world")')
        self.assertEqual(result["status"], "success")
        self.assertIn("hello world", result["output"])
        self.assertEqual(result["tool_calls_made"], 0)

    def test_no_tool_call_script_does_not_wait_for_rpc_accept_timeout(self):
        """A no-tool script should not wait seconds for the idle RPC accept thread."""
        start = time.monotonic()
        result = self._run('print("fast")')
        elapsed = time.monotonic() - start

        self.assertEqual(result["status"], "success")
        self.assertIn("fast", result["output"])
        self.assertLess(elapsed, 2.0, f"execute_code took {elapsed:.3f}s")

    def test_repo_root_modules_are_importable(self):
        """Sandboxed scripts can import modules that live at the repo root."""
        result = self._run('import hermes_constants; print(hermes_constants.__file__)')
        self.assertEqual(result["status"], "success")
        self.assertIn("hermes_constants.py", result["output"])

    def test_single_tool_call(self):
        """Script calls terminal and prints the result."""
        code = """
from hermes_tools import terminal
result = terminal("echo hello")
print(result.get("output", ""))
"""
        result = self._run(code)
        self.assertEqual(result["status"], "success")
        self.assertIn("mock output for: echo hello", result["output"])
        self.assertEqual(result["tool_calls_made"], 1)


    def test_concurrent_tool_calls_match_responses(self):
        """Regression for the UDS RPC race: multiple threads inside the
        sandbox calling terminal() concurrently must each receive their own
        response, not another thread's.

        Before the fix, `_sock` and the recv-loop were shared without a
        lock, so responses (written FIFO by the single-threaded server)
        got delivered to whichever client thread happened to win the
        recv() race. That surfaced as each thread seeing another thread's
        output.

        The mock dispatcher sleeps briefly to guarantee the requests
        overlap on the socket.
        """
        code = '''
import threading
from concurrent.futures import ThreadPoolExecutor
from hermes_tools import terminal

N = 10

def call(i):
    r = terminal(f"echo TAG-{i}")
    return i, r.get("output", "")

with ThreadPoolExecutor(max_workers=N) as ex:
    results = list(ex.map(call, range(N)))

mismatches = [(i, out) for i, out in results if f"TAG-{i}" not in out]
if mismatches:
    print(f"MISMATCH {len(mismatches)}/{N}: {mismatches[:3]}")
else:
    print(f"OK {N}/{N}")
'''

        def slow_mock(function_name, function_args, task_id=None, user_task=None):
            import time as _t
            if function_name == "terminal":
                _t.sleep(0.05)  # ensure requests overlap on the socket
                cmd = function_args.get("command", "")
                # Echo semantics: strip leading "echo " and return the rest
                out = cmd[5:] if cmd.startswith("echo ") else f"mock: {cmd}"
                return json.dumps({"output": out, "exit_code": 0})
            return _mock_handle_function_call(
                function_name, function_args, task_id=task_id, user_task=user_task
            )

        with patch("model_tools.handle_function_call", side_effect=slow_mock):
            raw = execute_code(
                code=code,
                task_id="test-concurrent",
                enabled_tools=list(SANDBOX_ALLOWED_TOOLS),
            )
        result = json.loads(raw)
        self.assertEqual(result["status"], "success", msg=result)
        self.assertIn("OK 10/10", result["output"],
                      msg=f"Concurrent tool calls mismatched: {result['output']!r}")


    def test_stderr_on_error(self):
        """Traceback from stderr is included in the response."""
        code = """
import sys
print("before error")
raise RuntimeError("deliberate crash")
"""
        result = self._run(code)
        self.assertEqual(result["status"], "error")
        self.assertIn("before error", result["output"])
        self.assertIn("RuntimeError", result.get("error", "") + result.get("output", ""))


    def test_shell_quote_helper(self):
        """shell_quote properly escapes dangerous characters."""
        code = """
from hermes_tools import shell_quote
# String with backticks, quotes, and special chars
dangerous = '`rm -rf /` && $(whoami) "hello"'
escaped = shell_quote(dangerous)
print(escaped)
# Verify it's wrapped in single quotes with proper escaping
assert "rm -rf" in escaped
assert escaped.startswith("'")
"""
        result = self._run(code)
        self.assertEqual(result["status"], "success")


    def test_retry_helper_all_fail(self):
        """retry raises the last error when all attempts fail."""
        code = """
from hermes_tools import retry
def always_fail():
    raise ValueError("nope")
try:
    retry(always_fail, max_attempts=2, delay=0.01)
    print("should not reach here")
except ValueError as e:
    print(f"caught: {e}")
"""
        result = self._run(code)
        self.assertEqual(result["status"], "success")
        self.assertIn("caught: nope", result["output"])


class TestStubSchemaDrift(unittest.TestCase):
    """Verify that _TOOL_STUBS in code_execution_tool.py stay in sync with
    the real tool schemas registered in tools/registry.py.

    If a tool gains a new parameter but the sandbox stub isn't updated,
    the LLM will try to use the parameter (it sees it in the system prompt)
    and get a TypeError.  This test catches that drift.
    """

    # Parameters that are internal (injected by the handler, not user-facing)
    _INTERNAL_PARAMS = {"task_id", "user_task"}
    # Parameters intentionally blocked in the sandbox
    _BLOCKED_TERMINAL_PARAMS = {"background", "pty", "notify_on_complete", "watch_patterns"}

    def test_stubs_cover_all_schema_params(self):
        """Every user-facing parameter in the real schema must appear in the
        corresponding _TOOL_STUBS entry."""
        import re
        from tools.code_execution_tool import _TOOL_STUBS

        # Import the registry and trigger tool registration
        from tools.registry import registry
        import tools.file_tools  # noqa: F401 - registers read_file, write_file, patch, search_files
        import tools.web_tools  # noqa: F401 - registers web_search, web_extract

        for tool_name, (func_name, sig, doc, args_expr) in _TOOL_STUBS.items():
            entry = registry._tools.get(tool_name)
            if not entry:
                # Tool might not be registered yet (e.g., terminal uses a
                # different registration path).  Skip gracefully.
                continue

            schema_props = entry.schema.get("parameters", {}).get("properties", {})
            schema_params = set(schema_props.keys()) - self._INTERNAL_PARAMS
            if tool_name == "terminal":
                schema_params -= self._BLOCKED_TERMINAL_PARAMS

            # Extract parameter names from the stub signature string
            # Match word before colon: "pattern: str, target: str = ..."
            stub_params = set(re.findall(r'(\w+)\s*:', sig))

            missing = schema_params - stub_params
            self.assertEqual(
                missing, set(),
                f"Stub for '{tool_name}' is missing parameters that exist in "
                f"the real schema: {missing}. Update _TOOL_STUBS in "
                f"code_execution_tool.py to include them."
            )


    def test_generated_module_accepts_all_params(self):
        """The generated hermes_tools.py module should accept all current params
        without TypeError when called with keyword arguments."""
        src = generate_hermes_tools_module(list(SANDBOX_ALLOWED_TOOLS))

        # Compile the generated module to check for syntax errors
        compile(src, "hermes_tools.py", "exec")

        # Verify specific parameter signatures are in the source
        # search_files must accept context, offset, output_mode
        self.assertIn("context", src)
        self.assertIn("offset", src)
        self.assertIn("output_mode", src)

        # patch must accept mode and patch params
        self.assertIn("mode", src)


# ---------------------------------------------------------------------------
# build_execute_code_schema
# ---------------------------------------------------------------------------

class TestBuildExecuteCodeSchema(unittest.TestCase):
    """Tests for build_execute_code_schema — the dynamic schema generator."""

    def test_default_includes_all_tools(self):
        schema = build_execute_code_schema()
        desc = schema["description"]
        for name, _ in _TOOL_DOC_LINES:
            self.assertIn(name, desc, f"Default schema should mention '{name}'")

    def test_schema_structure(self):
        schema = build_execute_code_schema()
        self.assertEqual(schema["name"], "execute_code")
        self.assertIn("parameters", schema)
        self.assertIn("code", schema["parameters"]["properties"])
        self.assertEqual(schema["parameters"]["required"], ["code"])

    def test_subset_only_lists_enabled_tools(self):
        enabled = {"terminal", "read_file"}
        schema = build_execute_code_schema(enabled)
        desc = schema["description"]
        self.assertIn("terminal(", desc)
        self.assertIn("read_file(", desc)
        self.assertNotIn("web_search(", desc)
        self.assertNotIn("web_extract(", desc)
        self.assertNotIn("write_file(", desc)


    def test_none_defaults_to_all_tools(self):
        schema_none = build_execute_code_schema(None)
        schema_all = build_execute_code_schema(SANDBOX_ALLOWED_TOOLS)
        self.assertEqual(schema_none["description"], schema_all["description"])


# ---------------------------------------------------------------------------
# Environment variable filtering (security critical)
# ---------------------------------------------------------------------------

@unittest.skipIf(sys.platform == "win32", "UDS not available on Windows")
class TestEnvVarFiltering(unittest.TestCase):
    """Verify that execute_code filters environment variables correctly.

    The child process should NOT receive API keys, tokens, or secrets.
    It should receive safe vars like PATH, HOME, LANG, etc.
    """

    def _get_child_env(self, extra_env=None):
        """Run a script that dumps its environment and return the env dict."""
        code = (
            "import os, json\n"
            "print(json.dumps(dict(os.environ)))\n"
        )
        env_backup = os.environ.copy()
        try:
            if extra_env:
                os.environ.update(extra_env)
            with patch("model_tools.handle_function_call", return_value='{}'), \
                 patch("tools.code_execution_tool._load_config",
                       return_value={"timeout": 10, "max_tool_calls": 50}):
                raw = execute_code(code, task_id="test-env",
                                   enabled_tools=list(SANDBOX_ALLOWED_TOOLS))
        finally:
            os.environ.clear()
            os.environ.update(env_backup)

        result = json.loads(raw)
        self.assertEqual(result["status"], "success", result.get("error", ""))
        return json.loads(result["output"].strip())

    def test_api_keys_excluded(self):
        child_env = self._get_child_env({
            "OPENAI_API_KEY": "sk-secret123",
            "ANTHROPIC_API_KEY": "sk-ant-secret",
            "FIRECRAWL_API_KEY": "fc-secret",
        })
        self.assertNotIn("OPENAI_API_KEY", child_env)
        self.assertNotIn("ANTHROPIC_API_KEY", child_env)
        self.assertNotIn("FIRECRAWL_API_KEY", child_env)

    def test_tokens_excluded(self):
        child_env = self._get_child_env({
            "GITHUB_TOKEN": "ghp_secret",
            "MODAL_TOKEN_ID": "tok-123",
            "MODAL_TOKEN_SECRET": "tok-sec",
        })
        self.assertNotIn("GITHUB_TOKEN", child_env)
        self.assertNotIn("MODAL_TOKEN_ID", child_env)
        self.assertNotIn("MODAL_TOKEN_SECRET", child_env)


    def test_hermes_rpc_socket_injected(self):
        child_env = self._get_child_env()
        self.assertIn("HERMES_RPC_SOCKET", child_env)


    def test_timezone_injected_when_set(self):
        env_backup = os.environ.copy()
        try:
            os.environ["HERMES_TIMEZONE"] = "America/New_York"
            child_env = self._get_child_env()
            self.assertEqual(child_env.get("TZ"), "America/New_York")
        finally:
            os.environ.clear()
            os.environ.update(env_backup)

    def test_timezone_not_set_when_empty(self):
        env_backup = os.environ.copy()
        try:
            os.environ.pop("HERMES_TIMEZONE", None)
            child_env = self._get_child_env()
            if "TZ" in child_env:
                self.assertNotEqual(child_env["TZ"], "")
        finally:
            os.environ.clear()
            os.environ.update(env_backup)


# ---------------------------------------------------------------------------
# execute_code edge cases
# ---------------------------------------------------------------------------

class TestExecuteCodeEdgeCases(unittest.TestCase):

    def test_windows_returns_error(self):
        """When SANDBOX_AVAILABLE is False (e.g. when the backend deems
        the sandbox unusable for this environment), execute_code returns
        an error JSON with a readable message pointing the caller at
        regular tool calls.  Previously this was a Windows-only gate;
        execute_code now works on Windows via loopback TCP, so the
        error is only emitted when SANDBOX_AVAILABLE is explicitly
        flipped off (e.g. for future platform-specific disables)."""
        with patch("tools.code_execution_tool.SANDBOX_AVAILABLE", False):
            result = json.loads(execute_code("print('hi')", task_id="test"))
            self.assertIn("error", result)
            self.assertIn("unavailable", result["error"].lower())


    @unittest.skipIf(sys.platform == "win32", "UDS not available on Windows")
    def test_nonoverlapping_tools_fallback(self):
        """When enabled_tools has no overlap with SANDBOX_ALLOWED_TOOLS,
        should fall back to all allowed tools."""
        code = (
            "from hermes_tools import terminal\n"
            "print('fallback ok')\n"
        )
        with patch("model_tools.handle_function_call",
                    return_value=json.dumps({"ok": True})):
            result = json.loads(execute_code(
                code, task_id="test-nonoverlap",
                enabled_tools=["vision_analyze", "browser_snapshot"],
            ))
        self.assertEqual(result["status"], "success")
        self.assertIn("fallback ok", result["output"])


# ---------------------------------------------------------------------------
# _load_config
# ---------------------------------------------------------------------------

class TestLoadConfig(unittest.TestCase):
    def test_returns_empty_dict_when_cli_config_unavailable(self):
        from tools.code_execution_tool import _load_config
        with patch.dict("sys.modules", {"cli": None}):
            result = _load_config()
            self.assertIsInstance(result, dict)


    def test_does_not_import_interactive_cli(self):
        from tools.code_execution_tool import _load_config
        mock_cli = MagicMock()
        mock_cli.CLI_CONFIG = {"code_execution": {"timeout": 999}}
        with patch.dict("sys.modules", {"cli": mock_cli}), \
             patch("hermes_cli.config.read_raw_config", return_value={}):
            result = _load_config()
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# Interrupt event
# ---------------------------------------------------------------------------

@unittest.skipIf(sys.platform == "win32", "UDS not available on Windows")
class TestInterruptHandling(unittest.TestCase):
    def test_interrupt_event_stops_execution(self):
        """When interrupt is set for the execution thread, execute_code should stop."""
        code = "import time; time.sleep(60); print('should not reach')"
        from tools.interrupt import set_interrupt

        # Capture the main thread ID so we can target the interrupt correctly.
        # execute_code runs in the current thread; set_interrupt needs its ID.
        main_tid = threading.current_thread().ident

        def set_interrupt_after_delay():
            import time as _t
            _t.sleep(1)
            set_interrupt(True, main_tid)

        t = threading.Thread(target=set_interrupt_after_delay, daemon=True)
        t.start()

        try:
            with patch("model_tools.handle_function_call",
                        return_value=json.dumps({"ok": True})), \
                 patch("tools.code_execution_tool._load_config",
                       return_value={"timeout": 30, "max_tool_calls": 50}):
                result = json.loads(execute_code(
                    code, task_id="test-interrupt",
                    enabled_tools=list(SANDBOX_ALLOWED_TOOLS),
                ))
            self.assertEqual(result["status"], "interrupted")
            self.assertIn("interrupted", result["output"])
        finally:
            set_interrupt(False, main_tid)
            t.join(timeout=3)


class TestHeadTailTruncation(unittest.TestCase):
    """Tests for head+tail truncation of large stdout in execute_code."""

    def _run(self, code):
        with patch("model_tools.handle_function_call", side_effect=_mock_handle_function_call):
            result = execute_code(
                code=code,
                task_id="test-task",
                enabled_tools=list(SANDBOX_ALLOWED_TOOLS),
            )
        return json.loads(result)

    def test_short_output_not_truncated(self):
        """Output under MAX_STDOUT_BYTES should not be truncated."""
        result = self._run('print("small output")')
        self.assertEqual(result["status"], "success")
        self.assertIn("small output", result["output"])
        self.assertNotIn("TRUNCATED", result["output"])


    def test_remote_large_output_gets_truncation_metadata(self):
        """Remote backend output capping is explicit in the JSON result."""
        class FakeEnv:
            def __init__(self):
                self.commands = []

            def get_temp_dir(self):
                return "/tmp"

            def execute(self, command, cwd=None, timeout=None):
                self.commands.append((command, cwd, timeout))
                if "command -v python3" in command:
                    return {"output": "OK\n"}
                if "python3 script.py" in command:
                    return {"output": "HEAD\n" + ("x" * 80_000) + "\nTAIL\n", "returncode": 0}
                return {"output": ""}

        fake_thread = MagicMock()

        with patch("tools.code_execution_tool._load_config", return_value={"timeout": 30, "max_tool_calls": 5}), \
             patch("tools.code_execution_tool._get_or_create_env", return_value=(FakeEnv(), "ssh")), \
             patch("tools.code_execution_tool._ship_file_to_remote"), \
             patch("tools.code_execution_tool.threading.Thread", return_value=fake_thread):
            result = json.loads(_execute_remote("print('large')", "task-1", ["terminal"]))

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(result["stdout_truncated"])
        self.assertIn("HEAD", result["output"])
        self.assertIn("TAIL", result["output"])
        self.assertGreater(result["stdout_bytes_total"], result["stdout_bytes_captured"])
        self.assertGreater(result["stdout_bytes_omitted"], 0)
        self.assertIn("execute_code stdout was truncated", result["warning"])


class TestRpcTokenAuthorization(unittest.TestCase):
    """The per-session RPC token must gate socket dispatch (fail-closed).

    Regression coverage for the execute_code tool-socket hardening: a
    request without the matching HERMES_RPC_TOKEN must be rejected before
    the tool is dispatched, while a request carrying the correct token
    round-trips normally.
    """

    def _drive_server(self, rpc_token, requests):
        """Run _rpc_server_loop against a real AF_UNIX socketpair.

        Sends each dict in *requests* as a newline-delimited JSON message
        and returns the list of decoded JSON responses.
        """
        from tools.code_execution_tool import _rpc_server_loop

        # socketpair gives us a connected client end and a "server" end we
        # can hand to accept() by wrapping it in a tiny listener shim.
        srv, cli = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

        class _OneShotListener:
            """Minimal object exposing the .accept()/.settimeout() the loop uses."""

            def __init__(self, conn):
                self._conn = conn
                self._served = False

            def settimeout(self, _t):
                pass

            def accept(self):
                if self._served:
                    raise socket.timeout()
                self._served = True
                return self._conn, ("peer", 0)

        listener = _OneShotListener(srv)
        stop_event = threading.Event()
        tool_call_log = []
        tool_call_counter = [0]

        def _run():
            with patch(
                "model_tools.handle_function_call",
                side_effect=_mock_handle_function_call,
            ):
                _rpc_server_loop(
                    listener,
                    "test-task",
                    tool_call_log,
                    tool_call_counter,
                    max_tool_calls=10,
                    allowed_tools=frozenset({"terminal"}),
                    stop_event=stop_event,
                    rpc_token=rpc_token,
                )

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        responses = []
        try:
            for req in requests:
                cli.sendall((json.dumps(req) + "\n").encode())
            cli.settimeout(5)
            buf = b""
            while len(responses) < len(requests):
                chunk = cli.recv(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if line:
                        responses.append(json.loads(line.decode()))
        finally:
            stop_event.set()
            cli.close()
            srv.close()
            t.join(timeout=5)
        return responses

    def test_missing_token_rejected(self):
        """A request with no token is rejected as Unauthorized."""
        resp = self._drive_server(
            "secret-token", [{"tool": "terminal", "args": {"command": "echo hi"}}]
        )
        self.assertEqual(len(resp), 1)
        self.assertIn("Unauthorized", resp[0].get("error", ""))


    def test_generated_module_sends_token(self):
        """The generated hermes_tools module reads HERMES_RPC_TOKEN and sends it."""
        src = generate_hermes_tools_module(["terminal"], transport="uds")
        self.assertIn("HERMES_RPC_TOKEN", src)
        self.assertIn('"token"', src)


if __name__ == "__main__":
    unittest.main()
