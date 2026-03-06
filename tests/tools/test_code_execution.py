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

import json
import sys
import time
import unittest
from unittest.mock import patch

from tools.code_execution_tool import (
    SANDBOX_ALLOWED_TOOLS,
    execute_code,
    generate_hermes_tools_module,
    check_sandbox_requirements,
    EXECUTE_CODE_SCHEMA,
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
    if function_name == "search":
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

    def test_generates_subset(self):
        src = generate_hermes_tools_module(["terminal", "web_search"])
        self.assertIn("def terminal(", src)
        self.assertIn("def web_search(", src)
        self.assertNotIn("def read_file(", src)

    def test_empty_list_generates_nothing(self):
        src = generate_hermes_tools_module([])
        self.assertNotIn("def terminal(", src)
        self.assertIn("def _call(", src)  # infrastructure still present

    def test_non_allowed_tools_ignored(self):
        src = generate_hermes_tools_module(["vision_analyze", "terminal"])
        self.assertIn("def terminal(", src)
        self.assertNotIn("def vision_analyze(", src)

    def test_rpc_infrastructure_present(self):
        src = generate_hermes_tools_module(["terminal"])
        self.assertIn("HERMES_RPC_SOCKET", src)
        self.assertIn("AF_UNIX", src)
        self.assertIn("def _connect(", src)
        self.assertIn("def _call(", src)

    def test_convenience_helpers_present(self):
        """Verify json_parse, shell_quote, and retry helpers are generated."""
        src = generate_hermes_tools_module(["terminal"])
        self.assertIn("def json_parse(", src)
        self.assertIn("def shell_quote(", src)
        self.assertIn("def retry(", src)
        self.assertIn("import json, os, socket, shlex, time", src)


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

    def test_multi_tool_chain(self):
        """Script calls multiple tools sequentially."""
        code = """
from hermes_tools import terminal, read_file
r1 = terminal("ls")
r2 = read_file("test.py")
print(f"terminal: {r1['output'][:20]}")
print(f"file lines: {r2['total_lines']}")
"""
        result = self._run(code)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tool_calls_made"], 2)

    def test_syntax_error(self):
        """Script with a syntax error returns error status."""
        result = self._run("def broken(")
        self.assertEqual(result["status"], "error")
        self.assertIn("SyntaxError", result.get("error", "") + result.get("output", ""))

    def test_runtime_exception(self):
        """Script with a runtime error returns error status."""
        result = self._run("raise ValueError('test error')")
        self.assertEqual(result["status"], "error")

    def test_excluded_tool_returns_error(self):
        """Script calling a tool not in the allow-list gets an error from RPC."""
        code = """
from hermes_tools import terminal
result = terminal("echo hi")
print(result)
"""
        # Only enable web_search -- terminal should be excluded
        result = self._run(code, enabled_tools=["web_search"])
        # terminal won't be in hermes_tools.py, so import fails
        self.assertEqual(result["status"], "error")

    def test_empty_code(self):
        """Empty code string returns an error."""
        result = json.loads(execute_code("", task_id="test"))
        self.assertIn("error", result)

    def test_output_captured(self):
        """Multiple print statements are captured in order."""
        code = """
for i in range(5):
    print(f"line {i}")
"""
        result = self._run(code)
        self.assertEqual(result["status"], "success")
        for i in range(5):
            self.assertIn(f"line {i}", result["output"])

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

    def test_timeout_enforcement(self):
        """Script that sleeps too long is killed."""
        code = "import time; time.sleep(999)"
        with patch("model_tools.handle_function_call", side_effect=_mock_handle_function_call):
            # Override config to use a very short timeout
            with patch("tools.code_execution_tool._load_config", return_value={"timeout": 2, "max_tool_calls": 50}):
                result = json.loads(execute_code(
                    code=code,
                    task_id="test-task",
                    enabled_tools=list(SANDBOX_ALLOWED_TOOLS),
                ))
        self.assertEqual(result["status"], "timeout")
        self.assertIn("timed out", result.get("error", ""))

    def test_web_search_tool(self):
        """Script calls web_search and processes results."""
        code = """
from hermes_tools import web_search
results = web_search("test query")
print(f"Found {len(results.get('results', []))} results")
"""
        result = self._run(code)
        self.assertEqual(result["status"], "success")
        self.assertIn("Found 1 results", result["output"])

    def test_json_parse_helper(self):
        """json_parse handles control characters that json.loads(strict=True) rejects."""
        code = r"""
from hermes_tools import json_parse
# This JSON has a literal tab character which strict mode rejects
text = '{"body": "line1\tline2\nline3"}'
result = json_parse(text)
print(result["body"])
"""
        result = self._run(code)
        self.assertEqual(result["status"], "success")
        self.assertIn("line1", result["output"])

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

    def test_retry_helper_success(self):
        """retry returns on first success."""
        code = """
from hermes_tools import retry
counter = [0]
def flaky():
    counter[0] += 1
    return f"ok on attempt {counter[0]}"
result = retry(flaky)
print(result)
"""
        result = self._run(code)
        self.assertEqual(result["status"], "success")
        self.assertIn("ok on attempt 1", result["output"])

    def test_retry_helper_eventual_success(self):
        """retry retries on failure and succeeds eventually."""
        code = """
from hermes_tools import retry
counter = [0]
def flaky():
    counter[0] += 1
    if counter[0] < 3:
        raise ConnectionError(f"fail {counter[0]}")
    return "success"
result = retry(flaky, max_attempts=3, delay=0.01)
print(result)
"""
        result = self._run(code)
        self.assertEqual(result["status"], "success")
        self.assertIn("success", result["output"])

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
    _BLOCKED_TERMINAL_PARAMS = {"background", "check_interval", "pty"}

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

    def test_stubs_pass_all_params_to_rpc(self):
        """The args_dict_expr in each stub must include every parameter from
        the signature, so that all params are actually sent over RPC."""
        import re
        from tools.code_execution_tool import _TOOL_STUBS

        for tool_name, (func_name, sig, doc, args_expr) in _TOOL_STUBS.items():
            stub_params = set(re.findall(r'(\w+)\s*:', sig))
            # Check that each param name appears in the args dict expression
            for param in stub_params:
                self.assertIn(
                    f'"{param}"',
                    args_expr,
                    f"Stub for '{tool_name}' has parameter '{param}' in its "
                    f"signature but doesn't pass it in the args dict: {args_expr}"
                )

    def test_search_files_target_uses_current_values(self):
        """search_files stub should use 'content'/'files', not old 'grep'/'find'."""
        from tools.code_execution_tool import _TOOL_STUBS
        _, sig, doc, _ = _TOOL_STUBS["search_files"]
        self.assertIn('"content"', sig,
                      "search_files stub should default target to 'content', not 'grep'")
        self.assertNotIn('"grep"', sig,
                         "search_files stub still uses obsolete 'grep' target value")
        self.assertNotIn('"find"', doc,
                         "search_files stub docstring still uses obsolete 'find' target value")

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


if __name__ == "__main__":
    unittest.main()
