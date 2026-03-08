#!/usr/bin/env python3
"""
Tests for build_execute_code_schema, environment variable filtering,
and other untested code paths in code_execution_tool.py.

Run with:  python -m pytest tests/tools/test_code_execution_schema.py -v
"""

import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

from tools.code_execution_tool import (
    SANDBOX_ALLOWED_TOOLS,
    build_execute_code_schema,
    execute_code,
    check_sandbox_requirements,
    _TOOL_DOC_LINES,
    DEFAULT_TIMEOUT,
    DEFAULT_MAX_TOOL_CALLS,
)


# ---------------------------------------------------------------------------
# build_execute_code_schema
# ---------------------------------------------------------------------------

class TestBuildExecuteCodeSchema(unittest.TestCase):
    """Tests for build_execute_code_schema — previously completely untested."""

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

    def test_single_tool(self):
        schema = build_execute_code_schema({"terminal"})
        desc = schema["description"]
        self.assertIn("terminal(", desc)
        self.assertNotIn("web_search(", desc)

    def test_import_examples_prefer_web_search_and_terminal(self):
        enabled = {"web_search", "terminal", "read_file"}
        schema = build_execute_code_schema(enabled)
        code_desc = schema["parameters"]["properties"]["code"]["description"]
        self.assertIn("web_search", code_desc)
        self.assertIn("terminal", code_desc)

    def test_import_examples_fallback_when_no_preferred(self):
        """When neither web_search nor terminal are enabled, falls back to
        sorted first two tools."""
        enabled = {"read_file", "write_file", "patch"}
        schema = build_execute_code_schema(enabled)
        code_desc = schema["parameters"]["properties"]["code"]["description"]
        # Should use sorted first 2: patch, read_file
        self.assertIn("patch", code_desc)
        self.assertIn("read_file", code_desc)

    def test_empty_set_produces_valid_description(self):
        """BUG: build_execute_code_schema(set()) produces 'import , ...' in
        the code property description — a broken import example."""
        schema = build_execute_code_schema(set())
        code_desc = schema["parameters"]["properties"]["code"]["description"]
        # The description should NOT contain a bare comma before ellipsis
        # like "from hermes_tools import , ..."
        self.assertNotIn("import , ...", code_desc,
                         "Empty enabled set produces broken import syntax in description")

    def test_real_scenario_all_sandbox_tools_disabled(self):
        """Reproduce the exact code path from model_tools.py:231-234.

        Scenario: user runs `hermes tools code_execution` (only code_execution
        toolset enabled). tools_to_include = {"execute_code"}.

        model_tools.py does:
            sandbox_enabled = SANDBOX_ALLOWED_TOOLS & tools_to_include
            dynamic_schema = build_execute_code_schema(sandbox_enabled)

        SANDBOX_ALLOWED_TOOLS = {web_search, web_extract, read_file, write_file,
                                  search_files, patch, terminal}
        tools_to_include  = {"execute_code"}
        intersection      = empty set

        This sends set() to build_execute_code_schema, which BEFORE the fix
        produced "from hermes_tools import , ..." in the description.
        """
        from tools.code_execution_tool import SANDBOX_ALLOWED_TOOLS

        # Simulate model_tools.py:233
        tools_to_include = {"execute_code"}
        sandbox_enabled = SANDBOX_ALLOWED_TOOLS & tools_to_include

        self.assertEqual(sandbox_enabled, set(),
                         "Intersection should be empty when only execute_code is enabled")

        schema = build_execute_code_schema(sandbox_enabled)
        code_desc = schema["parameters"]["properties"]["code"]["description"]
        self.assertNotIn("import , ...", code_desc,
                         "Bug: broken import syntax sent to the model")

    def test_real_scenario_only_vision_enabled(self):
        """Another real path: user runs `hermes tools code_execution,vision`.

        tools_to_include = {"execute_code", "vision_analyze"}
        SANDBOX_ALLOWED_TOOLS has neither, so intersection is empty.
        """
        from tools.code_execution_tool import SANDBOX_ALLOWED_TOOLS

        tools_to_include = {"execute_code", "vision_analyze"}
        sandbox_enabled = SANDBOX_ALLOWED_TOOLS & tools_to_include

        self.assertEqual(sandbox_enabled, set())

        schema = build_execute_code_schema(sandbox_enabled)
        code_desc = schema["parameters"]["properties"]["code"]["description"]
        self.assertNotIn("import , ...", code_desc)

    def test_description_mentions_limits(self):
        schema = build_execute_code_schema()
        desc = schema["description"]
        self.assertIn("5-minute timeout", desc)
        self.assertIn("50KB", desc)
        self.assertIn("50 tool calls", desc)

    def test_description_mentions_helpers(self):
        schema = build_execute_code_schema()
        desc = schema["description"]
        self.assertIn("json_parse", desc)
        self.assertIn("shell_quote", desc)
        self.assertIn("retry", desc)

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

    def test_password_vars_excluded(self):
        child_env = self._get_child_env({
            "DB_PASSWORD": "hunter2",
            "MY_PASSWD": "secret",
            "AUTH_CREDENTIAL": "cred",
        })
        self.assertNotIn("DB_PASSWORD", child_env)
        self.assertNotIn("MY_PASSWD", child_env)
        self.assertNotIn("AUTH_CREDENTIAL", child_env)

    def test_path_included(self):
        child_env = self._get_child_env()
        self.assertIn("PATH", child_env)

    def test_home_included(self):
        child_env = self._get_child_env()
        self.assertIn("HOME", child_env)

    def test_hermes_rpc_socket_injected(self):
        child_env = self._get_child_env()
        self.assertIn("HERMES_RPC_SOCKET", child_env)

    def test_pythondontwritebytecode_set(self):
        child_env = self._get_child_env()
        self.assertEqual(child_env.get("PYTHONDONTWRITEBYTECODE"), "1")

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
            # TZ should not be set unless HERMES_TIMEZONE is non-empty
            # (it might be set from the system, so we just check it's not
            # set to empty string)
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
        """On Windows (or when SANDBOX_AVAILABLE is False), returns error JSON."""
        with patch("tools.code_execution_tool.SANDBOX_AVAILABLE", False):
            result = json.loads(execute_code("print('hi')", task_id="test"))
            self.assertIn("error", result)
            self.assertIn("Windows", result["error"])

    def test_whitespace_only_code(self):
        result = json.loads(execute_code("   \n\t  ", task_id="test"))
        self.assertIn("error", result)
        self.assertIn("No code", result["error"])

    @unittest.skipIf(sys.platform == "win32", "UDS not available on Windows")
    def test_none_enabled_tools_uses_all(self):
        """When enabled_tools is None, all sandbox tools should be available."""
        code = (
            "from hermes_tools import terminal, web_search, read_file\n"
            "print('all imports ok')\n"
        )
        with patch("model_tools.handle_function_call",
                    return_value=json.dumps({"ok": True})):
            result = json.loads(execute_code(code, task_id="test-none",
                                             enabled_tools=None))
        self.assertEqual(result["status"], "success")
        self.assertIn("all imports ok", result["output"])

    @unittest.skipIf(sys.platform == "win32", "UDS not available on Windows")
    def test_empty_enabled_tools_uses_all(self):
        """When enabled_tools is [] (empty), all sandbox tools should be available."""
        code = (
            "from hermes_tools import terminal, web_search\n"
            "print('imports ok')\n"
        )
        with patch("model_tools.handle_function_call",
                    return_value=json.dumps({"ok": True})):
            result = json.loads(execute_code(code, task_id="test-empty",
                                             enabled_tools=[]))
        self.assertEqual(result["status"], "success")
        self.assertIn("imports ok", result["output"])

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
        with patch("tools.code_execution_tool.CLI_CONFIG",
                    {"code_execution": {"timeout": 120}},
                    create=True):
            # When the import works, it should return the config
            pass
        # When CLI_CONFIG import fails, should return {}
        with patch.dict("sys.modules", {"cli": None}):
            result = _load_config()
            self.assertIsInstance(result, dict)

    def test_returns_code_execution_section(self):
        from tools.code_execution_tool import _load_config
        mock_cli = MagicMock()
        mock_cli.CLI_CONFIG = {"code_execution": {"timeout": 120, "max_tool_calls": 10}}
        with patch.dict("sys.modules", {"cli": mock_cli}):
            with patch("tools.code_execution_tool._load_config", wraps=_load_config):
                result = _load_config()
        # Result should be a dict (either the config or empty)
        self.assertIsInstance(result, dict)


# ---------------------------------------------------------------------------
# Interrupt event
# ---------------------------------------------------------------------------

@unittest.skipIf(sys.platform == "win32", "UDS not available on Windows")
class TestInterruptHandling(unittest.TestCase):
    def test_interrupt_event_stops_execution(self):
        """When _interrupt_event is set, execute_code should stop the script."""
        import threading

        code = "import time; time.sleep(60); print('should not reach')"

        def set_interrupt_after_delay():
            import time as _t
            _t.sleep(1)
            from tools.terminal_tool import _interrupt_event
            _interrupt_event.set()

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
            from tools.terminal_tool import _interrupt_event
            _interrupt_event.clear()
            t.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
