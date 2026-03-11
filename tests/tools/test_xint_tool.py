"""Tests for tools/xint_tool.py."""

import json
import subprocess
from unittest.mock import Mock, patch

from tools.xint_tool import (
    check_xint_requirements,
    xint_tool,
    _find_xint_executable,
    XINT_SCHEMA,
)


class TestExecutableResolution:
    def test_prefers_xint_rs_in_auto_mode(self):
        with patch("tools.xint_tool.shutil.which") as mock_which:
            mock_which.side_effect = lambda name: {
                "xint-rs": "/usr/local/bin/xint-rs",
                "xint": "/usr/local/bin/xint",
            }.get(name)
            assert _find_xint_executable("auto") == "/usr/local/bin/xint-rs"

    def test_falls_back_to_xint_when_rs_missing(self):
        with patch("tools.xint_tool.shutil.which") as mock_which:
            mock_which.side_effect = lambda name: {
                "xint-rs": None,
                "xint": "/usr/local/bin/xint",
            }.get(name)
            assert _find_xint_executable("auto") == "/usr/local/bin/xint"

    def test_requirement_check_false_when_missing(self):
        with patch("tools.xint_tool.shutil.which", return_value=None):
            assert check_xint_requirements() is False


class TestXintTool:
    def test_returns_install_error_when_missing(self):
        with patch("tools.xint_tool.shutil.which", return_value=None):
            out = json.loads(xint_tool(action="help"))
            assert out["success"] is False
            assert "unavailable" in out["error"].lower()
            assert "install_refs" in out

    def test_executes_successfully(self):
        completed = Mock()
        completed.returncode = 0
        completed.stdout = '{"ok":true}'
        completed.stderr = ""

        with patch("tools.xint_tool.shutil.which", side_effect=lambda n: "/usr/local/bin/xint-rs" if n == "xint-rs" else None), \
             patch("tools.xint_tool.subprocess.run", return_value=completed) as mock_run:
            out = json.loads(xint_tool(action="health", args=["--json"], parse_json=True))

        assert out["success"] is True
        assert out["exit_code"] == 0
        assert out["parsed_json"] == {"ok": True}
        mock_run.assert_called_once()
        called_cmd = mock_run.call_args.kwargs
        assert called_cmd["timeout"] == 120

    def test_timeout_error(self):
        with patch("tools.xint_tool.shutil.which", side_effect=lambda n: "/usr/local/bin/xint-rs" if n == "xint-rs" else None), \
             patch("tools.xint_tool.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["xint-rs"], timeout=3)):
            out = json.loads(xint_tool(action="search", timeout=3))

        assert out["success"] is False
        assert "timed out" in out["error"].lower()

    def test_rejects_non_list_args(self):
        with patch("tools.xint_tool.shutil.which", return_value="/usr/local/bin/xint-rs"):
            out = json.loads(xint_tool(action="search", args="--json"))  # type: ignore[arg-type]
            assert out["success"] is False
            assert "array" in out["error"].lower()


class TestXintSchema:
    def test_schema_name(self):
        assert XINT_SCHEMA["name"] == "xint"

    def test_schema_requires_action(self):
        assert "action" in XINT_SCHEMA["parameters"]["required"]
