"""Tests for tools/workspace_tool.py — agent rebinding its own session's workspace."""

import json
import os
from unittest import mock

from tools.workspace_tool import (
    set_workspace_tool,
    check_workspace_requirements,
    SET_WORKSPACE_SCHEMA,
)


class TestSetWorkspaceTool:
    """set_workspace_tool — agent changes its session's working directory."""

    def test_no_callback_returns_error(self):
        result = json.loads(set_workspace_tool(cwd="/tmp"))
        assert "error" in result
        assert "only available" in result["error"].lower()

    def test_missing_cwd_returns_error(self):
        result = json.loads(set_workspace_tool(cwd=None, callback=lambda cwd: {}))
        assert "error" in result
        assert "required" in result["error"].lower()

    def test_empty_cwd_returns_error(self):
        result = json.loads(set_workspace_tool(cwd="  ", callback=lambda cwd: {}))
        assert "error" in result
        assert "required" in result["error"].lower()

    def test_callback_returns_cwd_and_branch(self):
        def cb(cwd):
            assert cwd == "/home/user/repo"
            return {"cwd": "/home/user/repo", "branch": "main"}

        result = json.loads(set_workspace_tool(cwd="/home/user/repo", callback=cb))
        assert result["cwd"] == "/home/user/repo"
        assert result["branch"] == "main"

    def test_callback_returning_error_dict_surfaces_error(self):
        def cb(cwd):
            return {"error": "no such directory: /nonexistent"}

        result = json.loads(set_workspace_tool(cwd="/nonexistent", callback=cb))
        assert "error" in result
        assert "no such directory" in result["error"]

    def test_callback_raising_returns_error(self):
        def cb(cwd):
            raise RuntimeError("gateway crashed")

        result = json.loads(set_workspace_tool(cwd="/tmp", callback=cb))
        assert "error" in result
        assert "gateway crashed" in result["error"]

    def test_callback_returning_falsy_returns_error(self):
        result = json.loads(set_workspace_tool(cwd="/tmp", callback=lambda cwd: None))
        assert "error" in result
        assert "failed" in result["error"].lower() or "timed out" in result["error"].lower()

    def test_cwd_stripped_before_passing_to_callback(self):
        calls = []

        def cb(cwd):
            calls.append(cwd)
            return {"cwd": cwd, "branch": ""}

        set_workspace_tool(cwd="  /tmp/test  ", callback=cb)
        assert calls == ["/tmp/test"]


class TestCheckWorkspaceRequirements:
    """check_workspace_requirements — desktop/TUI only."""

    def test_returns_true_when_hermes_desktop_is_1(self):
        with mock.patch.dict(os.environ, {"HERMES_DESKTOP": "1"}):
            assert check_workspace_requirements() is True

    def test_returns_true_when_hermes_desktop_is_true(self):
        with mock.patch.dict(os.environ, {"HERMES_DESKTOP": "true"}):
            assert check_workspace_requirements() is True

    def test_returns_false_when_hermes_desktop_is_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            assert check_workspace_requirements() is False

    def test_returns_false_when_hermes_desktop_is_0(self):
        with mock.patch.dict(os.environ, {"HERMES_DESKTOP": "0"}):
            assert check_workspace_requirements() is False


class TestSetWorkspaceSchema:
    """SET_WORKSPACE_SCHEMA — schema matches expected structure."""

    def test_schema_name_is_set_workspace(self):
        assert SET_WORKSPACE_SCHEMA["name"] == "set_workspace"

    def test_schema_requires_cwd_parameter(self):
        assert "cwd" in SET_WORKSPACE_SCHEMA["parameters"]["required"]
        assert "cwd" in SET_WORKSPACE_SCHEMA["parameters"]["properties"]

    def test_schema_has_description(self):
        assert "description" in SET_WORKSPACE_SCHEMA
        assert len(SET_WORKSPACE_SCHEMA["description"]) > 10
