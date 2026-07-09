"""Tests for compact Matrix tools (tools/matrix_tool.py)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import matrix_tool as mt
from tools.matrix_tool import (
    _ADMIN_ACTIONS,
    _CORE_ACTIONS,
    _authorize_room_id,
    _gate,
    check_matrix_tool_requirements,
)


@pytest.fixture(autouse=True)
def _clear_matrix_env(monkeypatch):
    for key in (
        "MATRIX_TOOLS_ALLOW_CROSS_ROOM",
        "MATRIX_TOOLS_ALLOW_CROSS_ROOM_DESTRUCTIVE",
        "MATRIX_TOOLS_ALLOW_REDACTION",
        "MATRIX_TOOLS_ALLOW_INVITES",
        "MATRIX_TOOLS_ALLOW_ROOM_CREATE",
        "MATRIX_ALLOW_PUBLIC_ROOMS",
        "MATRIX_ALLOWED_ROOMS",
        "MATRIX_ACCESS_TOKEN",
        "MATRIX_HOMESERVER",
    ):
        monkeypatch.delenv(key, raising=False)
    yield


class TestMatrixToolRequirements:
    def test_session_platform_matrix(self):
        with patch(
            "tools.matrix_tool.get_session_env",
            side_effect=lambda k, d="": "matrix" if k == "HERMES_SESSION_PLATFORM" else d,
        ):
            assert check_matrix_tool_requirements() is True

    def test_token_and_homeserver(self, monkeypatch):
        with patch("tools.matrix_tool.get_session_env", return_value=""):
            monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_x")
            monkeypatch.setenv("MATRIX_HOMESERVER", "https://example.org")
            assert check_matrix_tool_requirements() is True

    def test_missing(self):
        with patch("tools.matrix_tool.get_session_env", return_value=""):
            assert check_matrix_tool_requirements() is False


class TestAuthorizeRoom:
    def test_defaults_to_session_room(self):
        with patch(
            "tools.matrix_tool.get_session_env",
            side_effect=lambda k, d="": "!cur:example.org" if k == "HERMES_SESSION_CHAT_ID" else d,
        ), patch("tools.matrix_tool._matrix_tools_cfg", return_value={}):
            room, err = _authorize_room_id("")
            assert err == ""
            assert room == "!cur:example.org"

    def test_cross_room_denied_by_default(self):
        with patch(
            "tools.matrix_tool.get_session_env",
            side_effect=lambda k, d="": "!cur:example.org" if k == "HERMES_SESSION_CHAT_ID" else d,
        ), patch("tools.matrix_tool._matrix_tools_cfg", return_value={}):
            room, err = _authorize_room_id("!other:example.org")
            assert room == ""
            assert "current room" in err

    def test_cross_room_allowed_via_config(self):
        with patch(
            "tools.matrix_tool.get_session_env",
            side_effect=lambda k, d="": "!cur:example.org" if k == "HERMES_SESSION_CHAT_ID" else d,
        ), patch(
            "tools.matrix_tool._matrix_tools_cfg",
            return_value={"allow_cross_room": True},
        ):
            room, err = _authorize_room_id("!other:example.org")
            assert err == ""
            assert room == "!other:example.org"

    def test_destructive_cross_room_needs_extra_gate(self):
        with patch(
            "tools.matrix_tool.get_session_env",
            side_effect=lambda k, d="": "!cur:example.org" if k == "HERMES_SESSION_CHAT_ID" else d,
        ), patch(
            "tools.matrix_tool._matrix_tools_cfg",
            return_value={"allow_cross_room": True},
        ):
            room, err = _authorize_room_id("!other:example.org", destructive=True)
            assert room == ""
            assert "allow_cross_room_destructive" in err


class TestCoreAndAdminActions:
    def test_send_reaction(self):
        adapter = MagicMock()
        adapter._send_reaction = AsyncMock(return_value="$rxn")
        with patch("tools.matrix_tool._matrix_adapter", return_value=(adapter, "")), patch(
            "tools.matrix_tool.get_session_env",
            side_effect=lambda k, d="": {
                "HERMES_SESSION_PLATFORM": "matrix",
                "HERMES_SESSION_CHAT_ID": "!room:example.org",
            }.get(k, d),
        ), patch("tools.matrix_tool._matrix_tools_cfg", return_value={}), patch(
            "tools.matrix_tool._run", return_value="$rxn"
        ):
            out = mt._run_matrix_action(
                "send_reaction",
                _CORE_ACTIONS,
                "matrix",
                event_id="$evt",
                emoji="✅",
            )
        data = json.loads(out)
        assert data.get("success") is True

    def test_redaction_gated(self):
        with patch("tools.matrix_tool._matrix_tools_cfg", return_value={}), patch(
            "tools.matrix_tool.get_session_env",
            side_effect=lambda k, d="": {
                "HERMES_SESSION_PLATFORM": "matrix",
                "HERMES_SESSION_CHAT_ID": "!room:example.org",
            }.get(k, d),
        ):
            out = mt._run_matrix_action(
                "redact_message",
                _ADMIN_ACTIONS,
                "matrix_admin",
                event_id="$evt",
            )
        assert "allow_redaction" in out

    def test_redaction_enabled_via_config(self):
        adapter = MagicMock()
        adapter.redact_message = AsyncMock(return_value=True)
        with patch("tools.matrix_tool._matrix_adapter", return_value=(adapter, "")), patch(
            "tools.matrix_tool.get_session_env",
            side_effect=lambda k, d="": {
                "HERMES_SESSION_PLATFORM": "matrix",
                "HERMES_SESSION_CHAT_ID": "!room:example.org",
            }.get(k, d),
        ), patch(
            "tools.matrix_tool._matrix_tools_cfg",
            return_value={"allow_redaction": True},
        ), patch("tools.matrix_tool._run", return_value=True):
            out = mt._run_matrix_action(
                "redact_message",
                _ADMIN_ACTIONS,
                "matrix_admin",
                event_id="$evt",
                reason="spam",
            )
        data = json.loads(out)
        assert data.get("success") is True


class TestGatePrecedence:
    def test_yaml_overrides_env(self, monkeypatch):
        monkeypatch.setenv("MATRIX_TOOLS_ALLOW_REDACTION", "true")
        with patch(
            "tools.matrix_tool._matrix_tools_cfg",
            return_value={"allow_redaction": False},
        ):
            assert _gate("allow_redaction", "MATRIX_TOOLS_ALLOW_REDACTION", False) is False

    def test_env_when_yaml_absent(self, monkeypatch):
        monkeypatch.setenv("MATRIX_TOOLS_ALLOW_REDACTION", "true")
        with patch("tools.matrix_tool._matrix_tools_cfg", return_value={}):
            assert _gate("allow_redaction", "MATRIX_TOOLS_ALLOW_REDACTION", False) is True


class TestToolsetsWiring:
    def test_toolsets_define_matrix_tools(self):
        from toolsets import TOOLSETS

        assert "matrix" in TOOLSETS
        assert TOOLSETS["matrix"]["tools"] == ["matrix"]
        assert "matrix_admin" in TOOLSETS
        assert TOOLSETS["matrix_admin"]["tools"] == ["matrix_admin"]
        hm = TOOLSETS["hermes-matrix"]["tools"]
        assert "matrix" in hm
        assert "matrix_admin" in hm

    def test_default_config_has_tools_gates(self):
        from hermes_cli.config import DEFAULT_CONFIG

        tools = DEFAULT_CONFIG["matrix"]["tools"]
        assert tools["allow_redaction"] is False
        assert tools["allow_public_rooms"] is False


class TestGateParsing:
    def test_string_false_is_false(self):
        with patch(
            "tools.matrix_tool._matrix_tools_cfg",
            return_value={"allow_redaction": "false"},
        ):
            assert _gate("allow_redaction", "MATRIX_TOOLS_ALLOW_REDACTION", False) is False

    def test_string_true_is_true(self):
        with patch(
            "tools.matrix_tool._matrix_tools_cfg",
            return_value={"allow_redaction": "true"},
        ):
            assert _gate("allow_redaction", "MATRIX_TOOLS_ALLOW_REDACTION", False) is True
