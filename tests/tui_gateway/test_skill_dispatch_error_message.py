"""Tests for skill command dispatch error messages (#43053).

When a skill is registered (shows in autocomplete) but fails to load at
dispatch time, the user should see a descriptive error — not the misleading
"not a quick/plugin/skill command" message.
"""

from unittest.mock import patch

import pytest


def _call_dispatch(name: str, arg: str = ""):
    """Call the ``command.dispatch`` handler and return the raw result dict.

    The handler is registered via ``@method("command.dispatch")`` so we
    import the module and invoke ``dispatch`` directly.  Returns the
    JSON-RPC envelope: ``{"jsonrpc": "2.0", "id": N, "result": ...}``
    on success or ``{"jsonrpc": "2.0", "id": N, "error": ...}`` on error.
    """
    from tui_gateway.server import dispatch

    rid = 1
    params = {"name": name, "arg": arg, "session_id": "test-sid"}
    req = {"id": rid, "method": "command.dispatch", "params": params}
    return dispatch(req)


# ── Happy-path: skill found and loaded ─────────────────────────────────


def test_skill_command_returns_skill_type():
    """When a skill is found AND loads, dispatch returns type=skill."""
    fake_skill_info = {
        "/nano-pdf": {
            "name": "nano-pdf",
            "description": "Edit PDFs",
            "skill_md_path": "/fake/SKILL.md",
            "skill_dir": "/fake",
        }
    }
    fake_message = "[Skill loaded: nano-pdf]"

    with patch(
        "agent.skill_commands.scan_skill_commands",
        return_value=fake_skill_info,
    ), patch(
        "agent.skill_commands.build_skill_invocation_message",
        return_value=fake_message,
    ):
        result = _call_dispatch("nano-pdf")

    assert "result" in result, f"Expected success but got: {result}"
    assert result["result"]["type"] == "skill"
    assert result["result"]["message"] == fake_message
    assert result["result"]["name"] == "nano-pdf"


# ── Bug #43053: skill found but failed to load ────────────────────────


def test_skill_found_but_load_fails_returns_descriptive_error():
    """When scan finds the skill but build_skill_invocation_message returns
    None, the user should see a descriptive error — NOT the generic
    'not a quick/plugin/skill command' message."""
    fake_skill_info = {
        "/nano-pdf": {
            "name": "nano-pdf",
            "description": "Edit PDFs",
            "skill_md_path": "/fake/SKILL.md",
            "skill_dir": "/fake",
        }
    }

    with patch(
        "agent.skill_commands.scan_skill_commands",
        return_value=fake_skill_info,
    ), patch(
        "agent.skill_commands.build_skill_invocation_message",
        return_value=None,
    ):
        result = _call_dispatch("nano-pdf")

    assert "error" in result, f"Expected error but got: {result}"
    err_msg = result["error"]["message"]
    # Must mention the skill name and explain the failure
    assert "nano-pdf" in err_msg
    assert "failed to load" in err_msg
    # Must NOT use the misleading generic message
    assert "not a quick/plugin/skill command" not in err_msg


def test_skill_load_exception_returns_descriptive_error():
    """When scan_skill_commands or build_skill_invocation_message raises,
    the error message should include the exception text."""
    with patch(
        "agent.skill_commands.scan_skill_commands",
        side_effect=RuntimeError("permission denied"),
    ):
        result = _call_dispatch("nano-pdf")

    assert "error" in result, f"Expected error but got: {result}"
    err_msg = result["error"]["message"]
    assert "permission denied" in err_msg
    assert "not a quick/plugin/skill command" not in err_msg


# ── Unregistered command still shows generic message ───────────────────


def test_unknown_command_still_shows_generic_message():
    """Commands that are NOT skills should still get the generic error."""
    with patch(
        "agent.skill_commands.scan_skill_commands",
        return_value={},
    ):
        result = _call_dispatch("nonexistent-command-xyz")

    assert "error" in result, f"Expected error but got: {result}"
    err_msg = result["error"]["message"]
    assert "not a quick/plugin/skill command" in err_msg
