"""A plugin slash command may answer with a UI directive instead of text.

``slash.exec`` has to hand that directive to the client verbatim: the Desktop/TUI dispatch
``send``/``skill`` directives (submit ``message`` as a prompt, render ``display``) exactly like
``command.dispatch`` results. Stringifying the handler's answer is what left a plugin's skill
commands printing their prompt into the transcript instead of running.
"""

from __future__ import annotations

from unittest.mock import patch

from tui_gateway import server


def _slash_exec(handler, command: str = "ponytail-audit", sid: str = "plugin-directive-sid"):
    """``slash.exec`` for *command*, with *handler* standing in for the plugin's registered one."""
    server._sessions[sid] = {"session_key": sid, "agent": None}
    try:
        with patch("hermes_cli.plugins.get_plugin_command_handler", return_value=handler):
            return server.handle_request({
                "id": "r1",
                "method": "slash.exec",
                "params": {"command": command, "session_id": sid},
            })
    finally:
        server._sessions.pop(sid, None)


def test_plugin_directive_reaches_the_client_verbatim():
    """The handler's directive is the result — not text wrapped in ``output``."""
    directive = {
        "type": "skill",
        "message": "Load and follow the Hermes plugin skill `ponytail:ponytail-audit`. Audit the repo.",
        "name": "ponytail-audit",
        "display": "/ponytail-audit",
    }

    resp = _slash_exec(lambda _arg: dict(directive))

    assert resp["result"] == directive


def test_plugin_text_result_is_unchanged():
    """Plain text still arrives as ``output`` (every existing plugin keeps working)."""
    resp = _slash_exec(lambda _arg: "Ponytail mode: full. Use `/ponytail lite|full|ultra|off`.")

    assert resp["result"] == {"output": "Ponytail mode: full. Use `/ponytail lite|full|ultra|off`."}


def test_incomplete_directive_is_not_a_directive():
    """No payload, no directive: an empty message must never submit an empty prompt."""
    resp = _slash_exec(lambda _arg: {"type": "skill", "message": "   "})

    assert set(resp["result"]) == {"output"}