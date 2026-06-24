"""Tests for the ``/learn`` slash command in the TUI gateway.

``/learn`` must be listed in ``_PENDING_INPUT_COMMANDS`` because the CLI
handler (``_handle_learn_command``) queues the learn prompt on
``_pending_input`` — a queue that the slash-worker subprocess does **not**
consume.  Routing through ``command.dispatch`` returns ``{"type": "send",
"message": …}`` so the frontend can submit the prompt directly.
"""

import importlib

import pytest

server = importlib.import_module("tui_gateway.server")


def test_learn_in_pending_input_commands():
    """Registry sanity: /learn must be in _PENDING_INPUT_COMMANDS so
    slash.exec routes it to command.dispatch instead of the worker."""
    assert "learn" in server._PENDING_INPUT_COMMANDS


def test_command_dispatch_learn_returns_send_type():
    """command.dispatch for /learn returns a ``send`` payload with a message
    that the frontend can submit as a user prompt."""
    resp = server._methods["command.dispatch"](
        "test-rid",
        {"name": "learn", "arg": "create a skill from https://example.com", "session_id": ""},
    )
    assert "result" in resp, f"Expected JSON-RPC result, got: {resp}"
    payload = resp["result"]
    assert payload["type"] == "send"
    assert isinstance(payload["message"], str)
    assert len(payload["message"]) > 0
    assert "skill" in payload["message"].lower() or "learn" in payload["message"].lower()


def test_command_dispatch_learn_no_arg():
    """command.dispatch for /learn with no argument still returns a valid
    send payload (the prompt describes learning from the current conversation)."""
    resp = server._methods["command.dispatch"](
        "test-rid",
        {"name": "learn", "arg": "", "session_id": ""},
    )
    assert "result" in resp, f"Expected JSON-RPC result, got: {resp}"
    payload = resp["result"]
    assert payload["type"] == "send"
    assert isinstance(payload["message"], str)
    assert len(payload["message"]) > 0
