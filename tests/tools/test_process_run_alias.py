"""process(action='run') should redirect models to the terminal tool."""

import json

import pytest


def test_process_run_action_hints_terminal_tool():
    from tools.process_registry import _handle_process

    raw = _handle_process({"action": "run", "command": "pwd"})
    assert isinstance(raw, str)
    # tool_error wraps JSON payload
    try:
        payload = json.loads(raw)
        text = json.dumps(payload)
    except Exception:
        text = raw
    assert "run" in text.lower()
    assert "terminal" in text.lower()
    assert "Unknown process action" not in text


def test_process_execute_alias_same_hint():
    from tools.process_registry import _handle_process

    raw = _handle_process({"action": "EXECUTE"})
    text = raw if isinstance(raw, str) else json.dumps(raw)
    try:
        text = json.dumps(json.loads(raw))
    except Exception:
        pass
    assert "terminal" in text.lower()


def test_unknown_action_still_lists_valid():
    from tools.process_registry import _handle_process

    raw = _handle_process({"action": "frobnicate"})
    text = raw if isinstance(raw, str) else str(raw)
    assert "Unknown process action" in text
    assert "list" in text
