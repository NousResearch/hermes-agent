"""config.set show_message_tokens — the persistent half of the TUI /tokens toggle.

Regression cover for the review finding on PR #55805: the Ink side
(``ui-tui/src/app/slash/commands/core.ts``) sends ``config.set`` with key
``show_message_tokens`` when the user runs ``/tokens always``, but the RPC had
no handler and fell through to ``unknown config key``. Because the caller
swallowed the rejection, ``/tokens always`` reported a persistence that never
happened and the preference silently vanished on restart.

The key writes ``display.show_message_tokens``, which is what
``useConfigSync``'s ``applyDisplay`` reads back into ``ui.showTokens``.
"""

from __future__ import annotations

import pytest

import tui_gateway.server as srv


def _call(params: dict) -> dict:
    """Invoke the config.set RPC and return the raw JSON-RPC envelope."""
    return srv._methods["config.set"](1, params)


@pytest.fixture
def cfg(monkeypatch):
    """Capture _write_config_key writes over a mutable in-memory config."""
    state: dict = {"display": {}}
    written: list[tuple[str, object]] = []

    def _fake_write(key_path: str, value):
        written.append((key_path, value))
        node = state
        parts = key_path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    monkeypatch.setattr(srv, "_load_cfg", lambda: state)
    monkeypatch.setattr(srv, "_write_config_key", _fake_write)
    return state, written


# ---------------------------------------------------------------------------
# The regression the review flagged
# ---------------------------------------------------------------------------


def test_key_is_not_rejected_as_unknown(cfg):
    """The exact payload the Ink /tokens always sends must be accepted."""
    env = _call({"key": "show_message_tokens", "value": "on"})
    assert "error" not in env, env
    assert env["result"]["key"] == "show_message_tokens"


def test_always_persists_under_display(cfg):
    """`/tokens always` -> display.show_message_tokens = True (what applyDisplay reads)."""
    state, written = cfg
    env = _call({"key": "show_message_tokens", "value": "on"})
    assert env["result"]["value"] == "on"
    assert ("display.show_message_tokens", True) in written
    assert state["display"]["show_message_tokens"] is True


# ---------------------------------------------------------------------------
# Value parsing — same grammar as the sibling boolean display keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["on", "true", "yes"])
def test_truthy_aliases(cfg, value):
    _, written = cfg
    assert _call({"key": "show_message_tokens", "value": value})["result"]["value"] == "on"
    assert written[-1] == ("display.show_message_tokens", True)


@pytest.mark.parametrize("value", ["off", "false", "no"])
def test_falsy_aliases(cfg, value):
    _, written = cfg
    assert _call({"key": "show_message_tokens", "value": value})["result"]["value"] == "off"
    assert written[-1] == ("display.show_message_tokens", False)


@pytest.mark.parametrize("value", ["", "toggle"])
def test_toggle_flips_current(cfg, value):
    state, written = cfg
    state["display"]["show_message_tokens"] = True
    assert _call({"key": "show_message_tokens", "value": value})["result"]["value"] == "off"
    assert written[-1] == ("display.show_message_tokens", False)


def test_toggle_from_absent_defaults_off_then_on(cfg):
    """No stored key reads as False, so a bare toggle turns it on."""
    _, written = cfg
    assert _call({"key": "show_message_tokens", "value": ""})["result"]["value"] == "on"
    assert written[-1] == ("display.show_message_tokens", True)


def test_invalid_value_errors_and_writes_nothing(cfg):
    _, written = cfg
    env = _call({"key": "show_message_tokens", "value": "sometimes"})
    assert env["error"]["code"] == 4002
    # Must be the value complaint, NOT the unknown-key fall-through — both
    # share code 4002, so assert on the message or this passes vacuously.
    assert env["error"]["message"] == "unknown show_message_tokens value: sometimes"
    assert written == []
