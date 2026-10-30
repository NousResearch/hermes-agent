"""The live adviser event is for plugin rows only.

``plugin_message_callback`` also fires when a mid-turn steer is delivered at a tool
boundary. Those rows carry ``display_kind="steer"`` and are the operator's words, so
the gateway must not announce them as plugin advice.
"""

from __future__ import annotations

from unittest.mock import patch

from tui_gateway import agent_callbacks


def _announce(emitted: list[tuple], row) -> None:
    with patch.object(agent_callbacks, "_emit", create=True,
                      side_effect=lambda *args: emitted.append(args)):
        agent_callbacks._announce_plugin_row("test-session", row)


def test_adviser_row_emits_message_adviser():
    emitted: list[tuple] = []
    _announce(emitted, {
        "content": "[adviser — advisory only]\n- [blocker · reviewer] Stop the migration.\n[end adviser]",
        "display_kind": "adviser",
        "plugin_message_id": "abc123",
        "role": "user",
    })

    assert len(emitted) == 1
    event_type, sid, payload = emitted[0]
    assert event_type == "message.adviser"
    assert sid == "test-session"
    assert payload["id"] == "abc123"
    assert payload["severity"] == "blocker"
    assert payload["label"] == "blocker · reviewer"


def test_plain_steer_row_is_not_announced_as_adviser_guidance():
    emitted: list[tuple] = []
    _announce(emitted, {
        "content": "[steer] Ship the smaller patch.",
        "display_kind": "steer",
        "role": "user",
    })

    assert emitted == []


def test_non_dict_row_is_ignored():
    emitted: list[tuple] = []
    _announce(emitted, None)

    assert emitted == []
