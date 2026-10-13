"""``manage_connections`` action ``display``: the desktop directory card.

Contracts:
- on desktop the call emits a card with no targets, waits for Continue or the deadline, and reports what is
  connected, computed from the gateway list read at the start and at settle; the deadline adds timed_out
- without a card surface it returns at once with the status list and where to connect
"""

import json
from unittest.mock import patch

import pytest

from tools.connectors import live
from tools.connectors.run import apply_answer
from tools.connectors.tool import manage_connections


@pytest.fixture(autouse=True)
def _clean_live():
    live.reset_for_tests()
    yield
    live.reset_for_tests()


class ListFake:
    def __init__(self, connected):
        self.connected = set(connected)

    def list_connectors(self, *, timeout=None):
        return [{"connector": s, "connected": s in self.connected} for s in ("gmail", "linear", "slack")]


def _display(gw, *, callback, card_surface):
    return json.loads(manage_connections(
        {"action": "display", "connectors": ["slack", "linear"]}, client_factory=lambda: gw,
        connection_callback=callback, card_surface=card_surface, session_id="s1",
    ))


@pytest.mark.parametrize("ending", ["continue", "deadline"])
def test_display_waits_for_the_user_and_reports_what_they_connected(ending):
    gw = ListFake({"gmail"})
    seen = []

    def card(payload):
        # The user connects Slack while the card is open, then presses Continue or walks away.
        seen.append(payload)
        gw.connected.add("slack")
        if ending == "continue":
            apply_answer(live.get("s1", payload["op_id"]), json.dumps({"settled_by": "continue"}))

    with patch("tools.connectors.operation.OPERATION_DEADLINE_SECONDS", 2 if ending == "deadline" else 60):
        out = _display(gw, callback=card, card_surface=True)

    (payload,) = seen
    assert payload["targets"] == [] and payload["display"] == {"suggested": ["slack", "linear"]}
    expected = {"connected": ["gmail", "slack"], "newly_connected": ["slack"], "suggested": ["slack", "linear"]}
    assert out == ({**expected, "timed_out": True} if ending == "deadline" else expected)
    assert live.current("s1") is None


def test_display_without_a_card_returns_the_status_list_at_once():
    out = _display(ListFake({"gmail"}), callback=lambda _payload: None, card_surface=False)

    assert {row["connector"]: row["connected"] for row in out["connectors"]}["gmail"] is True
    assert "desktop app" in out["note"]
    assert live.current("s1") is None
