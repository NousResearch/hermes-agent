"""A delivered /steer reaches Desktop as ``steer.delivered`` so it can seal the live reply there."""

import pytest

from tui_gateway import server


def test_agent_callbacks_emit_steer_delivered_for_the_session(monkeypatch):
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emitted.append((event, sid, payload)))

    server._agent_cbs("s1")["steer_delivered_callback"](["prefer B", "keep it short"])

    assert emitted == [("steer.delivered", "s1", {"count": 2, "texts": ["prefer B", "keep it short"]})]


def test_steer_delivered_payload_names_the_drained_submissions():
    from tui_gateway.contracts.registry import EVENTS

    payload = EVENTS["steer.delivered"].payload.model_validate({"count": 2, "texts": ["a", "b"]})
    assert (payload.count, payload.texts) == (2, ["a", "b"])


class _SteerAgent:
    def __init__(self, announces: bool):
        if announces:
            self.steer_delivered_callback = lambda texts: None

    def steer(self, text):
        return True


@pytest.mark.parametrize("announces", [True, False])
def test_an_accepted_steer_says_whether_its_delivery_will_be_announced(announces):
    """Desktop holds the steer under the live reply only when ``steer.delivered`` will come;
    an agent without the sink keeps the client's seal-at-arrival behaviour."""
    from tests.tui_gateway.test_tui_gateway_server import _session

    server._sessions["sid"] = _session(agent=_SteerAgent(announces), running=True)
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.steer", "params": {"session_id": "sid", "text": "prefer B"}}
        )
    finally:
        server._sessions.pop("sid", None)

    assert resp["result"]["status"] == "queued"
    assert resp["result"].get("delivery_event") is (True if announces else None)
