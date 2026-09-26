"""Retryable receive failures must not consume peer-session messages.

The registry/SessionDB round trip is covered by test_peer_messaging.py. These
receive-side tests use real inbox files and inject failures at the read/steer
boundaries; no new delivery or acknowledgement protocol is assumed.
"""

import json
from pathlib import Path

import pytest


class _ReceivingAgent:
    session_id = "target-session"

    def __init__(self):
        self.steers = []
        self._peer_inbox_last_drain_mono = float("-inf")

    def steer(self, text):
        self.steers.append(text)
        return True


@pytest.fixture
def inbox_env(tmp_path, monkeypatch):
    import tools.peer_messaging_tool as pm

    monkeypatch.setattr(pm, "_inbox_root", lambda: tmp_path)
    inbox = pm._inbox_dir(_ReceivingAgent.session_id)
    inbox.mkdir()
    message = inbox / "001-valid.json"
    message.write_text(json.dumps({
        "from_session_id": "sender-session",
        "message": "I am changing the schema; please leave it alone.",
    }), encoding="utf-8")
    return pm, inbox, message


@pytest.mark.parametrize("failure", ["read", "rejected", "raised"])
def test_receive_failure_retains_message_for_retry(inbox_env, monkeypatch, failure):
    pm, _, message = inbox_env
    original = message.read_bytes()
    agent = _ReceivingAgent()

    with monkeypatch.context() as patch:
        if failure == "read":
            read_text = Path.read_text

            def unavailable(path, *args, **kwargs):
                if path == message:
                    raise PermissionError("temporary inbox read failure")
                return read_text(path, *args, **kwargs)

            patch.setattr(Path, "read_text", unavailable)
        else:
            def refuse(text):
                if failure == "raised":
                    raise RuntimeError("steer unavailable before acceptance")
                return False

            patch.setattr(agent, "steer", refuse)
        assert pm.inject_peer_messages(agent) is False

    assert message.exists(), "retryable failures must leave the message queued"
    assert message.read_bytes() == original
    assert agent.steers == []

    agent._peer_inbox_last_drain_mono = float("-inf")
    assert pm.inject_peer_messages(agent) is True
    assert len(agent.steers) == 1
    assert "sender-session" in agent.steers[0]
    assert json.loads(original)["message"] in agent.steers[0]
    assert not message.exists()
    # A fresh receiver must not redeliver a successfully consumed file.
    assert pm.inject_peer_messages(_ReceivingAgent()) is False


@pytest.mark.parametrize("poison", [b"{", b"\xff", b"[]"])
def test_bad_payload_does_not_discard_a_retryable_neighbor(inbox_env, monkeypatch, poison):
    pm, inbox, message = inbox_env
    malformed = inbox / "000-invalid.json"
    malformed.write_bytes(poison)
    agent = _ReceivingAgent()

    with monkeypatch.context() as patch:
        patch.setattr(agent, "steer", lambda text: False)
        assert pm.inject_peer_messages(agent) is False

    assert not malformed.exists(), "invalid payloads must not wedge the inbox"
    assert message.exists(), "a valid neighbor still needs successful acceptance"
    agent._peer_inbox_last_drain_mono = float("-inf")
    assert pm.inject_peer_messages(agent) is True
    assert len(agent.steers) == 1
    assert not message.exists()
