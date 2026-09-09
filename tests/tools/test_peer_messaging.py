"""Peer session messaging (tools/peer_messaging_tool.py) — invariant tests.

1. End-to-end mailbox contract: a message sent to a known session via the registry
   handler is drained by ``inject_peer_messages`` into the target agent's steer
   channel with peer provenance declared, and the inbox file is consumed.
2. Trust/scope contract: sending to an unknown session id fails without creating a
   mailbox, and a session with no inbox drains nothing.
"""

import json
import time

import pytest


@pytest.fixture
def peer_env(tmp_path, monkeypatch):
    """Temp HERMES_HOME with a real SessionDB containing one live target session."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_state import SessionDB
    db = SessionDB(db_path=home / "state.db")
    db.create_session("target-session-1", source="cli")
    db.close()
    return home


class _StubAgent:
    def __init__(self, session_id):
        self.session_id = session_id
        self.steers = []

    def steer(self, text):
        self.steers.append(text)
        return True


def _dispatch(name, args, session_id):
    from tools.registry import registry
    return json.loads(registry.dispatch(name, args, session_id=session_id))


def test_peer_send_delivers_via_steer_and_consumes_inbox(peer_env):
    import tools.peer_messaging_tool as pm

    result = _dispatch(
        "peer_send",
        {"target_session_id": "target-session-1", "message": "heads up: schema changed"},
        session_id="sender-session",
    )
    assert result.get("success") is True

    inbox = pm._inbox_dir("target-session-1")
    assert [p for p in inbox.iterdir() if p.suffix == ".json"], "message file must be queued"

    agent = _StubAgent("target-session-1")
    assert pm.inject_peer_messages(agent) is True
    delivered = "\n".join(agent.steers)
    # Provenance: attributed to the sender session, not the user.
    assert "sender-session" in delivered
    assert "heads up: schema changed" in delivered
    assert "user authority" in delivered  # explicit non-user provenance framing
    # Consumed: nothing left to deliver, and the rate limit halts an immediate re-poll.
    assert not [p for p in inbox.iterdir() if p.suffix == ".json"]
    agent2 = _StubAgent("target-session-1")
    agent2._peer_inbox_last_drain_mono = time.monotonic()
    assert pm.inject_peer_messages(agent2) is False


def test_unknown_target_rejected_and_no_inbox_is_a_noop(peer_env):
    import tools.peer_messaging_tool as pm

    result = _dispatch(
        "peer_send",
        {"target_session_id": "no-such-session", "message": "hello"},
        session_id="sender-session",
    )
    assert result.get("success") is False or "error" in result
    assert not pm._inbox_dir("no-such-session").exists(), "failed send must not create a mailbox"

    # A session nobody messaged drains nothing and never steers.
    agent = _StubAgent("never-messaged")
    assert pm.inject_peer_messages(agent) is False
    assert agent.steers == []
