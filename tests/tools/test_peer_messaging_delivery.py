"""Failed hints and explicit ACKs do not destroy unread coordination messages.

Supersedes the earlier consume-after-steer tests: steer now carries only a
fixed inbox hint, and even successful steering must not consume the envelope.
"""
import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def receive_env(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    import tools.peer_messaging_tool as pm

    monkeypatch.setattr(pm, "_inbox_root", lambda: tmp_path / "runtime" / "peer_messages")
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("target", source="cli")
    db.close()
    entry = pm.mailbox.enqueue(pm._inbox_dir("target"), sender="sender", target="target", text="do not edit api.py")
    agent = SimpleNamespace(session_id="target", valid_tool_names={"peer_receive"},
                            _peer_inbox_last_drain_mono=float("-inf"))
    return pm, agent, entry


@pytest.mark.parametrize("outcome", ["accepted", "rejected", "raised"])
def test_hint_never_consumes_and_restart_can_read_same_id(receive_env, outcome):
    pm, agent, entry = receive_env
    hints = []

    def steer(note):
        hints.append(note)
        if outcome == "raised":
            raise RuntimeError("hint admission unavailable")
        return outcome == "accepted"

    agent.steer = steer
    assert pm.inject_peer_messages(agent) is (outcome == "accepted")
    assert hints == [pm._INBOX_HINT]
    path = pm._inbox_dir("target") / f"{entry['message_id']}.json"
    assert path.exists()
    # No in-memory receipt is required to resume after a failed hint/result.
    for _ in range(2):
        received = json.loads(pm.peer_receive(current_session_id="target"))
        assert received["messages"] == [entry]
    acknowledged = json.loads(pm.peer_receive(action="ack", message_ids=[entry["message_id"]],
                                              current_session_id="target"))
    assert acknowledged["acknowledged"] == [entry["message_id"]]
    assert not path.exists()


@pytest.mark.parametrize("poison", [b"{", b"\xff", b"[]"])
def test_bad_payload_does_not_discard_unacknowledged_neighbor(receive_env, poison):
    pm, _, entry = receive_env
    bad = pm._inbox_dir("target") / "00000000000000000000_deadbeef.json"
    bad.write_bytes(poison)
    result = json.loads(pm.peer_receive(current_session_id="target"))
    assert result["messages"] == [entry]
    assert not bad.exists()
    assert (pm._inbox_dir("target") / f"{entry['message_id']}.json").exists()


def test_same_pending_batch_does_not_spam_each_activity_tick(receive_env):
    pm, agent, _ = receive_env
    hints = []
    agent.steer = lambda note: hints.append(note) or True
    assert pm.inject_peer_messages(agent) is True
    agent._peer_inbox_last_drain_mono = float("-inf")
    assert pm.inject_peer_messages(agent) is False
    pm.mailbox.enqueue(pm._inbox_dir("target"), sender="sender", target="target", text="new information")
    agent._peer_inbox_last_drain_mono = float("-inf")
    assert pm.inject_peer_messages(agent) is True
    assert hints == [pm._INBOX_HINT, pm._INBOX_HINT]
