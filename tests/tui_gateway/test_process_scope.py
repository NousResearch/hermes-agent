"""``process.list`` / ``process.kill`` scoping across gateway processes.

A ``terminal(background=true)`` job started for a Telegram conversation runs in
the messaging gateway's own ``ProcessRegistry``. Desktop talks to a different
gateway process, so its ``process.list`` used to see nothing: the local registry
was filtered by an exact ``session_key`` match against a runtime row that never
existed here.

Contract pinned below:

1. the handler imports peers from the shared checkpoint before answering,
2. ownership is judged on the DURABLE conversation identity as well as the
   runtime key, so an unrelated conversation still sees nothing,
3. when the ephemeral runtime row is gone the handler may fall back to the
   durable id — but only for the expected ``4001 session not found`` error, and
   never for an empty or control-character id,
4. ``process.kill`` uses exactly the same ownership rule.
"""

from __future__ import annotations

import json
import os
import time
from unittest.mock import patch

import pytest

from gateway.status import get_process_start_time
from tools.process_registry import ProcessRegistry, ProcessSession
import tui_gateway.server as server

RID = "rid-1"
DURABLE_ID = "20260901_120000_abcdef"
LIVE_PID = os.getpid()


@pytest.fixture()
def registry(tmp_path):
    """A private registry + checkpoint standing in for the process-wide pair."""
    reg = ProcessRegistry()
    with patch("tools.process_registry.process_registry", reg), \
            patch("tools.process_registry.CHECKPOINT_PATH", tmp_path / "processes.json"):
        reg.checkpoint_path = tmp_path / "processes.json"
        yield reg


def _track(registry, **kw) -> ProcessSession:
    session = ProcessSession(id=kw.pop("id", "proc_local1234"), started_at=time.time(), **kw)
    registry._running[session.id] = session
    return session


def _publish_peer(registry, **overrides) -> None:
    entry = {
        "session_id": "proc_peerabc12345",
        "command": "npm run dev",
        "pid": LIVE_PID,
        "pid_scope": "host",
        "host_start_time": get_process_start_time(LIVE_PID),
        "started_at": time.time(),
        "task_id": "t-telegram",
        "owner_task_id": "t-telegram",
        "session_key": "telegram:4242",
        "parent_session_id": DURABLE_ID,
        "output_tail": "listening on :3000\n",
    }
    entry.update(overrides)
    registry.checkpoint_path.write_text(json.dumps([entry]), encoding="utf-8")


def _list(params: dict) -> dict:
    return server._methods["process.list"](RID, params)


def _kill(params: dict) -> dict:
    return server._methods["process.kill"](RID, params)


def _ids(reply: dict) -> set:
    return {row["session_id"] for row in reply["result"]["processes"]}


class TestLiveRuntimeScope:
    def test_own_processes_still_listed_with_an_output_tail(self, registry):
        _track(registry, session_key="runtime-key", command="npm test", output_buffer="ok\n")
        with patch.dict(server._sessions, {"s1": {"session_key": "runtime-key"}}, clear=True):
            reply = _list({"session_id": "s1"})

        row = reply["result"]["processes"][0]
        assert row["session_id"] == "proc_local1234"
        assert row["output_tail"] == "ok\n"

    def test_peer_gateway_process_is_discovered_for_the_same_conversation(self, registry):
        _publish_peer(registry)
        # Desktop resumed the stored conversation, so its runtime session_key IS
        # the durable id the peer stamped as parent_session_id.
        with patch.dict(server._sessions, {"s1": {"session_key": DURABLE_ID}}, clear=True):
            reply = _list({"session_id": "s1"})

        row = next(r for r in reply["result"]["processes"] if r["session_id"] == "proc_peerabc12345")
        assert row["status"] == "running"
        assert row["output_tail"] == "listening on :3000\n"
        assert row["peer"] is True

    def test_a_session_homed_on_another_profile_gets_no_peer_rows(self, registry, tmp_path):
        """``CHECKPOINT_PATH`` resolves once, against the launch profile. Rather than
        serve one profile's process registry to another, the peer import is skipped —
        profiles are independent islands, and local rows are unaffected."""
        _publish_peer(registry)
        _track(registry, session_key=DURABLE_ID, command="npm test")
        session = {"session_key": DURABLE_ID, "profile_home": str(tmp_path / "profiles" / "work")}
        with patch.dict(server._sessions, {"s1": session}, clear=True):
            reply = _list({"session_id": "s1"})

        assert _ids(reply) == {"proc_local1234"}

    def test_a_launch_profile_session_still_imports_peers(self, registry):
        _publish_peer(registry)
        with patch.dict(server._sessions, {"s1": {"session_key": DURABLE_ID, "profile_home": None}}, clear=True):
            reply = _list({"session_id": "s1"})

        assert _ids(reply) == {"proc_peerabc12345"}

    def test_unrelated_conversation_sees_nothing(self, registry):
        _publish_peer(registry)
        _track(registry, session_key="telegram:4242", command="npm test")
        with patch.dict(server._sessions, {"s2": {"session_key": "20260101_000000_other"}}, clear=True):
            reply = _list({"session_id": "s2"})

        assert _ids(reply) == set()


class _StoredSessions:
    """Minimal state.db stand-in: only these ids are stored conversations."""

    def __init__(self, *ids: str) -> None:
        self._ids = set(ids)

    def get_session(self, session_id: str):
        return {"id": session_id} if session_id in self._ids else None


class TestDurableFallback:
    def test_durable_id_resolves_when_the_runtime_row_is_gone(self, registry):
        _publish_peer(registry)
        with patch.dict(server._sessions, {}, clear=True), \
                patch.object(server, "_get_db", return_value=_StoredSessions(DURABLE_ID)):
            reply = _list({"session_id": DURABLE_ID})

        assert _ids(reply) == {"proc_peerabc12345"}

    def test_fallback_does_not_leak_another_conversations_processes(self, registry):
        _publish_peer(registry)
        other = "20260101_000000_other"
        with patch.dict(server._sessions, {}, clear=True), \
                patch.object(server, "_get_db", return_value=_StoredSessions(DURABLE_ID, other)):
            reply = _list({"session_id": other})

        assert _ids(reply) == set()

    def test_a_dead_runtime_id_still_gets_the_session_not_found_verdict(self, registry):
        """The desktop's gone-latch and passive session heal hang off this 4001;
        a reaped runtime id must never be reinterpreted as a durable scope."""
        _publish_peer(registry)
        with patch.dict(server._sessions, {}, clear=True), \
                patch.object(server, "_get_db", return_value=_StoredSessions(DURABLE_ID)):
            reply = _list({"session_id": "a1b2c3d4"})

        assert reply["error"]["code"] == 4001
        assert "result" not in reply

    @pytest.mark.parametrize("bad", ["", "   ", "\x00", "sess\x07id", "sess\nid"])
    def test_empty_and_control_character_ids_are_rejected(self, registry, bad):
        _publish_peer(registry, parent_session_id=bad, session_key=bad)
        # Even if such an id somehow named a stored row, it never reaches the lookup.
        with patch.dict(server._sessions, {}, clear=True), \
                patch.object(server, "_get_db", return_value=_StoredSessions(bad, bad.strip())):
            reply = _list({"session_id": bad})

        assert reply["error"]["code"] == 4001

    def test_only_session_not_found_opens_the_fallback(self, registry):
        _publish_peer(registry)
        other = server._err(RID, 5032, "agent initialization timed out")
        with patch.object(server, "_sess", return_value=(None, other)), \
                patch.object(server, "_get_db", return_value=_StoredSessions(DURABLE_ID)):
            reply = _list({"session_id": DURABLE_ID})

        # A build failure is not permission to answer from the durable scope.
        assert reply["error"]["code"] == 5032
        assert "result" not in reply


class TestKillScope:
    def test_kill_rejects_a_process_owned_by_another_conversation(self, registry):
        _track(registry, session_key="telegram:9999", command="npm run dev")
        with patch.dict(server._sessions, {"s1": {"session_key": DURABLE_ID}}, clear=True):
            reply = _kill({"session_id": "s1", "process_id": "proc_local1234"})

        assert reply["error"]["code"] == 4044

    def test_kill_accepts_a_process_matched_by_durable_identity(self, registry):
        _track(registry, session_key="telegram:4242", parent_session_id=DURABLE_ID,
               command="npm run dev", exited=True, exit_code=0)
        with patch.dict(server._sessions, {"s1": {"session_key": DURABLE_ID}}, clear=True):
            reply = _kill({"session_id": "s1", "process_id": "proc_local1234"})

        assert reply["result"]["status"] == "already_exited"

    def test_kill_of_a_peer_mirror_never_signals_the_foreign_pid(self, registry):
        _publish_peer(registry)
        with patch.dict(server._sessions, {"s1": {"session_key": DURABLE_ID}}, clear=True), \
                patch.object(ProcessRegistry, "_terminate_host_pid") as terminate:
            reply = _kill({"session_id": "s1", "process_id": "proc_peerabc12345"})

        terminate.assert_not_called()
        assert reply["result"]["status"] == "error"
