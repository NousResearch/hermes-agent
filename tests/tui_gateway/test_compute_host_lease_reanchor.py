"""Blocker 2 (#99719): mid-turn compression re-anchors the REAL active-session
registry lease, not the child's disabled sentinel.

Under ``dashboard.turn_isolation`` a named-profile turn is admitted once in the
serving process (which holds the one real, file-locked active-session lease) and
dispatched to a compute-host CHILD. When context compression rotates the session
id A->B mid-turn INSIDE the child, the child's disabled sentinel
``transfer_active_session`` is a local no-op that never touches the registry: the
real lease stays on A while the child writes continuation B, so a second process
can acquire B (double-writer, reopening the #99719 hole).

The fix is a child-proposes / owner-transfers / owner-acks handshake fired at the
pre-commit hook point (before ``publish_compression_child``): the child BLOCKS and
sends a re-anchor request over a BUILT reverse-RPC; the serving owner atomically
moves the real registry lease A->B (enabled ``transfer_active_session``) and acks;
only then does the child publish B.

R6  - B-refused-until-transfer: the owner's atomic transfer is what protects B.
R7  - committed-transfer-then-lost-ack: a lost ack is reconciled by ONE bounded
      status re-query (idempotent, outcome-recorded), never a double-writer.
R7b - pre-commit nack: the rotation aborts to A (conversation_compression.py
      5312-5329); B is never durably created.
"""

import io
import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tui_gateway import compute_host, server


def _isolated_session(sid="sid", key="A", lease=None):
    return {
        "agent": None,
        "session_key": key,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": True,
        "_compute_host_active": True,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "_sid": sid,
        "active_session_lease": lease,
        "active_session_admission": (
            {
                "lease_id": getattr(lease, "lease_id", ""),
                "session_id": key,
                "generation": 0,
            }
            if lease is not None
            else None
        ),
    }


class TestR6BRefusedUntilTransfer:
    def test_r6_b_refused_until_owner_transfers(self, monkeypatch, tmp_path):
        # The owner (serving process) holds the real enabled lease on A. A
        # distinct writer cannot take A. The child proposes a re-anchor A->B;
        # the OWNER performs the atomic transfer_active_session A->B. Only AFTER
        # that commit is B protected (a distinct writer claiming B is REFUSED)
        # and A freed. FAILS on HEAD: the server-side reanchor handler does not
        # exist, so B is never protected by an atomic owner-side transfer.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        lease, refusal = server._claim_active_session_slot(
            "A", live_session_id="sid", surface="tui"
        )
        assert lease is not None and lease.enabled and refusal is None

        session = _isolated_session(lease=lease)
        with server._sessions_lock:
            server._sessions["sid"] = session
        self.addfinalizer = None

        try:
            # A is protected against a distinct second writer.
            s2, r2 = server._claim_active_session_slot(
                "A", live_session_id="distinct-writer", surface="tui"
            )
            assert s2 is None and r2 is not None

            # The child proposes; the owner atomically transfers A->B.
            outcome = server._handle_active_session_reanchor(
                {
                    "type": "active_session.reanchor",
                    "session_id": "sid",
                    "request_id": "req-1",
                    "old_id": "A",
                    "new_id": "B",
                    "lease_id": lease.lease_id,
                    "generation": 0,
                }
            )
            assert outcome["applied"] is True

            # B is now protected; A is free.
            b2, br = server._claim_active_session_slot(
                "B", live_session_id="distinct-writer", surface="tui"
            )
            assert b2 is None and br is not None, "B not protected after transfer"

            a3, ar = server._claim_active_session_slot(
                "A", live_session_id="distinct-writer", surface="tui"
            )
            assert a3 is not None and ar is None, "A not freed after transfer"
        finally:
            with server._sessions_lock:
                server._sessions.pop("sid", None)

    def test_r6_idempotent_duplicate_returns_recorded_outcome(
        self, monkeypatch, tmp_path
    ):
        # A duplicate proposal / status re-query for the same
        # (request_id, generation) returns the recorded outcome WITHOUT a second
        # transfer (same-id A->B is a no-op applied:true). FAILS on HEAD.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        lease, _ = server._claim_active_session_slot(
            "A", live_session_id="sid", surface="tui"
        )
        session = _isolated_session(lease=lease)
        with server._sessions_lock:
            server._sessions["sid"] = session
        try:
            first = server._handle_active_session_reanchor(
                {
                    "type": "active_session.reanchor",
                    "session_id": "sid",
                    "request_id": "req-dup",
                    "old_id": "A",
                    "new_id": "B",
                    "lease_id": lease.lease_id,
                    "generation": 0,
                }
            )
            assert first["applied"] is True
            status = server._handle_active_session_reanchor(
                {
                    "type": "active_session.reanchor.status",
                    "session_id": "sid",
                    "request_id": "req-dup",
                    "lease_id": lease.lease_id,
                    "generation": 0,
                }
            )
            assert status["applied"] is True
        finally:
            with server._sessions_lock:
                server._sessions.pop("sid", None)


class TestR7CommittedTransferThenLostAck:
    def test_r7_committed_transfer_then_lost_ack(self, monkeypatch, tmp_path):
        # The server commits A->B under the file lock, then the ack is LOST. The
        # child's ONE bounded status re-query for the same (request_id,
        # generation) returns applied:true, so the child PROCEEDS to publish B;
        # no third party can acquire B at any point. FAILS on HEAD: neither the
        # timeout constant nor the child re-anchor method exists.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(compute_host, "_REANCHOR_ACK_TIMEOUT_SECONDS", 0.2)

        owner_lease, _ = server._claim_active_session_slot(
            "A", live_session_id="sid", surface="tui"
        )
        assert owner_lease is not None and owner_lease.enabled

        host = compute_host.ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
        committed = {"done": False}

        def _emit(frame):
            msg = frame.get("message") or {} if frame.get("type") == "rpc" else {}
            params = msg.get("params") or {}
            kind = params.get("type")
            if kind == "active_session.reanchor":
                # Owner commits the real registry transfer, then DROPS the ack.
                from hermes_cli.active_sessions import transfer_active_session

                assert transfer_active_session(
                    owner_lease, session_id="B", metadata={"live_session_id": "sid"}
                )
                committed["done"] = True
                # ack lost: deliver nothing.
            elif kind == "active_session.reanchor.status":
                # The bounded status re-query is authoritative: committed.
                host.handle_frame(
                    {
                        "type": "active_session.reanchor.ack",
                        "request_id": params.get("request_id"),
                        "generation": params.get("generation"),
                        "applied": True,
                    }
                )

        monkeypatch.setattr(host, "emit", _emit)
        try:
            applied = host._reanchor_active_session(
                "sid",
                "A",
                "B",
                {"lease_id": owner_lease.lease_id, "generation": 0},
            )
            assert applied is True
            assert committed["done"]

            # No third party can ever hold B (the real lease is on B).
            b2, br = server._claim_active_session_slot(
                "B", live_session_id="third-party", surface="tui"
            )
            assert b2 is None and br is not None, "third party acquired B"

            # A is free (the lease moved off it).
            a2, ar = server._claim_active_session_slot(
                "A", live_session_id="third-party", surface="tui"
            )
            assert a2 is not None and ar is None
        finally:
            host.close()


class TestGenerationRefreshedAcrossRotations:
    def test_second_rotation_quotes_refreshed_generation_and_commits(
        self, monkeypatch, tmp_path
    ):
        # #99719 round-2: two SEQUENTIAL mid-turn rotations. The first re-anchor
        # A->B commits; the owner bumps its generation 0->1 and forwards
        # new_generation in the ack. The child must WRITE that back into the
        # shared admission dict so the SECOND re-anchor B->C quotes generation=1
        # (the refreshed value) and COMMITS. FAILS on HEAD: the ack drops
        # new_generation and the admission snapshot stays 0, so the second
        # rotation is nacked on a generation mismatch (quoted=0 current=1).
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(compute_host, "_REANCHOR_ACK_TIMEOUT_SECONDS", 0.2)

        owner_lease, _ = server._claim_active_session_slot(
            "A", live_session_id="sid", surface="tui"
        )
        assert owner_lease is not None and owner_lease.enabled

        session = _isolated_session(lease=owner_lease)
        with server._sessions_lock:
            server._sessions["sid"] = session

        host = compute_host.ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)

        # The shared admission object (session["_delegated_admission"] by ref).
        admission = {"lease_id": owner_lease.lease_id, "generation": 0}

        def _emit(frame):
            msg = frame.get("message") or {} if frame.get("type") == "rpc" else {}
            params = msg.get("params") or {}
            kind = params.get("type")
            if kind in ("active_session.reanchor", "active_session.reanchor.status"):
                # Owner handles the proposal and forwards the ack (with the
                # bumped new_generation, exactly like _emit_active_session_reanchor_ack).
                outcome = server._handle_active_session_reanchor(params)
                host.handle_frame(
                    {
                        "type": "active_session.reanchor.ack",
                        "request_id": outcome.get("request_id"),
                        "generation": outcome.get("generation"),
                        "applied": bool(outcome.get("applied")),
                        "new_generation": outcome.get("new_generation"),
                    }
                )

        monkeypatch.setattr(host, "emit", _emit)
        try:
            # First rotation A->B commits; admission generation refreshes 0->1.
            applied1 = host._reanchor_active_session("sid", "A", "B", admission)
            assert applied1 is True
            assert admission["generation"] == 1, "admission not refreshed after A->B"

            # Second rotation B->C quotes the REFRESHED generation and commits.
            applied2 = host._reanchor_active_session("sid", "B", "C", admission)
            assert applied2 is True, "second rotation nacked on stale generation"
            assert admission["generation"] == 2

            # The registry lease moved B->C: C is protected, B is free.
            c2, cr = server._claim_active_session_slot(
                "C", live_session_id="third-party", surface="tui"
            )
            assert c2 is None and cr is not None, "C not protected after B->C"
            b2, br = server._claim_active_session_slot(
                "B", live_session_id="third-party", surface="tui"
            )
            assert b2 is not None and br is None, "B not freed after B->C"
        finally:
            host.close()
            with server._sessions_lock:
                server._sessions.pop("sid", None)


class TestFailClosedHookInstall:
    class _RejectingAgent:
        # An agent object that refuses the re-anchor attribute assignment.
        __slots__ = ()

        def __setattr__(self, name, value):
            raise AttributeError(f"cannot set {name}")

    def test_install_raises_when_assignment_fails_with_admission(self):
        # #99719: a session carrying a delegated admission whose agent rejects
        # the hook assignment must FAIL CLOSED -- _install_pre_rotation_reanchor
        # raises so the turn's outer except emits turn.error rather than
        # proceeding hook-less (a mid-turn rotation would then reopen the
        # double-writer window). FAILS on HEAD: the assignment error is swallowed
        # (except Exception: pass) and the turn proceeds unprotected.
        host = compute_host.ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
        try:
            session = {
                "agent": self._RejectingAgent(),
                "_delegated_admission": {"lease_id": "L", "generation": 0},
            }
            with pytest.raises(Exception):
                host._install_pre_rotation_reanchor(server, "sid", session)
        finally:
            host.close()

    def test_install_is_noop_without_admission(self):
        # A session with NO delegated admission (serving/in-process path) is a
        # clean no-op even if the agent would reject the assignment -- the early
        # return fires before any attribute is touched, so no turn is failed.
        host = compute_host.ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
        try:
            session = {"agent": self._RejectingAgent(), "_delegated_admission": None}
            host._install_pre_rotation_reanchor(server, "sid", session)  # no raise
        finally:
            host.close()


def _make_agent(session_db, session_id="A"):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.compression_in_place = False
        return agent


def _stub_compressor():
    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail question"},
    ]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    return compressor


class TestR7bPreCommitNackAborts:
    def test_r7b_precommit_nack_aborts_rotation_to_A(self):
        # A nacking _pre_rotation_reanchor (owner raised BEFORE the file-locked
        # commit, so applied:false) must ABORT the rotation at
        # conversation_compression.py:5312-5329: agent stays on A, B is never
        # published. FAILS on HEAD: no pre-rotation hook exists, so the rotation
        # proceeds and agent.session_id changes.
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = _make_agent(db, session_id="A")
            agent.context_compressor = _stub_compressor()
            agent._pre_rotation_reanchor = (
                lambda old_session_id, new_session_id: False
            )

            original_sid = agent.session_id
            published = {"n": 0}
            orig_publish = db.publish_compression_child

            def _count_publish(*a, **k):
                published["n"] += 1
                return orig_publish(*a, **k)

            with patch.object(
                db, "publish_compression_child", side_effect=_count_publish
            ):
                agent._compress_context(
                    [{"role": "user", "content": f"m{i}"} for i in range(6)],
                    "sys",
                    approx_tokens=10_000,
                )

            assert agent.session_id == original_sid, "rotation not aborted on nack"
            assert published["n"] == 0, "published B despite nack"

    def test_r7b_hook_raise_aborts_rotation_to_A(self):
        # A RuntimeError from the hook (e.g. ack timeout after the status
        # re-query also failed) is equally fail-closed. FAILS on HEAD (no hook).
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = _make_agent(db, session_id="A")
            agent.context_compressor = _stub_compressor()

            def _raise(old_session_id, new_session_id):
                raise RuntimeError("re-anchor ack timed out")

            agent._pre_rotation_reanchor = _raise
            original_sid = agent.session_id
            published = {"n": 0}
            orig_publish = db.publish_compression_child

            def _count_publish(*a, **k):
                published["n"] += 1
                return orig_publish(*a, **k)

            with patch.object(
                db, "publish_compression_child", side_effect=_count_publish
            ):
                agent._compress_context(
                    [{"role": "user", "content": f"m{i}"} for i in range(6)],
                    "sys",
                    approx_tokens=10_000,
                )

            assert agent.session_id == original_sid
            assert published["n"] == 0


class TestReanchorHookHappyPath:
    def test_hook_fires_before_publish_and_allows_rotation(self):
        # A truthy hook lets the rotation commit; the hook fires BEFORE
        # publish_compression_child (the sole point where B is known but not yet
        # durable). FAILS on HEAD (no hook, so ordering can't be observed).
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = _make_agent(db, session_id="A")
            agent.context_compressor = _stub_compressor()

            events = []
            agent._pre_rotation_reanchor = (
                lambda old_session_id, new_session_id: events.append(
                    ("reanchor", old_session_id, new_session_id)
                )
                or True
            )
            orig_publish = db.publish_compression_child

            def _record_publish(*a, **k):
                events.append(("publish",))
                return orig_publish(*a, **k)

            original_sid = agent.session_id
            with patch.object(
                db, "publish_compression_child", side_effect=_record_publish
            ):
                agent._compress_context(
                    [{"role": "user", "content": f"m{i}"} for i in range(6)],
                    "sys",
                    approx_tokens=10_000,
                )

            assert agent.session_id != original_sid, "rotation did not commit"
            kinds = [e[0] for e in events]
            assert "reanchor" in kinds and "publish" in kinds
            assert kinds.index("reanchor") < kinds.index("publish"), (
                "re-anchor hook must fire before publish_compression_child"
            )
            reanchor_event = next(e for e in events if e[0] == "reanchor")
            assert reanchor_event[1] == original_sid
            assert reanchor_event[2] == agent.session_id
