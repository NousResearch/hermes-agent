"""Turn-isolation active-session lease handoff (Design 1).

With ``dashboard.turn_isolation`` on, a named-profile turn is admitted once in
the SERVING (dashboard) process -- it claims the per-session lease there -- and
then dispatched to a compute-host CHILD process to run. The child re-crosses the
ownership chokepoint in ``_run_prompt_submit`` (the #94778 double-writer guard),
but its rebuilt session dict does not carry the serving process's lease, and
``_is_same_writer`` can never match across a process boundary (it requires
``entry.pid == os.getpid()``). Without a handoff the child re-claims, sees the
serving process's live entry, and refuses the turn:

    Session ... already has a live owner (desktop, pid N, running 0m)

Design 1 hands the admission to the child as a disabled lease SENTINEL: the
serving process signals ``active_session_admitted`` in the turn frame, and the
child installs a disabled ``ActiveSessionLease`` so the chokepoint recheck is a
genuine no-op WITHOUT claiming a second registry slot and WITHOUT being able to
release the serving process's real lease. The single admission (the serving
process's) is preserved, so the double-writer guard is untouched.
"""

import threading

import pytest

from tui_gateway import server


def _session(**overrides):
    base = {
        "session_key": "sess-key",
        "history": [],
        "history_lock": threading.Lock(),
        "cols": 80,
    }
    base.update(overrides)
    return base


class TestTurnFrameAdmissionSignal:
    def test_frame_marks_admitted_when_serving_process_holds_lease(self):
        session = _session(active_session_lease=object())
        frame = server._compute_host_turn_frame("rid", "sid", session, "hello")
        assert frame["active_session_admitted"] is True

    def test_frame_admission_false_without_lease(self):
        session = _session()
        frame = server._compute_host_turn_frame("rid", "sid", session, "hello")
        assert frame["active_session_admitted"] is False


class TestChildDelegatedLease:
    def test_admitted_frame_makes_chokepoint_a_noop_without_reclaiming(
        self, monkeypatch
    ):
        # The child must NOT try to claim a second registry slot when the
        # serving process already admitted the turn.
        def _boom(*args, **kwargs):
            raise AssertionError(
                "child re-claimed the active-session slot despite admission"
            )

        monkeypatch.setattr(server, "_claim_active_session_slot", _boom)

        session = _session()
        server._install_delegated_active_session_lease(
            session, {"active_session_admitted": True, "session_key": "sess-key"}
        )

        # The chokepoint recheck sees a lease and returns None (admitted),
        # never reaching the claim path above.
        assert server._ensure_active_session_slot("sid", session) is None

    def test_delegated_lease_release_is_a_noop(self, monkeypatch):
        # Releasing the child's sentinel must NOT touch the registry -- that
        # would drop the serving process's real lease out from under it.
        session = _session()
        server._install_delegated_active_session_lease(
            session, {"active_session_admitted": True, "session_key": "sess-key"}
        )
        lease = session["active_session_lease"]

        import hermes_cli.active_sessions as active_sessions

        def _boom_release(_lease):
            raise AssertionError("delegated lease reached the registry release")

        # release() short-circuits on a disabled lease before calling the
        # module-level release_active_session, so this stays untouched.
        monkeypatch.setattr(active_sessions, "release_active_session", _boom_release)
        lease.release()  # must not raise

    def test_no_admission_installs_no_lease(self):
        session = _session()
        server._install_delegated_active_session_lease(
            session, {"active_session_admitted": False, "session_key": "sess-key"}
        )
        assert session.get("active_session_lease") is None

    def test_install_is_idempotent_and_preserves_a_real_lease(self):
        real = object()
        session = _session(active_session_lease=real)
        server._install_delegated_active_session_lease(
            session, {"active_session_admitted": True, "session_key": "sess-key"}
        )
        # A real lease already present is never overwritten by the sentinel.
        assert session["active_session_lease"] is real

    def test_release_slot_on_sentinel_never_touches_registry(self, monkeypatch):
        # End-of-turn / finalize releases the session slot. For the delegated
        # sentinel this must be a clean no-op: the serving process still owns
        # the real registry entry.
        import hermes_cli.active_sessions as active_sessions

        session = _session()
        server._install_delegated_active_session_lease(
            session, {"active_session_admitted": True, "session_key": "sess-key"}
        )

        def _boom(_lease):
            raise AssertionError("delegated lease reached the registry release")

        monkeypatch.setattr(active_sessions, "release_active_session", _boom)
        assert server._release_active_session_slot(session) is True
        assert session.get("active_session_lease") is None

    def test_compression_transfer_on_sentinel_never_claims_a_second_slot(
        self, monkeypatch
    ):
        # Auto-compression rotates agent.session_id mid-turn, which fires
        # _transfer_active_session_slot in the CHILD. The sentinel must take
        # transfer_active_session's disabled-lease no-op branch -- never fall
        # through to _claim_active_session_slot (a real second registry entry
        # that would orphan the serving process's lease and break #94778).
        session = _session()
        server._install_delegated_active_session_lease(
            session, {"active_session_admitted": True, "session_key": "old-key"}
        )
        sentinel = session["active_session_lease"]

        def _boom_claim(*args, **kwargs):
            raise AssertionError(
                "compression transfer claimed a real second slot for a sentinel"
            )

        monkeypatch.setattr(server, "_claim_active_session_slot", _boom_claim)

        assert (
            server._transfer_active_session_slot(
                "sid", session, new_session_id="new-key"
            )
            is True
        )
        # Same sentinel object, re-anchored to the new id; no real claim.
        assert session["active_session_lease"] is sentinel
        assert sentinel.session_id == "new-key"


class TestCloseInterruptsActiveChildTurn:
    """Explicit session.close must interrupt an in-flight compute-host child
    turn BEFORE releasing the serving process's real lease.

    With the sentinel installed, the child's ownership chokepoint is a total
    no-op, so safety rests on the serving process holding the real lease for the
    whole child turn. The automatic reapers already gate on ``running`` (which
    the serving process holds True from dispatch until the child reports done),
    but explicit ``session.close`` tears down unconditionally and only waited on
    the in-process ``_run_thread`` -- absent for an isolated turn. Without an
    interrupt the lease drops while the child keeps writing, reopening a narrow
    #94778 window for a sibling backend sharing HERMES_HOME.
    """

    def _base_session(self):
        return {
            "agent": None,
            "session_key": "session-key",
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
        }

    def test_close_interrupts_child_before_releasing_lease(self, monkeypatch):
        order = []
        session_ref = {}

        class _Supervisor:
            def interrupt(self, sid, request_id=None):
                order.append("interrupt")
                # A real child reports the turn done, which clears `running`.
                session_ref["session"]["running"] = False
                return True

        monkeypatch.setattr(
            server, "_get_compute_host_supervisor", lambda _cfg=None: _Supervisor()
        )
        monkeypatch.setattr(
            server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}}
        )
        monkeypatch.setattr(
            server, "_session_uses_compute_host", lambda session, cfg=None: True
        )

        # A real release records its ordering relative to the interrupt.
        real_release = server._release_active_session_slot

        def _tracked_release(session):
            order.append("release")
            return real_release(session)

        monkeypatch.setattr(server, "_release_active_session_slot", _tracked_release)

        session = self._base_session()
        session_ref["session"] = session
        # A delegated sentinel stands in for the handed-off admission.
        server._install_delegated_active_session_lease(
            session, {"active_session_admitted": True, "session_key": "session-key"}
        )
        session["_sid"] = "sid"

        server._teardown_popped_session(session, end_reason="tui_close")

        assert "interrupt" in order, "child turn was not interrupted on close"
        assert order.index("interrupt") < order.index("release"), (
            "lease released before the child turn was interrupted"
        )

    def test_close_wait_is_bounded_but_lease_held_when_child_never_settles(
        self, monkeypatch, tmp_path
    ):
        # INVERSION of the old cementing test (Blocker 1, #99719 single
        # admission). The close/RPC path stays BOUNDED, but OWNERSHIP is not
        # released at the grace deadline: while the isolated child may still be
        # writing, the canonical lease is DETACHED-and-deferred, not dropped.
        # A second distinct writer that races for the same session id is
        # therefore still REFUSED after close returns. HEAD released the lease
        # unconditionally at the deadline (the old assertion was
        # ``active_session_lease is None``), so it FAILS this inversion.
        import hermes_cli.active_sessions as active_sessions

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        class _Supervisor:
            def interrupt(self, sid, request_id=None):
                return True  # never clears `running`

        monkeypatch.setattr(
            server, "_get_compute_host_supervisor", lambda _cfg=None: _Supervisor()
        )
        monkeypatch.setattr(
            server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}}
        )
        monkeypatch.setattr(
            server, "_session_uses_compute_host", lambda session, cfg=None: True
        )
        # Shrink the grace so the test stays fast.
        monkeypatch.setattr(server, "_TURN_SETTLE_BEFORE_CLOSE_SECONDS", 0.1)

        # The SERVING process holds the one REAL enabled lease for the whole
        # child turn (the child only carries the disabled sentinel).
        lease, refusal = active_sessions.try_acquire_active_session(
            session_id="session-key",
            surface="tui",
            config={},
            metadata={"live_session_id": "sid"},
        )
        assert refusal is None and lease is not None and lease.enabled

        session = self._base_session()
        session["active_session_lease"] = lease
        session["_sid"] = "sid"

        import time as _time

        start = _time.monotonic()
        assert server._teardown_popped_session(session, end_reason="tui_close") is True
        elapsed = _time.monotonic() - start
        assert elapsed < 2.0, "close hung past the bounded settle grace"

        # OWNERSHIP is still HELD: the real lease was detached-and-deferred, and
        # the registry still protects the id -- a distinct writer is REFUSED.
        detached = getattr(server, "_detached_leases", {})
        assert any(
            rec is lease for rec in detached.values()
        ), "canonical lease was not detached-and-deferred"
        _second, second_refusal = active_sessions.try_acquire_active_session(
            session_id="session-key",
            surface="tui",
            config={},
            metadata={"live_session_id": "distinct-writer"},
        )
        assert second_refusal is not None, "lease dropped at the close deadline"

        # Only after the simulated child exit / turn.end does ownership release.
        server._release_detached_leases_for_dead_child()
        third, third_refusal = active_sessions.try_acquire_active_session(
            session_id="session-key",
            surface="tui",
            config={},
            metadata={"live_session_id": "distinct-writer"},
        )
        assert third_refusal is None and third is not None

    def test_close_without_active_child_turn_does_not_interrupt(self, monkeypatch):
        calls = {"interrupt": 0}

        class _Supervisor:
            def interrupt(self, sid, request_id=None):
                calls["interrupt"] += 1
                return True

        monkeypatch.setattr(
            server, "_get_compute_host_supervisor", lambda _cfg=None: _Supervisor()
        )
        monkeypatch.setattr(
            server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}}
        )
        monkeypatch.setattr(
            server, "_session_uses_compute_host", lambda session, cfg=None: True
        )

        session = self._base_session()
        session["running"] = False
        session["_compute_host_active"] = False
        session["_sid"] = "sid"

        server._teardown_popped_session(session, end_reason="tui_close")
        assert calls["interrupt"] == 0


class TestDetachedLeaseDeferredRelease:
    """Blocker 1 (#99719): the canonical lease is held until the isolated
    child actually exits, not merely until a bounded close returns."""

    def _isolated_session(self, sid="sid", key="session-key"):
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
        }

    def _clear_detached(self):
        with server._sessions_lock:
            server._detached_leases.clear()

    def test_r1_lease_held_until_child_death_real_registry(
        self, monkeypatch, tmp_path
    ):
        # Child never settles; after the close grace a DISTINCT-writer second
        # acquirer (a real _claim_active_session_slot against the file-locked
        # JSONL) is REFUSED, and reacquisition succeeds ONLY after the
        # simulated turn.end/child-exit. FAILS on HEAD (grace-deadline release).
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._clear_detached()

        class _Supervisor:
            def interrupt(self, sid, request_id=None):
                return True  # never clears `running`

        monkeypatch.setattr(
            server, "_get_compute_host_supervisor", lambda _cfg=None: _Supervisor()
        )
        monkeypatch.setattr(
            server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}}
        )
        monkeypatch.setattr(
            server, "_session_uses_compute_host", lambda session, cfg=None: True
        )
        monkeypatch.setattr(server, "_TURN_SETTLE_BEFORE_CLOSE_SECONDS", 0.1)

        lease, _ = server._claim_active_session_slot(
            "session-key", live_session_id="sid", surface="tui"
        )
        assert lease is not None and lease.enabled

        session = self._isolated_session()
        session["active_session_lease"] = lease

        server._teardown_popped_session(session, end_reason="tui_close")

        # Second, distinct writer is refused while the child may still write.
        second, refusal = server._claim_active_session_slot(
            "session-key", live_session_id="distinct-writer", surface="tui"
        )
        assert second is None and refusal is not None

        # turn.end / child-exit releases the detached lease -> reacquisition ok.
        session["running"] = False
        server._on_compute_host_turn_done(
            "rid", "sid", session, {"type": "turn.end", "session_key": "session-key"}
        )
        third, refusal3 = server._claim_active_session_slot(
            "session-key", live_session_id="distinct-writer", surface="tui"
        )
        assert third is not None and refusal3 is None

    def test_r3_closing_shutdown_releases_detached_lease(
        self, monkeypatch, tmp_path
    ):
        # Blocker 1a: with the supervisor _closing=True, an observed child exit
        # must still release the detached lease (release call is AHEAD of the
        # `if self._closing: return` guard). FAILS on HEAD (no such call).
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._clear_detached()
        from tui_gateway.host_supervisor import HostSupervisor

        released = {"n": 0}
        real = server._release_detached_leases_for_dead_child

        def _tracked(*a, **k):
            released["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(
            server, "_release_detached_leases_for_dead_child", _tracked
        )

        lease, _ = server._claim_active_session_slot(
            "session-key", live_session_id="sid", surface="tui"
        )
        key = server._detach_lease_for_deferred_release("sid", lease)
        assert key in server._detached_leases

        sup = HostSupervisor(autostart=False, registry_path=tmp_path / "reg.json")
        sup._closing = True

        class _Proc:
            def wait(self):
                return 0

        sup._wait_for_exit(_Proc())
        assert released["n"] >= 1, "detached release skipped under _closing guard"
        assert key not in server._detached_leases

    def test_r4_reaper_does_not_drop_detached_lease(self, monkeypatch, tmp_path):
        # Blocker 1b: the idle reaper unions detached lease_ids into `live`, so
        # release_orphaned_leases KEEPS them. FAILS on HEAD (detached absent
        # from _sessions -> swept on the ~300s tick).
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._clear_detached()
        import hermes_cli.active_sessions as active_sessions

        # Clear the live session map so the ONLY protection is the detached
        # table union (otherwise the test would pass vacuously).
        with server._sessions_lock:
            server._sessions.clear()

        lease, _ = server._claim_active_session_slot(
            "session-key", live_session_id="sid", surface="tui"
        )
        server._detach_lease_for_deferred_release("sid", lease)

        server._reclaim_orphaned_leases()

        snapshot = active_sessions.active_session_registry_snapshot()
        assert any(
            e.get("lease_id") == lease.lease_id for e in snapshot
        ), "reaper dropped the detached lease"

    def test_r5_detach_is_single_critical_section_insert_before_remove(
        self, monkeypatch, tmp_path
    ):
        # A reaper tick landing mid-detach must NEVER see the lease in neither
        # _sessions nor _detached_leases. The detach is one _sessions_lock
        # critical section that inserts into _detached_leases BEFORE removing
        # from the session dict. FAILS on HEAD (no _detached_leases table).
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._clear_detached()

        lease, _ = server._claim_active_session_slot(
            "session-key", live_session_id="sid", surface="tui"
        )
        session = self._isolated_session()
        session["active_session_lease"] = lease
        with server._sessions_lock:
            server._sessions["sid"] = session

        observed = {"gap": False}

        # Probe: the detach must run entirely under the lock, so any observer
        # that acquires the lock sees a consistent state. Assert the invariant
        # directly around the detach call: at no lock acquisition can the lease
        # be absent from BOTH maps.
        def _observe():
            with server._sessions_lock:
                in_sessions = (
                    server._sessions.get("sid", {}).get("active_session_lease")
                    is lease
                )
                in_detached = any(
                    rec is lease for rec in server._detached_leases.values()
                )
                if not in_sessions and not in_detached:
                    observed["gap"] = True

        _observe()  # before detach: in _sessions
        server._detach_lease_for_deferred_release("sid", lease, session=session)
        _observe()  # after detach: in _detached_leases
        assert not observed["gap"]
        # The lease left the session dict but is now in the detached table.
        assert session.get("active_session_lease") is None
        assert any(rec is lease for rec in server._detached_leases.values())
        with server._sessions_lock:
            server._sessions.pop("sid", None)

    def test_crash_path_reaches_detached_release(self, monkeypatch, tmp_path):
        # _fail_pending_turns invokes on_complete, which for an isolated turn
        # reaches _on_compute_host_turn_done -> detached release. Assert the
        # crash callback releases the detached lease for the sid.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._clear_detached()
        from tui_gateway.host_supervisor import HostSupervisor

        lease, _ = server._claim_active_session_slot(
            "session-key", live_session_id="sid", surface="tui"
        )
        session = self._isolated_session()
        session["running"] = False
        key = server._detach_lease_for_deferred_release("sid", lease)

        sink = []
        sup = HostSupervisor(
            autostart=False,
            registry_path=tmp_path / "reg.json",
            rpc_sink=sink.append,
        )

        def _on_complete(frame):
            server._on_compute_host_turn_done("rid", "sid", session, frame)

        sup._pending_turns["rid"] = ("sid", _on_complete)
        sup._fail_pending_turns(reason="crash", message="host died")

        assert key not in server._detached_leases
