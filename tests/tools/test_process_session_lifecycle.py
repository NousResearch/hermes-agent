"""
Tests for the tmux-derived process-session lifecycle prototype.

Run with: pytest test_process_session_lifecycle.py -v

No subprocesses are spawned — the FSM is intentionally testable without
real processes so we can exercise edge cases (orphaned pipe, premature
cleanup, conflicting exit codes, subscriber races) deterministically.
"""

from __future__ import annotations

import threading

import pytest

from tools.process_session_lifecycle import (
    DeadSessionSummary,
    IdempotentSubscriberRegistry,
    LifecycleState,
    LifecycleTransitionError,
    ProcessSessionLifecycle,
    RetentionPolicy,
    SubscriberBackpressure,
)


# ---------------------------------------------------------------------------
# LifecycleState ordering / enum sanity
# ---------------------------------------------------------------------------


class TestLifecycleStateEnum:
    def test_states_are_strings(self):
        assert LifecycleState.RUNNING.value == "running"
        assert LifecycleState.EXITED.value == "exited"
        assert LifecycleState.STREAM_DRAINED.value == "stream_drained"
        assert LifecycleState.CLOSED.value == "closed"

    def test_state_count(self):
        # If a state is added, tests must be updated deliberately.
        assert len(list(LifecycleState)) == 4


# ---------------------------------------------------------------------------
# Forward transitions: RUNNING -> EXITED -> STREAM_DRAINED -> CLOSED
# ---------------------------------------------------------------------------


class TestForwardTransitions:
    def test_initial_state_is_running(self):
        lc = ProcessSessionLifecycle("proc_a", "true")
        assert lc.state == LifecycleState.RUNNING
        assert lc.exit_code is None
        assert lc.exited_at is None

    def test_running_to_exited(self):
        lc = ProcessSessionLifecycle("proc_a", "true")
        assert lc.mark_exited(0) is True
        assert lc.state == LifecycleState.EXITED
        assert lc.exit_code == 0
        assert lc.exit_reason == "exited"
        assert lc.exited_at is not None

    def test_exited_to_stream_drained(self):
        lc = ProcessSessionLifecycle("proc_a", "true")
        lc.mark_exited(0)
        assert lc.mark_stream_drained() is True
        assert lc.state == LifecycleState.STREAM_DRAINED
        assert lc.stream_drained_at is not None

    def test_stream_drained_to_closed_success(self):
        lc = ProcessSessionLifecycle("proc_a", "true")
        lc.mark_exited(0)
        lc.mark_stream_drained()
        summary = lc.close(output_tail="ok\n")
        assert lc.state == LifecycleState.CLOSED
        # Default retention is RETAIN_ON_FAILURE; success returns None.
        assert summary is None

    def test_full_failure_path_retains_summary(self):
        lc = ProcessSessionLifecycle("proc_a", "true")
        lc.mark_exited(1)
        lc.mark_stream_drained()
        summary = lc.close(output_tail="boom\n")
        assert summary is not None
        assert isinstance(summary, DeadSessionSummary)
        assert summary.exit_code == 1
        assert summary.is_failure is True
        assert summary.output_tail == "boom\n"


# ---------------------------------------------------------------------------
# Invalid transitions
# ---------------------------------------------------------------------------


class TestInvalidTransitions:
    def test_cannot_drain_stream_before_exit(self):
        """The core invariant: don't lose exit status by cleaning stream first."""
        lc = ProcessSessionLifecycle("proc_a", "sleep 1")
        with pytest.raises(LifecycleTransitionError, match="before recording exit"):
            lc.mark_stream_drained()

    def test_cannot_abandon_stream_before_exit(self):
        lc = ProcessSessionLifecycle("proc_a", "sleep 1")
        with pytest.raises(LifecycleTransitionError):
            lc.abandon_stream()

    def test_cannot_close_from_running(self):
        lc = ProcessSessionLifecycle("proc_a", "sleep 1")
        with pytest.raises(LifecycleTransitionError):
            lc.close()

    def test_cannot_close_from_exited_without_drain(self):
        lc = ProcessSessionLifecycle("proc_a", "true")
        lc.mark_exited(0)
        with pytest.raises(LifecycleTransitionError, match="stream must be drained"):
            lc.close()

    def test_mark_exited_with_conflicting_payload_raises(self):
        lc = ProcessSessionLifecycle("proc_a", "true")
        lc.mark_exited(0)
        with pytest.raises(LifecycleTransitionError, match="conflicting payload"):
            lc.mark_exited(1)

    def test_mark_exited_with_conflicting_reason_raises(self):
        lc = ProcessSessionLifecycle("proc_a", "true")
        lc.mark_exited(0, exit_reason="exited")
        with pytest.raises(LifecycleTransitionError):
            lc.mark_exited(0, exit_reason="killed")


# ---------------------------------------------------------------------------
# Idempotency under reader/reconciler races
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_mark_exited_is_idempotent_with_consistent_payload(self):
        lc = ProcessSessionLifecycle("proc_a", "true")
        assert lc.mark_exited(0, exit_reason="exited") is True
        assert lc.mark_exited(0, exit_reason="exited") is False
        assert lc.state == LifecycleState.EXITED

    def test_mark_stream_drained_is_idempotent(self):
        lc = ProcessSessionLifecycle("proc_a", "true")
        lc.mark_exited(0)
        assert lc.mark_stream_drained() is True
        assert lc.mark_stream_drained() is False

    def test_close_is_idempotent(self):
        lc = ProcessSessionLifecycle("proc_a", "true")
        lc.mark_exited(1)
        lc.mark_stream_drained()
        first = lc.close(output_tail="x")
        second = lc.close(output_tail="y")  # ignored; first wins
        assert first is second
        # The retained tail is the first one — repeated close calls don't
        # mutate the summary, which mirrors tmux's frozen dead-pane text.
        assert first.output_tail == "x"

    def test_mark_exited_after_close_no_op(self):
        lc = ProcessSessionLifecycle("proc_a", "true")
        lc.mark_exited(0)
        lc.mark_stream_drained()
        lc.close()
        # Idempotent re-entry with consistent payload returns False, no raise.
        assert lc.mark_exited(0, exit_reason="exited") is False


# ---------------------------------------------------------------------------
# Abandon path (orphaned-pipe / issue-#17327 analogue)
# ---------------------------------------------------------------------------


class TestAbandonPath:
    def test_abandon_after_exit_advances_to_drained(self):
        lc = ProcessSessionLifecycle("proc_a", "true")
        lc.mark_exited(0)
        assert lc.abandon_stream(reason="orphaned descendant") is True
        assert lc.state == LifecycleState.STREAM_DRAINED

    def test_abandon_records_note_for_summary(self):
        lc = ProcessSessionLifecycle(
            "proc_a", "true", retention=RetentionPolicy.ALWAYS
        )
        lc.mark_exited(0)
        lc.abandon_stream(reason="orphaned descendant held stdout")
        summary = lc.close()
        assert summary is not None
        # Note is folded into the immutable summary tuple.
        assert any("orphaned descendant" in n for n in summary.notes)

    def test_abandon_is_idempotent(self):
        lc = ProcessSessionLifecycle("proc_a", "true")
        lc.mark_exited(0)
        assert lc.abandon_stream() is True
        assert lc.abandon_stream() is False


# ---------------------------------------------------------------------------
# Retention policy
# ---------------------------------------------------------------------------


class TestRetentionPolicy:
    def test_retain_on_failure_skips_success(self):
        lc = ProcessSessionLifecycle("proc_ok", "true")
        lc.mark_exited(0)
        lc.mark_stream_drained()
        assert lc.close() is None
        assert lc.summary() is None

    def test_retain_on_failure_keeps_nonzero_exit(self):
        lc = ProcessSessionLifecycle("proc_fail", "false")
        lc.mark_exited(2)
        lc.mark_stream_drained()
        s = lc.close()
        assert s is not None
        assert s.exit_code == 2

    def test_retain_on_failure_keeps_killed(self):
        lc = ProcessSessionLifecycle("proc_killed", "sleep 999")
        lc.mark_exited(-15, exit_signal=15, exit_reason="killed")
        lc.mark_stream_drained()
        s = lc.close()
        assert s is not None
        assert s.exit_reason == "killed"
        assert s.is_failure is True

    def test_retain_on_failure_keeps_lost_handle(self):
        lc = ProcessSessionLifecycle("proc_lost", "detached")
        lc.mark_exited(None, exit_reason="lost")
        lc.abandon_stream(reason="handle gone after gateway restart")
        s = lc.close()
        assert s is not None
        assert s.exit_code is None
        assert s.is_failure is True

    def test_retain_always_keeps_success(self):
        lc = ProcessSessionLifecycle(
            "proc_ok", "true", retention=RetentionPolicy.ALWAYS
        )
        lc.mark_exited(0)
        lc.mark_stream_drained()
        s = lc.close()
        assert s is not None
        assert s.exit_code == 0
        assert s.is_failure is False

    def test_retain_never_drops_everything(self):
        lc = ProcessSessionLifecycle(
            "proc_fail", "false", retention=RetentionPolicy.NEVER
        )
        lc.mark_exited(1)
        lc.mark_stream_drained()
        assert lc.close() is None


# ---------------------------------------------------------------------------
# DeadSessionSummary content
# ---------------------------------------------------------------------------


class TestDeadSessionSummary:
    def test_summary_carries_timestamps(self):
        lc = ProcessSessionLifecycle("proc_a", "false")
        lc.mark_exited(1)
        lc.mark_stream_drained()
        s = lc.close()
        assert s is not None
        assert s.started_at <= s.exited_at <= s.closed_at

    def test_output_tail_is_trimmed_to_limit(self):
        lc = ProcessSessionLifecycle(
            "proc_a", "false", output_tail_limit_bytes=64
        )
        lc.mark_exited(1)
        lc.mark_stream_drained()
        big = "X" * 4096
        s = lc.close(output_tail=big)
        assert s is not None
        # We keep the LAST bytes, not the first.
        assert s.output_tail == "X" * 64
        assert s.output_tail_bytes == 64

    def test_summary_is_frozen(self):
        lc = ProcessSessionLifecycle("proc_a", "false")
        lc.mark_exited(1)
        lc.mark_stream_drained()
        s = lc.close()
        assert s is not None
        with pytest.raises((AttributeError, Exception)):
            s.exit_code = 0  # type: ignore[misc]

    def test_summary_includes_backpressure_snapshot(self):
        lc = ProcessSessionLifecycle("proc_a", "false")

        def slow(kind: str, payload: str) -> None:
            # First broadcast happens at EXITED. Simulate a sink that
            # crashes on every event — backpressure should accumulate.
            raise RuntimeError("simulated slow sink")

        lc.subscribers.register("sink-1", slow)
        lc.mark_exited(1)  # broadcasts → sink raises → discard recorded
        lc.mark_stream_drained()
        s = lc.close()
        assert s is not None
        assert "sink-1" in s.subscriber_backpressure
        assert s.subscriber_backpressure["sink-1"]["discarded"] >= 1

    def test_summary_carries_log_pointer(self):
        lc = ProcessSessionLifecycle("proc_a", "false")
        lc.mark_exited(1)
        lc.mark_stream_drained()
        s = lc.close(log_pointer="/var/log/hermes/proc_a.log")
        assert s is not None
        assert s.log_pointer == "/var/log/hermes/proc_a.log"


# ---------------------------------------------------------------------------
# IdempotentSubscriberRegistry
# ---------------------------------------------------------------------------


class TestSubscriberRegistry:
    def test_register_returns_handle(self):
        reg = IdempotentSubscriberRegistry()
        h = reg.register("sub-1", lambda k, p: None)
        assert h.subscriber_id == "sub-1"

    def test_register_same_id_returns_existing(self):
        reg = IdempotentSubscriberRegistry()

        def cb1(k, p):
            return None

        def cb2(k, p):
            return None

        h1 = reg.register("sub-1", cb1)
        h2 = reg.register("sub-1", cb2)
        assert h1 is h2
        # The original callback wins — idempotent register does NOT replace.
        assert h2.callback is cb1

    def test_register_replace_true_swaps_callback(self):
        reg = IdempotentSubscriberRegistry()

        def cb1(k, p):
            return None

        def cb2(k, p):
            return None

        h1 = reg.register("sub-1", cb1)
        h2 = reg.register("sub-1", cb2, replace=True)
        assert h2.callback is cb2
        # Replace creates a new handle.
        assert h1 is not h2

    def test_register_empty_id_raises(self):
        reg = IdempotentSubscriberRegistry()
        with pytest.raises(ValueError):
            reg.register("", lambda k, p: None)

    def test_unregister_idempotent(self):
        reg = IdempotentSubscriberRegistry()
        reg.register("sub-1", lambda k, p: None)
        assert reg.unregister("sub-1") is True
        assert reg.unregister("sub-1") is False  # already gone, no raise

    def test_broadcast_invokes_all_subscribers(self):
        reg = IdempotentSubscriberRegistry()
        received = []
        reg.register("a", lambda k, p: received.append(("a", k, p)))
        reg.register("b", lambda k, p: received.append(("b", k, p)))
        reg.broadcast("event", "hello")
        assert ("a", "event", "hello") in received
        assert ("b", "event", "hello") in received

    def test_broadcast_isolates_failing_sink(self):
        reg = IdempotentSubscriberRegistry()
        received = []
        reg.register("bad", lambda k, p: (_ for _ in ()).throw(RuntimeError("boom")))
        reg.register("good", lambda k, p: received.append((k, p)))
        reg.broadcast("event", "payload")
        # Healthy sink still got the event.
        assert ("event", "payload") in received
        # Bad sink's backpressure was incremented.
        snap = reg.backpressure_snapshot()
        assert snap["bad"]["discarded"] == 1

    def test_register_after_close_raises(self):
        reg = IdempotentSubscriberRegistry()
        reg._close()
        with pytest.raises(RuntimeError, match="closed"):
            reg.register("sub-1", lambda k, p: None)


# ---------------------------------------------------------------------------
# Subscriber lifecycle hooks via ProcessSessionLifecycle
# ---------------------------------------------------------------------------


class TestSubscriberLifecycle:
    def test_cannot_subscribe_after_close(self):
        lc = ProcessSessionLifecycle("proc_a", "false")
        lc.mark_exited(1)
        lc.mark_stream_drained()
        lc.close()
        with pytest.raises(RuntimeError, match="closed"):
            lc.subscribers.register("late", lambda k, p: None)

    def test_can_subscribe_after_exited_before_drained(self):
        """A reader can attach between exit and drain to read final bytes —
        important for callers that want to capture late output."""
        lc = ProcessSessionLifecycle("proc_a", "false")
        lc.mark_exited(0)
        # Should not raise.
        lc.subscribers.register("late", lambda k, p: None)
        lc.mark_stream_drained()
        lc.close()

    def test_lifecycle_transitions_broadcast_to_subscribers(self):
        lc = ProcessSessionLifecycle("proc_a", "false")
        seen = []
        lc.subscribers.register("watcher", lambda k, p: seen.append((k, p)))
        lc.mark_exited(1)
        lc.mark_stream_drained()
        lc.close()
        kinds = [k for k, _ in seen]
        # We should have observed 3 lifecycle transition events.
        assert kinds.count("lifecycle_transition") == 3
        # Payload encodes the state.
        states = [p.split(":")[1] for k, p in seen if k == "lifecycle_transition"]
        assert states == ["exited", "stream_drained", "closed"]


# ---------------------------------------------------------------------------
# Backpressure
# ---------------------------------------------------------------------------


class TestBackpressure:
    def test_record_discard_increments(self):
        bp = SubscriberBackpressure()
        bp.record_discard(byte_count=100)
        assert bp.discarded == 1
        assert bp.bytes_discarded == 100
        assert bp.too_far_behind is False

    def test_too_far_behind_threshold(self):
        bp = SubscriberBackpressure(too_far_behind_threshold=3)
        bp.record_discard()
        bp.record_discard()
        assert bp.too_far_behind is False
        bp.record_discard()
        assert bp.too_far_behind is True

    def test_snapshot_is_dict(self):
        bp = SubscriberBackpressure()
        bp.record_discard(byte_count=10)
        snap = bp.snapshot()
        assert snap["discarded"] == 1
        assert snap["bytes_discarded"] == 10
        assert snap["too_far_behind"] == 0


# ---------------------------------------------------------------------------
# Add-note behavior
# ---------------------------------------------------------------------------


class TestNotes:
    def test_note_appears_in_summary(self):
        lc = ProcessSessionLifecycle(
            "proc_a", "false", retention=RetentionPolicy.ALWAYS
        )
        lc.add_note("triggered by user cancel")
        lc.mark_exited(0)
        lc.mark_stream_drained()
        s = lc.close()
        assert s is not None
        assert "triggered by user cancel" in s.notes

    def test_note_after_close_raises(self):
        lc = ProcessSessionLifecycle("proc_a", "false")
        lc.mark_exited(0)
        lc.mark_stream_drained()
        lc.close()
        with pytest.raises(LifecycleTransitionError, match="summary is frozen"):
            lc.add_note("too late")


# ---------------------------------------------------------------------------
# Thread safety — concurrent mark_exited from reader and reconciler
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_mark_exited_one_winner(self):
        """Reader thread and reconciler both call mark_exited(0). Only one
        wins; the other observes a no-op. No race, no exception."""
        lc = ProcessSessionLifecycle("proc_a", "true")
        winners: list[bool] = []
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            try:
                won = lc.mark_exited(0)
                winners.append(won)
            except Exception:
                winners.append(False)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert sum(1 for w in winners if w is True) == 1
        assert sum(1 for w in winners if w is False) == 7
        assert lc.state == LifecycleState.EXITED

    def test_concurrent_register_same_id_returns_same_handle(self):
        reg = IdempotentSubscriberRegistry()

        def cb(k, p):
            return None

        handles = []
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            h = reg.register("sub-1", cb)
            handles.append(h)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        first = handles[0]
        assert all(h is first for h in handles)


# ---------------------------------------------------------------------------
# Snapshot / observability
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_snapshot_carries_everything(self):
        lc = ProcessSessionLifecycle("proc_a", "false")
        lc.subscribers.register("sub-1", lambda k, p: None)
        lc.mark_exited(7, exit_reason="exited")
        snap = lc.snapshot()
        assert snap.session_id == "proc_a"
        assert snap.command == "false"
        assert snap.state == LifecycleState.EXITED
        assert snap.exit_code == 7
        assert snap.subscriber_ids == ("sub-1",)

    def test_snapshot_is_consistent_across_states(self):
        lc = ProcessSessionLifecycle("proc_a", "false")
        assert lc.snapshot().state == LifecycleState.RUNNING
        lc.mark_exited(0)
        assert lc.snapshot().state == LifecycleState.EXITED
        lc.mark_stream_drained()
        assert lc.snapshot().state == LifecycleState.STREAM_DRAINED
        lc.close()
        assert lc.snapshot().state == LifecycleState.CLOSED
