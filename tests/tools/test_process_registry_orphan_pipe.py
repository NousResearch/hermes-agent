"""Regression test: orphan-held stdout pipe vs. finished-handle release.

Reviewer claim (PR #75162 review, 2026-08-09 — MongLong0214):

  ``ProcessRegistry._reader_loop()`` deliberately stops draining shortly after
  the direct child exits even when a background descendant still owns the
  pipe's write end (issue #68915 idle-escape: 3 x 0.2s select cycles with no
  data, then break WITHOUT EOF). ``_move_to_finished()`` then calls
  ``_release_finished_handles()``, which closes ``Popen.stdout`` (and the PTY)
  unconditionally. With the parent-side read end gone, a descendant that
  writes to the pipe LATER receives EPIPE/SIGPIPE and dies — the registry's
  handle release terminates background processes it never owned.

This test builds a REAL process tree (no mocks) exercising exactly that
sequence:

  1. spawn a shell that backgrounds a descendant which INHERITS stdout:
        ( sleep 2; echo LATE_OUTPUT; touch <marker> ) &
     The direct shell exits immediately; the descendant sleeps, then writes.
  2. wait for the registry to finish the session (reader idle-escape fires
     within ~0.7s of child exit).
  3. the descendant writes LATE_OUTPUT ~1.3s AFTER the release window that
     existed before the fix.
  4. assert the descendant SURVIVES (marker written after its late write)
     AND that its late output was captured into session.output_buffer.

Chosen late-output policy (asserted, guard fix 6 — bounded drain owner): the
descendant must survive the handle release AND its late output must be
captured within the ``_ORPHAN_DRAIN_WINDOW_S`` drain window. The drain owner
keeps the read end open for that bounded window (default 5.0s > the
descendant's 2s sleep), drains the late write into ``output_buffer``, then
closes the handle — bounded, never indefinite.

POSIX-only: relies on select() over pipe FDs and SIGPIPE semantics.
"""

import os
import shlex
import signal
import sys
import time

import pytest

from tools.process_registry import ProcessRegistry


@pytest.fixture()
def registry():
    """Create a fresh ProcessRegistry."""
    return ProcessRegistry()


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    """Poll a predicate until it returns truthy or the timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only: select()/SIGPIPE")
class TestOrphanDescendantPipeRelease:
    """A backgrounded descendant holding the stdout pipe must survive the
    finished-handle release AND have its late output captured (PR #75162
    review claim, guard fix 6 — bounded drain owner)."""

    def test_orphan_descendant_survives_handle_release(self, registry, tmp_path):
        marker = tmp_path / "descendant-survived.marker"
        # Bounded descendant: sleeps 2s (past the pre-fix ~0.7s release
        # window, inside the 5.0s _ORPHAN_DRAIN_WINDOW_S drain window), writes
        # LATE_OUTPUT to the inherited stdout pipe, then writes the survivor
        # marker. If SIGPIPE/EPIPE kills it at the late write, the marker is
        # never created.
        cmd = (
            "( sleep 2; echo LATE_OUTPUT; touch "
            f"{shlex.quote(str(marker))} ) &"
        )
        session = registry.spawn_local(cmd, cwd=str(tmp_path))
        try:
            # Steps 1-2: direct shell exits immediately; the reader loop
            # idle-escapes (issue #68915) and the session moves to finished
            # within ~0.7s. Completion is prompt — the drain owner only
            # defers the handle close, never the finished status.
            assert _wait_until(
                lambda: registry.poll(session.id)["status"] == "exited",
                timeout=10.0,
            ), f"session never reached exited: {registry.poll(session.id)!r}"

            proc = session.process
            stdout_closed = bool(
                proc is not None and proc.stdout is not None and proc.stdout.closed
            )

            # Step 3-4: the descendant writes at t=2s, ~1.3s after the
            # pre-fix release window. It must survive (marker proves it
            # completed its post-write action) AND its late output must be
            # captured by the bounded drain owner.
            survived = _wait_until(lambda: marker.exists(), timeout=8.0)
            captured = _wait_until(
                lambda: "LATE_OUTPUT" in (session.output_buffer or ""),
                timeout=3.0,
            )

            assert survived, (
                "ORPHAN-DESCENDANT POLICY VIOLATION: the backgrounded "
                "descendant was TERMINATED (EPIPE/SIGPIPE) writing its late "
                "output after ProcessRegistry closed the read end — the "
                "reviewer's PR #75162 claim reproduces against local commit "
                "0de805e55.\n"
                "  chosen late-output policy: descendant MUST survive AND its "
                "late output MUST be captured within the drain window\n"
                f"  marker file present: {marker.exists()} ({marker})\n"
                f"  proc.stdout.closed after finish: {stdout_closed}\n"
                f"  LATE_OUTPUT observable in output_buffer: "
                f"{'LATE_OUTPUT' in (session.output_buffer or '')}\n"
                f"  output_buffer tail: {session.output_buffer[-200:]!r}\n"
                f"  session: {registry.poll(session.id)!r}"
            )
            assert captured, (
                "ORPHAN-DRAIN CAPTURE VIOLATION: the descendant SURVIVES but "
                "its late output (LATE_OUTPUT) was NOT captured into "
                "session.output_buffer — the bounded drain owner failed to "
                "drain the orphan-held pipe within _ORPHAN_DRAIN_WINDOW_S.\n"
                f"  proc.stdout.closed after finish: {stdout_closed}\n"
                f"  output_buffer tail: {session.output_buffer[-200:]!r}\n"
                f"  session: {registry.poll(session.id)!r}"
            )
        finally:
            registry.kill_process(session.id)
            # Bounded straggler cleanup: the descendant dies on its own by
            # t=2s, but kill the process group anyway so a surviving/hung
            # descendant can never leak out of the test.
            try:
                if session.process is not None:
                    os.killpg(os.getpgid(session.process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass

    def test_drain_owner_is_bounded_and_never_indefinite(
        self, registry, tmp_path, monkeypatch
    ):
        """A descendant that holds the pipe forever must NOT park the drain
        owner forever: after _ORPHAN_DRAIN_WINDOW_S (monkeypatched small) the
        handle closes regardless, the drain-owner state clears, and the test
        exits cleanly — bounded, never indefinite. This is the FD-hygiene
        property that guard fix 1 (finished sessions must stop holding FDs)
        must not regress."""
        import tools.process_registry as _pr

        # Small window: the descendant holds the pipe for 30s (forever,
        # relative to the 1.5s drain window).
        monkeypatch.setattr(_pr, "_ORPHAN_DRAIN_WINDOW_S", 1.5)

        cmd = "( sleep 30 ) &"
        session = registry.spawn_local(cmd, cwd=str(tmp_path))
        try:
            # Direct shell exits; reader idle-escapes; session finishes.
            assert _wait_until(
                lambda: registry.poll(session.id)["status"] == "exited",
                timeout=10.0,
            ), f"session never reached exited: {registry.poll(session.id)!r}"

            proc = session.process
            assert proc is not None and proc.stdout is not None

            # The drain owner must actually ENGAGE (the deferral is real —
            # the read end is still open right after finish, still held by
            # the bounded owner). The window is 1.5s and we poll immediately
            # after exit, so this is deterministically observable before the
            # deadline.
            engaged = _wait_until(
                lambda: bool(session._drain_owner_active), timeout=1.0
            )
            assert engaged, (
                "ORPHAN-DRAIN ENGAGE VIOLATION: the bounded drain owner never "
                "engaged after the orphan-escape — the handle would be closed "
                "immediately and the descendant SIGPIPE'd on its next write."
            )

            # The window is a HARD bound: the handle closes within window +
            # margin even though the descendant still holds the write end.
            closed = _wait_until(
                lambda: proc.stdout is None or proc.stdout.closed,
                timeout=5.0,  # 1.5s window + generous margin
            )
            assert closed, (
                "ORPHAN-DRAIN BOUND VIOLATION: the drain owner held the pipe "
                "open past _ORPHAN_DRAIN_WINDOW_S — an orphaned descendant "
                "that holds the pipe forever parked the FD indefinitely "
                "(regresses guard fix 1's FD-hygiene intent)."
            )
            assert not session._drain_owner_active, (
                "drain-owner state must be cleared after the window elapses "
                "(the FD charge is returned when the handle actually closes)"
            )
        finally:
            registry.kill_process(session.id)
            # Reap the forever-descendant so it can never leak out of the
            # test run.
            try:
                if session.process is not None:
                    os.killpg(os.getpgid(session.process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
