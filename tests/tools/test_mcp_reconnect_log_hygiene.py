"""Tests for MCP reconnect log hygiene and backoff jitter (#65673, #66092).

The retry/park machinery used to emit one WARNING per retry attempt — a
flapping server produced thousands of identical lines (#62212: 6212 spawns
in 63h). Now:

- per-attempt retry logs are DEBUG;
- state transitions carry exactly one WARNING each
  (connected→degraded, degraded→parked, parked→revived);
- backoff sleeps get ±20% jitter so herds of servers don't retry in
  lockstep.

#105190 extends the same contract across self-probe cycles: a parked server
that stays dead wakes every ``_PARKED_RETRY_INTERVAL`` (#57129) and used to
re-log the SAME park WARNING on every cycle (4,615 identical lines in 3 weeks
on the reporting host). Now each park entry warns exactly once per episode
(an episode ends when a revival proves healthy); repeats stay DEBUG with the
message text unchanged, so the retry cadence remains diagnosable.
"""

import asyncio
import logging
import time
from types import SimpleNamespace

import pytest

from tools.mcp_tool import MCPServerTask
from tools.mcp_tool_common import _jittered

# Captured at import, before any test patches asyncio.sleep: the pump helper below
# needs the REAL sleep so event-loop timers keep elapsing while we poll.
_REAL_SLEEP = asyncio.sleep


async def _pump_until(predicate, timeout: float = 4.0) -> bool:
    """Advance the event loop with real (tiny) sleeps until ``predicate()`` is true.

    Polling with ``sleep(0)`` yields control but never advances the loop clock, so
    timer-based waits — the parked self-probe ``asyncio.wait(timeout=...)`` among
    them — cannot fire; on a slow shared runner the poll loop exhausts its
    iterations before a single 10ms park interval elapses (CI flake on #105190).
    Bounding by wall clock instead of iteration count keeps that deterministic.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await _REAL_SLEEP(0.001)
        if predicate():
            return True
    return predicate()


# ── Jitter ───────────────────────────────────────────────────────────────────

class TestJitter:
    def test_jitter_within_20_percent(self):
        for _ in range(200):
            v = _jittered(10.0)
            assert 8.0 <= v <= 12.0

    def test_jitter_varies(self):
        values = {_jittered(10.0) for _ in range(50)}
        assert len(values) > 1, "jitter produced constant values"


# ── Log levels: retry chatter DEBUG, transitions WARNING ─────────────────────

@pytest.mark.no_isolate
def test_retry_attempts_log_debug_transitions_warn(monkeypatch, tmp_path, caplog):
    """Consecutive transient failures: each retry logs at DEBUG, and the
    degraded→parked transition logs exactly one WARNING."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_MAX_RECONNECT_RETRIES", 2)

    _real_sleep = asyncio.sleep

    async def _fast_sleep(_delay, *a, **kw):
        await _real_sleep(0)

    monkeypatch.setattr(mcp_tool.asyncio, "sleep", _fast_sleep)

    state = {"transport_calls": 0, "parked": False}

    async def _scenario():
        class _Task(MCPServerTask):
            def _is_http(self):
                return False

            def _deregister_tools(self):
                state["parked"] = True
                self._registered_tool_names = []

            async def _run_stdio(self, config):
                state["transport_calls"] += 1
                if state["transport_calls"] == 1:
                    self.session = object()
                    self._ready.set()
                    self._ever_connected = True
                    self.session = None
                raise ConnectionError("backend down")

        task = _Task("noisy")
        task._registered_tool_names = ["noisy__tool"]

        with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
            run_task = asyncio.ensure_future(task.run({"command": "x"}))
            for _ in range(1000):
                await _real_sleep(0)
                if state["parked"]:
                    break

        assert state["parked"]

        task._shutdown_event.set()
        task._reconnect_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            run_task.cancel()

    asyncio.run(_scenario())

    retry_records = [
        r for r in caplog.records if "connection lost (attempt" in r.getMessage()
    ]
    assert retry_records, "no per-attempt retry logs at all"
    assert all(r.levelno == logging.DEBUG for r in retry_records), (
        "per-attempt retry logs must be DEBUG, got: "
        + str({r.levelname for r in retry_records})
    )

    park_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "parking" in r.getMessage()
    ]
    assert len(park_warnings) == 1, (
        f"expected exactly 1 degraded→parked WARNING, got {len(park_warnings)}"
    )
    assert "degraded → parked" in park_warnings[0].getMessage()


@pytest.mark.no_isolate
def test_initial_retry_attempts_log_debug(monkeypatch, tmp_path, caplog):
    """Initial-connect per-attempt retries are DEBUG; only the final park
    (connecting→parked) is a WARNING."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    _real_sleep = asyncio.sleep

    async def _fast_sleep(_delay, *a, **kw):
        await _real_sleep(0)

    monkeypatch.setattr(mcp_tool.asyncio, "sleep", _fast_sleep)

    state = {"parked": False}

    async def _scenario():
        class _Task(MCPServerTask):
            def _is_http(self):
                return False

            def _deregister_tools(self):
                state["parked"] = True
                self._registered_tool_names = []

            async def _run_stdio(self, config):
                raise ConnectionError("dns blip")

        task = _Task("startup")

        with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
            run_task = asyncio.ensure_future(task.run({"command": "x"}))
            for _ in range(1000):
                await _real_sleep(0)
                if state["parked"]:
                    break

        assert state["parked"]

        task._shutdown_event.set()
        task._reconnect_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            run_task.cancel()

    asyncio.run(_scenario())

    attempt_records = [
        r for r in caplog.records
        if "initial connection failed (attempt" in r.getMessage()
    ]
    assert attempt_records
    assert all(r.levelno == logging.DEBUG for r in attempt_records)

    park_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and "connecting → parked" in r.getMessage()
    ]
    assert len(park_warnings) == 1


# ── Park-warning deduplication across self-probe cycles (#105190) ────────────
#
# A parked server wakes every _PARKED_RETRY_INTERVAL for a self-probe (#57129).
# A still-dead server re-enters run(), re-exhausts the park-entry ladder, and
# used to re-log the SAME park WARNING on every cycle. Contract: each park
# entry warns exactly once per episode (an episode ends at a proven-healthy
# revival); repeats stay DEBUG with the message text unchanged.


def _auth_error(status_code: int = 401):
    """The repo's own OAuth error type — verifies as auth → permanent under
    both ``_classify_mcp_failure`` and ``_is_auth_error`` (a bare 401/403
    ``.response`` feeds the classifier but not the auth matcher)."""
    from tools.mcp_oauth import OAuthNonInteractiveError

    err = OAuthNonInteractiveError(f"{status_code} auth rejected")
    err.response = SimpleNamespace(status_code=status_code)  # httpx-shaped marker for the classifier
    return err


@pytest.mark.no_isolate
def test_parked_initial_failure_warns_once_across_self_probe_cycles(monkeypatch, tmp_path, caplog):
    """Layer 1: initial-connect ladder exhaustion. The 'connecting → parked'
    WARNING fires once; later self-probe re-parks log DEBUG, not WARNING."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_MAX_INITIAL_CONNECT_RETRIES", 2)
    # Keep the parked self-probe cadence tiny so several cycles fit in the test.
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 0.01)

    _real_sleep = asyncio.sleep

    async def _fast_sleep(_delay, *a, **kw):
        await _real_sleep(0)

    monkeypatch.setattr(mcp_tool.asyncio, "sleep", _fast_sleep)

    state = {"transport_calls": 0, "parked": 0}

    async def _scenario():
        class _Task(MCPServerTask):
            def _is_http(self):
                return False

            def _deregister_tools(self):
                state["parked"] += 1
                self._registered_tool_names = []

            async def _run_stdio(self, config):
                state["transport_calls"] += 1
                raise ConnectionError("backend down")

        task = _Task("deadstart")
        task._registered_tool_names = ["deadstart__tool"]

        with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
            run_task = asyncio.ensure_future(task.run({"command": "x"}))
            assert await _pump_until(lambda: state["parked"] >= 3), (
                f"scenario never exercised repeated park cycles "
                f"(parks={state['parked']}, transport_calls={state['transport_calls']})"
            )

        assert state["parked"] >= 3

        task._shutdown_event.set()
        task._reconnect_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            run_task.cancel()

    asyncio.run(_scenario())

    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "connecting → parked" in r.getMessage()
    ]
    assert len(warnings) == 1, (
        f"expected exactly 1 'connecting → parked' WARNING across {state['parked']} parks, "
        f"got {len(warnings)}"
    )
    # Repeats must remain observable at DEBUG with the message text unchanged —
    # the issue asks that diagnosis still show the cadence.
    debug_repeats = [
        r for r in caplog.records
        if r.levelno == logging.DEBUG and "connecting → parked" in r.getMessage()
    ]
    assert debug_repeats, "repeated parks vanished from logs entirely (no DEBUG records)"


@pytest.mark.no_isolate
def test_parked_permanent_initial_failure_warns_once(monkeypatch, tmp_path, caplog):
    """Layer 2: permanent initial failure (auth 401/403). The actionable
    're-authenticate with `hermes mcp login`' guidance must not repeat forever
    while credentials stay bad."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 0.01)

    _real_sleep = asyncio.sleep

    async def _fast_sleep(_delay, *a, **kw):
        await _real_sleep(0)

    monkeypatch.setattr(mcp_tool.asyncio, "sleep", _fast_sleep)

    state = {"transport_calls": 0, "parked": 0}

    async def _scenario():
        class _Task(MCPServerTask):
            def _is_http(self):
                return False

            def _deregister_tools(self):
                state["parked"] += 1
                self._registered_tool_names = []

            async def _run_stdio(self, config):
                state["transport_calls"] += 1
                raise _auth_error(401)

        task = _Task("locked")

        with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
            run_task = asyncio.ensure_future(task.run({"command": "x"}))
            assert await _pump_until(lambda: state["parked"] >= 3), (
                f"scenario never exercised repeated park cycles (parks={state['parked']})"
            )

        assert state["parked"] >= 3

        task._shutdown_event.set()
        task._reconnect_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            run_task.cancel()

    asyncio.run(_scenario())

    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "connecting → parked" in r.getMessage()
    ]
    assert len(warnings) == 1, (
        f"expected exactly 1 permanent 'connecting → parked' WARNING, got {len(warnings)}"
    )
    assert any("hermes mcp login" in r.getMessage() for r in warnings), (
        "permanent auth park must carry the re-authenticate guidance"
    )


@pytest.mark.no_isolate
def test_parked_flapping_session_warns_once_per_episode(monkeypatch, tmp_path, caplog):
    """Layers 3+5: a server that connects, never proves healthy, and drops
    (rapid-drop budget / reconnect ladder exhausted) re-parks every self-probe
    cycle; each episode warns once. Also the revival edge: after the backend
    recovers and the session proves healthy, the NEXT park is a new episode
    and warns again."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_MAX_RECONNECT_RETRIES", 1)
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 0.01)

    _real_sleep = asyncio.sleep

    async def _fast_sleep(_delay, *a, **kw):
        await _real_sleep(0)

    monkeypatch.setattr(mcp_tool.asyncio, "sleep", _fast_sleep)

    state = {
        "transport_calls": 0,
        "parked": 0,
        "backend_up": False,
    }

    async def _scenario():
        class _Task(MCPServerTask):
            def _is_http(self):
                return False

            def _deregister_tools(self):
                state["parked"] += 1
                self._registered_tool_names = []

            async def _run_stdio(self, config):
                state["transport_calls"] += 1
                if not state["backend_up"]:
                    # Handshake succeeds then drops: unproven, charged, parks
                    # via the rapid-drop budget or the reconnect ladder.
                    self.session = object()
                    self._ready.set()
                    self._ever_connected = True
                    self.session = None
                    raise RuntimeError("handshake then drop")
                # Backend recovered: establish a session that proves healthy.
                self.session = object()
                self._ready.set()
                self._ever_connected = True
                self._mark_session_proven()
                return await self._wait_for_lifecycle_event()

        task = _Task("flapper")
        task._registered_tool_names = ["flapper__tool"]

        with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
            run_task = asyncio.ensure_future(task.run({"command": "x"}))
            # Phase 1: park + several dead self-probe cycles.
            assert await _pump_until(lambda: state["parked"] >= 3), (
                f"never parked repeatedly (parks={state['parked']})"
            )

            phase1_warnings = [
                r for r in caplog.records
                if r.levelno == logging.WARNING
                and ("degraded → parked" in r.getMessage() or "connected → parked" in r.getMessage())
            ]
            assert len(phase1_warnings) == 1, (
                f"expected exactly 1 park WARNING in episode 1, got {len(phase1_warnings)}: "
                + str([r.getMessage()[:90] for r in phase1_warnings])
            )

            # Phase 2: backend recovers; the self-probe revives the session.
            state["backend_up"] = True
            assert await _pump_until(lambda: task._session_proven), (
                "parked server never revived to a proven session"
            )

            # Phase 3: backend dies again — a NEW park episode must warn again.
            state["backend_up"] = False
            task._reconnect_event.set()  # end the lifecycle wait
            assert await _pump_until(lambda: state["parked"] >= 4), (
                f"never re-parked after revival (parks={state['parked']})"
            )

        task._shutdown_event.set()
        task._reconnect_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            run_task.cancel()

    asyncio.run(_scenario())

    park_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and ("degraded → parked" in r.getMessage() or "connected → parked" in r.getMessage())
    ]
    revived = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "revived" in r.getMessage()
    ]
    # Episode 1 (many parks) + episode 2 (fresh park after revival) = 2 WARNINGs.
    assert len(park_warnings) == 2, (
        f"expected exactly 2 park WARNINGs (one per episode), got {len(park_warnings)}: "
        + str([r.getMessage()[:90] for r in park_warnings])
    )
    assert revived, "revival WARNING missing — episode boundary never crossed"


@pytest.mark.no_isolate
def test_parked_permanent_error_on_proven_session_warns_once(monkeypatch, tmp_path, caplog):
    """Layer 4: permanent error on a proven session (connected → parked) also
    re-logged every self-probe cycle while the server stays dead."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_MAX_RECONNECT_RETRIES", 1)
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 0.01)

    _real_sleep = asyncio.sleep

    async def _fast_sleep(_delay, *a, **kw):
        await _real_sleep(0)

    monkeypatch.setattr(mcp_tool.asyncio, "sleep", _fast_sleep)

    state = {"transport_calls": 0, "parked": 0}

    async def _scenario():
        class _Task(MCPServerTask):
            def _is_http(self):
                return False

            def _deregister_tools(self):
                state["parked"] += 1
                self._registered_tool_names = []

            async def _run_stdio(self, config):
                state["transport_calls"] += 1
                if state["transport_calls"] == 1:
                    # First connect succeeds and proves healthy...
                    self.session = object()
                    self._ready.set()
                    self._ever_connected = True
                    self._mark_session_proven()
                    # ...then the next transport call hits a permanent error.
                    raise _auth_error(403)
                # Self-probe cycles: still permanently dead.
                raise _auth_error(403)

        task = _Task("revoked")
        task._registered_tool_names = ["revoked__tool"]

        with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
            run_task = asyncio.ensure_future(task.run({"command": "x"}))
            assert await _pump_until(lambda: state["parked"] >= 3), (
                f"scenario never exercised repeated park cycles (parks={state['parked']})"
            )

        assert state["parked"] >= 3

        task._shutdown_event.set()
        task._reconnect_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            run_task.cancel()

    asyncio.run(_scenario())

    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "connected → parked" in r.getMessage()
    ]
    assert len(warnings) == 1, (
        f"expected exactly 1 'connected → parked' WARNING, got {len(warnings)}"
    )


@pytest.mark.no_isolate
def test_park_failure_mode_change_still_warns(monkeypatch, tmp_path, caplog):
    """Layer 7: when the failure mode changes mid-episode (transient
    ConnectError → permanent 401 with re-auth guidance), the new, different
    park message is a new signal and must still WARN, not be suppressed as a
    repeat."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_MAX_RECONNECT_RETRIES", 1)
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 0.01)

    _real_sleep = asyncio.sleep

    async def _fast_sleep(_delay, *a, **kw):
        await _real_sleep(0)

    monkeypatch.setattr(mcp_tool.asyncio, "sleep", _fast_sleep)

    state = {"transport_calls": 0, "parked": 0}

    def _perm_403():
        return _auth_error(403)

    async def _scenario():
        class _Task(MCPServerTask):
            def _is_http(self):
                return False

            def _deregister_tools(self):
                state["parked"] += 1
                self._registered_tool_names = []

            async def _run_stdio(self, config):
                state["transport_calls"] += 1
                if state["parked"] >= 2:
                    # Mid-episode switch: credentials revoked after the first park.
                    raise _perm_403()
                if state["transport_calls"] == 1:
                    self.session = object()
                    self._ready.set()
                    self._ever_connected = True
                    self.session = None
                raise ConnectionError("transient network blip")

        task = _Task("shifting")
        task._registered_tool_names = ["shifting__tool"]

        with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
            run_task = asyncio.ensure_future(task.run({"command": "x"}))
            assert await _pump_until(lambda: state["parked"] >= 4), (
                f"scenario never exercised repeated park cycles (parks={state['parked']})"
            )

        assert state["parked"] >= 4

        task._shutdown_event.set()
        task._reconnect_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            run_task.cancel()

    asyncio.run(_scenario())

    degraded_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "degraded → parked" in r.getMessage()
    ]
    connected_park_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "connected → parked" in r.getMessage()
    ]
    # First episode: one transient 'degraded → parked'. Then the failure mode
    # changed to permanent — that different park message must still surface.
    assert len(degraded_warnings) == 1, (
        f"expected exactly 1 'degraded → parked' WARNING, got {len(degraded_warnings)}"
    )
    assert connected_park_warnings, (
        "permanent failure-mode change mid-episode was suppressed — the new "
        "park message must still WARN"
    )


@pytest.mark.no_isolate
def test_park_repeat_log_keeps_message_text(monkeypatch, tmp_path, caplog):
    """Layer 8: the DEBUG repeat keeps the full park message (server name,
    attempt count, state transition, root cause) so the cadence remains
    diagnosable — only the level drops, nothing else."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_MAX_INITIAL_CONNECT_RETRIES", 1)
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 0.01)

    _real_sleep = asyncio.sleep

    async def _fast_sleep(_delay, *a, **kw):
        await _real_sleep(0)

    monkeypatch.setattr(mcp_tool.asyncio, "sleep", _fast_sleep)

    state = {"transport_calls": 0, "parked": 0}

    async def _scenario():
        class _Task(MCPServerTask):
            def _is_http(self):
                return False

            def _deregister_tools(self):
                state["parked"] += 1
                self._registered_tool_names = []

            async def _run_stdio(self, config):
                state["transport_calls"] += 1
                raise ConnectionError("backend down")

        task = _Task("verbatim")
        task._registered_tool_names = ["verbatim__tool"]

        with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
            run_task = asyncio.ensure_future(task.run({"command": "x"}))
            assert await _pump_until(lambda: state["parked"] >= 3), (
                f"scenario never exercised repeated park cycles (parks={state['parked']})"
            )

        task._shutdown_event.set()
        task._reconnect_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            run_task.cancel()

    asyncio.run(_scenario())

    all_park_lines = [
        r for r in caplog.records if "connecting → parked" in r.getMessage()
    ]
    assert len(all_park_lines) >= 2, "expected a WARNING + at least one DEBUG repeat"
    warning_lines = [r for r in all_park_lines if r.levelno == logging.WARNING]
    debug_lines = [r for r in all_park_lines if r.levelno == logging.DEBUG]
    assert len(warning_lines) == 1
    assert debug_lines
    # The rendered message text is IDENTICAL — only the level changed.
    assert warning_lines[0].getMessage() == debug_lines[0].getMessage()
