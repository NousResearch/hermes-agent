"""The supervised-gateway identity verdict: memoize ONLY a confirmed True.

The verdict is invariant for one process with one environment and the probe behind it does
real filesystem and kernel-process work that can wedge, so the terminal guard (which calls it
on every command) must not re-run it once it has answered True.

A *negative* answer is not invariant and must never be memoized: the gateway PID file is
written after the process starts, so an early read can see a missing or stale file and answer
False. Freezing that False would disable the self-kill and ownership guards that key on this
identity for the rest of the process's life -- the exact hole the guards exist to close.
"""

import os

import pytest

import tools.process_registry as registry


@pytest.fixture(autouse=True)
def _fresh_verdict(monkeypatch):
    registry._reset_supervised_gateway_verdict_cache()
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.delenv("HERMES_SUPERVISED_CHILD", raising=False)
    yield
    registry._reset_supervised_gateway_verdict_cache()


def _stub(monkeypatch, *, launch=True, pids=None):
    """Stub the two gateway probes; ``pids`` is a list consumed one read per call."""
    calls = {"launch": 0, "pid": 0}

    def _launch(*_a, **_k):
        calls["launch"] += 1
        return launch

    queue = list(pids) if pids is not None else None

    def _pid(**_k):
        calls["pid"] += 1
        if queue:
            return queue.pop(0)
        return os.getpid()

    monkeypatch.setattr("gateway.restart.is_supervised_gateway_launch", _launch, raising=False)
    monkeypatch.setattr("gateway.status.get_running_pid", _pid, raising=False)
    return calls


def test_confirmed_true_is_probed_once(monkeypatch):
    """A confirmed identity must not be re-probed: the guard calls this on every command."""
    calls = _stub(monkeypatch)
    for _ in range(25):
        assert registry._is_supervised_gateway_process() is True
    assert calls["launch"] == 1, f"identity probe re-ran {calls['launch']}x for an invariant verdict"


def test_transient_false_does_not_freeze_the_verdict(monkeypatch):
    """The regression this test exists for.

    The gateway PID file can be missing or still name the previous pid right after boot. A
    False read at that moment must be retried, not cached: caching it makes every later call
    answer "not the gateway", which silently disables the self-kill guards for the process's
    whole life (#113667).
    """
    stale_then_live = _stub(monkeypatch, pids=[os.getpid() + 1])
    assert registry._is_supervised_gateway_process() is False, "stale pid should read as False"

    # The pid file now names us. This MUST re-probe, or the identity stays wedged False.
    assert registry._is_supervised_gateway_process() is True, (
        "a cached False survived a pid file that now names this process -- the self-kill "
        "guards are disabled for the rest of the process's life"
    )
    assert stale_then_live["launch"] == 2, "the second call must have re-run the probe"

    # ...and the now-confirmed True is reused.
    assert registry._is_supervised_gateway_process() is True
    assert stale_then_live["launch"] == 2, "a confirmed True must not re-probe"


def test_unsupervised_launch_is_not_memoized(monkeypatch):
    """launch=False is the other transient negative: an unsupervised shell reading the marker."""
    calls = _stub(monkeypatch, launch=False)
    assert registry._is_supervised_gateway_process() is False
    assert registry._is_supervised_gateway_process() is False
    assert calls["launch"] == 2, "an unconfirmed negative must be re-probed, not memoized"


def test_a_raising_probe_is_never_cached(monkeypatch):
    calls = {"n": 0}

    def _boom(*_a, **_k):
        calls["n"] += 1
        raise RuntimeError("probe blew up")

    monkeypatch.setattr("gateway.restart.is_supervised_gateway_launch", _boom, raising=False)
    monkeypatch.setattr("gateway.status.get_running_pid", lambda **_k: os.getpid(), raising=False)
    assert registry._is_supervised_gateway_process() is False
    assert registry._is_supervised_gateway_process() is False
    assert calls["n"] == 2, "a failed probe must be retried, not frozen into 'not the gateway'"


def test_marker_change_reprobes(monkeypatch):
    calls = _stub(monkeypatch)
    assert registry._is_supervised_gateway_process() is True
    monkeypatch.setenv("HERMES_SUPERVISED_CHILD", "1")
    assert registry._is_supervised_gateway_process() is True
    assert calls["launch"] == 2, "a changed marker must invalidate the cached verdict"


def test_no_gateway_marker_short_circuits_before_any_probe(monkeypatch):
    """Every non-gateway caller must stay free: no marker returns before importing gateway."""
    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
    calls = _stub(monkeypatch)
    assert registry._is_supervised_gateway_process() is False
    assert registry._is_supervised_gateway_process() is False
    assert calls["launch"] == 0, "a process with no gateway marker must not run the probe at all"


def test_verdict_cached_under_another_pid_is_not_reused(monkeypatch):
    calls = _stub(monkeypatch)
    registry._SUPERVISED_GATEWAY_VERDICT = ((os.getpid() + 1, "1", ""), True)
    assert registry._is_supervised_gateway_process() is True
    assert calls["launch"] == 1, "a verdict cached under a different pid must not be trusted"
