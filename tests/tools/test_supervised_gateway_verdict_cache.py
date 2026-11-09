"""The supervised-gateway identity verdict is memoized per process+environment.

``_is_supervised_gateway_process()`` runs on every terminal command (and every
``execute_code`` call, and every process-ownership check). Its probe does filesystem and
kernel-process work, and on macOS a single ``open()`` in that chain can block forever,
so a per-call probe turned one kernel wedge into a leak of one abandoned deadline thread
per tool call on a long-lived gateway (board #497: 140 parked
``deadline-terminal.pre-exec-guard`` threads on one pid).
"""

from __future__ import annotations

import pytest

import tools.process_registry as registry


@pytest.fixture(autouse=True)
def _fresh_verdict(monkeypatch):
    registry._reset_supervised_gateway_verdict_cache()
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    # A gateway's children inherit HERMES_SUPERVISED_CHILD, so start from a known state.
    monkeypatch.delenv("HERMES_SUPERVISED_CHILD", raising=False)
    yield
    registry._reset_supervised_gateway_verdict_cache()


def _stub_probe(monkeypatch, *, verdict: bool = True, explode: bool = False):
    """Count probe executions without importing the real gateway modules."""
    calls: list[str] = []

    def _probe(*_a, **_k):
        calls.append("probe")
        if explode:
            raise RuntimeError("probe blew up")
        return verdict

    monkeypatch.setattr("gateway.restart.is_supervised_gateway_launch", _probe, raising=False)
    monkeypatch.setattr("gateway.status.get_running_pid", lambda **_k: registry.os.getpid(), raising=False)
    return calls


def test_repeated_calls_probe_once(monkeypatch):
    calls = _stub_probe(monkeypatch)
    for _ in range(25):
        assert registry._is_supervised_gateway_process() is True
    assert len(calls) == 1, f"identity probe re-ran {len(calls)}x for an invariant verdict"


def test_marker_change_reprobes(monkeypatch):
    calls = _stub_probe(monkeypatch)
    assert registry._is_supervised_gateway_process() is True
    monkeypatch.setenv("HERMES_SUPERVISED_CHILD", "1")
    assert registry._is_supervised_gateway_process() is True
    assert len(calls) == 2, "a changed marker must invalidate the cached verdict"


def test_cached_verdict_for_another_pid_is_not_reused(monkeypatch):
    calls = _stub_probe(monkeypatch)
    other_pid = registry.os.getpid() + 1
    registry._SUPERVISED_GATEWAY_VERDICT = ((other_pid, "1", ""), True)
    assert registry._is_supervised_gateway_process() is True
    assert len(calls) == 1, "a verdict cached under a different pid must not be trusted"


def test_non_gateway_process_caches_false_without_probing(monkeypatch):
    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
    calls = _stub_probe(monkeypatch)
    assert registry._is_supervised_gateway_process() is False
    assert registry._is_supervised_gateway_process() is False
    assert calls == [], "a process with no gateway marker must not run the probe at all"


def test_a_raising_probe_is_never_cached(monkeypatch):
    calls = _stub_probe(monkeypatch, explode=True)
    assert registry._is_supervised_gateway_process() is False
    assert registry._is_supervised_gateway_process() is False
    assert len(calls) == 2, "a failed probe must be retried, not frozen into 'not the gateway'"


def test_transient_pid_mismatch_reprobes_then_caches_the_confirmed_true(monkeypatch):
    """A gateway PID file that is missing or still names the previous pid at startup must not
    freeze the process into "not the gateway" — that would disable the self-kill and ownership
    guards for the process's whole life. Once the file names us, the True verdict is reused."""
    launches: list[str] = []
    this_pid = registry.os.getpid()
    pid_reads = iter([this_pid + 1, this_pid, this_pid + 1])

    def _launch():
        launches.append("launch")
        return True

    monkeypatch.setattr("gateway.restart.is_supervised_gateway_launch", _launch)
    monkeypatch.setattr("gateway.status.get_running_pid", lambda **_k: next(pid_reads))

    assert registry._is_supervised_gateway_process() is False
    assert registry._is_supervised_gateway_process() is True
    assert registry._is_supervised_gateway_process() is True
    assert launches == ["launch", "launch"], "only a confirmed identity may be memoized"
