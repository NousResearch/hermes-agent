from __future__ import annotations

import json
from pathlib import Path
import pytest

from workstation.browser_supervisor import (
    BrowserRuntimeSupervisor,
    CircuitBreakerState,
    ControllerErrorKind,
)


def test_stale_descriptor_detection_with_dead_pid(tmp_path):
    """Scenario 3a: Stale descriptor with dead PID is actively detected and cleaned up."""
    control_file = tmp_path / "browser-control.json"
    dead_pid = 9999999  # Guaranteed non-existent PID
    control_file.write_text(
        json.dumps({
            "version": 1,
            "url": "http://127.0.0.1:54321",
            "token": "sec_dummy",
            "pid": dead_pid,
        })
    )

    sup = BrowserRuntimeSupervisor(control_path=control_file)
    assert not sup.is_pid_alive(dead_pid)

    desc, err_kind, detail = sup.inspect_descriptor()
    assert err_kind == ControllerErrorKind.PROCESS_DEAD
    assert "dead" in detail

    # Check that stale descriptor was unlinked
    assert not control_file.exists()


def test_supervisor_error_classification():
    sup = BrowserRuntimeSupervisor()
    exc_refused = OSError("[WinError 10061] No connection could be made because the target machine actively refused it")
    assert sup.classify_error(exc_refused) == ControllerErrorKind.CONNECTION_REFUSED

    exc_timeout = TimeoutError("The read operation timed out")
    assert sup.classify_error(exc_timeout) == ControllerErrorKind.TIMEOUT


def test_supervisor_circuit_breaker(tmp_path):
    """Scenario 3b: Circuit breaker trips after max retries and cools down."""
    sup = BrowserRuntimeSupervisor(
        control_path=tmp_path / "dummy.json",
        max_retries=2,
        backoff_factor=0.01,
        circuit_cooldown_seconds=0.2,
    )

    calls = 0

    def failing_action():
        nonlocal calls
        calls += 1
        raise OSError("[WinError 10061] connection actively refused")

    # First attempt fails after 2 tries
    with pytest.raises(OSError):
        sup.execute_with_supervision(failing_action)
    assert calls == 2
    assert sup._circuit_state == CircuitBreakerState.OPEN

    # While OPEN, immediate attempts are rejected without calling action
    with pytest.raises(RuntimeError, match="circuit breaker is OPEN"):
        sup.execute_with_supervision(failing_action)
    assert calls == 2  # Not called again!
