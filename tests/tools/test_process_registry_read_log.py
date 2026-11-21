"""Tests for ``ProcessRegistry.read_log`` pagination edge cases."""
import time

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture()
def registry():
    return ProcessRegistry()


def _session(output: str) -> ProcessSession:
    return ProcessSession(
        id="proc_test123",
        command="echo hello",
        task_id="t1",
        started_at=time.time(),
        output_buffer=output,
    )


def test_read_with_negative_offset_counts_from_end(registry):
    """Regression for #146914: a negative offset must count back from the end —
    lines[-N:0] is always empty, so models asking for "the last N lines" got nothing."""
    s = _session(output="\n".join(f"line {i}" for i in range(100)))
    registry._running[s.id] = s
    result = registry.read_log(s.id, offset=-10, limit=10)
    assert result["output"].splitlines() == [f"line {i}" for i in range(90, 100)]


def test_read_with_negative_offset_beyond_start_clamps(registry):
    s = _session(output="\n".join(f"line {i}" for i in range(5)))
    registry._running[s.id] = s
    result = registry.read_log(s.id, offset=-80, limit=80)
    assert result["output"].splitlines() == [f"line {i}" for i in range(5)]
