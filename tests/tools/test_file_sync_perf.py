"""Reproducible perf benchmark for file sync overhead.

Measures actual env.execute() wall-clock time, no LLM in the loop.
Run with: uv run pytest tests/tools/test_file_sync_perf.py -v -o "addopts=" -s

Requires backends to be configured (SSH host, Modal creds, etc).
Skip markers gate each backend.
"""

import statistics
import time

import pytest

# ---------------------------------------------------------------------------
# Backend fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def local_env():
    from tools.environments.local import LocalEnvironment
    env = LocalEnvironment(cwd="/tmp", timeout=30)
    yield env
    env.cleanup()


@pytest.fixture
def ssh_env():
    import os
    host = os.environ.get("TERMINAL_SSH_HOST")
    user = os.environ.get("TERMINAL_SSH_USER")
    if not host or not user:
        pytest.skip("TERMINAL_SSH_HOST and TERMINAL_SSH_USER required")
    from tools.environments.ssh import SSHEnvironment
    env = SSHEnvironment(host=host, user=user, cwd="/tmp", timeout=30)
    yield env
    env.cleanup()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time_executions(env, command: str, n: int = 10) -> list[float]:
    """Run *command* n times and return per-call wall-clock durations."""
    durations = []
    for _ in range(n):
        t0 = time.monotonic()
        result = env.execute(command, timeout=10)
        elapsed = time.monotonic() - t0
        durations.append(elapsed)
        assert result.get("returncode", result.get("exit_code", -1)) == 0, \
            f"command failed: {result}"
    return durations


def _report(label: str, durations: list[float]):
    """Print timing stats."""
    med = statistics.median(durations)
    mean = statistics.mean(durations)
    p95 = sorted(durations)[int(len(durations) * 0.95)]
    print(f"\n  {label}:")
    print(f"    n={len(durations)}  median={med*1000:.0f}ms  mean={mean*1000:.0f}ms  p95={p95*1000:.0f}ms")
    print(f"    raw: {[f'{d*1000:.0f}ms' for d in durations]}")
    return med


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLocalPerf:
    """Local baseline — no file sync, no network. Sets the floor."""

    def test_echo_latency(self, local_env):
        durations = _time_executions(local_env, "echo hello", n=20)
        med = _report("local echo", durations)
        # Spawn-per-call overhead should be < 500ms
        assert med < 0.5, f"local echo median {med*1000:.0f}ms exceeds 500ms"


@pytest.mark.ssh
class TestSSHPerf:
    """SSH with FileSyncManager — mtime skip should make sync ~0ms."""

    def test_echo_latency(self, ssh_env):
        """Sequential echo commands — measures per-command overhead including sync check."""
        durations = _time_executions(ssh_env, "echo hello", n=20)
        med = _report("ssh echo (with sync check)", durations)
        # SSH round-trip + spawn-per-call, but sync should be ~0ms (rate limited)
        assert med < 2.0, f"ssh echo median {med*1000:.0f}ms exceeds 2000ms"

    def test_sync_overhead_after_interval(self, ssh_env):
        """Measure sync cost when the rate-limit window has expired.

        Sleep past the 5s interval, then time the next command which
        triggers a real sync cycle (but with mtime skip, should be fast).
        """
        # Warm up
        ssh_env.execute("echo warmup", timeout=10)

        # Wait for sync interval to expire
        time.sleep(6)

        # This command will trigger a real sync cycle
        t0 = time.monotonic()
        result = ssh_env.execute("echo after-interval", timeout=10)
        elapsed = time.monotonic() - t0

        print(f"\n  ssh echo after 6s wait (sync triggered): {elapsed*1000:.0f}ms")
        assert result.get("returncode", result.get("exit_code", -1)) == 0

        # Even with sync triggered, mtime skip should keep it fast
        # Old rsync approach: ~2-3s. New mtime skip: should be < 1.5s
        assert elapsed < 1.5, f"sync-triggered command took {elapsed*1000:.0f}ms (expected < 1500ms)"

    def test_no_sync_within_interval(self, ssh_env):
        """Rapid sequential commands within 5s window — no sync at all."""
        # First command triggers sync
        ssh_env.execute("echo prime", timeout=10)

        # Immediately run 10 more — all within rate-limit window
        durations = _time_executions(ssh_env, "echo rapid", n=10)
        med = _report("ssh echo (within interval, no sync)", durations)

        # Should be pure SSH overhead, no sync
        assert med < 1.5, f"within-interval median {med*1000:.0f}ms exceeds 1500ms"
