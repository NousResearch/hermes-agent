"""The kanban rate limiters' MODULE DEFAULTS must not swallow the first call on a freshly booted host.

Both limiters compare a module-level seed against ``time.monotonic()``, whose origin is host boot on
Linux. A ``0.0`` seed reads as "attempted just now" while uptime is below the interval, so a worker
on a fresh host (or a CI microVM reaching the test 30-60 s after boot) drops its first heartbeat.
Each case runs in a fresh interpreter so the real import-time default is exercised, not a value an
earlier test left behind.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_PRELUDE = (
    "import os, time\n"
    "_real = time.monotonic; _boot = _real() - 5.0\n"
    "time.monotonic = lambda: _real() - _boot  # host booted 5 s ago, clock still advancing\n"
    "os.environ['HERMES_KANBAN_TASK'] = 't_fresh'\n"
    "from tools import kanban_tools as kt\n"
)

_CASES = {
    # The limiter stamps its seed only once the call is past the interval check.
    "auto_heartbeat": (
        "seed = kt._auto_heartbeat_last_attempt\n"
        "kt.heartbeat_current_worker_from_env()\n"
        "print('PASSED_LIMITER', kt._auto_heartbeat_last_attempt != seed)\n"
    ),
    "comment_poll": (
        "seed = kt._comment_poll_last_attempt\n"
        "print('PASSED_LIMITER', time.monotonic() - seed >= kt._COMMENT_POLL_MIN_INTERVAL_SECONDS)\n"
    ),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_first_call_is_not_rate_limited_on_a_freshly_booted_host(case, tmp_path):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("HERMES_KANBAN_", "HERMES_DELEGATED_"))}
    env["HERMES_HOME"] = str(tmp_path / ".hermes")
    env["PYTHONPATH"] = str(_REPO_ROOT)
    out = subprocess.run(
        [sys.executable, "-c", _PRELUDE + _CASES[case]], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "PASSED_LIMITER True" in out.stdout, (
        f"{case}: the first call was rate-limited away on a fresh-boot clock\n" + out.stdout[-500:]
    )
