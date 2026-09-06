"""Circuit-breaker half-open semantics for tirith_security (fork patch #44).

Pins the three T5-mandated behaviors:
1. any COMPLETED scan (exit 0/1/2) fully closes the breaker,
2. half-open probing is single-flight (claim re-arms the TTL),
3. a failed probe re-arms the timer (breaker stays open).
Plus the crash-streak reset on block/warn verdicts (pre-existing counter bug).
"""
import threading
import time

import pytest

import tools.tirith_security as ts

_real_time = time.time


class _Result:
    def __init__(self, code, stdout=""):
        self.returncode = code
        self.stdout = stdout


@pytest.fixture(autouse=True)
def _breaker_sandbox(monkeypatch):
    monkeypatch.setattr(
        ts,
        "_load_security_config",
        lambda: {
            "tirith_enabled": True,
            "tirith_path": "/fake/tirith",
            "tirith_timeout": 5,
            "tirith_fail_open": True,
        },
    )
    monkeypatch.setattr(ts, "is_platform_supported", lambda: True)
    monkeypatch.setattr(ts, "_resolve_tirith_path", lambda p: "/fake/tirith")
    ts._crash_count = 0
    ts._circuit_open = False
    ts._circuit_open_at = 0.0
    yield
    ts._crash_count = 0
    ts._circuit_open = False
    ts._circuit_open_at = 0.0


def _open_breaker(age_s=None):
    ts._crash_count = ts._CRASH_LIMIT
    ts._circuit_open = True
    ts._circuit_open_at = time.monotonic() - (
        ts._CIRCUIT_RETRY_S + 1 if age_s is None else age_s
    )


@pytest.mark.parametrize("code,action", [(0, "allow"), (1, "block"), (2, "warn")])
def test_completed_scan_closes_breaker(monkeypatch, code, action):
    """Half-open probe returning ANY verdict (allow/block/warn) closes the breaker."""
    _open_breaker()
    spawns = []
    monkeypatch.setattr(
        ts.subprocess, "run", lambda *a, **k: spawns.append(1) or _Result(code)
    )
    out = ts.check_command_security("echo hi")
    assert out["action"] == action
    assert spawns == [1]
    assert ts._circuit_open is False
    assert ts._circuit_open_at == 0.0
    assert ts._crash_count == 0


def test_breaker_within_ttl_never_spawns(monkeypatch):
    _open_breaker(age_s=1)  # freshly opened
    monkeypatch.setattr(
        ts.subprocess, "run", lambda *a, **k: pytest.fail("must not spawn inside TTL")
    )
    out = ts.check_command_security("echo hi")
    assert out["action"] == "allow"
    assert "circuit breaker" in out["summary"]


def test_half_open_claim_blocks_followup_within_ttl(monkeypatch):
    """A failed probe re-arms the timestamp; the next caller inside the fresh TTL
    must fail open WITHOUT spawning (single-flight via timestamp claim)."""
    _open_breaker()

    def boom(*a, **k):
        raise OSError("binary gone")

    monkeypatch.setattr(ts.subprocess, "run", boom)
    out1 = ts.check_command_security("echo hi")
    assert out1["action"] == "allow"
    assert ts._circuit_open is True  # re-armed, not closed

    spawns = []
    monkeypatch.setattr(
        ts.subprocess, "run", lambda *a, **k: spawns.append(1) or _Result(0)
    )
    out2 = ts.check_command_security("echo hi")
    assert out2["action"] == "allow"
    assert spawns == []  # inside re-armed TTL: no probe


def test_half_open_concurrent_second_caller_does_not_probe(monkeypatch):
    """While the claimer's probe is in flight, a concurrent caller fails open
    without spawning a second probe."""
    _open_breaker()
    entered = threading.Event()
    release = threading.Event()
    spawns = []

    def slow_run(*a, **k):
        spawns.append(1)
        entered.set()
        release.wait(5)
        return _Result(0)

    monkeypatch.setattr(ts.subprocess, "run", slow_run)
    results = {}
    t1 = threading.Thread(
        target=lambda: results.__setitem__("t1", ts.check_command_security("x"))
    )
    t1.start()
    assert entered.wait(5), "claimer never reached the probe"

    results["t2"] = ts.check_command_security("x")  # concurrent caller
    assert results["t2"]["action"] == "allow"
    assert len(spawns) == 1  # single-flight held

    release.set()
    t1.join(5)
    assert results["t1"]["action"] == "allow"
    assert ts._circuit_open is False  # claimer's success closed the breaker


def test_failed_probe_rearms_timer(monkeypatch):
    _open_breaker()

    def boom(*a, **k):
        raise OSError("binary gone")

    monkeypatch.setattr(ts.subprocess, "run", boom)
    before = time.monotonic()
    out = ts.check_command_security("x")
    after = time.monotonic()
    assert out["action"] == "allow"
    assert ts._circuit_open is True
    # Bounded on BOTH sides: >= before alone would accept an epoch value from
    # time.time() (always astronomically larger), i.e. it would not pin the
    # re-arm to the monotonic domain at all.
    assert before <= ts._circuit_open_at <= after  # timer re-armed for a fresh window


def test_block_verdict_resets_crash_streak(monkeypatch):
    """Pre-existing counter bug: a completed block/warn scan must clear the
    crash streak even when the breaker is not open."""
    ts._crash_count = ts._CRASH_LIMIT - 1
    monkeypatch.setattr(ts.subprocess, "run", lambda *a, **k: _Result(1))
    out = ts.check_command_security("x")
    assert out["action"] == "block"
    assert ts._crash_count == 0


def test_backwards_wall_clock_jump_does_not_wedge_breaker(monkeypatch):
    """An NTP/DST correction must not hold the breaker open past its TTL.

    The retry window is measured with time.monotonic(). Measured against
    time.time() instead, a backwards wall-clock step between opening the
    breaker and the next call makes the elapsed comparison negative, so the
    breaker stays open for the length of the correction — on a host that
    steps its clock back an hour, tirith stays disabled for an hour.

    The breaker is aged in whatever clock the implementation itself stored,
    so this test is clock-agnostic by construction: it is exactly due to
    probe either way, and only the wall-clock jump distinguishes them.
    """
    ts._crash_count = ts._CRASH_LIMIT
    ts._circuit_open = True
    ts._circuit_open_at = 0.0
    ts._record_tirith_crash()  # stamp _circuit_open_at with the real clock
    ts._circuit_open_at -= ts._CIRCUIT_RETRY_S + 1  # now exactly due to probe

    monkeypatch.setattr(time, "time", lambda: _real_time() - 3600.0)

    spawns = []
    monkeypatch.setattr(
        ts.subprocess, "run", lambda *a, **k: spawns.append(1) or _Result(0)
    )
    out = ts.check_command_security("echo hi")

    assert spawns == [1], "breaker stayed open past its TTL after the clock stepped back"
    assert out["action"] == "allow"
    assert ts._circuit_open is False
