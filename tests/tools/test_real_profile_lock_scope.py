"""Real-profile CDP resolution must not serialize unrelated browser calls on the launch lock (#106244).

Two failures shipped together:
1. ``_real_profile_cdp`` held ``_real_profile_cdp_lock`` across snapshot + launch + attach
   (minutes-scale, unbounded when a file op wedges inside the snapshot). Every other session's
   ``browser_exec`` blocked on that lock for the whole hold — each with its own 30s activity
   heartbeat stamping ``last_activity_at`` "now", pinning old sessions at the top of the sidebar
   for days. The fix scopes the lock to cache reads/publishes; the snapshot+launch runs under a
   separate launch lock whose waiters poll with a hard bound instead of parking forever.
2. ``_run_tool_activity_heartbeat`` (tested in tests/run_agent/) stamped ``last_activity_at``
   every 30s for as long as a wedged tool call lived — the pinned-at-"now" symptom itself.
"""

import threading
import time

import pytest


@pytest.fixture(autouse=True)
def _isolate_hermes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir(exist_ok=True)


def _make_module(monkeypatch, *, snapshot_fn):
    """Import the real module with browser-layer globals stubbed through ``_bt``/locals."""
    import tools.browser_tool_real_profile as rp
    import tools.browser_tool as bt
    import hermes_cli.browser_connect as bc

    monkeypatch.setattr(bt, "_real_profile_launch_lock", threading.Lock())
    monkeypatch.setattr(bt, "_real_profile_cdp_cache", {})

    monkeypatch.setattr(rp._cloud, "_use_real_profile", lambda: True)
    monkeypatch.setattr(rp._lp, "_using_lightpanda_engine", lambda: False)
    monkeypatch.setattr(rp, "_real_profile_unsupported_reason", lambda browser: None)
    monkeypatch.setattr(rp, "_cdp_http_ready", lambda cdp: True)
    monkeypatch.setattr(rp, "_surviving_chrome_cdp", lambda copy_dir: None)
    monkeypatch.setattr(rp, "_agent_browser_get_cdp", lambda session: None)
    monkeypatch.setattr(bc, "detect_default_chromium", lambda: "chrome", raising=False)
    monkeypatch.setattr(rp, "_real_profile_snapshot_error", lambda err: f"snapshot error: {err}")
    monkeypatch.setattr(rp, "_launch_real_profile_chrome", lambda b, d: (9223, None), raising=False)
    monkeypatch.setattr(rp, "_attach_agent_browser_to_real_profile",
                        lambda port, d: ("http://127.0.0.1:9223", None), raising=False)
    monkeypatch.setattr(bc, "chromium_executable", lambda browser: "/usr/bin/chrome", raising=False)
    monkeypatch.setattr(bc, "snapshot_real_profile", snapshot_fn, raising=False)
    monkeypatch.setattr(rp._session, "_prepare_session_socket_dir", lambda name: None)
    return rp, bt


def test_waiter_bounded_when_holder_errors(monkeypatch):
    """Holder's snapshot fails → it releases the launch lock without publishing; the waiter
    must take over and succeed, NOT park on the cache for the full 15-minute bound.

    Red on the pre-fix code differently: the waiter parked on ``_real_profile_cdp_lock``
    held across the holder's whole (failing) launch path.
    """
    calls = {"n": 0}
    started = threading.Event()

    def flaky_snapshot(browser, src=None):
        calls["n"] += 1
        if calls["n"] == 1:
            started.set()
            time.sleep(0.3)
            return None, "db locked"
        return "/tmp/fake-copy", None

    rp, bt = _make_module(monkeypatch, snapshot_fn=flaky_snapshot)
    monkeypatch.setattr(rp, "_REAL_PROFILE_LAUNCH_WAIT_POLLS", 50)
    monkeypatch.setattr(rp, "_REAL_PROFILE_LAUNCH_WAIT_POLL_S", 0.05)

    out = {}
    t1 = threading.Thread(target=lambda: out.__setitem__("first", rp._real_profile_cdp()))
    t1.start()
    assert started.wait(5), "first caller never reached the snapshot"
    time.sleep(0.05)
    t2 = threading.Thread(target=lambda: out.__setitem__("second", rp._real_profile_cdp()))
    t2.start()
    t1.join(15)
    t2.join(15)
    assert not t1.is_alive() and not t2.is_alive(), "a caller parked past the bound"

    first_cdp, first_err = out["first"]
    second_cdp, second_err = out["second"]
    assert first_cdp is None and "db locked" in (first_err or "")
    assert second_cdp == "http://127.0.0.1:9223", f"waiter should take over and succeed: {second_err!r}"
    assert calls["n"] == 2


def test_concurrent_callers_launch_once(monkeypatch):
    """Two racing callers: one snapshot+launch, the waiter reuses the published cache."""
    started = threading.Event()

    def slow_snapshot(browser, src=None):
        started.set()
        time.sleep(0.3)
        return "/tmp/fake-copy", None

    rp, bt = _make_module(monkeypatch, snapshot_fn=slow_snapshot)
    launches = []
    real_launch = rp._launch_real_profile_chrome

    def counting_launch(browser, copy_dir):
        launches.append(copy_dir)
        return real_launch(browser, copy_dir)

    monkeypatch.setattr(rp, "_launch_real_profile_chrome", counting_launch, raising=False)

    out = []
    threads = [threading.Thread(target=lambda: out.append(rp._real_profile_cdp())) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)
        assert not t.is_alive(), "racing caller parked past the bound"

    assert len(out) == 2
    for cdp, err in out:
        assert cdp == "http://127.0.0.1:9223", f"both callers succeed: {err!r}"
    assert len(launches) == 1, f"racing callers launched {len(launches)} browsers"


def test_second_caller_not_blocked_while_first_snapshots(monkeypatch):
    """The lock must guard cache reads/publishes only: a caller entering the snapshot must not
    block a second caller from PROGRESSING (reaching its own bounded wait), which is what
    pinned unrelated sessions at "now" pre-fix."""
    started = threading.Event()

    def slow_snapshot(browser, src=None):
        started.set()
        time.sleep(1.5)
        return "/tmp/fake-copy", None

    rp, bt = _make_module(monkeypatch, snapshot_fn=slow_snapshot)
    monkeypatch.setattr(rp, "_REAL_PROFILE_LAUNCH_WAIT_POLLS", 3)
    monkeypatch.setattr(rp, "_REAL_PROFILE_LAUNCH_WAIT_POLL_S", 0.05)

    out = {}
    t1 = threading.Thread(target=lambda: out.__setitem__("first", rp._real_profile_cdp()))
    t1.start()
    assert started.wait(5), "first caller never reached the snapshot"

    t0 = time.monotonic()
    t2 = threading.Thread(target=lambda: out.__setitem__("second", rp._real_profile_cdp()))
    t2.start()
    t2.join(10)
    elapsed = time.monotonic() - t0

    assert not t2.is_alive(), "second caller blocked on the launch lock while the first snapshots"
    second_cdp, second_err = out["second"]
    assert second_cdp is None and "not finished in time" in (second_err or ""), \
        f"second caller should fail on its bounded wait: {second_cdp!r} {second_err!r}"
    assert elapsed < 1.0, f"second caller took {elapsed:.2f}s (should be bounded by the poll window)"
    t1.join(15)
    first_cdp, first_err = out["first"]
    assert first_cdp == "http://127.0.0.1:9223", f"first caller should still succeed: {first_err!r}"