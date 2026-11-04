"""Regression tests for browser session cleanup (tools/browser_session_lifecycle.py)."""

import contextlib
import os
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools import browser_launcher as launcher
from tools import browser_session as bs
from tools import browser_session_lifecycle as lifecycle


class TestBrowserCleanup:
    def setup_method(self):
        self.orig_active_sessions = bs._active_sessions.copy()
        self.orig_session_last_activity = bs._session_last_activity.copy()
        self.orig_cleanup_done = bs._cleanup_done

    def teardown_method(self):
        bs._active_sessions.clear()
        bs._active_sessions.update(self.orig_active_sessions)
        bs._session_last_activity.clear()
        bs._session_last_activity.update(self.orig_session_last_activity)
        bs._cleanup_done = self.orig_cleanup_done

    def test_cleanup_browser_clears_tracking_state(self):
        bs._active_sessions["task-1"] = {
            "session_name": "sess-1",
            "bb_session_id": None,
            "launcher_key": "task-1",
            "features": {"local": True},
        }
        bs._session_last_activity["task-1"] = 123.0

        with patch("tools.browser_launcher.stop_browser") as mock_stop:
            lifecycle.cleanup_browser("task-1")

        assert "task-1" not in bs._active_sessions
        assert "task-1" not in bs._session_last_activity
        mock_stop.assert_called_once_with("task-1")


    def test_emergency_cleanup_clears_all_tracking_state(self):
        bs._cleanup_done = False
        bs._active_sessions["task-1"] = {"session_name": "sess-1"}
        bs._active_sessions["task-2"] = {"session_name": "sess-2"}
        bs._session_last_activity["task-1"] = 1.0
        bs._session_last_activity["task-2"] = 2.0

        with (
            patch("tools.browser_session_lifecycle.cleanup_all_browsers") as mock_cleanup_all,
            patch("tools.browser_session_lifecycle.reap_orphaned_browser_sessions") as mock_reap,
        ):
            lifecycle.emergency_cleanup_all_sessions()

        mock_cleanup_all.assert_called_once_with()
        mock_reap.assert_called_once_with()
        assert bs._active_sessions == {}
        assert bs._session_last_activity == {}
        assert bs._cleanup_done is True


class TestInactivityJanitorMultiplex:
    """#86402 / #100738: the process-global janitor thread has no profile scope."""

    def setup_method(self):
        from agent import secret_scope

        self.saved = {
            name: getattr(bs, name).copy()
            for name in (
                "_active_sessions", "_session_last_activity",
                "_session_owner_homes", "_cleanup_failures",
            )
        }
        self.orig_timeout = bs.BROWSER_SESSION_INACTIVITY_TIMEOUT
        bs.BROWSER_SESSION_INACTIVITY_TIMEOUT = 0
        for name in self.saved:
            getattr(bs, name).clear()
        secret_scope.set_multiplex_active(True)

    def teardown_method(self):
        from agent import secret_scope

        secret_scope.set_multiplex_active(False)
        bs.BROWSER_SESSION_INACTIVITY_TIMEOUT = self.orig_timeout
        for name, saved in self.saved.items():
            live = getattr(bs, name)
            live.clear()
            live.update(saved)

    def test_janitor_tears_down_under_owner_profile_scope(self, tmp_path, monkeypatch):
        from agent import secret_scope
        from hermes_constants import (
            get_hermes_home, reset_hermes_home_override, set_hermes_home_override,
        )

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("BROWSER_USE_API_KEY", raising=False)
        monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
        p1 = tmp_path / "profiles" / "p1"
        p1.mkdir(parents=True)
        (p1 / ".env").write_text("BROWSER_USE_API_KEY=p1-key\n", encoding="utf-8")

        # Profile p1's turn opens the session; the janitor later runs unscoped.
        home_tok = set_hermes_home_override(str(p1))
        scope_tok = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(p1))
        try:
            lifecycle.update_session_activity("t1")
            bs._active_sessions["t1"] = {"session_name": "s1", "bb_session_id": None,
                                         "launcher_key": "t1", "features": {"local": True}}
        finally:
            secret_scope.reset_secret_scope(scope_tok)
            reset_hermes_home_override(home_tok)
        bs._session_last_activity["t1"] -= 10

        seen = {}

        def fake_stop(launcher_key):
            seen["home"] = str(get_hermes_home())
            seen["key"] = secret_scope.get_secret("BROWSER_USE_API_KEY")

        with patch("tools.browser_launcher.stop_browser", side_effect=fake_stop):
            lifecycle.cleanup_inactive_sessions()

        assert seen == {"home": str(p1), "key": "p1-key"}
        assert "t1" not in bs._session_last_activity
        assert "t1" not in bs._active_sessions
        assert "t1" not in bs._session_owner_homes

    def test_repeated_failures_force_reap_and_close_cloud_session(self):
        from unittest.mock import MagicMock

        bs._active_sessions["t1"] = {"session_name": "s1", "bb_session_id": "bb-1"}
        bs._session_last_activity["t1"] = 1.0
        provider = MagicMock()

        with (
            patch("tools.browser_session_lifecycle.cleanup_single_session", side_effect=RuntimeError("boom")),
            patch("tools.browser_session_cloud.get_cloud_provider", return_value=provider),
        ):
            for _ in range(bs.MAX_INACTIVITY_CLEANUP_FAILURES - 1):
                lifecycle.cleanup_inactive_sessions()
            # An activity touch must NOT reset the failure budget.
            lifecycle.update_session_activity("t1")
            bs._session_last_activity["t1"] = 1.0
            assert bs._cleanup_failures["t1"] == bs.MAX_INACTIVITY_CLEANUP_FAILURES - 1
            assert "t1" in bs._active_sessions
            provider.close_session.assert_not_called()

            lifecycle.cleanup_inactive_sessions()

        provider.close_session.assert_called_once_with("bb-1")
        assert "t1" not in bs._active_sessions
        assert "t1" not in bs._session_last_activity
        assert "t1" not in bs._cleanup_failures


class TestAtexitStopSwallowsInterrupt:
    def test_second_ctrl_c_during_join_does_not_propagate(self, monkeypatch):
        """A second Ctrl+C while the atexit hook waits on the janitor must not escape as a
        traceback (#10764): the thread is a daemon, the interpreter is already exiting."""

        class _InterruptedJoin:
            def join(self, timeout=None):
                raise KeyboardInterrupt

        monkeypatch.setattr(bs, "_cleanup_thread", _InterruptedJoin())
        monkeypatch.setattr(bs, "_cleanup_running", True)
        lifecycle.stop_cleanup_thread()  # must not raise
        assert bs._cleanup_running is False


class TestAtexitLifecycleUnimportable:
    def test_hook_stays_silent_when_lifecycle_import_fails(self, monkeypatch):
        """Mid-`hermes update` the on-disk tree can be half-new, so the atexit hook's import of
        the lifecycle module raises; the hook must stay silent (#112437)."""
        monkeypatch.setitem(sys.modules, "tools.browser_session_lifecycle", None)  # import -> ImportError
        bs._atexit_cleanup()  # must not raise


@pytest.fixture
def isolated_sessions(monkeypatch):
    """Empty session + launcher registries, no janitor thread."""
    for name in ("_active_sessions", "_session_last_activity", "_session_owner_homes", "_cleanup_failures"):
        monkeypatch.setattr(bs, name, {})
    monkeypatch.setattr(launcher, "_browsers", {})
    monkeypatch.setattr(lifecycle, "start_cleanup_thread", lambda: None)
    return bs._active_sessions


def _attached_browser(key: str, cdp_url: str) -> launcher.LaunchedBrowser:
    return launcher.LaunchedBrowser(key=key, pid=os.getpid(), start_time=None, cdp_url=cdp_url,
                                    http_url="http://127.0.0.1:1", user_data_dir="/shared", owned=False,
                                    ephemeral=False, home_key="h")


class TestSharedBrowserSessions:
    """Several cache keys share one browser (Bot Desktop profile, browser.user_data_dir, real profile)."""

    def test_a_key_left_on_a_relaunched_shared_browser_rebuilds_on_the_live_endpoint(self, monkeypatch,
                                                                                    isolated_sessions):
        key = "profile:/shared"
        dead, live = "ws://127.0.0.1:9301/devtools/browser/dead", "ws://127.0.0.1:9302/devtools/browser/live"
        # The browser died and key 'b' already relaunched it: the launcher key is alive again, on a new endpoint.
        launcher._browsers[key] = _attached_browser(key, live)
        isolated_sessions.update({
            "a": {"session_name": "h_a", "cdp_url": dead, "launcher_key": key, "features": {"local": True}},
            "b": {"session_name": "h_b", "cdp_url": live, "launcher_key": key, "features": {"local": True}},
        })

        def build_from_launcher(cache_key, force_local):
            browser = launcher.get_browser(key)
            return {**bs._session_record("h", browser.cdp_url, {"local": True}), "launcher_key": key}
        monkeypatch.setattr(bs, "_create_session_for_key", build_from_launcher)

        assert bs.get_session_info("a")["cdp_url"] == live
        assert launcher.get_browser(key) is not None, "b's live browser must survive a's rebuild"


class _FakeSandbox:
    """The terminal sandbox's exec stream: one Chromium on the shared profile, launched by ``setsid``."""

    def __init__(self):
        self.running = None
        self.killed = []
        self.probes = 0  # liveness execs (the /dev/tcp check)
        self.probe_failure = None  # "exit-255" (docker/ssh transport error) or "timeout"
        self.forward_failures = 0  # connections through the port forward that never reached Chromium
        self.browser_id = "abc"  # a relaunch on the same port gets a new /devtools/browser/<id>
        self.launches = 0
        self.port_published = True  # False: the launched Chromium never writes DevToolsActivePort
        self.timeouts = set()  # argv[0] of execs ("cat", "kill") whose docker/ssh exec times out
        self.probe_barrier = None  # holds each liveness probe until a rival session probes too

    def run_in(self, env, argv, **kw):
        cmd = " ".join(argv)
        if _stop_target(argv) is not None and "kill" in self.timeouts:
            raise subprocess.TimeoutExpired(argv, kw.get("timeout"))
        if argv[0] in self.timeouts:
            raise subprocess.TimeoutExpired(argv, kw.get("timeout"))
        if "setsid" in cmd:
            self.running = 4242
            self.launches += 1
            return _Completed(0, b"4242 777\n")  # pid, start time
        if argv[0] == "cat" and not self.port_published:
            return _Completed(1)
        if "/dev/tcp" in cmd:
            self.probes += 1
            if self.probe_failure == "exit-255":
                return _Completed(255)
            if self.probe_failure == "timeout":
                raise subprocess.TimeoutExpired(argv, kw.get("timeout"))
        if "/dev/tcp" in cmd or argv[0] == "cat":
            found = f"9222\n/devtools/browser/{self.browser_id}\n".encode()
            answer = _Completed(0, found) if self.running else _Completed(3)
            if "/dev/tcp" in cmd and self.probe_barrier is not None:  # answered, then held until the rival probes
                with contextlib.suppress(threading.BrokenBarrierError):
                    self.probe_barrier.wait(timeout=1.0)
            return answer
        if _stop_target(argv) is not None:
            self.killed.append(_stop_target(argv))
            self.running = None
            return _Completed(0)
        return _Completed(1)


def _stop_target(argv):
    """The pid a ``stop_sandbox_browser`` exec signals (its identity check passed), else None."""
    if argv[:2] == ["bash", "-c"] and "kill -TERM" in argv[2] and len(argv) > 4:
        return int(argv[4])
    return None


class _Completed:
    def __init__(self, returncode, stdout=b""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, b""


@pytest.fixture
def fake_sandbox(monkeypatch, isolated_sessions):
    from tools import browser_session_sandbox
    from tools.bot_desktop import runtime, sandbox_host
    from tools.environments import streams
    sandbox, env = _FakeSandbox(), object()
    monkeypatch.setattr(browser_session_sandbox, "_launched", {}, raising=False)
    monkeypatch.setattr(browser_session_sandbox, "_confirmed", {}, raising=False)
    monkeypatch.setattr(streams, "run_in", sandbox.run_in)
    monkeypatch.setattr(streams, "forward_port", lambda env, port, user=None: 50000 + port)
    monkeypatch.setattr(streams, "forward_failures", lambda env, port: sandbox.forward_failures, raising=False)
    monkeypatch.setattr(streams, "sandbox_temp_dir", lambda env: "/tmp")  # no-tmp: ok — sandbox-side path
    monkeypatch.setattr(runtime, "_sandbox_env", lambda create=False: env)
    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": ":5"})
    monkeypatch.setattr(sandbox_host, "_user_for", lambda env: "hermes")
    monkeypatch.setattr(sandbox_host, "browser_profile_dir", lambda env: "/home/hermes/.bot-browser")
    monkeypatch.setattr(sandbox_host, "chromium_executable", lambda env: "/usr/bin/chromium")
    return sandbox


class TestSandboxSharedBrowser:
    """Every session placed in the terminal sandbox shares the one in-sandbox Chromium on the dock's profile."""

    def test_the_sandbox_browser_is_stopped_only_after_its_last_session_tears_down(self, fake_sandbox,
                                                                                 isolated_sessions):
        from tools.browser_session_sandbox import create_sandbox_session
        isolated_sessions["task-A"] = create_sandbox_session("task-A")  # launches it
        isolated_sessions["task-B"] = create_sandbox_session("task-B")  # attaches to it

        lifecycle.cleanup_single_session("task-A")
        assert fake_sandbox.killed == [] and fake_sandbox.running, "B still uses the sandbox browser"

        lifecycle.cleanup_single_session("task-B")
        assert fake_sandbox.killed == [4242]

    def test_a_sandbox_session_is_dead_once_its_chromium_is_gone(self, fake_sandbox):
        from tools.browser_session_sandbox import create_sandbox_session
        info = create_sandbox_session("task-A")
        assert bs._backend_dead(info) is False

        fake_sandbox.running = None  # crashed inside the sandbox
        fake_sandbox.forward_failures += 1  # the harness's next connection through the forward fails
        assert bs._backend_dead(info) is True

    def test_a_failed_or_timed_out_liveness_exec_keeps_the_live_browser(self, fake_sandbox, isolated_sessions,
                                                                      monkeypatch):
        """Only a definitive answer from inside the sandbox ("no DevTools port file", "port closed") means
        Chromium is gone; a docker/ssh exec that fails or times out says nothing about it."""
        from tools.browser_session_sandbox import create_sandbox_session
        monkeypatch.setattr(bs, "_create_session_for_key", lambda key, force_local: create_sandbox_session(key))
        first = bs.get_session_info("task-A")
        for failure in ("exit-255", "timeout"):
            fake_sandbox.forward_failures += 1  # a connection through the forward failed: liveness is re-checked
            fake_sandbox.probe_failure = failure
            assert bs.get_session_info("task-A") is first, failure
        assert fake_sandbox.killed == [] and fake_sandbox.running == 4242

    def test_a_healthy_sandbox_session_spends_no_exec_per_browser_call(self, fake_sandbox, isolated_sessions,
                                                                     monkeypatch):
        from tools.browser_session_sandbox import create_sandbox_session
        monkeypatch.setattr(bs, "_create_session_for_key", lambda key, force_local: create_sandbox_session(key))
        first = bs.get_session_info("task-A")
        fake_sandbox.probes = 0
        for _ in range(5):
            assert bs.get_session_info("task-A") is first
        assert fake_sandbox.probes == 0

    def test_a_browser_relaunched_on_the_same_port_is_rechecked_once_the_confirmation_expires(
            self, fake_sandbox, monkeypatch):
        """No forward failure surfaces for a relaunch the harness never connected through, so the confirmed
        answer is only trusted for a short while: healthy calls inside it spend no exec."""
        from tools import browser_session_sandbox
        from tools.browser_session_sandbox import create_sandbox_session
        clock = [1000.0]
        monkeypatch.setattr(browser_session_sandbox, "time", SimpleNamespace(monotonic=lambda: clock[0]))
        info = create_sandbox_session("task-A")
        fake_sandbox.browser_id = "relaunched"  # another process relaunched it on the same port
        fake_sandbox.probes = 0
        clock[0] += 5
        assert bs._backend_dead(info) is False and fake_sandbox.probes == 0, "inside the TTL: no exec"
        clock[0] += 60
        assert bs._backend_dead(info) is True
        assert fake_sandbox.probes == 1


class TestSandboxBrowserOwnership:
    """A sandbox Chromium this process spawned is always stoppable, and one container gets one launch."""

    @pytest.mark.parametrize("failure", ["never-ready", "interrupted"])
    def test_a_failed_start_stops_the_chromium_it_spawned(self, fake_sandbox, monkeypatch, failure):
        from tools import browser_session_sandbox
        from tools.browser_session_sandbox import create_sandbox_session
        from tools.environments import streams
        if failure == "never-ready":
            fake_sandbox.port_published = False
            monkeypatch.setattr(browser_session_sandbox, "_LAUNCH_TIMEOUT_S", 0.0)
            expected = RuntimeError
        else:
            def interrupted(env, port, user=None):
                raise KeyboardInterrupt
            monkeypatch.setattr(streams, "forward_port", interrupted)
            expected = KeyboardInterrupt
        with pytest.raises(expected):
            create_sandbox_session("task-A")
        assert fake_sandbox.killed == [4242]
        assert browser_session_sandbox._launched == {}

    def test_a_timed_out_port_wait_is_a_readiness_failure(self, fake_sandbox, monkeypatch):
        from tools import browser_session_sandbox
        from tools.browser_session_sandbox import create_sandbox_session
        fake_sandbox.timeouts.add("cat")
        monkeypatch.setattr(browser_session_sandbox, "_LAUNCH_TIMEOUT_S", 0.01)
        with pytest.raises(RuntimeError, match="did not expose a DevTools endpoint"):
            create_sandbox_session("task-A")
        assert fake_sandbox.killed == [4242]

    def test_a_timed_out_stop_does_not_crash_the_teardown(self, fake_sandbox, isolated_sessions):
        from tools.browser_session_sandbox import create_sandbox_session
        isolated_sessions["task-A"] = create_sandbox_session("task-A")
        fake_sandbox.timeouts.add("kill")
        lifecycle.cleanup_single_session("task-A")
        assert "task-A" not in isolated_sessions

    def test_sessions_racing_onto_one_container_launch_one_browser(self, fake_sandbox):
        from tools.browser_session_sandbox import create_sandbox_session
        fake_sandbox.probe_barrier = threading.Barrier(2)
        results, errors = [], []

        def create(key):
            try:
                results.append(create_sandbox_session(key))
            except Exception as exc:  # surfaced by the assertion below
                errors.append(exc)
        threads = [threading.Thread(target=create, args=(key,)) for key in ("task-A", "task-B")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert errors == [] and len(results) == 2
        assert fake_sandbox.launches == 1
        assert {info["cdp_url"] for info in results} == {"ws://127.0.0.1:59222/devtools/browser/abc"}


class _FakeContainers:
    """Several terminal sandboxes (one ``env`` object per container), each with its own Chromium."""

    def __init__(self):
        self.current = None
        self.running = {}  # env -> pid
        self.killed = []  # (env, pid)
        self._next_pid = 5000

    def run_in(self, env, argv, **kw):
        cmd = " ".join(argv)
        if "setsid" in cmd:
            self._next_pid += 1
            self.running.setdefault(env, self._next_pid)
            return _Completed(0, f"{self._next_pid} 777\n".encode())
        if "/dev/tcp" in cmd or argv[0] == "cat":
            pid = self.running.get(env)
            return _Completed(0, f"9222\n/devtools/browser/{pid}\n".encode()) if pid else _Completed(3)
        if _stop_target(argv) is not None:
            pid = _stop_target(argv)
            self.killed.append((env, pid))
            if self.running.get(env) == pid:
                del self.running[env]
            return _Completed(0)
        return _Completed(1)


@pytest.fixture
def fake_containers(monkeypatch, isolated_sessions):
    from tools import browser_session_sandbox
    from tools.bot_desktop import runtime, sandbox_host
    from tools.environments import streams
    containers = _FakeContainers()
    monkeypatch.setattr(browser_session_sandbox, "_launched", {}, raising=False)
    monkeypatch.setattr(browser_session_sandbox, "_confirmed", {}, raising=False)
    monkeypatch.setattr(streams, "run_in", containers.run_in)
    monkeypatch.setattr(streams, "forward_port", lambda env, port, user=None: 50000 + id(env) % 10000)
    monkeypatch.setattr(streams, "forward_failures", lambda env, port: 0, raising=False)
    monkeypatch.setattr(streams, "sandbox_temp_dir", lambda env: "/tmp")  # no-tmp: ok — sandbox-side path
    monkeypatch.setattr(runtime, "_sandbox_env", lambda create=False: containers.current)
    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": ":5"})
    monkeypatch.setattr(sandbox_host, "_user_for", lambda env: "hermes")
    monkeypatch.setattr(sandbox_host, "browser_profile_dir", lambda env: "/home/hermes/.bot-browser")
    monkeypatch.setattr(sandbox_host, "chromium_executable", lambda env: "/usr/bin/chromium")
    return containers


def _in_home(home, fn):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(str(home))
    try:
        return fn()
    finally:
        reset_hermes_home_override(token)


class TestSandboxBrowserPerContainer:
    """The sandbox Chromium belongs to its container, not to the profile that started it."""

    def test_profiles_sharing_one_container_share_its_browser_until_the_last_session(self, fake_containers,
                                                                                  isolated_sessions, tmp_path):
        from tools.browser_session_sandbox import create_sandbox_session
        shared = fake_containers.current = object()  # terminal.docker_shared_container_key
        isolated_sessions["a"] = _in_home(tmp_path / "a", lambda: create_sandbox_session("a"))
        isolated_sessions["b"] = _in_home(tmp_path / "b", lambda: create_sandbox_session("b"))
        pid = fake_containers.running[shared]

        _in_home(tmp_path / "a", lambda: lifecycle.cleanup_single_session("a"))
        assert fake_containers.killed == [], "profile b still uses the container's browser"
        _in_home(tmp_path / "b", lambda: lifecycle.cleanup_single_session("b"))
        assert fake_containers.killed == [(shared, pid)]

    def test_a_per_session_container_browser_stops_with_that_session(self, fake_containers, isolated_sessions):
        from tools.browser_session_sandbox import create_sandbox_session
        first = fake_containers.current = object()  # container_persistent: false
        isolated_sessions["s1"] = create_sandbox_session("s1")
        second = fake_containers.current = object()
        isolated_sessions["s2"] = create_sandbox_session("s2")
        pids = dict(fake_containers.running)

        lifecycle.cleanup_single_session("s1")
        assert fake_containers.killed == [(first, pids[first])]
        lifecycle.cleanup_single_session("s2")
        assert fake_containers.killed == [(first, pids[first]), (second, pids[second])]
