"""Tests for tools/browser_lightpanda.py — the ``lightpanda serve`` launcher
Browser Use mode uses when ``browser.engine`` is ``lightpanda``."""

import json
import os
import stat
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

import tools.browser_lightpanda as lp

# The autouse _isolate fixture swaps _binary_supports_http_cache for a lambda;
# the probe test needs the real (lru_cache-wrapped) function back.
_real_probe = lp._binary_supports_http_cache


class FakeProc:
    def __init__(self, pid=4242, exit_code=None):
        self.pid = pid
        self._rc = exit_code
        self.terminated = False
        self.killed = False

    def poll(self):
        return self._rc

    def terminate(self):
        self.terminated = True
        self._rc = -15

    def kill(self):
        self.killed = True
        self._rc = -9

    def wait(self, timeout=None):
        return self._rc


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr(lp, "_state_dir", lambda: state)
    # Never touch the developer's real ~/.local/bin/lightpanda.
    monkeypatch.setattr(lp, "_home_candidates", list)
    monkeypatch.setattr(lp, "_safe_start_time", lambda pid: 111)
    lp._binary_supports_http_cache.cache_clear()
    monkeypatch.setattr(lp, "_binary_supports_http_cache", lambda binary: True)
    with lp._servers_lock:
        lp._servers.clear()
    yield state
    with lp._servers_lock:
        lp._servers.clear()


@pytest.fixture(autouse=True)
def owned(monkeypatch):
    """The ownership seam, recorded: FakeProc pids must never reach the real owner-death supervisor
    (it would SIGKILL whatever process group really has that number when pytest exits)."""
    seen = []

    def fake_spawn_owned(argv, *, purpose, **kwargs):
        proc = lp.subprocess.Popen(argv, start_new_session=True, **kwargs)
        seen.append(("own", purpose, proc.pid))
        return proc

    monkeypatch.setattr(lp, "spawn_owned", fake_spawn_owned)
    monkeypatch.setattr(lp, "release_process_groups", lambda pgids: seen.append(("release", list(pgids))))
    return seen


def _exe(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class TestFindBinary:
    @pytest.mark.platforms("posix")
    def test_prefers_path(self, tmp_path, monkeypatch):
        exe = _exe(tmp_path / "bin" / "lightpanda")
        monkeypatch.setenv("PATH", str(tmp_path / "bin"))
        monkeypatch.setattr("tools.browser_tool_install._merge_browser_path", lambda p: p)
        assert lp.find_lightpanda_binary() == str(exe)

    @pytest.mark.platforms("posix")
    def test_falls_back_to_home_candidates(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setattr("tools.browser_tool_install._merge_browser_path", lambda p: p)
        exe = _exe(tmp_path / ".lightpanda" / "lightpanda")
        monkeypatch.setattr(lp, "_home_candidates", lambda: [exe])
        assert lp.find_lightpanda_binary() == str(exe)

    def test_none_when_absent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setattr("tools.browser_tool_install._merge_browser_path", lambda p: p)
        assert lp.find_lightpanda_binary() is None

    @pytest.mark.platforms("windows")
    def test_none_on_windows(self):
        assert lp.find_lightpanda_binary() is None



class TestLaunch:
    def _launch(self, monkeypatch, *, proc=None, ready=True, stderr=b"", **kw):
        calls = {}

        def fake_popen(argv, **kwargs):
            calls["argv"] = argv
            calls["kwargs"] = kwargs
            if stderr:
                kwargs["stderr"].write(stderr)
                kwargs["stderr"].flush()
            return proc or FakeProc()

        monkeypatch.setattr(lp, "find_lightpanda_binary", lambda: "/opt/lightpanda")
        monkeypatch.setattr(lp, "_pick_free_loopback_port", lambda: 43111)
        monkeypatch.setattr(lp, "_cdp_ready", lambda url: ready)
        monkeypatch.setattr(lp, "_browser_env", lambda: {"PATH": "/usr/bin"})
        monkeypatch.setattr(lp.subprocess, "Popen", fake_popen)
        server, err = lp.launch_lightpanda("lp_test", **kw)
        return server, err, calls

    @pytest.mark.platforms("posix")
    def test_missing_binary_returns_install_hint(self, monkeypatch):
        monkeypatch.setattr(lp, "find_lightpanda_binary", lambda: None)
        server, err = lp.launch_lightpanda("lp_test")
        assert server is None
        assert "browser.engine" in err
        assert lp.LIGHTPANDA_INSTALL_URL in err

    def test_spawns_serve_on_loopback_without_timeout_flag(self, monkeypatch, _isolate, owned):
        server, err, calls = self._launch(monkeypatch)
        assert err is None
        assert calls["argv"] == [
            "/opt/lightpanda", "serve", "--host", "127.0.0.1", "--port", "43111",
            "--http-cache-dir", str(_isolate / "http-cache"),
        ]
        kw = calls["kwargs"]
        assert kw["stdin"] is subprocess.DEVNULL
        assert kw["stdout"] is subprocess.DEVNULL
        assert hasattr(kw["stderr"], "write")  # a log file, never a pipe
        assert kw["env"] == {"PATH": "/usr/bin"}
        assert kw["start_new_session"] is True
        assert server.cdp_url == "http://127.0.0.1:43111"
        assert server.start_time == 111
        assert lp.get_server("lp_test") is server
        # The server is an owned child: it dies with this process however this process dies.
        assert owned == [("own", "lightpanda-serve", 4242)]

    def test_http_cache_dir_is_shared_across_sessions(self, monkeypatch, _isolate):
        _, _, first = self._launch(monkeypatch)
        with lp._servers_lock:
            lp._servers.clear()
        _, _, second = self._launch(monkeypatch)
        cache = str(_isolate / "http-cache")
        assert first["argv"][first["argv"].index("--http-cache-dir") + 1] == cache
        assert second["argv"][second["argv"].index("--http-cache-dir") + 1] == cache
        assert Path(cache).is_dir()

    def test_no_http_cache_flag_on_old_binary(self, monkeypatch, _isolate):
        monkeypatch.setattr(lp, "_binary_supports_http_cache", lambda binary: False)
        _, err, calls = self._launch(monkeypatch)
        assert err is None
        assert "--http-cache-dir" not in calls["argv"]
        assert calls["argv"][-1] == "43111"

    def test_http_cache_probe_caches_and_detects_flag(self, monkeypatch, tmp_path):
        monkeypatch.setattr(lp, "_binary_supports_http_cache", _real_probe)
        _real_probe.cache_clear()
        exe = _exe(tmp_path / "lightpanda")
        runs = []

        def fake_run(argv, **kwargs):
            runs.append(argv)
            return subprocess.CompletedProcess(
                argv, returncode=0,
                stdout="--http-cache-dir <PATH>" if len(runs) == 1 else "",
            )

        monkeypatch.setattr(lp.subprocess, "run", fake_run)
        assert lp._binary_supports_http_cache(str(exe)) is True
        assert lp._binary_supports_http_cache(str(exe)) is True  # cached, single probe
        assert len(runs) == 1

        lp._binary_supports_http_cache.cache_clear()

        def fake_run_old(argv, **kwargs):
            return subprocess.CompletedProcess(argv, returncode=0, stdout="no such flag")

        monkeypatch.setattr(lp.subprocess, "run", fake_run_old)
        assert lp._binary_supports_http_cache(str(exe)) is False

        def fake_run_hangs(argv, **kwargs):
            raise subprocess.TimeoutExpired(cmd=argv, timeout=3.0)

        lp._binary_supports_http_cache.cache_clear()
        monkeypatch.setattr(lp.subprocess, "run", fake_run_hangs)
        assert lp._binary_supports_http_cache(str(exe)) is False

    def test_block_private_networks_flag(self, monkeypatch):
        _, err, calls = self._launch(monkeypatch, block_private_networks=True)
        assert err is None
        assert calls["argv"][-1] == "--block-private-networks"

    def test_early_exit_reports_stderr_tail(self, monkeypatch, owned):
        server, err, _ = self._launch(
            monkeypatch, proc=FakeProc(exit_code=1), ready=False,
            stderr=b"info: starting\nFATAL app : unknown argument --bogus\n",
        )
        assert server is None
        assert "exited with code 1" in err
        assert "unknown argument --bogus" in err
        assert lp.get_server("lp_test") is None
        assert owned[-1] == ("release", [4242])

    def test_ready_timeout_terminates_child(self, monkeypatch, _isolate, owned):
        monkeypatch.setattr(lp, "_READY_TIMEOUT_S", 0.05)
        monkeypatch.setattr(lp, "_POLL_INTERVAL_S", 0.01)
        proc = FakeProc()
        server, err, _ = self._launch(monkeypatch, proc=proc, ready=False)
        assert server is None
        assert "did not expose" in err
        assert proc.terminated is True
        assert owned == [("own", "lightpanda-serve", 4242), ("release", [4242])]
        assert lp.get_server("lp_test") is None

    def test_spawn_failure_is_reported(self, monkeypatch):
        monkeypatch.setattr(lp, "find_lightpanda_binary", lambda: "/opt/lightpanda")
        monkeypatch.setattr(lp, "_pick_free_loopback_port", lambda: 1)
        monkeypatch.setattr(lp, "_browser_env", dict)

        def boom(*a, **k):
            raise OSError("exec format error")

        monkeypatch.setattr(lp.subprocess, "Popen", boom)
        server, err = lp.launch_lightpanda("lp_test")
        assert server is None
        assert "exec format error" in err


class TestStop:
    def _seed(self, _isolate, name="lp_test", alive=True):
        proc = FakeProc(pid=777, exit_code=None if alive else 0)
        server = lp.LightpandaServer(name, 43111, proc, str(_isolate / f"{name}.log"), 111)
        with lp._servers_lock:
            lp._servers[name] = server
        return server

    def test_stop_uses_tree_kill_with_expected_start_and_releases_group(self, _isolate, owned):
        server = self._seed(_isolate)
        killed = []
        with patch(
            "tools.process_registry.ProcessRegistry._terminate_host_pid",
            side_effect=lambda pid, expected_start=None: killed.append((pid, expected_start)),
        ):
            lp.stop_lightpanda("lp_test")
        assert killed == [(777, 111)]
        assert lp.get_server("lp_test") is None
        assert owned == [("release", [777])]
        assert server.proc.terminated is False  # tree-kill handled it

    def test_stop_falls_back_to_terminate_when_tree_kill_fails(self, _isolate):
        server = self._seed(_isolate)
        with patch(
            "tools.process_registry.ProcessRegistry._terminate_host_pid",
            side_effect=RuntimeError("psutil missing"),
        ):
            lp.stop_lightpanda("lp_test")
        assert server.proc.terminated is True

    def test_stop_dead_server_just_releases_group(self, _isolate, owned):
        self._seed(_isolate, alive=False)
        with patch("tools.process_registry.ProcessRegistry._terminate_host_pid") as kill:
            lp.stop_lightpanda("lp_test")
        kill.assert_not_called()
        assert owned == [("release", [777])]

    def test_stop_unknown_session_is_noop(self, owned):
        lp.stop_lightpanda("lp_nope")  # must not raise
        assert owned == []

    def test_stop_all(self, _isolate, owned):
        self._seed(_isolate, "lp_a")
        self._seed(_isolate, "lp_b")
        with patch("tools.process_registry.ProcessRegistry._terminate_host_pid") as kill:
            lp.stop_all_lightpanda()
        assert kill.call_count == 2
        assert lp.get_server("lp_a") is None and lp.get_server("lp_b") is None
        assert owned == [("release", [777]), ("release", [777])]
