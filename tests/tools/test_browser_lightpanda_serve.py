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
    monkeypatch.setattr(lp, "_home_candidates", lambda: [])
    monkeypatch.setattr(lp, "_safe_start_time", lambda pid: 111)
    with lp._servers_lock:
        lp._servers.clear()
    yield state
    with lp._servers_lock:
        lp._servers.clear()


def _exe(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class TestFindBinary:
    def test_prefers_path(self, tmp_path, monkeypatch):
        exe = _exe(tmp_path / "bin" / "lightpanda")
        monkeypatch.setenv("PATH", str(tmp_path / "bin"))
        monkeypatch.setattr("tools.browser_tool._merge_browser_path", lambda p: p)
        assert lp.find_lightpanda_binary() == str(exe)

    def test_falls_back_to_home_candidates(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setattr("tools.browser_tool._merge_browser_path", lambda p: p)
        exe = _exe(tmp_path / ".lightpanda" / "lightpanda")
        monkeypatch.setattr(lp, "_home_candidates", lambda: [exe])
        assert lp.find_lightpanda_binary() == str(exe)

    def test_none_when_absent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setattr("tools.browser_tool._merge_browser_path", lambda p: p)
        assert lp.find_lightpanda_binary() is None

    def test_none_on_windows(self, monkeypatch):
        monkeypatch.setattr(lp.os, "name", "nt")
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

    def test_missing_binary_returns_install_hint(self, monkeypatch):
        monkeypatch.setattr(lp, "find_lightpanda_binary", lambda: None)
        server, err = lp.launch_lightpanda("lp_test")
        assert server is None
        assert "browser.engine" in err
        assert lp.LIGHTPANDA_INSTALL_URL in err

    def test_spawns_serve_on_loopback_without_timeout_flag(self, monkeypatch, _isolate):
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
        if os.name != "nt":
            assert kw["start_new_session"] is True
        assert server.cdp_url == "http://127.0.0.1:43111"
        assert server.start_time == 111
        assert lp.get_server("lp_test") is server
        record = json.loads((_isolate / "lp_test.json").read_text(encoding="utf-8"))
        assert record["pid"] == 4242
        assert record["port"] == 43111
        assert record["owner_pid"] == os.getpid()
        assert record["start_time"] == 111

    def test_http_cache_dir_is_shared_across_sessions(self, monkeypatch, _isolate):
        _, _, first = self._launch(monkeypatch)
        with lp._servers_lock:
            lp._servers.clear()
        _, _, second = self._launch(monkeypatch)
        cache = str(_isolate / "http-cache")
        assert first["argv"][first["argv"].index("--http-cache-dir") + 1] == cache
        assert second["argv"][second["argv"].index("--http-cache-dir") + 1] == cache
        assert Path(cache).is_dir()
        assert not list(Path(cache).glob("*.json"))  # never confused with a session record

    def test_block_private_networks_flag(self, monkeypatch):
        _, err, calls = self._launch(monkeypatch, block_private_networks=True)
        assert err is None
        assert calls["argv"][-1] == "--block-private-networks"

    def test_early_exit_reports_stderr_tail(self, monkeypatch):
        server, err, _ = self._launch(
            monkeypatch, proc=FakeProc(exit_code=1), ready=False,
            stderr=b"info: starting\nFATAL app : unknown argument --bogus\n",
        )
        assert server is None
        assert "exited with code 1" in err
        assert "unknown argument --bogus" in err
        assert lp.get_server("lp_test") is None

    def test_ready_timeout_terminates_child(self, monkeypatch, _isolate):
        monkeypatch.setattr(lp, "_READY_TIMEOUT_S", 0.05)
        monkeypatch.setattr(lp, "_POLL_INTERVAL_S", 0.01)
        proc = FakeProc()
        server, err, _ = self._launch(monkeypatch, proc=proc, ready=False)
        assert server is None
        assert "did not expose" in err
        assert proc.terminated is True
        assert not (_isolate / "lp_test.json").exists()
        assert lp.get_server("lp_test") is None

    def test_spawn_failure_is_reported(self, monkeypatch):
        monkeypatch.setattr(lp, "find_lightpanda_binary", lambda: "/opt/lightpanda")
        monkeypatch.setattr(lp, "_pick_free_loopback_port", lambda: 1)
        monkeypatch.setattr(lp, "_browser_env", lambda: {})

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
        lp._write_record(server)
        with lp._servers_lock:
            lp._servers[name] = server
        return server

    def test_stop_uses_tree_kill_with_expected_start_and_removes_record(self, _isolate):
        server = self._seed(_isolate)
        killed = []
        with patch(
            "tools.process_registry.ProcessRegistry._terminate_host_pid",
            side_effect=lambda pid, expected_start=None: killed.append((pid, expected_start)),
        ):
            lp.stop_lightpanda("lp_test")
        assert killed == [(777, 111)]
        assert lp.get_server("lp_test") is None
        assert not (_isolate / "lp_test.json").exists()
        assert server.proc.terminated is False  # tree-kill handled it

    def test_stop_falls_back_to_terminate_when_tree_kill_fails(self, _isolate):
        server = self._seed(_isolate)
        with patch(
            "tools.process_registry.ProcessRegistry._terminate_host_pid",
            side_effect=RuntimeError("psutil missing"),
        ):
            lp.stop_lightpanda("lp_test")
        assert server.proc.terminated is True

    def test_stop_dead_server_just_drops_record(self, _isolate):
        self._seed(_isolate, alive=False)
        with patch("tools.process_registry.ProcessRegistry._terminate_host_pid") as kill:
            lp.stop_lightpanda("lp_test")
        kill.assert_not_called()
        assert not (_isolate / "lp_test.json").exists()

    def test_stop_unknown_session_is_noop(self):
        lp.stop_lightpanda("lp_nope")  # must not raise

    def test_stop_all(self, _isolate):
        self._seed(_isolate, "lp_a")
        self._seed(_isolate, "lp_b")
        with patch("tools.process_registry.ProcessRegistry._terminate_host_pid") as kill:
            lp.stop_all_lightpanda()
        assert kill.call_count == 2
        assert lp.get_server("lp_a") is None and lp.get_server("lp_b") is None


class TestReapOrphans:
    def _record(self, _isolate, name, *, pid=999, owner_pid=1, port=43111, start_time=111):
        (_isolate / f"{name}.json").write_text(
            json.dumps({"pid": pid, "port": port, "owner_pid": owner_pid,
                        "start_time": start_time, "started_at": 0}),
            encoding="utf-8",
        )
        return _isolate / f"{name}.json"

    def test_live_other_owner_is_skipped(self, _isolate):
        rec = self._record(_isolate, "lp_x", owner_pid=12345)
        with patch("gateway.status._pid_exists", return_value=True), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid") as kill:
            assert lp.reap_orphaned_lightpanda() == 0
        kill.assert_not_called()
        assert rec.exists()

    def test_own_tracked_session_is_skipped(self, _isolate):
        rec = self._record(_isolate, "lp_x", owner_pid=os.getpid())
        with lp._servers_lock:
            lp._servers["lp_x"] = lp.LightpandaServer("lp_x", 43111, FakeProc(), "", 111)
        with patch("tools.process_registry.ProcessRegistry._terminate_host_pid") as kill:
            assert lp.reap_orphaned_lightpanda() == 0
        kill.assert_not_called()
        assert rec.exists()

    def test_dead_owner_verified_process_is_killed(self, _isolate):
        rec = self._record(_isolate, "lp_x", owner_pid=12345, pid=999)
        with patch("gateway.status._pid_exists", return_value=False), \
             patch.object(lp, "_is_lightpanda_process", return_value=True), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid") as kill:
            assert lp.reap_orphaned_lightpanda() == 1
        kill.assert_called_once_with(999, expected_start=111)
        assert not rec.exists()

    def test_own_untracked_session_is_reaped(self, _isolate):
        """Owner alive but lost its in-memory tracking: reap, don't leak."""
        self._record(_isolate, "lp_x", owner_pid=os.getpid(), pid=999)
        with patch.object(lp, "_is_lightpanda_process", return_value=True), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid") as kill:
            assert lp.reap_orphaned_lightpanda() == 1
        kill.assert_called_once()

    def test_unverified_pid_is_never_signalled(self, _isolate):
        rec = self._record(_isolate, "lp_x", owner_pid=12345, pid=999)
        with patch("gateway.status._pid_exists", return_value=False), \
             patch.object(lp, "_is_lightpanda_process", return_value=False), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid") as kill:
            assert lp.reap_orphaned_lightpanda() == 0
        kill.assert_not_called()
        assert not rec.exists()

    def test_corrupt_record_is_removed(self, _isolate):
        rec = _isolate / "lp_bad.json"
        rec.write_text("{not json", encoding="utf-8")
        assert lp.reap_orphaned_lightpanda() == 0
        assert not rec.exists()


class TestProcessIdentity:
    class _P:
        def __init__(self, name, cmdline):
            self._name, self._cmdline = name, cmdline

        def name(self):
            return self._name

        def cmdline(self):
            return self._cmdline

    def test_matches_name_port_and_start_time(self):
        proc = self._P("lightpanda", ["/opt/lightpanda", "serve", "--host", "127.0.0.1", "--port", "43111"])
        with patch("psutil.Process", return_value=proc), \
             patch("gateway.status.get_process_start_time", return_value=111):
            assert lp._is_lightpanda_process(999, 43111, 111) is True
        with patch("psutil.Process", return_value=proc), \
             patch("gateway.status.get_process_start_time", return_value=222):
            assert lp._is_lightpanda_process(999, 43111, 111) is False

    def test_rejects_other_process_on_recycled_pid(self):
        proc = self._P("chrome", ["chrome", "--remote-debugging-port=43111"])
        with patch("psutil.Process", return_value=proc):
            assert lp._is_lightpanda_process(999, 43111, None) is False

    def test_rejects_lightpanda_on_other_port(self):
        proc = self._P("lightpanda", ["lightpanda", "serve", "--port", "1"])
        with patch("psutil.Process", return_value=proc):
            assert lp._is_lightpanda_process(999, 43111, None) is False
