"""Lifecycle tests for the real-profile agent-browser attach daemon."""

import json
import os
from unittest.mock import Mock

import pytest


def _state(tmp_path, session, *, owner_pid, owner_start, daemon_pid=None, owners=None):
    socket_dir = tmp_path / f"agent-browser-{session}"
    socket_dir.mkdir()
    (socket_dir / f"{session}.owners.json").write_text(
        json.dumps(
            {
                "session_name": session,
                "owners": owners or [{"pid": owner_pid, "start_time": owner_start}],
            }
        ),
        encoding="utf-8",
    )
    if daemon_pid is not None:
        (socket_dir / f"{session}.pid").write_text(str(daemon_pid), encoding="utf-8")
    return socket_dir


def _scope(tmp_path, monkeypatch, session="hermes-real-profile-testowner"):
    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(bt, "_REAL_PROFILE_SESSION", session)
    return session


def test_registers_exact_owner_incarnation(tmp_path, monkeypatch):
    from tools import browser_tool_real_profile_daemon as daemon

    session = _scope(tmp_path, monkeypatch)
    socket_dir = tmp_path / f"agent-browser-{session}"
    socket_dir.mkdir()
    monkeypatch.setattr(daemon, "_process_start_time", lambda pid: 123456)

    assert daemon.register(str(socket_dir), session)

    owner = json.loads((socket_dir / f"{session}.owners.json").read_text(encoding="utf-8"))
    assert owner == {
        "session_name": session,
        "owners": [{"pid": os.getpid(), "start_time": 123456}],
    }


def test_register_refuses_existing_daemon_without_owner_manifest(tmp_path, monkeypatch):
    from tools import browser_tool_real_profile_daemon as daemon

    session = _scope(tmp_path, monkeypatch)
    socket_dir = tmp_path / f"agent-browser-{session}"
    socket_dir.mkdir()
    (socket_dir / f"{session}.pid").write_text("222", encoding="utf-8")
    monkeypatch.setattr(daemon, "_process_start_time", lambda pid: 44)

    assert not daemon.register(str(socket_dir), session)
    assert not (socket_dir / f"{session}.owners.json").exists()


def test_register_preserves_another_live_owner(tmp_path, monkeypatch):
    from tools import browser_tool_real_profile_daemon as daemon

    session = _scope(tmp_path, monkeypatch)
    other = {"pid": 777, "start_time": 77}
    socket_dir = _state(tmp_path, session, owner_pid=777, owner_start=77, owners=[other])
    monkeypatch.setattr(
        daemon,
        "_process_start_time",
        lambda pid: 44 if pid == os.getpid() else 77,
    )
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: True)

    assert daemon.register(str(socket_dir), session)

    stored = json.loads((socket_dir / f"{session}.owners.json").read_text(encoding="utf-8"))
    assert stored["owners"] == [other, {"pid": os.getpid(), "start_time": 44}]


def test_reaper_removes_dead_owner_generation_and_exact_daemon(tmp_path, monkeypatch):
    from tools import browser_tool_real_profile_daemon as daemon
    from tools import browser_tool_lifecycle as lifecycle
    from tools.process_registry import ProcessRegistry

    session = _scope(tmp_path, monkeypatch)
    socket_dir = _state(tmp_path, session, owner_pid=111, owner_start=11, daemon_pid=222)
    live = {222}
    terminated = []
    monkeypatch.setattr(daemon, "_owner_alive", lambda owner: False)
    monkeypatch.setattr(daemon, "_process_start_time", lambda pid: 22 if pid == 222 else None)
    monkeypatch.setattr(lifecycle, "_verify_reapable_browser_daemon", lambda *args: True)
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: pid in live)

    def terminate(pid, expected_start=None):
        terminated.append((pid, expected_start))
        live.discard(pid)

    monkeypatch.setattr(ProcessRegistry, "_terminate_host_pid", terminate)

    assert daemon.reap_orphans() == 1
    assert terminated == [(222, 22)]
    assert not socket_dir.exists()


def test_reaper_preserves_live_owner(tmp_path, monkeypatch):
    from tools import browser_tool_real_profile_daemon as daemon
    from tools.process_registry import ProcessRegistry

    session = _scope(tmp_path, monkeypatch)
    socket_dir = _state(tmp_path, session, owner_pid=111, owner_start=11, daemon_pid=222)
    terminate = Mock()
    monkeypatch.setattr(daemon, "_owner_alive", lambda owner: True)
    monkeypatch.setattr(ProcessRegistry, "_terminate_host_pid", terminate)

    assert daemon.reap_orphans() == 0
    terminate.assert_not_called()
    assert socket_dir.exists()


def test_reaper_refuses_pid_reuse_between_verify_and_signal(tmp_path, monkeypatch):
    from tools import browser_tool_real_profile_daemon as daemon
    from tools import browser_tool_lifecycle as lifecycle
    from tools.process_registry import ProcessRegistry

    session = _scope(tmp_path, monkeypatch)
    socket_dir = _state(tmp_path, session, owner_pid=111, owner_start=11, daemon_pid=222)
    starts = iter((22, 23))
    terminate = Mock()
    monkeypatch.setattr(daemon, "_owner_alive", lambda owner: False)
    monkeypatch.setattr(daemon, "_process_start_time", lambda pid: next(starts))
    monkeypatch.setattr(lifecycle, "_verify_reapable_browser_daemon", lambda *args: True)
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: True)
    monkeypatch.setattr(ProcessRegistry, "_terminate_host_pid", terminate)

    assert daemon.reap_orphans() == 0
    terminate.assert_not_called()
    assert socket_dir.exists()


def test_reaper_refuses_identity_mismatch_including_user_chrome(tmp_path, monkeypatch):
    from tools import browser_tool_real_profile_daemon as daemon
    from tools import browser_tool_lifecycle as lifecycle
    from tools.process_registry import ProcessRegistry

    session = _scope(tmp_path, monkeypatch)
    socket_dir = _state(tmp_path, session, owner_pid=111, owner_start=11, daemon_pid=222)
    terminate = Mock()
    monkeypatch.setattr(daemon, "_owner_alive", lambda owner: False)
    monkeypatch.setattr(daemon, "_process_start_time", lambda pid: 22)
    monkeypatch.setattr(lifecycle, "_verify_reapable_browser_daemon", lambda *args: False)
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: True)
    monkeypatch.setattr(ProcessRegistry, "_terminate_host_pid", terminate)

    assert daemon.reap_orphans() == 0
    terminate.assert_not_called()
    assert socket_dir.exists()


@pytest.mark.linux_only
def test_controlled_orphan_daemon_is_reaped_end_to_end(tmp_path, monkeypatch):
    import shutil
    import subprocess

    from tools import browser_tool_real_profile_daemon as daemon

    session = _scope(tmp_path, monkeypatch)
    socket_dir = tmp_path / f"agent-browser-{session}"
    fixture = tmp_path / "agent-browser-fixture"
    shutil.copy2("/bin/sleep", fixture)
    env = dict(os.environ)
    env["AGENT_BROWSER_SOCKET_DIR"] = str(socket_dir)
    proc = subprocess.Popen([str(fixture), "60"], env=env)
    try:
        _state(tmp_path, session, owner_pid=999_999_999, owner_start=1, daemon_pid=proc.pid)
        assert daemon.reap_orphans() == 1
        proc.wait(timeout=5)
        assert proc.poll() is not None
        assert not socket_dir.exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_strict_identity_guard_spares_chrome_and_other_daemons(tmp_path, monkeypatch):
    import psutil
    from tools import browser_tool_lifecycle as lifecycle

    session = _scope(tmp_path, monkeypatch)
    socket_dir = str(tmp_path / f"agent-browser-{session}")

    visible_chrome = Mock()
    visible_chrome.name.return_value = "Google Chrome"
    visible_chrome.cmdline.return_value = ["Google Chrome", socket_dir, "agent-browser"]
    visible_chrome.environ.return_value = {"AGENT_BROWSER_SOCKET_DIR": socket_dir}
    monkeypatch.setattr(psutil, "Process", lambda pid: visible_chrome)
    assert not lifecycle._verify_reapable_browser_daemon(222, socket_dir, session)

    other_daemon = Mock()
    other_daemon.name.return_value = "agent-browser-linux-x64"
    other_daemon.cmdline.return_value = ["agent-browser-linux-x64", "daemon"]
    other_daemon.environ.return_value = {"AGENT_BROWSER_SOCKET_DIR": f"{socket_dir}-other"}
    monkeypatch.setattr(psutil, "Process", lambda pid: other_daemon)
    assert not lifecycle._verify_reapable_browser_daemon(222, socket_dir, session)

    exact_daemon = Mock()
    exact_daemon.name.return_value = "agent-browser-linux-x64"
    exact_daemon.cmdline.return_value = ["agent-browser-linux-x64", "daemon"]
    exact_daemon.environ.return_value = {"AGENT_BROWSER_SOCKET_DIR": socket_dir}
    monkeypatch.setattr(psutil, "Process", lambda pid: exact_daemon)
    assert lifecycle._verify_reapable_browser_daemon(222, socket_dir, session)


def test_pid_record_replacement_preserves_new_state(tmp_path, monkeypatch):
    from tools import browser_tool_real_profile_daemon as daemon
    from tools import browser_tool_lifecycle as lifecycle
    from tools.process_registry import ProcessRegistry

    session = _scope(tmp_path, monkeypatch)
    socket_dir = _state(tmp_path, session, owner_pid=111, owner_start=11, daemon_pid=222)
    live = {222}
    monkeypatch.setattr(daemon, "_owner_alive", lambda owner: False)
    monkeypatch.setattr(daemon, "_process_start_time", lambda pid: 22)
    monkeypatch.setattr(lifecycle, "_verify_reapable_browser_daemon", lambda *args: True)
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: pid in live)

    def terminate(pid, expected_start=None):
        live.discard(pid)
        (socket_dir / f"{session}.pid").write_text("333", encoding="utf-8")

    monkeypatch.setattr(ProcessRegistry, "_terminate_host_pid", terminate)

    assert daemon.reap_orphans() == 0
    assert socket_dir.exists()
    assert (socket_dir / f"{session}.pid").read_text(encoding="utf-8") == "333"


def test_ambiguous_owner_metadata_fails_closed(tmp_path, monkeypatch):
    from tools import browser_tool_real_profile_daemon as daemon

    session = _scope(tmp_path, monkeypatch)
    socket_dir = tmp_path / f"agent-browser-{session}"
    socket_dir.mkdir()
    (socket_dir / f"{session}.owners.json").write_text("{}", encoding="utf-8")

    assert daemon.reap_orphans() == 0
    assert socket_dir.exists()


def test_ambiguous_daemon_pid_fails_closed(tmp_path, monkeypatch):
    from tools import browser_tool_real_profile_daemon as daemon

    session = _scope(tmp_path, monkeypatch)
    socket_dir = _state(tmp_path, session, owner_pid=111, owner_start=11)
    (socket_dir / f"{session}.pid").write_text("not-a-pid", encoding="utf-8")
    monkeypatch.setattr(daemon, "_owner_alive", lambda owner: False)

    assert daemon.reap_orphans() == 0
    assert socket_dir.exists()


def test_close_refuses_while_another_live_owner_exists(tmp_path, monkeypatch):
    import tools.browser_tool as bt
    from tools import browser_tool_lifecycle as lifecycle
    from tools import browser_tool_real_profile as real_profile
    from tools import browser_tool_real_profile_daemon as daemon
    from tools import browser_tool_install as install

    session = _scope(tmp_path, monkeypatch)
    other = {"pid": 777, "start_time": 77}
    _state(tmp_path, session, owner_pid=777, owner_start=77, owners=[other])
    monkeypatch.setattr(daemon, "_process_start_time", lambda pid: 44 if pid == os.getpid() else 77)
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: True)
    monkeypatch.setattr(install, "_find_agent_browser", lambda: "/usr/bin/agent-browser")
    monkeypatch.setattr(lifecycle, "_start_browser_cleanup_thread", lambda: None)
    run = Mock()
    monkeypatch.setattr(bt.subprocess, "run", run)

    assert not real_profile._agent_browser_close_session(session)
    run.assert_not_called()


def test_cleanup_owned_preserves_another_live_hermes_owner(tmp_path, monkeypatch):
    from tools import browser_tool_real_profile_daemon as daemon
    from tools.process_registry import ProcessRegistry

    session = _scope(tmp_path, monkeypatch)
    current = {"pid": os.getpid(), "start_time": 44}
    other = {"pid": 777, "start_time": 77}
    socket_dir = _state(
        tmp_path,
        session,
        owner_pid=current["pid"],
        owner_start=current["start_time"],
        daemon_pid=222,
        owners=[current, other],
    )
    terminate = Mock()
    monkeypatch.setattr(daemon, "_process_start_time", lambda pid: 44 if pid == os.getpid() else 77)
    monkeypatch.setattr(daemon, "_owner_alive", lambda owner: owner == other)
    monkeypatch.setattr(ProcessRegistry, "_terminate_host_pid", terminate)

    assert daemon.cleanup_owned() is False
    terminate.assert_not_called()
    stored = json.loads((socket_dir / f"{session}.owners.json").read_text(encoding="utf-8"))
    assert stored["owners"] == [other]


def test_cleanup_owned_preserves_pidless_startup_state(tmp_path, monkeypatch):
    from tools import browser_tool_real_profile_daemon as daemon

    session = _scope(tmp_path, monkeypatch)
    socket_dir = _state(tmp_path, session, owner_pid=os.getpid(), owner_start=44)
    monkeypatch.setattr(daemon, "_process_start_time", lambda pid: 44)

    assert daemon.cleanup_owned() is False
    assert socket_dir.exists()


def test_cleanup_owned_is_idempotent_and_owner_bound(tmp_path, monkeypatch):
    from tools import browser_tool_lifecycle as lifecycle
    from tools import browser_tool_real_profile_daemon as daemon
    from tools.process_registry import ProcessRegistry

    session = _scope(tmp_path, monkeypatch)
    socket_dir = _state(
        tmp_path, session, owner_pid=os.getpid(), owner_start=44, daemon_pid=222
    )
    live = {222}
    monkeypatch.setattr(daemon, "_process_start_time", lambda pid: 44 if pid == os.getpid() else 22)
    monkeypatch.setattr(lifecycle, "_verify_reapable_browser_daemon", lambda *args: True)
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: pid in live)
    monkeypatch.setattr(
        ProcessRegistry,
        "_terminate_host_pid",
        lambda pid, expected_start=None: live.discard(pid),
    )

    assert daemon.cleanup_owned() is True
    assert daemon.cleanup_owned() is False
    assert not socket_dir.exists()


def test_public_orphan_reaper_routes_real_profile_to_dedicated_owner_logic(tmp_path, monkeypatch):
    from tools import browser_tool_lifecycle as lifecycle
    from tools import browser_tool_real_profile_daemon as daemon

    session = _scope(tmp_path, monkeypatch)
    _state(tmp_path, session, owner_pid=111, owner_start=11, daemon_pid=222)
    calls = []
    monkeypatch.setattr(daemon, "reap_orphans", lambda: calls.append(True) or 0)
    monkeypatch.setattr(
        lifecycle,
        "_reap_socket_dir",
        lambda *args: (_ for _ in ()).throw(AssertionError("legacy reaper handled real-profile state")),
    )
    monkeypatch.setattr("tools.browser_lightpanda.reap_orphaned_lightpanda", lambda: None)

    lifecycle._reap_orphaned_browser_sessions()

    assert calls == [True]


def test_cleanup_all_browsers_closes_attach_daemon(tmp_path, monkeypatch):
    import tools.browser_tool as bt
    from tools import browser_tool_lifecycle as lifecycle
    from tools import browser_tool_real_profile_daemon as daemon

    _scope(tmp_path, monkeypatch)
    monkeypatch.setattr(bt, "_active_sessions", {})
    cleaned = []
    monkeypatch.setattr(daemon, "cleanup_owned", lambda: cleaned.append(True))

    lifecycle.cleanup_all_browsers()

    assert cleaned == [True]
