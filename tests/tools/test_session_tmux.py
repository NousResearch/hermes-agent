import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from tools import session_tmux as tm

pytestmark = pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux not installed")


@pytest.fixture
def owner(monkeypatch, request):
    root = request.getfixturevalue("tmp_path")
    monkeypatch.setattr(tm, "get_hermes_home", lambda: root)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    name = "test-" + uuid.uuid4().hex
    yield name
    sock = tm._socket(name)
    if sock:
        subprocess.run(["tmux", "-S", str(sock), "kill-server"], check=False, capture_output=True, timeout=2)
    directory = tm._directory(name)
    if directory:
        shutil.rmtree(directory)


def run_terminal(owner, command):
    directory = tm.session_directory(owner)
    assert directory is not None
    env = dict(os.environ, TMUX_TMPDIR=str(directory))
    env.pop("TMUX", None)
    result = subprocess.run(["/bin/sh", "-c", command], env=env, check=False, capture_output=True, text=True, timeout=5)
    return {"exit_code": result.returncode, "stderr": result.stderr}


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    return None


def test_terminal_tracks_only_owned_default_socket(owner):
    result = run_terminal(owner, "tmux new-session -d -s job 'sleep 30'")
    assert result.get("exit_code") == 0, result
    rows = tm.list_session_panes(owner)
    assert len(rows) == 1 and rows[0]["status"] == "running", rows
    assert rows[0]["session_id"].startswith("tmux:")
    assert rows[0]["command"] == "tmux: sleep"
    assert rows[0]["output_preview"] == ""
    assert (owner, 1) in tm.running_pane_owners()
    assert tm.list_session_panes(owner + '-foreign') == []
    assert tm.stop_session_pane(owner + '-foreign', rows[0]['session_id'])['status'] == 'not_found'
    assert tm.stop_session_pane(owner, rows[0]['session_id'])['status'] == 'killed'
    assert tm.list_session_panes(owner) == []


def test_idle_shell_and_send_keys_activity(owner):
    result = run_terminal(owner, "tmux new-session -d -s job")
    assert result.get("exit_code") == 0, result
    assert wait_for(lambda: tm.list_session_panes(owner) and tm.list_session_panes(owner)[0]['status'] == 'idle')
    result = run_terminal(owner, "tmux send-keys -t job 'sleep 1' Enter")
    assert result.get("exit_code") == 0, result
    assert wait_for(lambda: tm.list_session_panes(owner)[0]['status'] == 'running')
    assert wait_for(lambda: tm.list_session_panes(owner)[0]['status'] == 'idle')


def test_foreign_named_socket_not_adopted_and_stale_id(owner):
    assert run_terminal(owner, "tmux -L foreign new-session -d -s x 'sleep 30'")['exit_code'] == 0
    assert tm.list_session_panes(owner) == []
    assert run_terminal(owner, "tmux new-session -d -s x 'sleep 30'")['exit_code'] == 0
    old = tm.list_session_panes(owner)[0]['session_id']
    socket = tm._socket(owner)
    subprocess.run(['tmux', '-S', str(socket), 'kill-server'], check=True, timeout=2)
    assert run_terminal(owner, "tmux new-session -d -s y 'sleep 30'")['exit_code'] == 0
    new = tm.list_session_panes(owner)[0]['session_id']
    assert old != new
    assert tm.stop_session_pane(owner, old)['status'] == 'not_found'
    assert tm.list_session_panes(owner)[0]['session_id'] == new
    directory = tm._directory(owner)
    subprocess.run(['tmux', '-S', str(directory / f'tmux-{os.getuid()}' / 'foreign'), 'kill-server'], check=False, timeout=2)


def test_invalid_owner_and_bad_ids(owner):
    assert tm.list_session_panes('../other') == []
    assert tm.stop_session_pane(owner, 'tmux:../../other') == {'status': 'not_found'}


def test_long_profile_restart_and_isolation(owner, monkeypatch, tmp_path):
    import sys
    long_home = tmp_path / ("a" * 120) / "profile"
    monkeypatch.setattr(tm, "get_hermes_home", lambda: long_home)
    assert run_terminal(owner, "tmux new-session -d -s job 'sleep 30'")["exit_code"] == 0
    directory = tm.session_directory(owner)
    try:
        assert len(os.fsencode(str(directory / f"tmux-{os.getuid()}" / "default"))) < 100
        assert tm.running_pane_owners() == [(owner, 1)]
        script = ("from pathlib import Path; from tools import session_tmux as t; "
                  f"t.get_hermes_home=lambda:Path({str(long_home)!r}); "
                  f"print(t.session_directory({owner!r})); print(t.running_pane_owners())")
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
        assert str(directory) in result.stdout and repr((owner, 1)) in result.stdout
        monkeypatch.setattr(tm, "get_hermes_home", lambda: tmp_path / "other")
        assert tm.running_pane_owners() == []
        assert tm._socket(owner) is None
        assert tm.session_directory(owner) != directory
    finally:
        subprocess.run(["tmux", "-S", str(directory / f"tmux-{os.getuid()}" / "default"), "kill-server"], check=False, capture_output=True)
        shutil.rmtree(directory)


def test_empty_namespace_does_not_query_tmux(owner, monkeypatch):
    assert tm.session_directory(owner)
    monkeypatch.setattr(tm, "_run", lambda *a, **k: pytest.fail("no socket should cause no process scans"))
    assert tm.running_pane_owners() == []


def test_squat_and_session_collision_fail_closed(owner, monkeypatch):
    directory = tm.session_directory(owner)
    assert directory
    (directory / "owner").write_text("different-owner")
    assert tm.session_directory(owner) is None
    assert tm.list_session_panes(owner) == []
    assert tm.running_pane_owners() == []
    shutil.rmtree(directory)
    root = tm._profile_root()
    root.chmod(0o755)
    try:
        assert tm.session_directory(owner) is None
    finally:
        root.chmod(0o700)


def test_canonical_profile_alias(owner, monkeypatch, tmp_path):
    home = tmp_path / 'real'
    home.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(home, target_is_directory=True)
    monkeypatch.setattr(tm, 'get_hermes_home', lambda: home)
    first = tm.session_directory(owner)
    monkeypatch.setattr(tm, 'get_hermes_home', lambda: alias)
    assert tm.session_directory(owner) == first
    assert tm.session_directory(owner + '-other') != first


def test_stale_socket_is_ignored(owner):
    import socket
    directory = tm.session_directory(owner)
    parent = directory / f'tmux-{os.getuid()}'
    parent.mkdir(mode=0o700)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(parent / 'default'))
    assert tm.running_pane_owners() == []
    assert tm.list_session_panes(owner) == []


def test_concurrent_namespace_creation(owner):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=16) as pool:
        directories = list(pool.map(tm.session_directory, [owner] * 32))
    assert all(path == directories[0] and path is not None for path in directories)
