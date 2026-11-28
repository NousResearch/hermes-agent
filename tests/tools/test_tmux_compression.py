"""A real tmux server follows only durable compression continuations."""
import contextvars
import os
import shutil
import subprocess
import sys

import pytest
import hermes_bootstrap  # Bootstrap before the per-test real-home I/O guard.

from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_session
from tools import session_tmux as tm

pytestmark = pytest.mark.skipif(shutil.which('tmux') is None, reason='tmux not installed')


def test_compression_restart_and_fork_isolation(tmp_path, monkeypatch):
    active_home = contextvars.ContextVar('test_profile_home', default=tmp_path / 'wrong-profile')
    active_home.set(tmp_path)
    monkeypatch.setattr(tm, 'get_hermes_home', active_home.get)
    scope = profile_scope(home=tmp_path)
    with open_persistence(scope) as db:
        seed_session(db, 'root', 'cli')
        # A stale sibling predating the actual handoff must not capture the namespace.
        seed_session(db, 'stale', 'cli', parent_session_id='root', end_reason='ws_orphan_reap')
        directory = tm.session_directory('root')
        env = dict(os.environ, TMUX_TMPDIR=str(directory))
        env.pop('TMUX', None)
        subprocess.run(['tmux', 'new-session', '-d', '-s', 'job', 'sleep 60'], env=env, check=True)
        try:
            for parent, child in [('root', 'middle'), ('middle', 'tip')]:
                db.lineage.publish_compression_child(parent, CompressionChild(session_id=child, source='cli'),
                    [NewMessage.from_live({'role': 'user', 'content': 'summary'})], require_compression_lease=False)
            seed_session(db, 'fork', 'cli', parent_session_id='root', model_config={'_branched_from': 'root'})
            seed_session(db, 'delegate', 'tool', parent_session_id='root')
            seed_session(db, 'reset', 'cli', parent_session_id='root', model_config={'_reset_from': 'root'})
            seed_session(db, 'new', 'cli')
            assert tm.session_directory('tip') == directory
            assert tm.session_directory('middle') == directory
            for other in ('fork', 'delegate', 'reset', 'new', 'stale'):
                assert tm.session_directory(other) != directory
                assert tm.list_session_panes(other) == []
            rows = tm.list_session_panes('tip')
            assert len(rows) == 1 and rows[0]['status'] == 'running'
            pane = rows[0]['session_id']
            assert tm.stop_session_pane('fork', pane)['status'] == 'not_found'
            assert ('tip', 1) in tm.running_pane_owners()
            script = "from pathlib import Path; from tools import session_tmux as t; " + \
                f"t.get_hermes_home=lambda:Path({str(tmp_path)!r}); " + \
                "print(t.session_directory('tip')); print(t.running_pane_owners()); print(len(t.list_session_panes('tip')))"
            result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=True)
            assert str(directory) in result.stdout and "('tip', 1)" in result.stdout
            assert result.stdout.rstrip().endswith('1')
            socket = directory / f'tmux-{os.getuid()}' / 'default'
            import pty
            master, slave = pty.openpty()
            client = subprocess.Popen(['tmux', '-S', str(socket), 'attach-session', '-t', 'job'],
                                      stdin=slave, stdout=slave, stderr=slave,
                                      env=dict(env, TERM='xterm'))
            os.close(slave)
            try:
                import time
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    clients = subprocess.run(['tmux', '-S', str(socket), 'list-clients', '-F', '#{client_tty}'],
                                             capture_output=True, text=True, check=True)
                    if clients.stdout:
                        break
                    time.sleep(0.05)
                assert clients.stdout, 'tmux client never attached'
                subprocess.run(['tmux', '-S', str(socket), 'detach-client', '-t', clients.stdout.strip()], check=True)
                assert client.wait(timeout=5) == 0
            finally:
                if client.poll() is None:
                    client.kill()
                    client.wait()
                os.close(master)
            # An attach changes socket permissions, not the identity of a live job.
            assert tm.list_session_panes('tip')[0]['session_id'] == pane
            assert tm.stop_session_pane('tip', pane)['status'] == 'killed'
            assert tm.list_session_panes('root') == []
        finally:
            subprocess.run(['tmux', '-S', str(directory / f'tmux-{os.getuid()}' / 'default'),
                            'kill-server'], capture_output=True, check=False)
            shutil.rmtree(tm._profile_root(), ignore_errors=True)


def test_busy_lineage_reader_does_not_launch_on_wrong_socket(tmp_path, monkeypatch):
    from tools.environments.local import _terminal_env, terminal_tmux_scope

    monkeypatch.setattr(tm, 'get_hermes_home', lambda: tmp_path)
    (tmp_path / 'state.db').touch()
    def unavailable(*args, **kwargs):
        raise TimeoutError('reader busy')
    monkeypatch.setattr('persistence_assembly.open_persistence_reader', unavailable)
    from gateway.session_context import scoped_current_session_id
    with scoped_current_session_id('tip'), terminal_tmux_scope('tip', local=True):
        env = _terminal_env(dict(os.environ, TMUX='ambient,123,0'))
    assert 'TMUX' not in env and env['TMUX_TMPDIR'] == os.devnull
    failed = subprocess.run(['tmux', 'new-session', '-d', 'true'], env=env, capture_output=True, check=False)
    assert failed.returncode != 0
    assert tm._directory('tip') is None


def test_tool_owned_compression_tip(tmp_path, monkeypatch):
    monkeypatch.setattr(tm, "get_hermes_home", lambda: tmp_path)
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, "agent", "tool")
        directory = tm.session_directory("agent")
        env = dict(os.environ, TMUX_TMPDIR=str(directory))
        env.pop("TMUX", None)
        subprocess.run(["tmux", "new-session", "-d", "sleep 60"], env=env, check=True)
        try:
            db.lineage.publish_compression_child("agent", CompressionChild(session_id="agent-tip", source="tool"),
                [NewMessage.from_live({"role": "user", "content": "summary"})], require_compression_lease=False)
            assert tm.session_directory("agent-tip") == directory
            assert ("agent-tip", 1) in tm.running_pane_owners()
            pane = tm.list_session_panes("agent-tip")[0]["session_id"]
            assert tm.stop_session_pane("agent-tip", pane)["status"] == "killed"
        finally:
            subprocess.run(["tmux", "-S", str(directory / f"tmux-{os.getuid()}" / "default"),
                            "kill-server"], capture_output=True, check=False)
            shutil.rmtree(tm._profile_root(), ignore_errors=True)


pytest_plugins = ["tests.tui_gateway.test_subagent_roster_profiles"]


def test_real_roster_reports_tip(two_profiles, monkeypatch):
    from hermes_constants import get_hermes_home
    _, roster = two_profiles
    home = get_hermes_home() / "profiles" / "bot2"
    monkeypatch.setattr(tm, "get_hermes_home", lambda: home)
    with open_persistence(profile_scope(home=home)) as db:
        directory = tm.session_directory("bot2-conv")
        env = dict(os.environ, TMUX_TMPDIR=str(directory))
        env.pop("TMUX", None)
        subprocess.run(["tmux", "new-session", "-d", "sleep 60"], env=env, check=True)
        try:
            db.lineage.publish_compression_child("bot2-conv", CompressionChild(session_id="bot2-tip", source="telegram"),
                [NewMessage.from_live({"role": "user", "content": "summary"})], require_compression_lease=False)
            assert roster("bot2") == {"bot2-tip": 1}
            assert roster() == {}
        finally:
            subprocess.run(["tmux", "-S", str(directory / f"tmux-{os.getuid()}" / "default"),
                            "kill-server"], capture_output=True, check=False)
            shutil.rmtree(tm._profile_root(), ignore_errors=True)
