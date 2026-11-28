"""HTTP attachment reads reach an existing environment in a real compute subprocess."""
import base64
import sys

import pytest


CHILD = r'''
import base64, json, os, shlex, subprocess, sys
from pathlib import Path
wire = sys.stdout
from tui_gateway import server
from tui_gateway.compute_host import ComputeHost
from tools import terminal_tool
from hermes_cli.web_server_files import _agent_path_scope
from tools.environments.base import BaseEnvironment

class Namespace(BaseEnvironment):
    cwd = '/workspace'
    timeout = 3600
    _remote_home = '/workspace'
    _hermes_backend_name = 'test_remote'
    def __init__(self, root): self.root = root
    def cleanup(self): pass
    def execute(self, command, **kwargs):
        assert any(terminal_tool._calls_in_flight.values()), 'read must hold reaper guard'
        assert 0 < kwargs['timeout'] <= 10, 'file commands need a bounded deadline'
        source = self.root + '/file'
        command = command.replace('/workspace/file.mp4', shlex.quote(source))
        result = subprocess.run(['bash', '-c', command], capture_output=True, text=True, timeout=10)
        output = result.stdout.replace(base64.b64encode(source.encode()).decode(),
                                       base64.b64encode(b'/workspace/file.mp4').decode())
        return {'output': output, 'returncode': result.returncode}

root = sys.argv[1]
for profile in ('alpha', 'beta'):
    home = root + '/' + profile
    server._sessions['live-' + profile] = {'session_key': 'durable', 'profile_home': home}
    with _agent_path_scope(Path(home)):
        terminal_tool._active_environments[terminal_tool._qualify_task_key('session:durable')] = Namespace(home)
def forbidden(*args, **kwargs): raise AssertionError('must not create environment')
terminal_tool.ensure_task_env = forbidden
host = ComputeHost(stdout=wire, heartbeat_secs=0)
host.emit({'type': 'hello', 'build_sha': 'unknown'})
for line in sys.stdin:
    frame = json.loads(line)
    if frame['type'] == 'fixture.destroy':
        with _agent_path_scope(Path(root + '/alpha')):
            terminal_tool._active_environments.clear()
        host.emit({'type': 'control.ack', 'request_id': frame['request_id']})
    else:
        host.handle_frame(frame)
    if frame['type'] == 'shutdown': break
'''


@pytest.mark.platforms('posix')
def test_http_reads_existing_compute_environment_with_exact_profile_and_session(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from fastapi import FastAPI
    from starlette.testclient import TestClient
    from hermes_cli.web_routers import files
    from tui_gateway import server
    from tui_gateway.host_supervisor import HostSupervisor

    for profile in ('alpha', 'beta'):
        home = tmp_path / profile
        home.mkdir()
        (home / 'config.yaml').write_text('terminal:\n  backend: test_remote\n')
        (home / 'file').write_bytes(profile.encode() * 20000 + b'\x00\xff')
        monkeypatch.setitem(server._sessions, 'live-' + profile, {
            'session_key': 'durable', 'profile_home': str(home), '_compute_host_active': True,
        })
    supervisor = HostSupervisor(
        argv=[sys.executable, '-c', CHILD, str(tmp_path)],
        registry_path=tmp_path / 'registry.json', respawn_max=0,
        env={'HERMES_COMPUTE_HOST_CHILD': '1'}, autostart=False,
    )
    supervisor.start()  # fixture setup only: HTTP reads may never call start()
    monkeypatch.setattr(supervisor, 'start', lambda: pytest.fail('file read started compute host'))
    monkeypatch.setattr(server, '_compute_host_supervisor', supervisor)
    monkeypatch.setattr(files, '_profile_scope', lambda profile=None: nullcontext())
    monkeypatch.setattr(files, '_fs_profile_home', lambda profile=None: tmp_path / (profile or 'alpha'))
    monkeypatch.setattr(files, 'load_config', lambda: {'terminal': {'backend': 'test_remote'}})
    monkeypatch.setattr(files, 'load_env', dict)
    async def owned(session_id, profile):
        return {'id': session_id}
    monkeypatch.setattr(files, '_fs_owned_session', owned)
    app = FastAPI()
    app.include_router(files.router)
    try:
        with TestClient(app) as client:
            for profile in ('alpha', 'beta'):
                query = {'path': '/workspace/file.mp4', 'session_id': 'durable', 'profile': profile}
                result = client.get('/api/fs/read-data-url', params=query)
                assert result.status_code == 200, result.text
                assert base64.b64decode(result.json()['dataUrl'].split(',')[1]) == profile.encode() * 20000 + b'\x00\xff'
                result = client.get('/api/fs/download', params=query)
                assert result.status_code == 200, result.text
                assert result.content == profile.encode() * 20000 + b'\x00\xff'
                media = client.get('/api/files/stream', params=query, headers={'Range': 'bytes=65530-70000'})
                assert media.status_code == 206, media.text
                assert media.content == (profile.encode() * 20000 + b'\x00\xff')[65530:70001]
            identity = {'sid': 'live-alpha', 'session_key': 'durable',
                        'profile_home': str(tmp_path / 'alpha'), 'backend': 'test_remote'}
            opened = supervisor.terminal_file({**identity, 'operation': 'open'}, process=supervisor._proc)
            token = opened['result']['env_token']
            for invalid in (
                {'profile_home': str(tmp_path / 'beta'), 'operation': 'open'},
                {'session_key': 'other', 'operation': 'open'},
                {'sid': 'absent', 'operation': 'open'},
                {'operation': 'execute', 'command': 'touch /must-not-execute'},
                {'operation': 'read', 'path': '/workspace/file.mp4', 'offset': 0, 'length': 65537},
                {'operation': 'read', 'path': '/workspace/file.mp4', 'offset': -1, 'length': 1},
            ):
                rejected = supervisor.terminal_file({**identity, 'env_token': token, **invalid},
                                                    process=supervisor._proc)
                assert rejected['type'] == 'terminal.file.error', rejected
            foreign = client.get('/api/fs/download', params={
                'path': '/workspace/file.mp4', 'session_id': 'other', 'profile': 'alpha'})
            assert foreign.status_code == 503, foreign.text
            supervisor._await_reply({'type': 'fixture.destroy', 'request_id': 'destroy'}, 'destroy', 10)
            gone = client.get('/api/fs/read-data-url', params={
                'path': '/workspace/file.mp4', 'session_id': 'durable', 'profile': 'alpha'})
            assert gone.status_code == 503, gone.text
            supervisor.shutdown()
            stopped = client.get('/api/fs/read-data-url', params={
                'path': '/workspace/file.mp4', 'session_id': 'durable', 'profile': 'alpha'})
            assert stopped.status_code == 503, stopped.text
    finally:
        supervisor.shutdown()


def test_absent_supervisor_does_not_start_one(tmp_path, monkeypatch):
    from hermes_cli.terminal_workspace_fs import get_terminal_workspace_fs
    from hermes_cli.ssh_workspace_fs import SshWorkspaceFsError
    from hermes_cli.web_server_files import _agent_path_scope
    from tui_gateway import server
    from tools.terminal_scope import set_terminal_scope, reset_terminal_scope
    monkeypatch.setattr(server, '_compute_host_supervisor', None)
    monkeypatch.setattr(server, '_get_compute_host_supervisor', lambda: pytest.fail('created supervisor'))
    monkeypatch.setitem(server._sessions, 'live', {
        'session_key': 'durable', 'profile_home': str(tmp_path), '_compute_host_active': True,
    })
    (tmp_path / 'config.yaml').write_text('terminal:\n  backend: test_remote\n')
    token = set_terminal_scope({'TERMINAL_ENV': 'test_remote'})
    try:
        with _agent_path_scope(tmp_path), pytest.raises(SshWorkspaceFsError, match='unavailable'):
            get_terminal_workspace_fs({'id': 'durable'})
    finally:
        reset_terminal_scope(token)
