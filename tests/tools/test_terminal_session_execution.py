"""Actual terminal child processes, not mocked subprocess environments."""
import json
import os
import shlex
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_cli.session_execution import (
    SessionExecutionContext, register_session_execution_context, remove_session_execution_context,
)
from tools.terminal_tool import terminal_tool


@pytest.mark.linux_only
def test_local_foreground_background_pty_and_snapshot_are_session_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("ROUTING_VALUE", "host")
    monkeypatch.setenv("ROUTING_HOST_SOCKET", "host-socket")
    wrapper = tmp_path / "launch.py"
    wrapper.write_text('import os,sys; os.environ["LAUNCH_OWNER"]=sys.argv[1]; os.execvpe(sys.argv[2],sys.argv[2:],os.environ)')
    code = 'import os,json;print("PROBE="+json.dumps([os.getenv("ROUTING_VALUE"),os.getenv("ROUTING_HOST_SOCKET")]));print("OWNER="+str(os.getenv("LAUNCH_OWNER")));print("PATH="+str(os.getenv("PATH")));print("UNBUFFERED="+str(os.getenv("PYTHONUNBUFFERED")))'
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    from tools.process_registry import process_registry

    def run(sid, background=False, pty=False, command=command):
        result = json.loads(terminal_tool(command, task_id="task-" + sid, session_id=sid,
                                         workdir=str(tmp_path), background=background, pty=pty))
        assert not result.get("error"), result
        if background:
            result = process_registry.wait(result["session_id"], timeout=15)
        assert result.get("exit_code") == 0, result
        return result["output"]

    for sid in ("alpha", "beta"):
        register_session_execution_context(sid, SessionExecutionContext(
            env_set={"ROUTING_VALUE": sid, "PATH": "/usr/bin", **({"PYTHONUNBUFFERED": "0"} if sid == "beta" else {})},
            env_unset={"ROUTING_HOST_SOCKET"} | ({"PYTHONUNBUFFERED"} if sid == "alpha" else set()),
            command_prefix=(sys.executable, str(wrapper), sid)), task_ids=("task-" + sid,))
    try:
        from tools.file_tools import _get_file_ops
        _get_file_ops("task-alpha")  # a file call must not seed the cache with a host shell
        with ThreadPoolExecutor(max_workers=2) as pool:
            outputs = list(pool.map(run, ("alpha", "beta")))
        for sid, output in zip(("alpha", "beta"), outputs):
            assert f'PROBE=["{sid}", null]' in output
            assert 'OWNER=' + sid in output
        run("alpha", command="export ROUTING_VALUE=wrong ROUTING_HOST_SOCKET=wrong")
        for background, pty in ((False, False), (True, False), (True, True)):
            output = run("alpha", background, pty)
            assert 'PROBE=["alpha", null]' in output
            assert 'OWNER=alpha' in output
            assert 'PATH=/usr/bin\n' in output.replace('\r', '')
            assert 'UNBUFFERED=None' in output
        assert 'PROBE=["host", "host-socket"]' in run("unbound")
        assert 'UNBUFFERED=0' in run("beta", background=True)
        register_session_execution_context("alpha", SessionExecutionContext(env_set={"ROUTING_VALUE": "new"}),
                                           task_ids=("task-alpha",))
        assert 'PROBE=["new", "host-socket"]' in run("alpha")
        assert os.environ["ROUTING_VALUE"] == "host"
    finally:
        for sid in ("alpha", "beta"):
            remove_session_execution_context(sid)


@pytest.mark.linux_only
@pytest.mark.parametrize('mode', ['snapshot', 'login-fallback', 'nonlogin-fallback'])
def test_foreground_routing_wins_after_shell_startup(tmp_path, monkeypatch, mode):
    from hermes_cli.session_execution import resolve_session_execution_context
    from tools.environments.local import LocalEnvironment

    # Harmless stand-ins for display/control variables; never use a real GUI endpoint.
    startup = tmp_path / 'startup.sh'
    startup.write_text('export ROUTING_VALUE=startup ROUTING_HOST_SOCKET=startup\n'
                       'export OPENAI_API_KEY=startup-inert\n')
    monkeypatch.setenv('BASH_ENV', str(startup))
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setattr('tools.environments.local._resolve_shell_init_files', lambda: [str(startup)])
    expected = "owner's value\nwith $literal ; shell syntax"
    register_session_execution_context('shell-startup', SessionExecutionContext(
        env_set={'ROUTING_VALUE': expected, 'OPENAI_API_KEY': 'must-still-be-sanitized'},
        env_unset={'ROUTING_HOST_SOCKET'}))
    routed = host = None
    try:
        routed = LocalEnvironment(cwd=str(tmp_path), execution_context=
                                  resolve_session_execution_context(session_id='shell-startup'))
        assert routed._snapshot_ready
        host = LocalEnvironment(cwd=str(tmp_path))
        if mode != 'snapshot':
            routed._snapshot_ready = host._snapshot_ready = False
            routed._prefer_nonlogin = host._prefer_nonlogin = mode == 'nonlogin-fallback'
        code = 'import os,json; print(json.dumps([os.getenv(k) for k in ("ROUTING_VALUE", "ROUTING_HOST_SOCKET", "OPENAI_API_KEY")]))'
        command = f'{shlex.quote(sys.executable)} -c {shlex.quote(code)}'
        for _ in range(2):
            result = routed.execute(command)
            assert result['returncode'] == 0, result
            assert json.loads(result['output']) == [expected, None, None]
        host_result = host.execute(command)
        assert host_result['returncode'] == 0, host_result
        assert json.loads(host_result['output']) == ['startup', 'startup', 'startup-inert']
    finally:
        for env in (routed, host):
            if env is not None:
                env.cleanup()
        remove_session_execution_context('shell-startup')


@pytest.mark.linux_only
@pytest.mark.parametrize('operation', ['set', 'unset'])
def test_readonly_startup_routing_fails_before_user_command(tmp_path, monkeypatch, operation):
    from hermes_cli.session_execution import resolve_session_execution_context
    from tools.environments.local import LocalEnvironment

    startup = tmp_path / 'readonly.sh'
    startup.write_text('export ROUTING_VALUE=startup\nreadonly ROUTING_VALUE\n')
    monkeypatch.setenv('BASH_ENV', str(startup))
    monkeypatch.setattr('tools.environments.local._resolve_shell_init_files', lambda: [str(startup)])
    register_session_execution_context('readonly-routing', SessionExecutionContext(
        env_set={'ROUTING_VALUE': 'owner'} if operation == 'set' else {},
        env_unset={'ROUTING_VALUE'} if operation == 'unset' else set()))
    env = None
    marker = tmp_path / 'must-not-run'
    try:
        env = LocalEnvironment(cwd=str(tmp_path), execution_context=
                               resolve_session_execution_context(session_id='readonly-routing'))
        result = env.execute(f'touch {shlex.quote(str(marker))}')
        assert not marker.exists(), result
        assert result['returncode'] == 126, result
    finally:
        if env is not None:
            env.cleanup()
        remove_session_execution_context('readonly-routing')


@pytest.mark.linux_only
@pytest.mark.parametrize('replaced', [False, True], ids=['invalid', 'replaced-invalid'])
@pytest.mark.parametrize('backend,override', [
    ('local', 'none'), ('local', 'raw-cwd'), ('local', 'collapsed-cwd'),
    ('docker', 'none'), ('docker', 'image'), ('docker-isolated', 'none'),
])
def test_host_local_ignores_execution_lease_without_losing_task_routing(
        tmp_path, monkeypatch, replaced, backend, override):
    from hermes_cli.session_execution import SessionExecutionError
    from tools import terminal_tool as tt
    from tools.terminal_scope import set_terminal_scope, reset_terminal_scope
    from tools.process_registry import process_registry

    token = set_terminal_scope({
        'TERMINAL_ENV': 'docker' if backend.startswith('docker') else 'local',
        'TERMINAL_CWD': str(tmp_path),
        'TERMINAL_CONTAINER_PERSISTENT': 'false' if backend == 'docker-isolated' else 'true',
        'TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE': 'true',
    })
    monkeypatch.setattr(tt, '_task_env_overrides', {})
    monkeypatch.setattr(tt, '_session_cwd', {})
    checks = []
    valid = [True]

    def validate():
        checks.append(valid[0])
        return valid[0]

    task_id = 'control-task'
    marker = tmp_path / 'control-ran'
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    tt.record_session_cwd(task_id, str(tmp_path))
    key = None
    try:
        if override == 'image':
            tt._task_env_overrides[task_id] = {'docker_image': 'inert-image', 'cwd': str(workspace)}
        elif override == 'raw-cwd':
            tt._task_env_overrides[task_id] = {'cwd': str(workspace)}
        key = tt._resolve_container_task_id(task_id)
        if override == 'collapsed-cwd':
            tt._task_env_overrides[key] = {'cwd': str(workspace)}
        register_session_execution_context('control-owner', SessionExecutionContext(validate=validate),
                                           task_ids=(task_id,))
        if replaced:
            register_session_execution_context('control-owner', SessionExecutionContext(validate=validate),
                                               task_ids=(task_id,))
        valid[0] = False
        with pytest.raises(SessionExecutionError, match='validation failed'):
            tt._plan_execution('true', task_id=task_id, session_id='control-owner',
                               timeout=10, background=True, _host_local=False)
        checked = len(checks)
        plan = tt._plan_execution('true', task_id=task_id, session_id='control-owner',
                                  timeout=10, background=True, _host_local=True)
        assert len(checks) == checked, 'control-plane planning must not validate a realm lease'
        assert plan.execution_context is None
        assert plan.env_type == 'local'
        assert plan.effective_task_id == f'host-local-{key}'
        assert plan.cwd == str(workspace if override != 'none' else tmp_path)
        # A real child proves no late helper re-enters execution-context lookup.
        result = json.loads(tt.terminal_tool(f'touch {shlex.quote(str(marker))}',
            task_id=task_id, session_id='control-owner', background=True, _host_local=True,
            workdir=str(tmp_path)))
        assert not result.get('error'), result
        assert process_registry.wait(result['session_id'], timeout=15)['exit_code'] == 0
        assert marker.exists()
        assert len(checks) == checked
    finally:
        remove_session_execution_context('control-owner')
        # Release only this test's host-local cache; never resolve the invalid task again.
        if key is not None:
            env = tt._active_environments.pop(f'host-local-{key}', None)
            tt._last_activity.pop(f'host-local-{key}', None)
            if env is not None:
                env.cleanup()
        reset_terminal_scope(token)


@pytest.mark.linux_only
def test_host_local_keeps_raw_cache_and_approval_boundaries(tmp_path, monkeypatch):
    from hermes_cli.session_execution import resolve_session_execution_context
    from tools import terminal_tool as tt
    from tools.environments.local import LocalEnvironment
    from tools.process_registry import process_registry
    from tools.registry import registry

    monkeypatch.setenv('TERMINAL_ENV', 'local')
    monkeypatch.setattr(tt, '_active_environments', {})
    monkeypatch.setattr(tt, '_last_activity', {})
    monkeypatch.setattr(tt, '_task_env_overrides', {})
    monkeypatch.setattr(tt, '_session_cwd', {})
    task_id = 'cache-control-task'
    tt.record_session_cwd(task_id, str(tmp_path))
    valid = [True]
    register_session_execution_context('cache-owner', SessionExecutionContext(validate=lambda: valid[0]),
                                       task_ids=(task_id,))
    try:
        # ACP/workspace callers may already have a raw task cache entry.
        raw = LocalEnvironment(cwd=str(tmp_path), execution_context=
                               resolve_session_execution_context(task_id=task_id))
        tt._active_environments[task_id] = raw
        valid[0] = False
        marker = tmp_path / 'control-ran'
        command = f'touch {shlex.quote(str(marker))}'
        result = json.loads(tt.terminal_tool(command, task_id=task_id, session_id='cache-owner',
                                            background=True, _host_local=True, workdir=str(tmp_path)))
        assert not result.get('error'), result
        assert process_registry.wait(result['session_id'], timeout=15)['exit_code'] == 0
        assert marker.exists()
        assert tt._active_environments[task_id] is raw
        marker.unlink()

        # The model-facing registry must never expose this internal exemption.
        ordinary = json.loads(registry.dispatch('terminal', {'command': command, '_host_local': True},
                                                 task_id=task_id, session_id='cache-owner'))
        assert 'validation failed' in ordinary['error']
        assert not marker.exists()

        guards = []

        def deny(command, env_type, **kwargs):
            guards.append(env_type)
            return {'approved': False, 'description': 'control-plane approval denied'}

        monkeypatch.setattr(tt, '_check_all_guards', deny)
        denied = json.loads(tt.terminal_tool(command, task_id=task_id, session_id='cache-owner',
                                            background=True, _host_local=True, workdir=str(tmp_path)))
        assert denied['status'] == 'blocked', denied
        assert guards == ['local']
        assert not marker.exists()
    finally:
        remove_session_execution_context('cache-owner')
        for env in tt._active_environments.values():
            env.cleanup()


@pytest.mark.linux_only
@pytest.mark.parametrize('background,pty', [(False, False), (True, False), (True, True)])
def test_invalid_context_never_falls_back_to_host(tmp_path, monkeypatch, background, pty):
    monkeypatch.setenv('TERMINAL_ENV', 'local')
    alive = [True]
    register_session_execution_context('refused', SessionExecutionContext(validate=lambda: alive[0]))
    alive[0] = False
    target = tmp_path / 'must-not-exist'
    try:
        result = json.loads(terminal_tool(f'touch {shlex.quote(str(target))}', session_id='refused',
                                          task_id='refused', workdir=str(tmp_path), background=background, pty=pty))
        assert result.get('error'), result
        assert not target.exists()
    finally:
        remove_session_execution_context('refused')
