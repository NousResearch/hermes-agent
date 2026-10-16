"""T3: a local session's tool subprocesses (terminal, background processes, execute_code) start with
its client's allowlisted toolchain environment (PATH, an activated venv, nvm/pyenv/conda), not the
gateway daemon's. Applied to that session only; no full shell snapshot. As on main, the activated venv
reaches terminal children through PATH: ``VIRTUAL_ENV`` itself stays a stripped runtime marker there."""
import asyncio
import os
from types import SimpleNamespace

import pytest

from tests.tools._child_env_fixtures import child_env, run_code

CLIENT = {'PATH': '/client/venv/bin:/usr/bin:/bin', 'VIRTUAL_ENV': '/client/venv', 'NVM_DIR': '/client/.nvm'}


@pytest.fixture
def daemon_env(monkeypatch):
    monkeypatch.setenv('PATH', '/daemon/bin:/usr/bin:/bin')
    monkeypatch.delenv('VIRTUAL_ENV', raising=False)
    monkeypatch.delenv('NVM_DIR', raising=False)


def _parts(env):
    return env['PATH'].split(os.pathsep)


def test_terminal_run_env_starts_from_the_sessions_toolchain(daemon_env):
    from agent.session_client_env import client_env_scope
    from tools.environments.local import _make_run_env
    with client_env_scope(CLIENT):
        env = _make_run_env({})
        assert '/client/venv/bin' in _parts(env) and '/daemon/bin' not in _parts(env)
        assert env['NVM_DIR'] == '/client/.nvm'
        # The backend's own terminal.env still wins over the client's.
        assert _make_run_env({'NVM_DIR': '/configured'})['NVM_DIR'] == '/configured'
    with client_env_scope({'HERMES_EPHEMERAL_SYSTEM_PROMPT': 'p'}):  # behaviour names are not tool env
        assert 'HERMES_EPHEMERAL_SYSTEM_PROMPT' not in _make_run_env({})
    env = _make_run_env({})  # messaging/cron/Desktop: the gateway's own environment
    assert '/daemon/bin' in _parts(env) and 'NVM_DIR' not in env


def test_background_process_env_starts_from_the_sessions_toolchain(daemon_env):
    from agent.session_client_env import client_env_scope
    from tools.process_registry import ProcessRegistry
    with client_env_scope(CLIENT):
        env = ProcessRegistry._spawn_env({})
    assert '/client/venv/bin' in _parts(env) and env['NVM_DIR'] == '/client/.nvm'
    assert '/daemon/bin' in _parts(ProcessRegistry._spawn_env({}))


def test_execute_code_child_env_starts_from_the_sessions_toolchain(daemon_env, tmp_path):
    import sys
    from agent.session_client_env import client_env_scope
    from tools.code_execution_env import _build_child_env
    kwargs = dict(rpc_endpoint='sock', rpc_token='t', tmpdir=str(tmp_path), child_python=sys.executable)
    with client_env_scope(CLIENT):
        env = _build_child_env(**kwargs)
    assert '/client/venv/bin' in _parts(env) and env['VIRTUAL_ENV'] == '/client/venv'
    assert '/daemon/bin' in _parts(_build_child_env(**kwargs))


# execute_code's secret names (BEARER, CREDS, DSN, WEBHOOK...) and Hermes control names never travel, even
# under a built-in prefix (NODE_, CONDA_) or a profile's configured extra names.
REFUSED = {'NODE_BEARER': 'b', 'NODE_CREDS': 'c', 'CONDA_DSN': 'postgres://u:p@h/db', 'DEPLOY_WEBHOOK': 'https://h',
           'HERMES_KANBAN_DB': '/board.db', 'HERMES_DELEGATED_CHILD_CONTEXT': '1'}


def test_credential_and_control_names_are_refused_by_execute_codes_secret_policy():
    from agent.session_client_env import accept_client_env, collect_client_env
    config = {'terminal': {'client_env_passthrough': ['DEPLOY_WEBHOOK', 'HERMES_KANBAN_DB', 'MY_TOOLCHAIN_HOME']}}
    sent = {**REFUSED, 'NVM_DIR': '/client/.nvm', 'MY_TOOLCHAIN_HOME': '/opt/tc', 'PATH': '/client/bin'}
    kept = {'NVM_DIR': '/client/.nvm', 'MY_TOOLCHAIN_HOME': '/opt/tc', 'PATH': '/client/bin'}
    assert accept_client_env(sent, config) == kept  # the gateway, before anything is stored
    assert collect_client_env(sent, config) == kept  # the client, before anything is sent


def test_execute_code_scrubs_after_merging_the_sessions_toolchain(daemon_env, tmp_path):
    """A bound map (say, one committed before this rule) still cannot carry a secret or a control marker
    into the sandbox; the client's non-secret toolchain names (NVM_DIR) do reach it."""
    import sys
    from agent.session_client_env import client_env_scope
    from tools.code_execution_env import _build_child_env
    with client_env_scope({**CLIENT, **REFUSED}):
        env = _build_child_env(rpc_endpoint='sock', rpc_token='t', tmpdir=str(tmp_path), child_python=sys.executable)
    assert env['NVM_DIR'] == '/client/.nvm' and '/client/venv/bin' in _parts(env)
    assert not set(REFUSED) & set(env)


@pytest.mark.platforms('linux', 'macos')
@pytest.mark.usefixtures('child_env')  # yields tmp_path: an isolated environment for the real kernel
def test_execute_code_runs_each_sessions_own_interpreter(tmp_path):
    """Project mode picks the interpreter from the session's toolchain: each client's venv, and for a
    client with no venv the gateway's own Python, never the venv of the client that started the daemon."""
    import sys
    import venv
    from pathlib import Path
    from agent.session_client_env import client_env_scope
    roots = {name: tmp_path / name for name in ('alpha', 'beta', 'daemon')}
    for root in roots.values():
        venv.EnvBuilder(with_pip=False, symlinks=False).create(root)
    os.environ['VIRTUAL_ENV'] = str(roots['daemon'])  # inherited from whichever client started the gateway
    probe = 'import json, sys; print(json.dumps([sys.executable, sys.prefix]))'

    def child(client):
        with client_env_scope(client):
            return [Path(p) for p in run_code(probe)]
    for name in ('alpha', 'beta'):
        exe, prefix = child({'PATH': f'{roots[name] / "bin"}{os.pathsep}{os.environ["PATH"]}',
                             'VIRTUAL_ENV': str(roots[name])})
        assert exe == roots[name] / 'bin' / 'python' and prefix.resolve() == roots[name].resolve()
    exe, prefix = child({'PATH': os.environ['PATH']})  # a client with no activated venv
    assert prefix.resolve() == Path(sys.prefix).resolve()
    exe, prefix = child(None)  # messaging/cron/Desktop: the gateway's own environment, as before
    assert prefix.resolve() == roots['daemon'].resolve()


async def _owner(home, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    monkeypatch.setattr(run, '_hermes_home', home)
    monkeypatch.setenv('HERMES_HOME', str(home))
    store = SessionStore(home / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None,
                             _evict_cached_agent=lambda route: None)
    owner = await initialize_session_authority(runner, profile_id=str(home), instance_id='fixture')
    return owner, AuthorityConnection(owner, object(), {'user_id': 'owner'})


@pytest.mark.asyncio
async def test_a_sessions_terminal_runs_with_its_clients_path(tmp_path, monkeypatch, daemon_env):
    """Two CLI sessions on one gateway: each terminal command sees its own client's PATH and venv."""
    from gateway.session_local import create_local_session
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(
        'platform_toolsets: {cli: []}\nmodel: {provider: custom, default: m, base_url: http://x.invalid/v1}\n'
        'terminal: {client_env_passthrough: [MY_TOOLCHAIN_HOME, MY_API_TOKEN]}\n')
    bins = {name: tmp_path / name / 'bin' for name in ('alpha', 'beta')}
    for path in bins.values():
        path.mkdir(parents=True)
    monkeypatch.setenv('PATH', f'/daemon/bin{os.pathsep}{os.environ["PATH"].split(os.pathsep, 1)[-1]}')
    owner, connection = await _owner(home, monkeypatch)
    try:
        policies = {}
        for name, path in bins.items():
            ref = create_local_session(owner, connection.actor, {
                'request_id': name, 'source': 'cli', 'cwd': str(home), 'toolsets': [],
                'client_env': {'PATH': f'{path}{os.pathsep}/usr/bin{os.pathsep}/bin', 'NVM_DIR': str(path.parent),
                               'MY_TOOLCHAIN_HOME': f'/opt/{name}', 'MY_API_TOKEN': 'secret', 'UNLISTED': 'x'}})
            policies[name] = restore_policy(local_receipt(owner.db, ref.session_id)['policy'])
        # Listed toolchain names travel; a credential-looking name never lands in the stored policy.
        assert set(policies['alpha'].client_env()) == {'PATH', 'NVM_DIR', 'MY_TOOLCHAIN_HOME'}
        outputs = await asyncio.to_thread(_overlapping_terminal_turns, owner, policies)
        for name, runs in outputs.items():
            for out in runs:  # the first command, then again on the cached environment
                path, nvm, toolchain = out.strip().splitlines()[-1].split('|')
                assert str(bins[name]) in path.split(os.pathsep) and '/daemon/bin' not in path.split(os.pathsep)
                assert nvm == str(bins[name].parent) and toolchain == f'/opt/{name}'
    finally:
        from tools.terminal_tool_lifecycle import cleanup_all_environments
        cleanup_all_environments()
        await connection.close()
        owner.db.close()


def _overlapping_terminal_turns(owner, policies):
    """Each session's turn runs the real terminal entry point (environment cache, shell snapshot)
    with the SAME task id, both turns alive at once, each twice so the second hits the cache."""
    import json
    import threading
    from gateway.session_context import clear_session_vars, set_session_vars
    from gateway.session_policy import policy_scope
    from tools.terminal_tool import terminal_tool
    barrier, outputs, errors = threading.Barrier(len(policies)), {}, []

    def turn(name, policy):
        tokens = set_session_vars(platform='local', session_key=f'agent:main:local:dm:{name}', session_id=name)
        try:
            with policy_scope(policy, authority=owner):
                command = 'echo "$PATH|$NVM_DIR|$MY_TOOLCHAIN_HOME"'
                first = json.loads(terminal_tool(command, task_id='same-task'))['output']
                barrier.wait(timeout=60)  # both sessions now hold a cached environment for 'same-task'
                outputs[name] = (first, json.loads(terminal_tool(command, task_id='same-task'))['output'])
        except Exception as exc:  # surfaced below: an assertion in a thread would be lost
            errors.append(exc)
            barrier.abort()
        finally:
            clear_session_vars(tokens)
    threads = [threading.Thread(target=turn, args=item) for item in policies.items()]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not errors, errors
    return outputs
