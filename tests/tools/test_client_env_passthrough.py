"""A sandbox allowlist must not erase an already-admitted local client toolchain."""

import os
import sys
from contextlib import contextmanager

import hermes_yaml as yaml
import pytest

from agent import secret_scope as ss
from agent.session_client_env import (
    accept_client_env, client_env_rules, client_env_scope, collect_client_env,
)
from tests.tools import _child_env_fixtures
from tests.tools._child_env_fixtures import observe_child, observe_terminal
from tools import env_passthrough


NAMES: tuple[str, ...] = ('CLIENT_WORKSPACE', 'CLIENT_EMPTY')
child_env = _child_env_fixtures.child_env


@contextmanager
def _routed_scope(secrets):
    multiplex = ss.set_multiplex_context(True)
    token = ss.set_secret_scope(secrets)
    try:
        yield
    finally:
        ss.reset_secret_scope(token)
        ss.reset_multiplex_context(multiplex)


def _configure(home, *, registration='config'):
    config = {'terminal': {'client_env_passthrough': list(NAMES),
                           'env_passthrough': list(NAMES) if registration == 'config' else []}}
    home.mkdir(parents=True, exist_ok=True)
    (home / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    if registration == 'skill':
        env_passthrough.register_env_passthrough(NAMES)
    return config


def _child_environment(surface, tmp_path, extra=None):
    from tools.code_execution_env import _build_child_env
    from tools.environments.local import _make_run_env
    from tools.process_registry import ProcessRegistry

    builders = {
        'foreground': lambda: _make_run_env(extra or {}),
        'background': lambda: ProcessRegistry._spawn_env(extra or {}),
        'execute_code': lambda: _build_child_env(rpc_endpoint='probe', rpc_token='probe',
                                               tmpdir=str(tmp_path), child_python=sys.executable),
    }
    return builders[surface]()


@pytest.mark.parametrize('surface', ['foreground', 'background', 'execute_code'])
@pytest.mark.parametrize('registration', ['config', 'skill'])
def test_allowlisted_local_client_values_reach_each_child(child_env, surface, registration):
    """Missing/conflicting profile values cannot replace a local client's values, including empty."""
    config = _configure(child_env / 'hermes', registration=registration)
    rules = client_env_rules(config)
    for secrets in ({}, {'CLIENT_WORKSPACE': 'profile-value', 'CLIENT_EMPTY': 'profile-nonempty'}):
        with _routed_scope(secrets):
            for workspace in ('workspace-a', 'workspace-b'):
                sent = collect_client_env({'CLIENT_WORKSPACE': workspace, 'CLIENT_EMPTY': ''}, config)
                client = accept_client_env(sent, config)
                with client_env_scope(client, rules=rules):
                    env = _child_environment(surface, child_env)
                    assert observe_child(env, NAMES) == client
                    if surface != 'execute_code':
                        # The backend's explicit env continues to win over the client toolchain.
                        extra = {'CLIENT_WORKSPACE': 'backend-value', 'CLIENT_EMPTY': ''}
                        env = _child_environment(surface, child_env, extra)
                        assert observe_child(env, NAMES) == extra
    assert all(name not in os.environ for name in NAMES)


@pytest.mark.platforms('linux', 'macos')
def test_cached_shell_keeps_client_and_profile_secret_provenance_separate(child_env, monkeypatch):
    """A reused shell changes client identity, clears omitted values, and retains scoped secrets only."""
    from tools.environments.local import LocalEnvironment

    config = _configure(child_env / 'hermes')
    config['terminal']['env_passthrough'].extend(['SERVICE_TOKEN', 'OPENAI_API_KEY'])
    (child_env / 'hermes' / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    monkeypatch.setenv('CLIENT_WORKSPACE', 'daemon-workspace')
    monkeypatch.setenv('SERVICE_TOKEN', 'daemon-service')
    monkeypatch.setenv('OPENAI_API_KEY', 'daemon-provider')
    rules = client_env_rules(config)
    names = [*NAMES, 'SERVICE_TOKEN', 'OPENAI_API_KEY']
    # Construct before binding the client, so the initial snapshot cannot supply the right value.
    env = LocalEnvironment(cwd=str(child_env))
    try:
        with _routed_scope({'SERVICE_TOKEN': 'profile-service'}):
            for workspace in ('workspace-a', 'workspace-b'):
                client = accept_client_env({'CLIENT_WORKSPACE': workspace, 'CLIENT_EMPTY': '',
                                            'SERVICE_TOKEN': 'client-service', 'OPENAI_API_KEY': 'client-provider'}, config)
                assert client == {'CLIENT_WORKSPACE': workspace, 'CLIENT_EMPTY': ''}
                with client_env_scope(client, rules=rules):
                    expected = {**client, 'SERVICE_TOKEN': 'profile-service', 'OPENAI_API_KEY': None}
                    assert observe_terminal(env, names) == expected
                    # Other resolver callers (remote/voice) have not merged the client
                    # toolchain. Its presence must not authorize their ambient fallback.
                    assert env_passthrough.resolve_passthrough_value('CLIENT_WORKSPACE', 'ambient') is None
            # An omitted client name is not recovered from the daemon or previous snapshot.
            with client_env_scope({}, rules=rules):
                assert observe_terminal(env, names) == {
                    'CLIENT_WORKSPACE': None, 'CLIENT_EMPTY': None,
                    'SERVICE_TOKEN': 'profile-service', 'OPENAI_API_KEY': None,
                }
        with _routed_scope({}):
            # A remote client never acquires the local client's exception, even for an old map.
            for client, local in ((None, True), ({'CLIENT_WORKSPACE': 'remote-workspace'}, False)):
                with client_env_scope(client, local=local, rules=rules):
                    assert observe_terminal(env, names) == dict.fromkeys(names)
    finally:
        env.cleanup()
