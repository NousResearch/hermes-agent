"""Foreign-profile catalogs use the target authority, not the launcher's secrets."""

import os
from contextvars import Context
from types import SimpleNamespace

import pytest

from agent import secret_scope as secrets
from gateway import hosted_room_execution_policy as policies
from gateway.run import _profile_runtime_scope
from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from tools.terminal_scope import get_terminal_scope, reset_terminal_scope, set_terminal_scope
from tui_gateway import methods_groups


@pytest.mark.parametrize('ambient_has_token,target_has_token', [(True, False), (False, True)])
@pytest.mark.parametrize('fail', [False, True])
def test_rpc_policy_matches_target_api_authority_and_restores_context(
    tmp_path, monkeypatch, ambient_has_token, target_has_token, fail
):
    launcher, target = tmp_path / 'launcher', tmp_path / 'target'
    launcher.mkdir()
    target.mkdir()
    (launcher / 'config.yaml').write_text('agent:\n  max_turns: 99\n')
    (target / 'config.yaml').write_text('agent:\n  max_turns: 7\napprovals:\n  mode: manual\n')
    (target / '.env').write_text('HASS_TOKEN=target-test-token\n' if target_has_token else '')
    monkeypatch.setenv('HERMES_PROFILE', 'launcher')
    if ambient_has_token:
        monkeypatch.setenv('HASS_TOKEN', 'ambient-test-token')
    else:
        monkeypatch.delenv('HASS_TOKEN', raising=False)
    monkeypatch.setattr(secrets, '_MULTIPLEX_ACTIVE', False)
    monkeypatch.setattr(methods_groups, '_bound_server', SimpleNamespace(
        _current_profile_name=lambda: 'launcher', _profile_home=lambda profile: target if profile == 'target' else launcher))
    outer_secrets = {'OUTER_TEST_SECRET': 'outer'}
    outer_terminal = {'TERMINAL_CWD': str(launcher)}
    home_token = set_hermes_home_override(launcher)
    secret_token = secrets.set_secret_scope(outer_secrets)
    terminal_token = set_terminal_scope(outer_terminal)
    real_resolver = policies.execution_policy_mapping
    try:
        # Simulate the API deployment mode only in this disposable test phase.
        with monkeypatch.context() as api_process:
            api_process.setattr(secrets, '_MULTIPLEX_ACTIVE', True)
            with _profile_runtime_scope(target):
                expected = real_resolver(target_profile='target')
                target_terminal = dict(get_terminal_scope())
        assert ('homeassistant' in expected['enabled_toolsets']) is target_has_token
        assert expected['max_iterations'] == 7
        environment = dict(os.environ)

        def resolve_in_target(**kwargs):
            actual = real_resolver(**kwargs)
            assert actual == expected
            assert get_hermes_home() == target
            assert get_terminal_scope() == target_terminal
            assert secrets.current_secret_scope() is not outer_secrets
            assert secrets.is_multiplex_active() is False
            if fail:
                raise RuntimeError('forced resolver failure')
            return actual

        monkeypatch.setattr(policies, 'execution_policy_mapping', resolve_in_target)
        try:
            if fail:
                with pytest.raises(RuntimeError, match='forced resolver failure'):
                    methods_groups._profile_execution_policy('target')
            else:
                assert methods_groups._profile_execution_policy('target') == expected
        finally:
            assert get_hermes_home() == launcher
            assert secrets.current_secret_scope() is outer_secrets
            assert get_terminal_scope() is outer_terminal
            assert secrets.is_multiplex_active() is False
            assert os.environ == environment
            assert secrets.get_secret('HASS_TOKEN') == ('ambient-test-token' if ambient_has_token else None)
    finally:
        reset_terminal_scope(terminal_token)
        secrets.reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


@pytest.mark.parametrize('multiplex', [False, True])
def test_strict_scope_nested_clear_failure_and_default_compatibility(monkeypatch, multiplex):
    monkeypatch.setattr(secrets, '_MULTIPLEX_ACTIVE', multiplex)
    monkeypatch.setenv('AMBIENT_TEST_SECRET', 'ambient')
    monkeypatch.setenv('API_SERVER_PORT', '8123')
    outer = {'PROFILE_TEST_SECRET': 'outer'}
    token = secrets.set_secret_scope(outer)
    environment = dict(os.environ)
    try:
        expected_miss = 'fallback' if multiplex else 'ambient'
        assert secrets.get_secret('AMBIENT_TEST_SECRET', 'fallback') == expected_miss
        with pytest.raises(RuntimeError, match='outer failure'):
            with secrets.strict_secret_scope({'PROFILE_TEST_SECRET': 'profile', 'API_SERVER_PORT': '9999'}):
                assert secrets.get_secret('PROFILE_TEST_SECRET') == 'profile'
                assert secrets.get_secret('AMBIENT_TEST_SECRET', 'fallback') == 'fallback'
                assert secrets.get_secret('API_SERVER_PORT') == '8123'
                assert secrets.is_multiplex_active() is multiplex
                fresh_context = Context()
                if multiplex:
                    with pytest.raises(secrets.UnscopedSecretError):
                        fresh_context.run(secrets.get_secret, 'AMBIENT_TEST_SECRET')
                else:
                    assert fresh_context.run(secrets.get_secret, 'AMBIENT_TEST_SECRET') == 'ambient'
                with secrets.strict_secret_scope({'PROFILE_TEST_SECRET': 'nested'}):
                    assert secrets.get_secret('PROFILE_TEST_SECRET') == 'nested'
                    cleared = secrets.set_secret_scope(None)
                    try:
                        with pytest.raises(secrets.UnscopedSecretError):
                            secrets.get_secret('AMBIENT_TEST_SECRET', 'fallback')
                        assert secrets.get_secret('API_SERVER_PORT') == '8123'
                    finally:
                        secrets.reset_secret_scope(cleared)
                    assert secrets.get_secret('PROFILE_TEST_SECRET') == 'nested'
                with pytest.raises(RuntimeError, match='inner failure'):
                    with secrets.strict_secret_scope({}):
                        raise RuntimeError('inner failure')
                assert secrets.get_secret('PROFILE_TEST_SECRET') == 'profile'
                assert secrets.get_secret('AMBIENT_TEST_SECRET', 'fallback') == 'fallback'
                raise RuntimeError('outer failure')
        assert secrets.current_secret_scope() is outer
        assert secrets.get_secret('AMBIENT_TEST_SECRET', 'fallback') == expected_miss
        assert secrets.is_multiplex_active() is multiplex
        assert os.environ == environment
    finally:
        secrets.reset_secret_scope(token)
