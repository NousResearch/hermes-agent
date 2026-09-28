"""The embedded TUI cannot revive a retired personality overlay."""
from tui_gateway import server


def test_personality_and_custom_prompt_route_to_employee_config(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    for key in ('personality', 'prompt'):
        result = server._methods['config.set'](1, {'key': key, 'value': 'Legacy identity'})
        assert result['error']['code'] == 4002
        assert 'employee.instructions' in result['error']['message']
    assert not (tmp_path / 'config.yaml').exists()


def test_retired_commands_cannot_reach_tui_handlers(monkeypatch):
    from hermes_cli.commands import EMPLOYEE_EXCLUDED_COMMAND_NAMES
    monkeypatch.setattr(server, '_sess_nowait', lambda params, rid: ({}, None))
    for name in EMPLOYEE_EXCLUDED_COMMAND_NAMES:
        for method, params in (
            ('command.dispatch', {'name': name}),
            ('slash.exec', {'command': f'/{name} add anything'}),
        ):
            result = server._methods[method](1, params)
            assert result['error']['code'] == 4018
            assert 'unavailable' in result['error']['message']


def test_dedicated_management_rpcs_enforce_employee_surface(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    for action in ('list', 'search', 'install', 'inspect', 'browse'):
        result = server._methods['skills.manage'](1, {'action': action, 'query': 'old'})
        assert result['error']['code'] == 4017
    for action in ('add', 'remove'):
        result = server._methods['cron.manage'](1, {'action': action, 'name': 'old'})
        assert result['error']['code'] == 4016
    result = server._methods['cron.manage'](1, {'action': 'list'})
    assert 'result' in result
    assert not (tmp_path / 'cron' / 'jobs.json').exists()


def test_profile_rpc_rejects_retired_authoring_before_creating_profile(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    for method, params in (
        ('profiles.create', {'name': 'old', 'soul': 'Legacy persona'}),
        ('profiles.configure', {'name': 'old', 'soul': 'Legacy persona'}),
        ('profiles.configure', {'name': 'old', 'disabled_skills': []}),
    ):
        result = server._methods[method](1, params)
        assert 'error' in result
        assert 'employee.instructions' in result['error']['message']
    assert not (tmp_path / 'profiles' / 'old').exists()
