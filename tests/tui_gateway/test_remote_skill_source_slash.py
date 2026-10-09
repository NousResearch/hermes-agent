"""Exercise the real RPC backend shared by TUI and Desktop; not a UI E2E test."""
from tui_gateway import server
from hermes_cli import plugins
from hermes_constants import get_hermes_home


def test_ram_source_completion_and_dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    plugins._reset_plugin_managers_for_tests()
    manager = plugins.get_plugin_manager()
    manager._discovered = True
    ctx = plugins.PluginContext(plugins.PluginManifest(name='remote-probe', version='1'), manager)
    handle = ctx.register_skill_source('catalog',
        list_skills=lambda: [{'name': 'interview-me', 'description': 'Interview', 'uri': 'test://skill'}],
        load_skill=lambda uri: {'name': 'interview-me', 'content': 'Ask one question at a time.'})
    sid = 'remote-probe'
    server._sessions[sid] = {'session_key': sid, 'profile_home': str(get_hermes_home()), 'agent': None}
    try:
        catalog = server.handle_request({'id': 'c', 'method': 'commands.catalog', 'params': {'session_id': sid}})
        completion = server.handle_request({'id': 's', 'method': 'complete.slash',
            'params': {'session_id': sid, 'text': '/interview'}})
        dispatch = server.handle_request({'id': 'd', 'method': 'command.dispatch',
            'params': {'session_id': sid, 'name': 'interview-me', 'arg': 'plan my project'}})
        assert list(catalog['result']['skills']).count('/interview-me') == 1
        assert [item['text'] for item in completion['result']['items'] if item['kind'] == 'skill'] == ['interview-me']
        assert dispatch['result']['type'] == 'skill'
        assert 'Ask one question at a time.' in dispatch['result']['message']
        assert 'plan my project' in dispatch['result']['message']
        assert not list(tmp_path.rglob('SKILL.md'))
    finally:
        handle.dispose()
        server._sessions.pop(sid, None)
        plugins._reset_plugin_managers_for_tests()
