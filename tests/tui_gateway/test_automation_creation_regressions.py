"""Parent integration regressions for creation semantics."""
import importlib.util
from pathlib import Path
_spec = importlib.util.spec_from_file_location('automation_creation_fixtures', Path(__file__).with_name('test_session_control_create.py'))
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
server, session, hermes_home, _call = (_fixtures.server, _fixtures.session, _fixtures.hermes_home, _fixtures._call)
import pytest

@pytest.mark.parametrize('kind', ['loop', 'heartbeat'])
def test_scheduler_owns_first_tick(server, session, kind):
    sid, _, _ = session
    response = _call(server, 'session.control', session_id=sid, action=f'{kind}.create', args={'prompt':'Check status', 'interval_seconds':60})
    assert response['result']['dispatch']['type'] == 'exec', 'Do not submit unaccounted ticks from the renderer'

def test_goal_initial_display_is_not_a_slash_command(server, session):
    sid, _, _ = session
    response = _call(server, 'session.control', session_id=sid, action='goal.create', args={'prompt':'Fix login'})
    assert response['result']['dispatch']['type'] == 'send'
    assert response['result']['dispatch']['display'] == 'Fix login'

def test_concurrent_creates_cannot_replace(server, session, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from hermes_cli import heartbeat
    import time
    sid, key, _ = session
    load = heartbeat.load_heartbeat
    def delayed_load(key):
        state = load(key)
        time.sleep(0.04)
        return state
    monkeypatch.setattr(heartbeat, 'load_heartbeat', delayed_load)
    def create(i):
        return _call(server, 'session.control', session_id=sid, action='heartbeat.create', args={'prompt':f'worker {i}', 'interval_seconds':60})
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(create, range(4)))
    assert sum('result' in r for r in responses) == 1
    assert all(r['id'] == 91 for r in responses), 'Errors must resolve the original RPC request'

def test_goal_uses_configured_turn_budget(server, session, monkeypatch):
    sid, _, _ = session
    monkeypatch.setattr(server, '_load_cfg', lambda: {'goals': {'max_turns': 7}})
    response = _call(server, 'session.control', session_id=sid, action='goal.create', args={'prompt':'Fix login'})
    assert response['result']['control']['goal']['max_turns'] == 7

def test_creation_and_read_use_session_profile(server, session, tmp_path, monkeypatch):
    import hermes_state
    monkeypatch.setattr(hermes_state, '_IMPORT_DEFAULT_DB_PATH', hermes_state.DEFAULT_DB_PATH)
    from hermes_cli.heartbeat import load_heartbeat
    sid, key, entry = session
    other_home = tmp_path / 'other-profile'
    other_home.mkdir()
    entry['profile_home'] = other_home
    response = _call(server, 'session.control', session_id=sid, action='heartbeat.create', args={'prompt':'Owned profile', 'interval_seconds':60})
    assert response['result']['control']['heartbeat']['prompt'] == 'Owned profile'
    assert load_heartbeat(key) is None, 'Must not write ambient profile'
    response = _call(server, 'session.control', session_id=sid, action='heartbeat.pause')
    assert response['result']['control']['heartbeat']['status'] == 'paused'
    assert response['result']['control']['heartbeat']['prompt'] == 'Owned profile'

@pytest.mark.parametrize('kind', ['goal', 'loop', 'heartbeat'])
def test_unpersisted_creation_is_not_reported_success(server, session, monkeypatch, kind):
    import importlib
    module = importlib.import_module('hermes_cli.' + {'goal':'goals', 'loop':'loops', 'heartbeat':'heartbeat'}[kind])
    monkeypatch.setattr(module, 'save_' + kind, lambda *args, **kwargs: None)
    sid, _, _ = session
    response = _call(server, 'session.control', session_id=sid, action=f'{kind}.create', args={'prompt':'Persist me', 'interval_seconds':60})
    assert 'error' in response
