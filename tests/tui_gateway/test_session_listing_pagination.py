"""Legacy session.list actual dispatch: bounded raw scanning and compatible envelopes."""
import sqlite3

import pytest

import tui_gateway.server as srv
from hermes_state import SessionDB
from tests.gateway.test_session_listing_subagents import _row, _snapshot
from tests.gateway.test_session_listing_pagination import _stable
from tui_gateway.contracts.sessions import SessionListResult


@pytest.fixture
def db(tmp_path, monkeypatch):
    database = SessionDB(tmp_path / 'state.db')
    monkeypatch.setattr(srv, '_get_db', lambda: database)
    yield database
    database.close()


def _call(**params):
    return srv.handle_request({'id': 1, 'method': 'session.list', 'params': params})


@pytest.mark.parametrize('count', [0, 199, 200, 201, 250])
def test_legacy_real_dispatch_depth(db, count):
    for i in range(count):
        _row(db, f'good-{i:04}')
    _stable(db)
    expected = [f'good-{i:04}' for i in reversed(range(count))]
    before = _snapshot(db)
    default = _call(limit=None)['result']
    assert default['offset'] == 0 and default['limit'] == 200
    assert default['has_more'] == (count > 200)
    actual = []
    for offset in range(0, count + 51, 50):
        result = _call(offset=offset, limit=50)['result']
        SessionListResult.model_validate(result)
        assert [r['id'] for r in result['sessions']] == expected[offset:offset + 50]
        assert result['has_more'] == (count > offset + 50)
        actual.extend(r['id'] for r in result['sessions'])
    assert actual == expected and _snapshot(db) == before


@pytest.mark.parametrize('key,value', [('offset', x) for x in [-1, True, False, 1.5, '1', None, 9007199254740992]] + [('limit', x) for x in [0, 201, True, False, '50', 1.5]])
def test_legacy_invalid_before_db(db, monkeypatch, key, value):
    before = _snapshot(db)
    def unexpected(*args, **kwargs):
        pytest.fail('invalid request reached DB')
    monkeypatch.setattr(srv, '_get_db', unexpected)
    reply = _call(**{key: value})
    assert reply['error']['code'] == 4000
    assert reply['error']['message'].startswith('invalid params:')
    assert _snapshot(db) == before


def test_legacy_filter_aware_windows_and_failure(db, monkeypatch):
    for i in range(200):
        db.create_session(f'z-cron-{i:04}', source='cron')
    for i in range(201):
        _row(db, f'good-{i:04}')
    _row(db, 'internal', source='tool')
    _stable(db)
    calls, rich = [], db.list_sessions_rich
    def traced(**kwargs):
        rows = rich(**kwargs)
        calls.append((kwargs['offset'], len(rows)))
        return rows
    monkeypatch.setattr(db, 'list_sessions_rich', traced)
    before = _snapshot(db)
    result = _call(offset=199, limit=2)['result']
    assert [r['id'] for r in result['sessions']] == ['good-0001', 'good-0000']
    assert result['has_more'] is False
    assert [x[0] for x in calls] == [0, 200, 400]
    def failing(**kwargs):
        if kwargs['offset']:
            raise sqlite3.OperationalError('second chunk unavailable')
        return rich(**kwargs)
    monkeypatch.setattr(db, 'list_sessions_rich', failing)
    reply = _call()
    assert reply['error']['code'] == 5006 and 'result' not in reply
    assert _snapshot(db) == before


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('count', [199, 200, 201])
def test_legacy_mixed_boundaries_and_config(db, monkeypatch, enabled, count):
    from pathlib import Path
    Path(db.db_path).parent.joinpath('config.yaml').write_text(
        'sessions:\n  show_subagents: ' + str(enabled).lower() + '\n')
    for i in range(count):
        _row(db, f'good-{i:04}')
    _row(db, 'child', parent='good-0000', source='subagent')
    _row(db, 'inherited', parent='good-0000')
    _stable(db)
    expected = sorted([f'good-{i:04}' for i in range(count)] +
                      (['child', 'inherited'] if enabled else []), reverse=True)
    before = _snapshot(db)
    for offset, limit in [(198, 1), (199, 2), (200, 200)]:
        result = _call(offset=offset, limit=limit)['result']
        assert [r['id'] for r in result['sessions']] == expected[offset:offset + limit]
        assert result['has_more'] == (len(expected) > offset + limit)
    assert _snapshot(db) == before


def test_legacy_aliases_across_raw_windows(db, monkeypatch):
    _row(db, 'a')
    _row(db, 'b')
    rows = {r['id']: r for r in db.list_sessions_rich(limit=200)}
    denied = dict(rows['a'], id='internal', source='tool')
    raw = [denied] * 199 + [rows['b'], rows['b'], rows['a']] + [denied] * 200
    calls = []
    def chunks(**kwargs):
        calls.append(kwargs['offset'])
        return raw[kwargs['offset']:kwargs['offset'] + kwargs['limit']]
    monkeypatch.setattr(db, 'list_sessions_rich', chunks)
    result = _call(offset=1, limit=1)['result']
    assert [r['id'] for r in result['sessions']] == ['a']
    assert result['has_more'] is False and calls == [0, 200, 400]


def test_legacy_selected_profile_isolation(db, tmp_path, monkeypatch):
    from pathlib import Path
    from hermes_cli import profiles
    home = tmp_path / 'profile-home'
    selected = home / 'profiles' / 'ops'
    selected.mkdir(parents=True)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(srv, '_hermes_home', home)
    monkeypatch.setattr(profiles, '_get_profiles_root', lambda: home / 'profiles')
    _row(db, 'launch-private')
    foreign = SessionDB(selected / 'state.db')
    for i in range(201):
        _row(foreign, f'ops-{i:04}')
    _stable(foreign)
    before = _snapshot(foreign)
    result = _call(profile='ops', offset=200, limit=1)['result']
    assert [r['id'] for r in result['sessions']] == ['ops-0000']
    assert result['has_more'] is False
    assert _snapshot(foreign) == before
    assert _call(profile='missing')['error']['code'] == 4064
    assert _call(profile='../ops')['error']['code'] == 4064
    foreign.close()


def test_legacy_title_bounds_defaults_and_unknown_keys(db):
    _row(db, 'title')
    db.set_session_title('title', 'My chat')
    result = _call(title='My chat', offset=0, limit=1)['result']
    assert len(result['sessions']) == 1
    assert result['offset'] == 0 and result['limit'] == 1 and result['has_more'] is False
    assert _call(title='My chat', offset=1)['error']['code'] == 4000
    result = _call(offset=9007199254740991, limit=200)['result']
    assert result['sessions'] == [] and result['has_more'] is False
    assert _call(include_subagents=True)['error']['code'] == 4000
