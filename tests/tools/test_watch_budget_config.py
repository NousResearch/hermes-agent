"""Operator-configured watch budgets keep the existing breakers (#139152, #93513)."""
from concurrent.futures import ThreadPoolExecutor
import json
from queue import Empty

import pytest

from tools.process_registry import ProcessRegistry


def _config(home, value):
    from hermes_cli.config import atomic_config_write
    home.mkdir(parents=True, exist_ok=True)
    atomic_config_write(home / 'config.yaml', {'terminal': {'watch_lifetime_max_hits': value}})


def _session():
    session = ProcessRegistry._new_session('fixture', 'owner', 'owner', 'route', None)
    session.watch_patterns = ['READY']
    return session


def _events(registry):
    result = []
    while True:
        try:
            result.append(registry.completion_queue.get_nowait())
        except Empty:
            return result


def _match(registry, session, monkeypatch, now):
    monkeypatch.setattr('tools.process_registry.time.time', lambda: now)
    registry._check_watch_patterns(session, 'READY stage\n')


@pytest.mark.parametrize('budget', [1, 12, 64])
def test_configured_budget_controls_exact_cutoff_and_exit_promotion(tmp_path, monkeypatch, budget):
    home = tmp_path / 'profile'
    _config(home, budget)
    monkeypatch.setenv('HERMES_HOME', str(home))
    before = (home / 'config.yaml').read_bytes()
    registry, session = ProcessRegistry(), _session()
    for index in range(budget):
        _match(registry, session, monkeypatch, 1000 + index * 16)
        assert session._watch_disabled is (index == budget - 1)
    events = _events(registry)
    assert sum(event['type'] == 'watch_match' for event in events) == budget
    assert events[-1]['type'] == 'watch_disabled'
    assert f'lifetime cap of {budget}' in events[-1]['message']
    assert session.notify_on_complete
    _match(registry, session, monkeypatch, 1000 + budget * 16)
    assert _events(registry) == []
    assert (home / 'config.yaml').read_bytes() == before


@pytest.mark.parametrize('value', [0, -1, 65, True, False, None, 'unbounded', '12', 12.5, [], {}])
def test_invalid_config_keeps_default_breaker(tmp_path, monkeypatch, value):
    home = tmp_path / 'profile'
    _config(home, value)
    monkeypatch.setenv('HERMES_HOME', str(home))
    registry, session = ProcessRegistry(), _session()
    for index in range(9):
        _match(registry, session, monkeypatch, 1000 + index * 16)
    assert session.watch_lifetime_max_hits == 8
    assert session._watch_hits == 8 and session._watch_disabled
    assert sum(event['type'] == 'watch_match' for event in _events(registry)) == 8


def test_absent_config_retains_eight_notifications(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    registry, session = ProcessRegistry(), _session()
    for index in range(9):
        _match(registry, session, monkeypatch, 1000 + index * 16)
    assert session._watch_hits == 8 and session.notify_on_complete
    assert sum(event['type'] == 'watch_match' for event in _events(registry)) == 8


def test_profile_budgets_are_captured_before_reader_threads_and_config_edits(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    launch, secondary = tmp_path / 'launch', tmp_path / 'secondary'
    _config(launch, 12)
    _config(secondary, 24)
    monkeypatch.setenv('HERMES_HOME', str(launch))

    def spawn(home):
        token = set_hermes_home_override(home)
        try:
            return _session()
        finally:
            reset_hermes_home_override(token)

    with ThreadPoolExecutor(max_workers=3) as pool:
        a, b, again = list(pool.map(spawn, [launch, secondary, launch]))
    assert [s.watch_lifetime_max_hits for s in (a, b, again)] == [12, 24, 12]
    _config(launch, 1)
    assert _session().watch_lifetime_max_hits == 1
    # The reader is now outside either spawning scope. Never reload that profile's new value.
    registry = ProcessRegistry()
    for index in range(9):
        _match(registry, a, monkeypatch, 1000 + index * 16)
    assert a._watch_hits == 9 and not a._watch_disabled
    assert [s.watch_lifetime_max_hits for s in (a, b, again)] == [12, 24, 12]


def test_larger_budget_does_not_relax_cooldown_or_three_strike_breaker(tmp_path, monkeypatch):
    _config(tmp_path, 64)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    registry, session = ProcessRegistry(), _session()
    for window in range(3):
        _match(registry, session, monkeypatch, 1000 + window * 16)
        _match(registry, session, monkeypatch, 1001 + window * 16)
    events = _events(registry)
    assert sum(event['type'] == 'watch_match' for event in events) == 3
    assert session._watch_hits == 3 and session._watch_consecutive_strikes == 3
    assert session._watch_disabled and session.notify_on_complete
    assert 'consecutive rate-limit windows' in events[-1]['message']


def test_larger_budget_keeps_cross_process_flood_breaker(tmp_path, monkeypatch):
    from tools.process_registry import WATCH_GLOBAL_MAX_PER_WINDOW
    _config(tmp_path, 64)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    registry = ProcessRegistry()
    for _ in range(WATCH_GLOBAL_MAX_PER_WINDOW + 2):
        _match(registry, _session(), monkeypatch, 1000)
    events = _events(registry)
    assert sum(event['type'] == 'watch_match' for event in events) == WATCH_GLOBAL_MAX_PER_WINDOW
    assert sum(event['type'] == 'watch_overflow_tripped' for event in events) == 1


@pytest.mark.parametrize('stored, expected', [(12, 12), (0, 8), ('unbounded', 8)])
def test_recovery_keeps_recorded_limit_and_validates_bad_values(tmp_path, monkeypatch, stored, expected):
    _config(tmp_path, 32)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    checkpoint = tmp_path / 'processes.json'
    checkpoint.write_text(json.dumps([{
        'session_id': 'proc-recovered', 'command': 'fixture', 'pid': 123456,
        'watch_patterns': ['READY'], 'watch_lifetime_max_hits': stored,
    }]))
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, '_detached_host_fate', lambda *args: 'running')
    monkeypatch.setattr(registry, '_safe_host_start_time', lambda pid: None)
    assert registry.recover_from_checkpoint() == 1
    recovered = registry.get('proc-recovered')
    assert recovered.watch_lifetime_max_hits == expected
    assert json.loads(checkpoint.read_text())[0]['watch_lifetime_max_hits'] == expected


def test_legacy_checkpoint_retains_default_not_current_profile_override(tmp_path, monkeypatch):
    _config(tmp_path, 32)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    checkpoint = tmp_path / 'processes.json'
    checkpoint.write_text(json.dumps([{'session_id': 'proc-old', 'command': 'fixture', 'pid': 123456}]))
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, '_detached_host_fate', lambda *args: 'running')
    monkeypatch.setattr(registry, '_safe_host_start_time', lambda pid: None)
    assert registry.recover_from_checkpoint() == 1
    assert registry.get('proc-old').watch_lifetime_max_hits == 8


def test_unreadable_config_falls_back_without_echoing_its_contents(monkeypatch, caplog):
    def unreadable():
        raise OSError('synthetic-private-config-content')

    monkeypatch.setattr('hermes_cli.config_effective.load_user_config_effective', unreadable)
    assert _session().watch_lifetime_max_hits == 8
    assert 'synthetic-private-config-content' not in caplog.text


def test_reading_default_budget_does_not_initialize_a_missing_profile(tmp_path, monkeypatch):
    missing = tmp_path / 'profiles' / 'not-created'
    monkeypatch.setenv('HERMES_HOME', str(missing))
    assert _session().watch_lifetime_max_hits == 8
    assert not missing.exists()
