import pytest
from types import SimpleNamespace

from tools.computer_use.execution_revision import admit, validate, StaleRevision


def backend(**kw):
    return SimpleNamespace(_active_pid=kw.get('pid'), _active_window_id=kw.get('window'),
                           backend_generation=kw.get('generation', 1), _snapshot_tokens=kw.get('tokens', {}))


def test_valid_admit_validate_succeeds(monkeypatch):
    monkeypatch.setattr('hermes_constants.hermes_home_key', lambda: 'profile-a')
    b = backend(pid=3, window='w', tokens={1: 'tok'})
    rev = admit(b, 'click', {'element': 1})
    validate(rev, b, deps=('profile', 'display', 'backend', 'epoch', 'target', 'snapshot'))


def test_display_change_is_stale(monkeypatch):
    monkeypatch.setattr('hermes_constants.hermes_home_key', lambda: 'p')
    monkeypatch.delenv('WAYLAND_DISPLAY', raising=False)
    monkeypatch.delenv('DISPLAY', raising=False)
    b = backend(); rev = admit(b, 'capture', {})
    monkeypatch.setenv('DISPLAY', ':99')
    with pytest.raises(StaleRevision) as exc: validate(rev, b, deps=('display',))
    assert exc.value.reason == 'display_changed'


def test_generation_change_is_stale():
    b = backend(); rev = admit(b, 'capture', {})
    b.backend_generation += 1
    with pytest.raises(StaleRevision) as exc: validate(rev, b, deps=('backend',))
    assert exc.value.reason == 'backend_changed'


def test_profile_change_is_stale(monkeypatch):
    monkeypatch.setattr('hermes_constants.hermes_home_key', lambda: 'a')
    b = backend(); rev = admit(b, 'list_apps', {})
    monkeypatch.setattr('hermes_constants.hermes_home_key', lambda: 'b')
    with pytest.raises(StaleRevision) as exc: validate(rev, b, deps=('profile',))
    assert exc.value.reason == 'profile_changed'


def test_none_epoch_is_not_zero_and_provider_mismatch_rejects(monkeypatch):
    monkeypatch.setattr('hermes_constants.hermes_home_key', lambda: 'p')
    b = backend(); rev = admit(b, 'capture', {})
    assert rev.control_epoch is None
    b.control_epoch_provider = lambda: 4
    with pytest.raises(StaleRevision) as exc: validate(rev, b, deps=('epoch',))
    assert exc.value.reason == 'epoch_changed'


@pytest.fixture(autouse=True)
def _noop_backend(grant_computer_use_approvals):
    from tools.computer_use.tool import reset_backend_for_tests
    from unittest.mock import patch
    reset_backend_for_tests()
    with patch.dict('os.environ', {'HERMES_COMPUTER_USE_BACKEND': 'noop'}, clear=False):
        yield
    reset_backend_for_tests()


def test_handle_capture_generation_stale_has_no_publication_side_effects(monkeypatch):
    from tools.computer_use import tool
    import json
    backend = tool._get_backend()
    original = backend.capture
    def capture(*a, **kw):
        result = original(*a, **kw)
        backend.backend_generation = 99
        return result
    monkeypatch.setattr(backend, 'capture', capture)
    counters = {name: 0 for name in ('persist', 'spill', 'aux')}
    monkeypatch.setattr(tool, '_persist_capture_image', lambda cap: counters.__setitem__('persist', counters['persist'] + 1))
    monkeypatch.setattr(tool, '_spill_elements_to_file', lambda cap: counters.__setitem__('spill', counters['spill'] + 1))
    monkeypatch.setattr(tool, '_route_capture_through_aux_vision', lambda *a, **k: counters.__setitem__('aux', counters['aux'] + 1))
    out = json.loads(tool.handle_computer_use({'action': 'capture', 'mode': 'som'}))
    assert out['code'] == 'revision_stale'
    assert counters == {'persist': 0, 'spill': 0, 'aux': 0}


def test_handle_capture_unchanged_facts_is_normal(monkeypatch):
    from tools.computer_use import tool
    import json
    out = json.loads(tool.handle_computer_use({'action': 'capture', 'mode': 'som'}))
    assert out.get('code') != 'revision_stale'
    assert out.get('mode') == 'som'


def test_handle_list_apps_still_works():
    from tools.computer_use import tool
    import json
    out = json.loads(tool.handle_computer_use({'action': 'list_apps'}))
    assert 'apps' in out and 'count' in out


def test_capture_after_snapshot_change_still_publishes(monkeypatch):
    from tools.computer_use import tool
    import json
    backend = tool._get_backend()
    original = backend.capture
    def capture(*a, **kw):
        result = original(*a, **kw)
        backend._snapshot_tokens[1] = 'new-token'
        return result
    monkeypatch.setattr(backend, 'capture', capture)
    raw = tool.handle_computer_use({'action': 'click', 'element': 1, 'capture_after': True})
    out = json.loads(raw) if isinstance(raw, str) else raw
    assert out.get('code') != 'revision_stale'
    assert out.get('ok') is True or out.get('mode') or out.get('_multimodal')
