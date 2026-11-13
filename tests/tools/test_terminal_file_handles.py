"""Post-turn handles have fixed leases, capacity and replacement/session fencing."""
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from tools import terminal_file_handles as handles
from tools import terminal_tool
from tools.terminal_tool_lifecycle import cleanup_vm


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(handles, '_handles', OrderedDict())
    monkeypatch.setattr(terminal_tool, '_active_environments', {})
    monkeypatch.setattr(handles, '_MAX_HANDLES', 2)
    monkeypatch.setattr(handles, '_HANDLE_TTL', 10)
    now = [0]
    monkeypatch.setattr(handles.time, 'monotonic', lambda: now[0])
    return now


def retire(key, cleanup=lambda: None):
    env = SimpleNamespace(cleanup=cleanup)
    terminal_tool._active_environments[key] = env
    cleanup_vm(key, retain_file_handle=True)
    return env


def test_post_turn_handles_are_capacity_bounded_with_fixed_not_sliding_expiry(registry):
    first = retire('profile-a:first')
    registry[0] = 1
    second = retire('profile-b:second')
    assert handles.resolve('profile-a:first') is first
    retire('profile-a:third')
    assert handles.resolve('profile-a:first') is None
    registry[0] = 10
    assert handles.resolve('profile-b:second') is second
    registry[0] = 11
    assert handles.resolve('profile-b:second') is None
    assert handles.resolve('profile-a:third') is None


def test_active_replacement_never_falls_back_to_retired_transport(registry):
    retire('qualified')
    new = SimpleNamespace(cleanup=lambda: None)
    terminal_tool._active_environments['qualified'] = new
    assert handles.resolve('qualified') is new
    cleanup_vm('qualified')
    assert handles.resolve('qualified') is None


def test_session_close_during_turn_cleanup_cannot_resurrect_handle(registry):
    def close_session():
        cleanup_vm('qualified')
    retire('qualified', close_session)
    assert handles.resolve('qualified') is None


def test_idle_reaper_never_revives_previous_transport(registry):
    from tools.terminal_tool_lifecycle import _cleanup_inactive_envs
    retire('qualified')
    terminal_tool._active_environments['qualified'] = SimpleNamespace(cleanup=lambda: None)
    terminal_tool._last_activity['qualified'] = -1000
    _cleanup_inactive_envs(lifetime_seconds=0)
    assert handles.resolve('qualified') is None


def test_force_remove_does_not_retain_transport(registry):
    terminal_tool._active_environments['qualified'] = SimpleNamespace(cleanup=lambda: None)
    cleanup_vm('qualified', force_remove=True, retain_file_handle=True)
    assert handles.resolve('qualified') is None
