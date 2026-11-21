"""A host's nonblocking delegation contract never falls back to joined execution."""
import contextvars
import json
import threading
import time
from types import SimpleNamespace

import pytest

from tools import delegate_tool_dispatch as dispatch


class Child:
    def __init__(self, index):
        self._subagent_id = f'child-{index}'
        self.closed = 0

    def close(self):
        self.closed += 1


def batch(count=1, *, parent=None):
    parent = parent or SimpleNamespace(_active_children=[], _active_children_lock=threading.Lock())
    children = [Child(i) for i in range(count)]
    parent._active_children.extend(children)
    tasks = [{'goal': f'task {i}', 'group': str(i)} for i in range(count)]
    return dispatch._Batch(
        task_list=tasks, children=[(i, tasks[i], child) for i, child in enumerate(children)],
        parent_agent=parent, creds={'model': 'test'}, context=None, top_role='leaf', max_children=count,
        live_deleg_id=None, live_writers=[], live_paths=[], origin_wake_sid='', origin_ui_session_id='',
        origin_owner_transport=None, origin_owner_session_record=None, origin_session_history_delivery=False,
        overall_start=time.monotonic())


@pytest.fixture
def execution(monkeypatch, tmp_path):
    (tmp_path / 'config.yaml').write_text('delegation:\n  independent_completions: true\n', encoding='utf-8')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(dispatch, '_resolve_async_wake_sid', lambda *args, **kwargs: '')
    inline = []
    monkeypatch.setattr(dispatch, '_execute_and_aggregate', lambda unit, **kwargs:
                        inline.append(unit.parent_agent) or {'results': []})
    return inline


@pytest.mark.parametrize('reason', ['pinned', 'oneshot', 'no_wake', 'capacity'])
def test_required_background_refuses_and_releases_instead_of_joining(execution, monkeypatch, reason):
    work = batch()
    if reason == 'pinned':
        from tools.delegate_tool import _ONESHOT_DESCRIPTION_DELIVERY
        work.parent_agent.tools = [{'function': {'name': 'delegate_task', 'description': _ONESHOT_DESCRIPTION_DELIVERY}}]
    elif reason == 'oneshot':
        work.parent_agent._oneshot_session = True
    elif reason == 'no_wake':
        monkeypatch.setattr(dispatch, '_resolve_async_wake_sid', lambda *args, **kwargs: None)
    attempts = []
    monkeypatch.setattr(dispatch, '_dispatch_unit', lambda *args:
                        attempts.append(args) or {'status': 'rejected', 'error': 'capacity'})
    with dispatch.require_background(work.parent_agent):
        result = json.loads(dispatch._run_batch(work, background=False))
    assert result['error']
    assert execution == []
    assert work.children[0][2].closed == 1
    assert work.parent_agent._active_children == []
    assert len(attempts) == (1 if reason == 'capacity' else 0)


def test_partial_dispatch_preserves_accepted_work_and_releases_only_remaining(execution, monkeypatch):
    work = batch(3)
    attempts = []

    def admit(unit, unit_id, slot_key, routing):
        attempts.append(unit)
        return ({'status': 'dispatched', 'delegation_id': 'accepted-unit'} if len(attempts) == 1
                else {'status': 'rejected', 'error': 'scheduler unavailable'})

    monkeypatch.setattr(dispatch, '_dispatch_unit', admit)
    with dispatch.require_background(work.parent_agent):
        result = json.loads(dispatch._run_batch(work, background=False))
    assert result['status'] == 'dispatched'
    assert result['delegation_id'] == 'accepted-unit'
    assert result['subagent_ids'] == ['child-0']
    assert result['count'] == 1
    assert [entry['task_index'] for entry in result['refused']] == [1, 2]
    assert [child.closed for _, _, child in work.children] == [0, 1, 1]
    assert len(attempts) == 2 and execution == []
    assert work.parent_agent._active_children == []


def test_scope_targets_exact_parent_and_resets_after_exception(execution):
    outer = batch()
    nested = batch()
    with pytest.raises(RuntimeError, match='release scope'):
        with dispatch.require_background(outer.parent_agent):
            # Copied contexts, as used by detached workers, do not constrain their own children.
            contextvars.copy_context().run(dispatch._run_batch, nested, False)
            raise RuntimeError('release scope')
    dispatch._run_batch(outer, False)
    assert execution == [nested.parent_agent, outer.parent_agent]


def test_ordinary_background_call_keeps_inline_fallback(execution, monkeypatch):
    work = batch()
    monkeypatch.setattr(dispatch, '_resolve_async_wake_sid', lambda *args, **kwargs: None)
    dispatch._run_batch(work, True)
    assert execution == [work.parent_agent]
