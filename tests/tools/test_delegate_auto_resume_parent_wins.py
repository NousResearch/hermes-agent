"""The parent wins: a stop or /stop that lands while the scheduler is starting an automatic
continue (after its verdict, before the child runs) cancels it, and the record stays as the parent left it."""
import json
from unittest.mock import patch
import pytest
from tests.tools.test_delegate_auto_resume import _settled, _enable, _repo, _AlwaysLimited, _scheduler
from tests.tools.test_delegate_continue import _parent, _ScriptedModel
from tests.tools.test_delegate_resumable import _db_rows

@pytest.mark.parametrize('action', ['stop_child', 'stop_session', 'new_spawn'])
def test_parent_action_after_scheduler_verdict(tmp_path, action):
    from hermes_agent.tools.delegate_tool import delegate_task
    from hermes_agent.tools.delegate_tool_auto_resume import tick, supersede_for_session
    from hermes_agent.tools.terminal_tool import record_session_cwd
    _enable()
    parent, dispatched = _parent(), []
    parent._current_task_id = 'parent-task'
    record_session_cwd('parent-task', str(_repo(tmp_path, 'repo')))
    limited = _AlwaysLimited()
    with patch('run_agent.AIAgent', side_effect=limited):
        out = json.loads(delegate_task(tasks=[{'goal': 'original task'}], background=True, parent_agent=parent))
        _settled(out['delegation_id'])
    sid = out['subagent_ids'][0]
    start = _scheduler(parent, dispatched)
    def interleaved(record, unit, message):
        if action == 'stop_child':
            result = json.loads(delegate_task(action='stop', subagent_id=sid, parent_agent=parent))
            assert result['status'] == 'dismissed', result
        elif action == 'stop_session':
            assert supersede_for_session(parent_session_id=parent.session_id, reason='/stop') == 1
        else:
            with patch('run_agent.AIAgent', side_effect=_ScriptedModel()):
                result = json.loads(delegate_task(tasks=[{'goal': 'replacement task'}], background=True, parent_agent=parent))
                _settled(result['delegation_id'])
        before = _db_rows(parent)[sid]
        print('PARENT_ACTION', action, 'OLD_STATE_BEFORE_AUTO_CLAIM', before[0])
        return start(record, unit, message)
    with patch('run_agent.AIAgent', side_effect=limited):
        resumed = tick(interleaved, capacity=3)
        if dispatched:
            _settled(dispatched[-1])
    print('AUTO_RESUMED', resumed, 'OLD_CHILD_TURNS', len(limited.turns))
    if action == 'new_spawn':  # explicit cancel only: another subagent in the same repo is not a cancel
        assert resumed == [sid] and len(limited.turns) == 2
        return
    assert resumed == [], 'parent action must cancel automatic resumption'
    state, _pinned, rec = _db_rows(parent)[sid]
    assert state == ('stopped' if action == 'stop_child' else 'superseded'), (state, rec.get('auto_note'))
    assert len(limited.turns) == 1  # the old child never ran again
    assert rec.get('auto_attempts', 0) == 0  # the cancelled attempt is refunded


def _limited_child(tmp_path, parent):
    from hermes_agent.tools.delegate_tool import delegate_task
    from hermes_agent.tools.terminal_tool import record_session_cwd
    parent._current_task_id = 'parent-task'
    record_session_cwd('parent-task', str(_repo(tmp_path, 'repo')))
    with patch('run_agent.AIAgent', side_effect=_AlwaysLimited()):
        out = json.loads(delegate_task(tasks=[{'goal': 'write the parser'}], background=True, parent_agent=parent))
        first = _settled(out['delegation_id'])
    return out['subagent_ids'][0], first


def test_a_stop_from_before_opt_in_is_never_resumed_once_auto_resume_is_turned_on(tmp_path):
    """auto_resume off at stop time: the notice told the parent to continue by hand, so no later opt-in resumes it."""
    from hermes_constants import get_hermes_home
    from hermes_agent.tools.delegate_tool_auto_resume import tick
    _enable()
    (get_hermes_home() / 'config.yaml').write_text(json.dumps({'delegation': {'worktree_isolation': True}}))
    parent, dispatched = _parent(), []
    sid, first = _limited_child(tmp_path, parent)
    assert 'auto_resume' not in first['results'][0]  # no promise was made
    _enable()
    with patch('run_agent.AIAgent', side_effect=_ScriptedModel()):
        assert tick(_scheduler(parent, dispatched), capacity=3) == [] and dispatched == []
    state, _pinned, record = _db_rows(parent)[sid]
    assert state == 'resumable' and 'never promised' in record['auto_note']  # still continuable by hand


def test_a_parent_with_no_gateway_scheduler_gets_the_manual_continue_hint(tmp_path):
    """A classic CLI parent (no gateway owns its profile, so no watcher) is never told the child WILL resume."""
    from hermes_agent.tools.delegate_tool_auto_resume import tick
    _enable(gateway=False)
    parent, dispatched = _parent(), []
    sid, first = _limited_child(tmp_path, parent)
    assert 'auto_resume' not in first['results'][0]
    assert 'WILL resume automatically' not in json.dumps(first)
    with patch('run_agent.AIAgent', side_effect=_ScriptedModel()):
        assert tick(_scheduler(parent, dispatched), capacity=3) == []
    assert _db_rows(parent)[sid][0] == 'resumable'


@pytest.mark.parametrize('action', ['slash_stop', 'stop_child'])
def test_a_stop_after_the_last_pre_dispatch_check_still_wins(tmp_path, action):
    """The parent stops it between the scheduler's last check and the child registering (nothing live to interrupt
    yet): the run-start write sees the decision, the child never runs, and the decision stays on the record."""
    import hermes_agent.tools.delegate_tool_auto_resume as ar
    from hermes_agent.tools.async_delegation import interrupt_for_session
    from hermes_agent.tools.delegate_tool import delegate_task
    _enable()
    parent, dispatched = _parent(), []
    sid, _first = _limited_child(tmp_path, parent)
    real = ar.confirm_start

    def confirm_then_parent_acts(path, s):
        out = real(path, s)
        if action == 'slash_stop':
            interrupt_for_session(parent_session_id=parent.session_id, reason='stop_command')
        else:
            assert json.loads(delegate_task(action='stop', subagent_id=s, parent_agent=parent))['status'] == 'dismissed'
        return out

    model = _ScriptedModel()
    with patch.object(ar, 'confirm_start', confirm_then_parent_acts), patch('run_agent.AIAgent', side_effect=model):
        ar.tick(_scheduler(parent, dispatched), capacity=3)
        if dispatched:
            _settled(dispatched[-1])
    assert len(model.turns) == 0, "the parent's stop was lost"
    state, _pinned, rec = _db_rows(parent)[sid]
    assert state == ('stopped' if action == 'stop_child' else 'superseded') and not rec.get('auto_attempts')


def test_a_promise_belongs_to_one_stop(tmp_path):
    """A resumed run's record carries no earlier stop's promise: should it die before its own stop is recorded,
    boot recovery decides afresh whether to promise anything."""
    from hermes_agent.tools.delegate_tool_auto_resume import tick
    _enable()
    parent, dispatched, seen = _parent(), [], []
    sid, _first = _limited_child(tmp_path, parent)
    assert _db_rows(parent)[sid][2].get('auto_promised') is True
    model = _ScriptedModel()

    def build(**kw):
        child = model(**kw)
        ok = child.run_conversation.side_effect
        child.run_conversation.side_effect = lambda *a, **k: (seen.append(_db_rows(parent)[sid]), ok(*a, **k))[1]
        return child

    with patch('run_agent.AIAgent', side_effect=build):
        assert tick(_scheduler(parent, dispatched), capacity=3) == [sid]
        _settled(dispatched[-1])
    state, _pinned, running = seen[0]
    assert state == 'running' and not running.get('auto_promised')


def test_an_automatic_run_whose_record_cannot_be_written_does_not_start(tmp_path):
    """Its run-start write (where the parent's last word is checked) failed: it fails closed instead of running
    with that check deferred into a retry, and the record goes back to waiting."""
    import hermes_agent.tools.delegate_tool_resumable as dr
    from hermes_agent.tools.delegate_tool_auto_resume import tick
    _enable()
    parent, dispatched = _parent(), []
    sid, _first = _limited_child(tmp_path, parent)
    real_write, failed = dr._write, []

    def busy_once(path, fn):
        if fn.__qualname__ == 'record_start.<locals>._do' and not failed:
            failed.append(1)
            return None  # what _write returns when the busy timeout expires
        return real_write(path, fn)

    model = _ScriptedModel()
    with patch.object(dr, '_write', busy_once), patch('run_agent.AIAgent', side_effect=model):
        tick(_scheduler(parent, dispatched), capacity=3)
        if dispatched:
            _settled(dispatched[-1])
    assert failed and len(model.turns) == 0
    assert _db_rows(parent)[sid][0] == 'resumable'


def test_a_cancelled_automatic_run_never_hands_back_the_parents_own_continue(tmp_path):
    """The parent dismissed the claimed child and then continued it by hand before the automatic run recorded
    itself: the automatic run stands down without touching the parent's claim."""
    import hermes_agent.tools.delegate_tool_auto_resume as ar
    from hermes_agent.tools.delegate_tool import delegate_task
    from hermes_agent.tools.delegate_tool_resumable import claim
    _enable()
    parent, dispatched = _parent(), []
    sid, _first = _limited_child(tmp_path, parent)
    real = ar.confirm_start

    def confirm_then_parent_takes_it(path, s):
        out = real(path, s)
        assert json.loads(delegate_task(action='stop', subagent_id=s, parent_agent=parent))['status'] == 'dismissed'
        assert claim(parent, s)[0] is not None  # the parent's own continue holds the row now
        return out

    model = _ScriptedModel()
    with patch.object(ar, 'confirm_start', confirm_then_parent_takes_it), patch('run_agent.AIAgent', side_effect=model):
        ar.tick(_scheduler(parent, dispatched), capacity=3)
        if dispatched:
            _settled(dispatched[-1])
    assert len(model.turns) == 0 and _db_rows(parent)[sid][0] == 'claimed'


def _busy_run_record(dr, on_busy=lambda: None):
    real_write, failed = dr._write, []

    def busy_once(path, fn):
        if fn.__qualname__ == 'record_start.<locals>._do' and not failed:
            failed.append(on_busy())
            return None  # what _write returns when the busy timeout expires
        return real_write(path, fn)
    return busy_once


def test_failing_closed_leaves_no_marker_that_turns_a_manual_continue_into_a_dismiss(tmp_path):
    import hermes_agent.tools.delegate_tool_resumable as dr
    from hermes_agent.tools.delegate_tool import delegate_task
    from hermes_agent.tools.delegate_tool_auto_resume import tick
    _enable()
    parent, dispatched = _parent(), []
    sid, _first = _limited_child(tmp_path, parent)
    with patch.object(dr, '_write', _busy_run_record(dr)), patch('run_agent.AIAgent', side_effect=_ScriptedModel()):
        tick(_scheduler(parent, dispatched), capacity=3)
        if dispatched:
            _settled(dispatched[-1])
    assert dr.claim(parent, sid)[0] is not None  # the parent continues it by hand
    out = json.loads(delegate_task(action='stop', subagent_id=sid, parent_agent=parent))
    assert out.get('status') == 'stop_recorded', out  # recorded for the hand-back, not dismissed as an automatic one
    assert _db_rows(parent)[sid][0] == 'claimed'


def test_a_stop_while_the_automatic_run_fails_closed_is_kept(tmp_path):
    import hermes_agent.tools.delegate_tool_resumable as dr
    from hermes_agent.tools.delegate_tool import delegate_task
    from hermes_agent.tools.delegate_tool_auto_resume import tick
    _enable()
    parent, dispatched = _parent(), []
    sid, _first = _limited_child(tmp_path, parent)
    stop = lambda: json.loads(delegate_task(action='stop', subagent_id=sid, parent_agent=parent))['status']
    model = _ScriptedModel()
    with patch.object(dr, '_write', _busy_run_record(dr, stop)), patch('run_agent.AIAgent', side_effect=model):
        tick(_scheduler(parent, dispatched), capacity=3)
        if dispatched:
            _settled(dispatched[-1])
        assert tick(_scheduler(parent, dispatched), capacity=3) == []
    assert len(model.turns) == 0 and _db_rows(parent)[sid][0] == 'stopped'
