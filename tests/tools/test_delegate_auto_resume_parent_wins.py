"""The parent wins: a stop, /stop or same-repo replacement that lands while the scheduler is starting an automatic
continue (after its verdict, before the child runs) cancels it, and the record stays as the parent left it."""
import json
from unittest.mock import patch
import pytest
from tests.tools.test_delegate_auto_resume import _enable, _repo, _AlwaysLimited, _scheduler
from tests.tools.test_delegate_continue import _parent, _completion, _ScriptedModel
from tests.tools.test_delegate_resumable import _db_rows

@pytest.mark.parametrize('action', ['stop_child', 'stop_session', 'new_spawn'])
def test_parent_action_after_scheduler_verdict(tmp_path, action):
    from tools.delegate_tool import delegate_task
    from tools.delegate_tool_auto_resume import tick, supersede_for_session
    from tools.terminal_tool import record_session_cwd
    _enable()
    parent, dispatched = _parent(), []
    parent._current_task_id = 'parent-task'
    record_session_cwd('parent-task', str(_repo(tmp_path, 'repo')))
    limited = _AlwaysLimited()
    with patch('run_agent.AIAgent', side_effect=limited):
        out = json.loads(delegate_task(tasks=[{'goal': 'original task'}], background=True, parent_agent=parent))
        _completion(out['delegation_id'])
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
                _completion(result['delegation_id'])
        before = _db_rows(parent)[sid]
        print('PARENT_ACTION', action, 'OLD_STATE_BEFORE_AUTO_CLAIM', before[0])
        return start(record, unit, message)
    with patch('run_agent.AIAgent', side_effect=limited):
        resumed = tick(interleaved, capacity=3)
        if dispatched:
            _completion(dispatched[-1])
    print('AUTO_RESUMED', resumed, 'OLD_CHILD_TURNS', len(limited.turns))
    assert resumed == [], 'parent action must cancel automatic resumption'
    state, _pinned, rec = _db_rows(parent)[sid]
    assert state == ('stopped' if action == 'stop_child' else 'superseded'), (state, rec.get('auto_note'))
    assert len(limited.turns) == 1  # the old child never ran again
    assert rec.get('auto_attempts', 0) == 0  # the cancelled attempt is refunded
