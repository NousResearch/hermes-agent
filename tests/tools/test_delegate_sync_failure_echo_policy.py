"""Salt R11-B1: synchronous child failure echo above the CLI spinner is an automatic diagnostic presentation
and honors display.suppress_warning_notifications under the parent's turn snapshot; the child result and the
relayed subagent.complete event are never gated."""
import json
import pytest


def configure(home, setting, monkeypatch):
    home.mkdir(exist_ok=True)
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home/'config.yaml').write_text(json.dumps({'display': {} if setting is None else {'suppress_warning_notifications':setting}}))



def test_sync_failure_real_run_to_spinner_three_modes(tmp_path, monkeypatch):
    from tools.delegate_tool import _run_single_child
    from tools.delegate_tool_progress import _build_child_progress_callback
    from agent.notification_presentation import notification_policy_snapshot
    from tests.tools.test_delegate_output_schema import _StubChild, _StubParent
    rows=[]
    for setting in (None,False,True):
        configure(tmp_path/f'sync-{setting}',setting,monkeypatch)
        import io
        from contextlib import redirect_stdout
        from agent.display import KawaiiSpinner
        from agent.notification_presentation import notification_config_snapshot
        buffer=io.StringIO();events=[]
        parent=_StubParent();parent.session_id='parent'
        with redirect_stdout(buffer):parent._delegate_spinner=KawaiiSpinner('delegating')
        parent.tool_progress_callback=lambda *a,**kw:events.append({'args':a,'kwargs':kw})
        child=_StubChild([])
        def fail(**kwargs):raise RuntimeError('SYNC_REAL_CHILD_CRASH')
        child.run_conversation=fail
        child.tool_progress_callback=_build_child_progress_callback(0,'actual failing child',parent)
        config=notification_config_snapshot()
        assert config['display'].get('suppress_warning_notifications') is setting
        with notification_policy_snapshot(parent,'cli',config):
            result=_run_single_child(0,'actual failing child',child,parent)
        assert result['status'] in ('error','failed') and 'SYNC_REAL_CHILD_CRASH' in str(result)
        rows.append({'setting':setting,'lines':buffer.getvalue().splitlines(),'events':events,'result':result})
    # absent/false: legacy echo present; true: echo suppressed. Result and relayed event untouched in all modes.
    assert any('SYNC_REAL_CHILD_CRASH' in s for s in rows[0]['lines']), rows[0]
    assert any('SYNC_REAL_CHILD_CRASH' in s for s in rows[1]['lines']), rows[1]
    assert not any('SYNC_REAL_CHILD_CRASH' in s for s in rows[2]['lines']), rows[2]
    for row in rows:
        assert any(e['args'][0] == 'subagent.complete' for e in row['events']), row['events']
