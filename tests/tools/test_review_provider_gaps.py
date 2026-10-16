"""Compatibility and nested operation authority at real dispatch seams."""
import json
import pytest
from tests.tools.target_selection_fixtures import lease_selection

@pytest.mark.parametrize('tool_name',['computer_use','terminal'])
def test_legacy_allocating_provider_never_runs_before_denial(tmp_path,monkeypatch,tool_name):
    from hermes_cli import session_execution as execution
    from tools.computer_use import tool,targets
    from tools import terminal_tool as tt,terminal_targets
    import tools.computer_use_tool
    from tools.registry import registry
    execution.register_session_execution_context('legacy-review',execution.SessionExecutionContext())
    lease=execution.resolve_session_execution_context(session_id='legacy-review')
    allocated=[]
    def legacy_resolver(**kw):
        allocated.append('allocation') # inert stand-in; no actual native compute
        return lease
    monkeypatch.setattr(tool,'_request_approval',lambda *_:json.dumps({'error':'denied'}))
    monkeypatch.setattr(tt,'_check_all_guards',lambda *a,**kw:{'approved':False,'description':'denied'})
    if tool_name=='computer_use':
        dispose=targets.register_target_resolver('review-legacy',legacy_resolver)
        args={'action':'click','coordinate':[1,2],'capture_after':False}
    else:
        dispose=terminal_targets.register_terminal_target_resolver('review-legacy',legacy_resolver)
        args={'command':'true','target':'review-legacy'}
    try:
        result=registry.dispatch(tool_name,args,session_id='legacy-caller',task_id='legacy-task')
        assert not allocated, 'denied call invoked old-style allocating resolver: '+str(result)
    finally:
        dispose(); execution.remove_session_execution_context('legacy-review')

@pytest.mark.parametrize('change',[False,True])
def test_explicit_mutable_cua_selection_rejects_changed_mapping(monkeypatch,change):
    from hermes_cli import session_execution as execution
    from tools.computer_use import tool,targets
    leases=[]
    for name in ('old','new'):
        execution.register_session_execution_context('review-'+name,execution.SessionExecutionContext())
        leases.append(execution.resolve_session_execution_context(session_id='review-'+name))
    selected=[leases[0]]; effects=[]
    class Backend(tool._NoopBackend):
        def __init__(self,mode,*,execution_context=None):
            super().__init__(); self.execution_context=execution_context
        def click(self,**kw):
            effects.append(self.execution_context.session_id)
            return super().click(**kw)
    monkeypatch.setattr(tool,'_new_backend',Backend)
    def approval(*a):
        if change:selected[0]=leases[1]
        return None
    monkeypatch.setattr(tool,'_request_approval',approval)
    dispose=targets.register_target_resolver('review-mutable',lambda **kw:selected[0],
        selector=lambda **kw:lease_selection(selected[0], current=lambda:selected[0]))
    try:
        result=tool.handle_computer_use({'action':'click','coordinate':[1,2],'capture_after':False},session_id='caller')
        assert effects==([] if change else ['review-old']),str(result)+' effects='+str(effects)
    finally:
        dispose()
        for name in ('old','new'):execution.remove_session_execution_context('review-'+name)

@pytest.mark.parametrize('kind',['terminal','cua'])
def test_nested_scope_cannot_refresh_stale_authority(kind):
    from hermes_cli import session_execution as execution
    from tools.terminal_targets import terminal_operation,access_epoch,check_terminal_operation
    from tools.computer_use.session_context import desktop_access,check_access_epoch
    epoch=[0]
    execution.register_session_execution_context('nested-review',execution.SessionExecutionContext(
        terminal_access_epoch=lambda:epoch[0], computer_use=execution.ComputerUseLaunchContext(access_epoch=lambda:epoch[0])))
    lease=execution.resolve_session_execution_context(session_id='nested-review')
    scope=(lambda:terminal_operation(lease,access_epoch(lease))) if kind=='terminal' else (lambda:desktop_access(lease))
    check=check_terminal_operation if kind=='terminal' else check_access_epoch
    try:
        with scope():
            epoch[0]+=2
            with pytest.raises(execution.SessionExecutionError):
                with scope(): check(lease)
    finally: execution.remove_session_execution_context('nested-review')
