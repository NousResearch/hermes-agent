"""Independent effect/cancellation and shell-semantic probes."""
import json,shlex
import pytest

from tests.tools.target_selection_fixtures import lease_selection
from tests.tools.test_core_routing_review import shell_route,dispatch


def test_cua_cancelled_start_cleans_fence_before_fresh_call(monkeypatch):
    from hermes_cli import session_execution as execution
    from tools.computer_use import tool,targets,session_context
    epoch=[0]; effects=[]; errors=[]; starts=[]
    execution.register_session_execution_context('review-cancel',execution.SessionExecutionContext(
        computer_use=execution.ComputerUseLaunchContext(private_daemon=True,access_epoch=lambda:epoch[0])))
    lease=execution.resolve_session_execution_context(session_id='review-cancel')
    class Backend(tool._NoopBackend):
        def __init__(self,mode,*,execution_context=None):
            super().__init__(); self.execution_context=execution_context
        def start(self):
            starts.append(True)
            if len(starts)==1: raise KeyboardInterrupt('cancel start')
        def click(self,**kw):
            effects.append(True); return super().click(**kw)
    monkeypatch.setattr(tool,'_new_backend',Backend)
    monkeypatch.setattr(tool,'_request_approval',lambda *_:None)
    dispose=targets.register_target_resolver('review-cancel',lambda **kw:lease, selector=lambda **kw: lease_selection(lease))
    saved=session_context._access_fence.get()
    args={'action':'click','coordinate':[1,2],'capture_after':False}
    try:
        try: tool.handle_computer_use(args,session_id='caller')
        except KeyboardInterrupt as exc: errors.append(exc) # Real callers/loggers may retain traceback.
        assert len(errors)==1
        epoch[0]+=2
        result=tool.handle_computer_use(args,session_id='caller')
        assert effects==[True], 'fresh call blocked after retained cancellation: '+str(result)
    finally:
        # Harness cleanup; preserve receipt, not the leaked context in other tests.
        for exc in errors: exc.__traceback__=None
        errors.clear()
        session_context._access_fence.set(saved)
        dispose(); execution.remove_session_execution_context('review-cancel')

@pytest.mark.parametrize('mode',['pipe','pty','promoted'])
@pytest.mark.parametrize('command,expected',[('',0), (' \n# comment only\n',0), ('set -e; false; printf SHOULD_NOT_RUN',1),('exit 7',7),('# comment only',0),('printf ok # trailing comment',0),('printf heredoc; cat <<\'END\'\ncontents\nEND',0)])
def test_routed_background_shell_semantics(shell_route,mode,command,expected):
    tt,home,nested,lease,epoch,paused=shell_route
    from tools.process_registry import process_registry
    args={'background':True,'pty':mode=='pty'} if mode!='promoted' else {'timeout':tt.FOREGROUND_MAX_TIMEOUT+1}
    result=dispatch(command,target='review',workdir=str(nested),**args)
    assert not result.get('error'),result
    completed=process_registry.wait(result['session_id'],timeout=10)
    assert completed['exit_code']==expected, completed
    assert 'SHOULD_NOT_RUN' not in completed['output']

@pytest.mark.parametrize('mode',['pipe','pty','promoted'])
def test_guest_missing_cwd_never_runs_payload(shell_route,mode):
    tt,home,nested,lease,epoch,paused=shell_route
    from tools.process_registry import process_registry
    marker=home/'effect'
    args={'background':True,'pty':mode=='pty'} if mode!='promoted' else {'timeout':tt.FOREGROUND_MAX_TIMEOUT+1}
    result=dispatch('touch '+shlex.quote(str(marker)),target='review',workdir=str(home/'absent'),**args)
    assert not result.get('error'),result
    completed=process_registry.wait(result['session_id'],timeout=10)
    assert completed['exit_code']!=0,completed
    assert not marker.exists()
