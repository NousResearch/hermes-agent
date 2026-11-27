"""Independent positive controls at shared source and shell boundaries."""
import asyncio, json, os, shlex, signal, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest
from tests.tools.test_core_routing_review import shell_route, dispatch

@pytest.mark.parametrize('mode',['pipe','pty','promoted'])
def test_guest_wrapper_preserves_tracked_exec_pid_and_signal(shell_route,mode):
    tt,home,nested,lease,epoch,paused=shell_route
    from tools.process_registry import process_registry
    ready=home/'ready.json'
    program="import os,signal,sys,time,json; from pathlib import Path; signal.signal(signal.SIGTERM,lambda *a:sys.exit(23)); Path("+repr(str(ready))+").write_text(json.dumps({'pid':os.getpid(),'cwd':os.getcwd()})); time.sleep(30)"
    args={'background':True,'pty':mode=='pty'} if mode!='promoted' else {'timeout':tt.FOREGROUND_MAX_TIMEOUT+1}
    result=dispatch('exec '+shlex.quote(sys.executable)+' -c '+shlex.quote(program),target='review',workdir=str(nested),**args)
    assert not result.get('error'),result
    proc=process_registry.get(result['session_id'])
    try:
        end=time.monotonic()+10
        while not ready.exists() and time.monotonic()<end: time.sleep(.02)
        assert ready.exists(),process_registry.poll(proc.id)
        observed=json.loads(ready.read_text())
        assert observed=={'pid':proc.pid,'cwd':str(nested)}
        os.kill(proc.pid,signal.SIGTERM)
        completed=process_registry.wait(proc.id,timeout=10)
        assert completed['exit_code']==23,completed
        assert proc.command.startswith('exec ')
    finally:
        if not proc.exited: os.kill(proc.pid,signal.SIGKILL); process_registry.wait(proc.id,timeout=10)


def test_concurrent_delayed_cua_keeps_each_selection_checker(monkeypatch):
    from hermes_cli import session_execution as execution
    from tools.computer_use.cua_backend_session import _AsyncBridge,_CuaDriverSession
    from tools.computer_use.session_context import desktop_access
    execution.register_session_execution_context('async-review',execution.SessionExecutionContext(
        computer_use=execution.ComputerUseLaunchContext(access_epoch=lambda:0)))
    lease=execution.resolve_session_execution_context(session_id='async-review')
    bridge=_AsyncBridge(); bridge.start()
    session=_CuaDriverSession(bridge,execution_context=lease)
    sent=[]; entered=[]; proceed=threading.Event(); ready=threading.Event(); valid=[True,True]
    class Transport:
        async def call_tool(self,name,args):
            sent.append(args['which'])
            return SimpleNamespace(content=[],isError=False,structuredContent={})
    session._session=Transport(); session._started=True
    original=session._call_tool_async
    async def delayed(name,args):
        entered.append(args['which'])
        if len(entered)==2: ready.set()
        end=time.monotonic()+10
        while not proceed.is_set():
            assert time.monotonic()<end
            await asyncio.sleep(.01)
        return await original(name,args)
    session._call_tool_async=delayed
    def invoke(i):
        def check():
            if not valid[i]: raise execution.SessionExecutionError('selected provider changed')
        with desktop_access(lease,check=check):
            return session.call_tool('get_desktop_state',{'which':i},timeout=15)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(invoke,i) for i in range(2)]
            try:
                assert ready.wait(10)
                valid[0]=False
            finally: proceed.set()
            with pytest.raises(execution.SessionExecutionError): futures[0].result(timeout=15)
            futures[1].result(timeout=15)
        assert sent==[1]
    finally:
        proceed.set(); bridge.stop(); execution.remove_session_execution_context('async-review')
