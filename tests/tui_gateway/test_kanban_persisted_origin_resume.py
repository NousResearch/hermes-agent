"""Test-only persisted-origin recovery probe; native dispatch, no core mocks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

CHILD = r'''
import json, os, sys, time, threading
from pathlib import Path
root=Path(os.environ['HOME']); home=Path(os.environ['HERMES_HOME'])
# Deny network and child execution before importing any product module.
blocked=[]
def guard(event,args):
    if event in ('socket.connect','socket.getaddrinfo','subprocess.Popen','os.system','os.posix_spawn'):
        blocked.append(event)
        raise RuntimeError('provider-free harness denied '+event)
sys.addaudithook(guard)
from hermes_state import SessionDB
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kbn
from tui_gateway import server
origin='fixture-persisted-origin'
marker='synthetic-terminal-lifecycle-proof'
receipt=root/'claim.json'
assert Path(kb.kanban_db_path()).resolve()==Path(os.environ['HERMES_KANBAN_DB']).resolve()
if sys.argv[1]=='claim':
    db=SessionDB(db_path=home/'state.db')
    db.create_session(origin,source='tui',session_key=origin)
    db.append_message(origin,role='user',content='Synthetic isolated lifecycle test task.')
    conn=kbc.connect()
    tid=kb.create_task(conn,title='synthetic lifecycle only',assignee='fixture_worker')
    kbn.add_notify_sub(conn,task_id=tid,platform='tui',chat_id=origin)
    db.append_message(origin,role='assistant',content='Tracking synthetic task '+tid+' on isolated default board.')
    before=kbn.list_notify_subs(conn,task_id=tid)[0]['last_event_id']
    assert kb.complete_task(conn,tid,summary=marker)
    conn.close()
    session={'session_key':origin,'history_lock':threading.Lock(),'running':True}
    server._notif_poll_kanban('fixture-reader',session)
    assert not session.get('_kanban_pending')
    conn=kbc.connect()
    cursor=kbn.list_notify_subs(conn,task_id=tid)[0]['last_event_id']
    assert cursor==before  # versioned source stays unacked until native admission
    data={'task_id':tid,'origin':origin,'cursor_before':before,'cursor_claimed':cursor,
          'admitted_before_loss':marker in json.dumps(db.get_messages(origin)),
          'blocked':blocked,'db':str(home/'state.db'),'board':os.environ['HERMES_KANBAN_DB']}
    assert not data['admitted_before_loss']
    db.close();conn.close()
    with receipt.open('w') as f:
        json.dump(data,f);f.flush();os.fsync(f.fileno())
    os._exit(0)
else:
    initial=json.loads(receipt.read_text()); tid=initial['task_id']
    if sys.argv[1].startswith('crash-'):
        original = SessionDB.append_notification_once
        def crash_after_commit(self, *args, **kwargs):
            message_id = original(self, *args, **kwargs)
            (root/'crash.json').write_text(json.dumps({'message_id': message_id}))
            os._exit(73)
        if sys.argv[1] == 'crash-before-cache':
            SessionDB.append_notification_once = crash_after_commit
        else:
            from hermes_cli import kanban_db_delivery
            def crash_before_ack(*args, **kwargs):
                (root/'crash.json').write_text(json.dumps({'point': 'before-ack'}))
                os._exit(73)
            kanban_db_delivery.acknowledge = crash_before_ack
    def resume():
        return server.handle_request({'id':'resume-probe','method':'session.resume',
          'params':{'session_id':origin,'eager_build':sys.argv[1]=='eager'}})
    responses=[resume()]
    runtime=(responses[0].get('result') or {}).get('session_id')
    # Cold resumes build asynchronously; wait for actual agent, not just RPC acknowledgement.
    deadline=time.monotonic()+12
    while runtime and time.monotonic()<deadline and not server._sessions.get(runtime,{}).get('agent'):
        time.sleep(.05)
    built=bool(server._sessions.get(runtime,{}).get('agent'))
    if runtime: responses.append(resume())
    db=SessionDB(db_path=home/'state.db')
    messages=db.get_messages(origin)
    conn=kbc.connect();task=kb.get_task(conn,tid)
    cursor=kbn.list_notify_subs(conn,task_id=tid)[0]['last_event_id']
    events=kb.list_events(conn,tid)
    result={'mode':sys.argv[1],'responses':responses,'agent_built':built,
      'same_origin':all((r.get('result') or {}).get('resumed')==origin for r in responses),
      'repeat_same_runtime':len(responses)==2 and responses[1].get('result',{}).get('session_id')==runtime,
      'admission_count':sum(marker in json.dumps(m) for m in messages),
      'task_status':task.status,'assignee':task.assignee,'cursor_after_resume':cursor,
      'cursor_claimed':initial['cursor_claimed'], 'blocked':blocked,
      'terminal_events':sum(e.kind=='completed' and e.payload.get('summary')==marker for e in events),
      'interrupted_marker':db.get_session(origin).get('interrupted_turn')}
    db.close();conn.close()
    (root/'result.json').write_text(json.dumps(result,indent=2,default=str))
    # Exit only the isolated fixture; never allow background work beyond the probe.
    os._exit(0)
'''

@pytest.mark.parametrize('mode', ['cold', 'eager'])
@pytest.mark.parametrize('crash_point', [None, 'crash-before-cache', 'crash-before-ack'])
def test_persisted_origin_admits_unacked_terminal_result(mode, crash_point):
    repo=Path(__file__).resolve().parents[2]
    root=Path(tempfile.mkdtemp(prefix='kanban-native-resume-'+mode+'-'))
    home=root/'isolated-profile';home.mkdir()
    (home/'config.yaml').write_text('model:\n  default: fixture-model\n  provider: custom:fixture\ncustom_providers:\n  - name: fixture\n    base_url: https://fixture.invalid/v1\n    api_key: fixture-not-a-credential\n    api_mode: chat_completions\n')
    script=root/'fixture.py';script.write_text(CHILD)
    env={'HOME':str(root),'HERMES_HOME':str(home),'HERMES_KANBAN_DB':str(root/'board.db'),
         'PATH':os.environ.get('PATH','/usr/bin:/bin'),'PYTHONPATH':str(repo),
         'PYTHONNOUSERSITE':'1','TZ':'UTC'}
    print('PRESERVED_FIXTURE',root,flush=True)
    for phase in (['claim'] + ([crash_point] if crash_point else []) + [mode]):
        run=subprocess.run([sys.executable,str(script),phase],cwd=repo,env=env,
                           capture_output=True,text=True,timeout=35)
        (root/(phase+'.stdout')).write_text(run.stdout)
        (root/(phase+'.stderr')).write_text(run.stderr)
        assert run.returncode==(73 if phase == crash_point else 0), (str(root),phase,run.stderr)
    result=json.loads((root/'result.json').read_text())
    print('NATIVE_RESULT',json.dumps(result),flush=True)
    assert result['agent_built'], ('Native agent build prerequisite failed',str(root),result)
    assert result['same_origin'] and result['repeat_same_runtime'], result
    assert result['assignee']=='fixture_worker' and result['task_status']=='done'
    assert result['terminal_events']==1
    assert result['cursor_after_resume'] > result['cursor_claimed'], result
    assert result['admission_count']==1, ('Native resume failed durable same-origin admission',str(root),result)
