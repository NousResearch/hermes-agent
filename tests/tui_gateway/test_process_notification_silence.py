"""A process wake may stay silent without changing what a human turn can say (#147996, #160238)."""
import queue
import threading
from types import SimpleNamespace

import pytest

import tui_gateway.server as srv
from hermes_cli.web_routers.sessions import _project_for_display
from tests.fixtures.storage_read import conversation
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle
from tests.gateway.test_authority_complete_persisted_turn import _flushing_agent
from tools.process_registry_notifications import format_process_notification


@pytest.mark.parametrize('kind,failed', [('completion', False), ('completion', True), ('user', False),
                                         ('heartbeat', False), ('watch_match', False), ('async_delegation', False)])
def test_process_notification_delivery(tmp_path, monkeypatch, kind, failed):
    events, prompts = [], []
    store = storage_handle(tmp_path)
    seed_session(store, 'stored', 'desktop')
    agent = _flushing_agent(store, 'stored')
    agent._session_title_hint = 'Scratch'
    monkeypatch.setattr(srv, '_emit', lambda event, _sid, payload=None: events.append((event, payload)))
    monkeypatch.setattr(srv, '_load_interim_assistant_messages', lambda: True)
    monkeypatch.setattr(srv, '_get_usage', lambda _agent: {})
    monkeypatch.setattr(srv, 'render_message', lambda _text, _cols: None)
    monkeypatch.setattr(srv, '_start_usage_ticker', lambda *_: (threading.Event(), SimpleNamespace(join=lambda: None)))

    def run_conversation(prompt, *, stream_callback, persist_user_message, persist_user_display_kind=None, **kwargs):
        prompts.append(prompt)
        for chunk in ('[SIL', 'ENT]'):
            stream_callback(chunk)
        rows = [{'role': 'user', 'content': persist_user_message, 'display_kind': persist_user_display_kind},
                {'role': 'assistant', 'content': '[SILENT]'}]
        agent._persist_user_message_idx = 0
        agent._flush_messages_to_session_db(rows, [])
        return {'final_response': '[SILENT]', 'messages': rows,
                **({'failed': True, 'error': 'Provider failed'} if failed else {})}

    agent.run_conversation = run_conversation
    session = {'agent': agent, 'session_key': 'stored', 'history_lock': threading.RLock(),
               'running': kind in {'heartbeat', 'watch_match', 'async_delegation'}}

    def submit(_rid, sid, session, text, *, display_kind=None, display_metadata=None, **kwargs):
        st = srv._TurnRun(agent, None, None, True)
        srv._invoke_agent(sid, session, st, text, text, None, [], display_kind, display_metadata)
        payload, _, _ = srv._complete_turn_payload(session, st, None, 80)
        events.append(('message.complete', payload))
        return True

    monkeypatch.setattr(srv, '_run_prompt_submit', submit)
    if kind == 'user':
        saved_prompt = 'Quote the literal token.'
        submit('request', 'ui', session, saved_prompt, display_kind='hidden')
    else:
        event = {'type': kind, 'session_id': 'proc_demo', 'exit_code': 0, 'command': 'echo done', 'output': 'done'}
        saved_prompt = format_process_notification(event)
        if kind == 'completion':
            registry = SimpleNamespace(completion_queue=queue.Queue(), is_completion_consumed=lambda _: False)
            srv._notif_dispatch_completions('ui', session, [(event, saved_prompt)], registry, None)
        else:
            srv._notif_dispatch_event('ui', session, event, saved_prompt)

    complete = events[-1][1]
    deltas = [p['text'] for name, p in events if name == 'message.delta']
    silent = kind != 'user' and not failed
    rows = conversation(store, 'stored')
    assert [(r['role'], r['content']) for r in rows] == [('user', saved_prompt), ('assistant', '[SILENT]')]
    if silent:
        assert (deltas, complete['text']) == ([], '')
        assert complete['silent'] is True
    else:
        assert complete['text'] == '[SILENT]'
        assert not complete.get('silent')
    assert (_project_for_display(rows)[-1].get('display_kind') == 'hidden') is silent
    assert any(r['role'] == 'assistant' for r in srv._history_to_messages(rows)) is not silent
    assert ('reply with exactly [SILENT]' in prompts[0]) == (kind in {'completion', 'heartbeat', 'watch_match'})
