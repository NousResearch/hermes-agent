"""A→B→A memory discussions keep creator provenance, history and draft budgets scoped."""
import json
import gateway.run  # Bootstrap before the per-test real-home I/O guard.

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
from gateway.session_context import set_session_vars, clear_session_vars
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from hermes_state import SessionDB
from tests.fakes.platforms.telegram_standin import TelegramStandin
from types import SimpleNamespace
from tools import write_approval as wa
from tools.memory_tool import load_on_disk_store
from plugins.platforms.telegram.memory_review_discussion import _claim


def test_discussion_history_and_drafts_route_a_b_a(tmp_path):
    with TelegramStandin() as server:
        homes = [tmp_path / 'alpha', tmp_path / 'beta']
        state = {}
        for index, home in enumerate(homes):
            home.mkdir()
            (home / 'config.yaml').write_text(json.dumps({'memory': {'write_approval': True, 'user_char_limit': 200 + index},
                'platforms': {'telegram': {'enabled': True, 'token': server.token,
                'home_channel': {'platform': 'telegram', 'chat_id': str(111 + index), 'name': home.name},
                'extra': {'base_url': server.api_base, 'base_file_url': server.file_base}}}}))
        for home in (homes[0], homes[1], homes[0]):
            index = homes.index(home)
            token = set_hermes_home_override(home)
            variables = set_session_vars(platform='telegram', user_id='111', chat_id=str(111 + index),
                                         message_id='900', session_id='creator')
            try:
                store = load_on_disk_store()
                if home not in state:
                    assert store.add('memory', f'Only {home.name} has this environment fact.')['success']
                    db = SessionDB(home / 'state.db')
                    db.create_session('creator', 'telegram')
                    history = f'Policy for {home.name}: its own historical instruction.'
                    db.append_message('creator', 'user', history, platform_message_id='900')
                    agent = SimpleNamespace(session_id='creator', platform='telegram', _parent_session_id='',
                        _user_id='111', _chat_id=str(111 + index), _thread_id='', _memory_store=store,
                        _session_db=db, _memory_manager=None)
                    result = json.loads(INLINE_TOOL_EXECUTORS['memory'](agent,
                        {'action': 'add', 'target': 'user', 'content': f'Policy for {home.name}: proposed wording.'},
                        InlineToolContext('test', 'creator-call-'+home.name)))
                    record = wa.get_pending('memory', result['pending_id'])
                    assert record['provenance']['tool_call_id'].endswith(home.name)
                    actor = {'platform': 'telegram', 'user_id': '111', 'chat_id': str(111 + index), 'thread_id': ''}
                    _claim(home, record['id'], 1, actor, 'callback-'+home.name, '900')
                    state[home] = agent, record, db, history
                agent, record, db, history = state[home]
                args = {'action': 'review', 'pending_id': record['id'], 'expected_revision': 1}
                context = json.loads(INLINE_TOOL_EXECUTORS['memory'](agent,args,InlineToolContext('test','review')))
                assert context['success']
                assert context['target_path'] == str(home / 'memories/USER.md')
                assert context['budget']['limit'] == 200 + index
                assert context['memory_files']['memory']['entries'] == [f'Only {home.name} has this environment fact.']
                assert homes[1-index].name not in json.dumps(context['memory_files'])
                assert history in json.dumps(context['history'])
                assert f'Policy for {homes[1-index].name}' not in json.dumps(context['history'])
                if home == homes[1]:
                    wrong = {**args, 'pending_id': state[homes[0]][1]['id']}
                    assert not json.loads(INLINE_TOOL_EXECUTORS['memory'](agent,wrong,InlineToolContext('test','wrong')))['success']
                suggested = json.loads(INLINE_TOOL_EXECUTORS['memory'](agent,
                    {**args, 'action': 'suggest', 'content': f'Only {home.name} owns this draft.'},
                    InlineToolContext('test', 'suggest-'+home.name)))
                assert suggested['success'] and suggested['delivered']
                candidate = wa.get_pending('memory', record['id'])['discussion_draft']
                assert candidate['payload']['content'] == f'Only {home.name} owns this draft.'
                assert candidate['receipt']['chat_id'] == str(111 + index)
                assert candidate['context']['session_id'] == 'creator'
                assert not store.user_entries
            finally:
                clear_session_vars(variables)
                reset_hermes_home_override(token)
        for agent, record, db, history in state.values():
            db.close()
