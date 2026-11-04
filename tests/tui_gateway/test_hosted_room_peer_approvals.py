"""The older lender still resolves the exact approval through its real HTTP handlers."""
import asyncio
import shlex
import subprocess
from unittest.mock import patch

import pytest

from gateway.session_group_controls import _execution_control
from tests.tui_gateway.test_hosted_room_two_gateway_scoped import _agent, _linked_home, _member_replies
from tests.waiting import async_wait_until
from tools.approval import _gateway_notify_cb, unregister_gateway_notify
from tools.approval_context import get_current_session_key
from tools.approval_gateway_wait import _await_gateway_decision


@pytest.mark.asyncio
@pytest.mark.platforms('posix')
async def test_older_lender_approval_runs_once_and_finishes(tmp_path):
    target, server, home = await _linked_home(tmp_path, active_poll_interval_seconds=0.05)
    effects = tmp_path / 'effects'
    command = f'echo ran >> {shlex.quote(str(effects))}'
    agent = _agent()

    def run(*args, **kwargs):
        route = get_current_session_key()
        decision = _await_gateway_decision(route, _gateway_notify_cb(route), {
            'command': command, 'description': 'Write the test receipt', 'timeout_s': 20})
        if decision['choice'] == 'once':
            subprocess.run(command, shell=True, check=True)
        return {'final_response': 'PEER_REPLY'}

    agent.run_conversation.side_effect = run
    try:
        with patch.object(target, '_create_agent', return_value=agent):
            home.start()
            home.send(room_id='room-1', event_id='ask', payload={'text': '@reviewer run', 'thread_id': 'thread'})
            actions = await async_wait_until(lambda: home.status('room-1')['pending_actions'],
                                            timeout=10, what='older lender approval card')
            action, = actions
            assert action['approval']['command'] == command
            assert action['approval']['choices'] == ['once', 'deny']
            assert not effects.exists()
            params = {key: action[key] for key in ('member_id', 'task_id', 'execution_generation', 'request_id')}
            answered = await asyncio.to_thread(_execution_control, home, 'groups.approve', {
                'room_id': 'room-1', 'choice': 'once', **params})
            assert answered['approved']
            replies = await _member_replies(home)
            assert [r['payload']['text'] for r in replies] == ['PEER_REPLY']
            assert effects.read_text().splitlines() == ['ran']
    finally:
        for route in target._run_approval_sessions.values():
            unregister_gateway_notify(route)
        home.stop(timeout=5)
        await asyncio.gather(*target._active_run_tasks.values(), return_exceptions=True)
        await server.close()
        target._run_idempotency_store.close()
