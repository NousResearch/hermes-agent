"""A slow lender must not hold approvals or Stop in another group chat."""
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer

import pytest

from gateway import hosted_room_driver as driver
from tests.tui_gateway.test_hosted_room_peer_http import FakePeer
from tests.tui_gateway.test_hosted_room_peer_recovery import _deferred_peer_turn
from tests.tui_gateway.test_hosted_room_service import _FakeRPC
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient


def _local_approval(service):
    service.local_profiles = lambda: ('default', 'ops')
    service.create_room(room_id='other', name='Other group', members=[
        {'member_id': 'default', 'profile': 'default', 'handle': 'home'},
        {'member_id': 'ops', 'profile': 'ops', 'handle': 'ops'}])
    service.send(room_id='other', event_id='ask', payload={'text': '@ops run', 'thread_id': 'ask'})
    task, = driver.list_tasks(service.db_path, room_id='other', status='queued')
    binding = next(b for b in service.bindings() if b.room_id == 'other')
    lease = service.runtime._ensure_lease(binding)
    driver.start_task(service.db_path, task['identity'], lease,
                      expected_cancel_generation=0, clock=service.runtime.clock)
    task = driver.get_task(service.db_path, task['identity'])
    service.runtime._report_pending_action(task, session_id='local', info={
        'pending_approval': {'request_id': 'local-prompt', 'command': 'echo local', 'choices': ['once', 'deny']}})
    return task


@pytest.mark.parametrize('slow_action', ['approval', 'stop'])
@pytest.mark.parametrize('other_action', ['approval', 'stop'])
def test_slow_lender_does_not_hold_another_room(tmp_path, slow_action, other_action):
    entered, release = threading.Event(), threading.Event()

    class SlowLender(FakePeer):
        runs = {'run-1': {'run_id': 'run-1', 'status': 'waiting_for_approval',
                'execution_generation': 9, 'approval': {
                    'request_id': 'peer-prompt', 'command': 'echo peer', 'choices': ['once', 'deny']}}}
        approvals = []

        def do_POST(self):
            if self.path == f'/v1/runs/run-1/{slow_action}':
                entered.set()
                if not release.wait(10):
                    return self._json({'error': 'test did not release lender'}, 503)
            return super().do_POST()

    server = ThreadingHTTPServer(('127.0.0.1', 0), SlowLender)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    try:
        client = PeerRunsHTTPClient(base_url=f'http://127.0.0.1:{server.server_port}', api_key='')
        service, db, identity = _deferred_peer_turn(tmp_path, client)
        task = driver.get_task(db, identity)
        route = service.peer_routes[('room-1', 'member-peer')]
        room = service._room('room-1')
        client.bind_room_scope(room_id='room-1', home_install_id=route.home_install_id,
            authority_gateway_id=room['authority_gateway_id'], authority_epoch=room['authority_epoch'],
            member_id=route.member_id, target_install_id=route.target_install_id, target_profile='reviewer')
        client._runs[(identity.task_id, task['execution_generation'])] = {
            'run_id': 'run-1', 'session_id': 'peer-session', 'room_id': 'room-1', 'target_profile': 'reviewer',
            'task_id': identity.task_id, 'execution_generation': task['execution_generation']}
        observed = client.status(room_id='room-1', profile='reviewer', session_id='peer-session',
            task_id=identity.task_id, execution_generation=task['execution_generation'], grant=route.grant)
        service.runtime._report_pending_action(task, session_id='peer-session', info=observed)
        local = _local_approval(service)
        service.rpc = service.runtime.rpc = rpc = _FakeRPC()

        def control(action, room_id, member_id, task, request_id):
            if action == 'stop':
                return service.stop_room(room_id, cancel_id='stop')
            return service.approve_room_task(room_id, member_id=member_id, task_id=task['identity'].task_id,
                execution_generation=task['execution_generation'], request_id=request_id, choice='once')

        with ThreadPoolExecutor(max_workers=2) as pool:
            slow = pool.submit(control, slow_action, 'room-1', 'member-peer', task, 'peer-prompt')
            other = None
            try:
                assert entered.wait(3), 'the lender never received the HTTP request'
                other = pool.submit(control, other_action, 'other', 'ops', local, 'local-prompt')
                result = other.result(timeout=3)
                assert result == ({'resolved': 1} if other_action == 'approval' else 1)
                assert rpc.approvals == ([{'session_id': 'local', 'request_id': 'local-prompt', 'choice': 'once'}]
                                         if other_action == 'approval' else [])
                assert not slow.done(), 'the lender must still be holding its response'
            finally:
                release.set()
                slow.result(timeout=5)
                if other is not None:
                    other.result(timeout=5)
        if slow_action == 'approval':
            assert SlowLender.approvals == [{'request_id': 'peer-prompt', 'execution_generation': 9, 'choice': 'once'}]
    finally:
        release.set()
        server.shutdown()
        serving.join(5)
        server.server_close()
