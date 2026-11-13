"""A slow admission replay preserves observation and the exact attempt's fences."""

import asyncio
import sqlite3
import time

import pytest

from hermes_agent.gateway import hosted_room_driver as driver, hosted_rooms
from tests.gateway import test_session_group_peers as peers
from hermes_agent.tui_gateway.hosted_room_peer_http import PeerRunsHTTPError

gateway, linked_peer = peers.gateway, peers.linked_peer


@pytest.mark.asyncio
@pytest.mark.parametrize("fence", [None, "lease", "authority", "cancel"])
async def test_slow_retry_rechecks_ownership_and_watches_the_same_run(
        gateway, monkeypatch, linked_peer, tmp_path, fence):
    service, db, identity = await peers._deferred_unreceipted_peer_turn(gateway, monkeypatch, linked_peer)
    runtime, binding = service.runtime, service.bindings()[0]
    now = [time.time()]
    runtime.clock = lambda: now[0]
    client = service.peer_clients[('linked', 'reviewer')]
    client.clock = runtime.clock
    request, keys = client._request, []

    def slow_reply(path, **kwargs):
        result = request(path, **kwargs)
        if path != '/v1/runs' or kwargs.get('method') != 'POST':
            return result
        keys.append(kwargs['headers']['Idempotency-Key'])
        if len(keys) == 1:
            now[0] += runtime.lease_ttl_seconds + 1
            raise PeerRunsHTTPError('admission reply timed out', retryable=True, ambiguous=True)
        if fence == 'lease':
            driver.acquire_lease(
                db, room_id='linked', gateway_id=binding.gateway_id, authority_epoch=binding.authority_epoch,
                process_generation='other-owner', ttl_seconds=30, clock=runtime.clock)
        elif fence == 'authority':
            hosted_rooms.claim_authority(
                db, room_id='linked', expected_gateway_id=binding.gateway_id,
                expected_epoch=binding.authority_epoch, new_gateway_id='new-authority',
                event_id='takeover', now=now[0])
        elif fence == 'cancel':
            driver.cancel_task(db, identity, cancel_id='cancel-during-replay',
                               expected_cancel_generation=0, clock=runtime.clock)
        return result

    monkeypatch.setattr(client, '_request', slow_reply)

    async def retry():
        return await asyncio.to_thread(service.retry_room_task, 'linked', member_id='reviewer',
                                       task_id=identity.task_id, execution_generation=1)

    if fence:
        error = {'lease': driver.LeaseHeldError, 'authority': driver.StaleLeaseError,
                 'cancel': driver.StaleTaskError}[fence]
        with pytest.raises(error):
            await retry()
        assert driver.get_task(db, identity)['status'] == ('cancelled' if fence == 'cancel' else 'deferred')
    else:
        retried = await retry()
        assert (retried['status'], retried['execution_generation']) == ('indeterminate', 1)
        assert retried['started_at'] == now[0]
        receipt = service.peer_admission_receipt('linked', retried)
        gateway.adapter._set_run_status(receipt['run_id'], 'completed', output='Recovered reply', completed=True)
        now[0] += 3  # The next observation is due; no wall-clock sleep is needed.
        await asyncio.to_thread(runtime._run_room_once, binding)
        settled = driver.get_task(db, identity)
        assert (settled['status'], settled['execution_generation']) == ('settled', 1)
        assert settled['result']['text'] == 'Recovered reply'
    assert keys == [f'room:{identity.task_id}:1'] * 2
    assert driver.get_task(db, identity)['execution_generation'] == 1
    with sqlite3.connect(tmp_path / 'runs.db') as conn:
        assert conn.execute('SELECT idempotency_key FROM run_idempotency').fetchall() == [(keys[0],)]
