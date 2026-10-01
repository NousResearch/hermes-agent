"""Exact peer receipts never turn an incomplete inventory into nonadmission proof."""
import sqlite3

import pytest

from gateway import hosted_rooms
from tests.tui_gateway.test_hosted_room_peer_http import _dispatch
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError


def admitted(tmp_path):
    db = tmp_path / 'state.db'
    client = PeerRunsHTTPClient(base_url='https://peer.example.test', api_key='', receipt_db_path=db)
    client._request = lambda *args, **kwargs: {'run_id': 'accepted-run', 'status': 'running'}
    client.dispatch(dispatch=_dispatch(), grant='test-grant')
    record, = hosted_rooms.list_remote_run_receipts(db)
    return db, record


def test_exact_receipt_after_more_than_one_inventory_batch_needs_no_reexecution(tmp_path):
    db, record = admitted(tmp_path)
    for index in range(200):
        hosted_rooms.upsert_remote_run_receipt(db, record={**record, 'task_id': f'other-{index}', 'run_id': f'run-{index}'})
    restarted = PeerRunsHTTPClient(base_url='https://peer.example.test', api_key='', receipt_db_path=db)
    restarted._request = lambda *args, **kwargs: pytest.fail('an exact durable receipt requires no admission request')
    assert restarted.recover_dispatch(dispatch=_dispatch(), grant='test-grant')['run_id'] == 'accepted-run'


def test_corrupt_exact_receipt_holds_admission_unknown_without_network_effect(tmp_path):
    db, _ = admitted(tmp_path)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE hosted_room_remote_runs SET run_id=''")
    restarted = PeerRunsHTTPClient(base_url='https://peer.example.test', api_key='', receipt_db_path=db)
    restarted._request = lambda *args, **kwargs: pytest.fail('corrupt custody does not authorize another POST')
    with pytest.raises(PeerRunsHTTPError) as error:
        restarted.recover_dispatch(dispatch=_dispatch(), grant='test-grant')
    assert error.value.ambiguous and not error.value.not_admitted


def test_missing_receipt_preserves_original_identity_during_inconclusive_reconciliation(tmp_path):
    db, _ = admitted(tmp_path)
    with sqlite3.connect(db) as conn:
        conn.execute('DELETE FROM hosted_room_remote_runs')
    restarted = PeerRunsHTTPClient(base_url='https://peer.example.test', api_key='', receipt_db_path=db)
    requests = []
    def unknown(path, **kwargs):
        requests.append(kwargs)
        raise PeerRunsHTTPError('response unknown', ambiguous=True)
    restarted._request = unknown
    with pytest.raises(PeerRunsHTTPError) as error:
        restarted.recover_dispatch(dispatch=_dispatch(), grant='test-grant')
    assert error.value.ambiguous and not error.value.not_admitted
    assert requests and all(r['headers']['Idempotency-Key'] == 'room:task-1:1' for r in requests)
    assert all(r['body']['hosted_room_dispatch'] == _dispatch() for r in requests)
