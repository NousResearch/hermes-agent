"""A failed source remains visible; sibling routes do not retry refusals forever."""

import io
import json
import sqlite3
import urllib.error

import pytest

from gateway import hosted_room_peer as peer, hosted_room_work_records as records
from tests.tui_gateway.test_hosted_room_replication import KEY, SECRET, append, pair
from tests.tui_gateway.test_work_record_delivery_fairness import copying, record_profiles
from tui_gateway import hosted_room_replication as publisher


def test_expired_source_anchor_is_visible_after_restart(copying):
    copying.pub._publish_one(KEY)
    copying.pub._publish_one(KEY)
    delivered = list(copying.records)
    # Exercise the retained-history boundary using a real missing source event.
    with sqlite3.connect(copying.source) as conn:
        conn.execute("DELETE FROM hosted_room_events WHERE room_id='room'")
    copying.pub._publish_one(KEY)
    restarted = publisher.HostedRoomReplicationPublisher(copying.source)
    status = restarted.status("room")
    assert status["routes"][0]["work_record_status"] == "source_prefix_expired"
    assert status["work_records_error"] == "source_prefix_expired"
    assert copying.records == delivered
    append(copying.source, "new-retained-anchor")
    restarted._publish_one(KEY)
    restarted._publish_one(KEY)
    assert restarted.status("room")["work_records_error"] is None
    assert restarted.status("room")["routes"][0]["work_record_status"] == "acked"
    assert copying.records[-1]["history"]["seq"] > delivered[-1]["history"]["seq"]


@pytest.mark.parametrize("after_failure", ["unchanged", "history_pending", "refused"])
def test_successful_recapture_clears_error_only_for_acknowledged_work(copying, monkeypatch, after_failure):
    copying.pub._publish_one(KEY)
    copying.pub._publish_one(KEY)
    delivered = list(copying.records)
    assert delivered

    def fail_capture(*args, **kwargs):
        raise sqlite3.OperationalError("temporary source read failure")

    with monkeypatch.context() as fault:
        fault.setattr(records, "capture_locked", fail_capture)
        copying.pub._publish_one(KEY)
    restarted = publisher.HostedRoomReplicationPublisher(copying.source)
    assert restarted.status("room")["work_records_error"] == "work_record_capture_unavailable"
    if after_failure == "history_pending":
        append(copying.source, "new-history-not-yet-acknowledged")
    elif after_failure == "refused":
        with sqlite3.connect(copying.source) as conn:
            conn.execute(f"UPDATE {records.PENDING_TABLE} SET status='rejected' WHERE room_id='room'")
    restarted._publish_one(KEY)
    status = publisher.HostedRoomReplicationPublisher(copying.source).status("room")
    assert copying.records == delivered
    if after_failure == "unchanged":
        assert status["work_records_error"] is None
        assert status["routes"][0]["work_record_status"] == "acked"
    else:
        assert status["work_records_error"] == "work_record_capture_unavailable"
        assert status["routes"][0]["work_record_status"] != "acked"
        assert status["work_records"][0]["status"] != "acked"


@pytest.mark.parametrize("failure", ["unauthorized", "rejected", "invalid_ack"])
def test_sibling_refusals_are_bounded_per_route_even_after_restart(pair, monkeypatch, failure):
    record_profiles(pair)
    attempts = []

    def transport(request, *, timeout):
        if not request.full_url.endswith("/work-records"):
            return pair.http(request, timeout=timeout)
        token = request.get_header("Authorization").removeprefix("HermesRoom ")
        claims = peer.decode_room_grant(SECRET, token, permission=records.PERMISSION)
        attempts.append(claims["member_id"])
        if failure == "invalid_ack":
            return io.BytesIO(b'{}')
        code = 403 if failure == "unauthorized" else 422
        raise urllib.error.HTTPError(request.full_url, code, "refused", {},
            io.BytesIO(json.dumps({"error": {"code": failure}}).encode()))

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", transport)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    for _ in range(8):
        pub._publish_one(KEY)
        pub._publish_one(("room", "z-other"))
    assert sorted(attempts) == ["reviewer", "z-other"]
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    for _ in range(4):
        pub._publish_one(KEY)
        pub._publish_one(("room", "z-other"))
    assert sorted(attempts) == ["reviewer", "z-other"]
