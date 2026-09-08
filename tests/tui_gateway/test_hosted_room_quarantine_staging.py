"""Authority loss during file staging must not turn into text-only execution."""

import pytest

from gateway import hosted_room_driver as state
from tests.gateway.test_hosted_room_driver import FakeClock
from tests.gateway.test_hosted_room_driver_quarantine import quarantine
from tests.tui_gateway.test_hosted_room_driver_runtime import BINDING, _identity, _runtime, db
from tests.tui_gateway.test_hosted_room_attachment_runtime import FakeSessionRPC, _admit


@pytest.mark.parametrize("fence", ["quarantine", "expiry"])
def test_authority_lost_during_staging_rolls_back_without_submission(db, fence):
    identity, clock, rpc = _identity(), FakeClock(), FakeSessionRPC()
    manifest = {"attachment_id": "att_11111111111111111111111111111111", "kind": "image",
                "name": "diagram.png", "size": 5, "mime": "image/png"}
    _admit(db, identity, attachments=[manifest])
    sid = rpc.add_session()
    rpc._pending_attachments[sid] = ["previous-file"]

    def load(_binding, _task):
        yield manifest, b"image"
        if fence == "quarantine":
            quarantine(db)
        else:
            clock.advance(31)

    runtime = _runtime(db, rpc, clock=clock, lease_ttl_seconds=30, attachment_loader=load)
    runtime._run_room_once(BINDING)
    assert any(name == "stage_attachment" for name, _ in rpc.calls)
    assert not any(name == "submit" for name, _ in rpc.calls)
    assert rpc._pending_attachments[sid] == ["previous-file"]
    assert not rpc._attachment_snapshots
    assert state.get_task(db, identity)["status"] == "running"
