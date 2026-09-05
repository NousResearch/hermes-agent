"""A fenced old attempt must retain a successor's accepted file input."""

import time
from contextlib import nullcontext
from pathlib import Path

import pytest

from gateway import hosted_room_driver as state, hosted_rooms, hosted_room_peer
from gateway.hosted_room_peer import HostedMemberDispatch, decode_room_grant, issue_room_grant
from gateway.platforms.api_server_room_attachments import RoomAttachmentSpool
from tui_gateway.hosted_room_driver import HostedRoomRuntime
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError
from tui_gateway.hosted_room_peer_status import _RouteStatusPeerClient
from tui_gateway.hosted_room_peer_transport import PeerHostedRoomTransport, PeerMemberRoute
from tests.tui_gateway.test_hosted_room_driver_runtime import BINDING, _identity, db  # noqa: F401


@pytest.mark.parametrize("response_loss", ["staging", "pre_dispatch"])
def test_lease_loss_cannot_delete_already_admitted_batch(
    db, tmp_path, monkeypatch, response_loss
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("socket.socket.connect", lambda *_a, **_k: pytest.fail("network forbidden"))
    initial = time.time()
    instant = [initial]
    clock = lambda: instant[0]
    original_needs_refresh = hosted_room_peer.room_grant_needs_dispatch_refresh
    monkeypatch.setattr(hosted_room_peer, "room_grant_needs_dispatch_refresh", lambda grant, **kw: original_needs_refresh(grant, now=clock(), **kw))
    identity = _identity()
    data = b"accepted input must remain readable"
    attachment = {"attachment_id": "att_11111111111111111111111111111111", "kind": "file", "name": "brief.txt", "size": len(data), "mime": "text/plain"}
    queued = state.admit_task(db, identity, payload={
        "target_profile": "ops", "target_member_id": "ops", "prompt": "Review the file",
        "source_event_seq": 1, "attachments": [attachment],
    }, clock=clock)
    scope = dict(room_id="room-1", home_install_id="install-home", authority_gateway_id=BINDING.gateway_id,
                 authority_epoch=1, member_id="ops", target_install_id="install-target", target_profile="ops",
                 execution_policy_digest="b" * 64)
    secret = b"s" * 32
    issued_at = initial if response_loss == "staging" else initial - 3299
    grant = issue_room_grant(secret, grant_id="old", issued_at=issued_at,
                             ttl_seconds=3600, status_expires_at=initial + 10000, **scope)
    spool = RoomAttachmentSpool(tmp_path / "target.db", root=tmp_path / "spool", clock=clock)
    accepted_paths, deletes = [], []

    def claims(dispatch):
        return {key: getattr(dispatch, key) for key in (
            "room_id", "home_install_id", "authority_gateway_id", "authority_epoch",
            "member_id", "target_install_id", "target_profile",
        )}

    class Peer(PeerRunsHTTPClient):
        def __init__(self):
            super().__init__(base_url="https://peer.invalid", api_key="", target_profile="ops", receipt_db_path=db)
            self.staged_dispatch = None

        def stage_attachments(self, *, dispatch, attachments, grant):
            checked = HostedMemberDispatch.from_mapping(dispatch)
            self.staged_dispatch = checked
            manifest = [{k: v for k, v in item.items() if k != "data"} for item in attachments]
            spool.prepare(checked, manifest)
            for item in attachments:
                spool.put(claims=claims(checked), task_id=checked.task_id,
                          execution_generation=checked.execution_generation, attachment_id=item["attachment_id"], data=item["data"])
            if response_loss == "staging":
                admit_successor(checked)
                raise PeerRunsHTTPError(
                    "the complete upload response was lost",
                    retryable=True,
                    ambiguous=True,
                )
            instant[0] = initial + 2
            return {"complete": True}

        def refresh_grant(self, **_kwargs):
            # The old worker's lease expires while its refresh is outstanding.
            admit_successor(self.staged_dispatch)
            raise PeerRunsHTTPError("refresh refused after target policy changed", status_code=403,
                                    error_code="room_reauthorization_required")

        def _request(self, path, **kwargs):
            if path == "/v1/runs" and kwargs.get("method") == "POST":
                checked = HostedMemberDispatch.from_mapping(kwargs["body"]["hosted_room_dispatch"])
                accepted_paths.extend(Path(item["path"]) for item in spool.materialize(checked))
                return {"run_id": "accepted-run", "status": "running"}
            assert kwargs.get("method") == "DELETE"
            token_claims = decode_room_grant(secret, kwargs["room_grant"], permission="status", now=clock())
            task, generation = path.rsplit("/", 2)[-2:]
            deletes.append((task, int(generation)))
            return {"removed": spool.discard_attempt(claims=token_claims, task_id=task, execution_generation=int(generation))}

    def admit_successor(dispatch):
        instant[0] = initial + 31
        successor = state.acquire_lease(
            db,
            room_id="room-1",
            gateway_id=BINDING.gateway_id,
            authority_epoch=1,
            process_generation="worker-b",
            ttl_seconds=30,
            clock=clock,
        )
        state.recover_room(db, successor, clock=clock)
        assert state.get_task(db, identity)["status"] == "indeterminate"
        result = Peer().recover_dispatch(dispatch=dispatch.as_mapping(), grant=grant)
        assert result["run_id"] == "accepted-run"
        assert accepted_paths and accepted_paths[0].read_bytes() == data

    peer = Peer()
    wrapper = _RouteStatusPeerClient(peer, grant=grant, on_ready=lambda **_: None,
        on_reauthorization=lambda **_: None, on_unavailable=lambda **_: None, on_refreshed=lambda *a, **k: None)
    route = PeerMemberRoute(home_install_id="install-home", member_id="ops", target_install_id="install-target",
        target_profile="ops", capability_digest="a" * 64, execution_policy_digest="b" * 64,
        cancellation_scope_id="cancel-1", trace_id="trace-1", grant=grant, attachments=True)
    transport = PeerHostedRoomTransport(binding=BINDING, route=route, client=wrapper,
        task_id=identity.task_id, execution_generation=1)
    runtime = HostedRoomRuntime(db_path=db, rooms=[BINDING], rpc=object(),
        transport_resolver=lambda *_: transport, turn_lock=lambda _: nullcontext(), clock=clock,
        process_generation="worker-a", attachment_loader=lambda *_: [(attachment, data)])
    lease = runtime._ensure_lease(BINDING)
    attempt = state.start_task(db, identity, lease, expected_cancel_generation=0, clock=clock)
    runtime._execute_attempt(BINDING, queued, attempt)
    current = state.get_task(db, identity)
    receipt = hosted_rooms.remote_run_receipt(db, record={**scope, "task_id": identity.task_id, "execution_generation": 1})
    assert current["status"] == "indeterminate" and receipt["run_id"] == "accepted-run"
    assert accepted_paths
    assert deletes == []
    assert accepted_paths[0].is_file(), "old fenced attempt deleted accepted run input"
