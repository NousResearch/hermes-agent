"""Deterministic loopback regressions adapted from RoomLink Review B (2026-09-04)."""

import hashlib
import http.server
import json
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from gateway import hosted_room_links, hosted_rooms
from gateway.hosted_room_driver import TaskIdentity
from gateway.hosted_room_peer import GatewayRoomCatalog, catalog_mapping, issue_room_grant
from tui_gateway.hosted_room_driver import HostedRoomBinding
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
from tui_gateway.hosted_room_service import HostedRoomService


@contextmanager
def endpoint():
    requests = []
    state = {"status": 200}
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            requests.append((self.path, self.headers.get("Authorization")))
            body = json.dumps({"revoked": True, "run_id": "run-1", "replayed": False}).encode()
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def registration(url, profile, name):
    catalog = GatewayRoomCatalog.from_mapping(catalog_mapping(
        installation_id="install-" + profile, target_profile=profile, persistent_process=True))
    token = issue_room_grant(
        b"review-b-test-secret-material-only", grant_id=name, room_id="room-1", home_install_id="install-home",
        authority_gateway_id="install-home", authority_epoch=1, member_id="member",
        target_install_id=catalog.installation_id, target_profile=profile,
        issued_at=time.time() - 10, ttl_seconds=3600,
    )
    return dict(
        room_id="room-1", member_id="member", target_url=url, catalog=catalog,
        client=PeerRunsHTTPClient(base_url=url, api_key="", target_profile=profile),
        route=PeerMemberRoute(
            home_install_id="install-home", member_id="member", target_install_id=catalog.installation_id,
            target_profile=profile, capability_digest=catalog.catalog_digest,
            execution_policy_digest=catalog.execution_policy.policy_digest,
            cancellation_scope_id="cancel-1", trace_id="trace-1", grant=token,
        ),
    )


@pytest.mark.parametrize("pause_during_resolution", [False, True])
def test_superseded_transport_never_admits_to_old_endpoint(tmp_path, monkeypatch, pause_during_resolution):
    monkeypatch.setattr(hosted_rooms, "local_authority_gateway_id", lambda: "install-home")
    with endpoint() as (old_url, old_requests, _), endpoint() as (new_url, new_requests, _):
        db = tmp_path / "state.db"
        service = HostedRoomService(SimpleNamespace(), db_path=db)
        old = registration(old_url + "/old", "reviewer", "old")
        new = registration(new_url + "/new", "reviewer-new", "new")
        service.register_peer_route(**old)
        worker = HostedRoomService(SimpleNamespace(), db_path=db)
        task = TaskIdentity("room-1", "task-1", "thread-1", "turn-1")
        reached, release = threading.Event(), threading.Event()
        original = service._tracked_peer_client
        def paused(*args, **kwargs):
            reached.set()
            assert release.wait(5)
            return original(*args, **kwargs)
        if pause_during_resolution:
            monkeypatch.setattr(service, "_tracked_peer_client", paused)
        outcome = {}
        def resolve():
            try:
                outcome["transport"] = service._resolve_member_transport(
                    HostedRoomBinding("room-1", "install-home", 1), {
                        "identity": task, "status": "queued", "execution_generation": 0,
                        "payload": {"target_member_id": "member", "source_event_seq": 1},
                    })
            except BaseException as exc:
                outcome["error"] = exc
        thread = threading.Thread(target=resolve)
        thread.start()
        try:
            if pause_during_resolution:
                assert reached.wait(5)
            else:
                thread.join(5)
                assert not thread.is_alive()
            worker.register_peer_route(**new, expected_grant_sha256=hashlib.sha256(old["route"].grant.encode()).hexdigest())
        finally:
            release.set()
            thread.join(5)
        assert "error" not in outcome, outcome.get("error")
        with pytest.raises(RuntimeError, match="route.*changed|route.*current"):
            outcome["transport"].submit(
                profile="reviewer", session_id="session-1", source="bot_room", prompt="Review.",
                task=task, execution_generation=1, on_terminal=lambda result: None,
            )
        assert old_requests == [("/old/p/reviewer/v1/room-members/grants/revoke-exact", f"HermesRoom {old['route'].grant}")]
        assert new_requests == []
        assert worker.revoke_room_routes("room-1") == 1
        assert new_requests[0][0] == "/new/p/reviewer-new/v1/room-members/grants/revoke"


@pytest.mark.parametrize("failure", ["remote", "local"])
def test_failed_superseded_revoke_retains_old_row_until_retry(tmp_path, monkeypatch, failure):
    with endpoint() as (old_url, requests, state), endpoint() as (new_url, _, _):
        service = HostedRoomService(SimpleNamespace(), db_path=tmp_path / "state.db")
        old = registration(old_url, "reviewer", "old")
        new = registration(new_url, "reviewer-new", "new")
        service.register_peer_route(**old)
        state["status"] = 503 if failure == "remote" else 200
        save = hosted_room_links.save_room_link
        def fail_save(*args, **kwargs):
            raise OSError("injected local route-store failure")
        if failure == "local":
            monkeypatch.setattr(hosted_room_links, "save_room_link", fail_save)
        expected = hashlib.sha256(old["route"].grant.encode()).hexdigest()
        with pytest.raises(PeerRunsHTTPError if failure == "remote" else OSError):
            service.register_peer_route(**new, expected_grant_sha256=expected)
        assert hosted_room_links.load_room_link(service.db_path, room_id="room-1", member_id="member").grant == old["route"].grant
        state["status"] = 200
        monkeypatch.setattr(hosted_room_links, "save_room_link", save)
        service.register_peer_route(**new, expected_grant_sha256=expected)
        assert hosted_room_links.load_room_link(service.db_path, room_id="room-1", member_id="member").grant == new["route"].grant
        assert len(requests) == 2
