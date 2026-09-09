"""Binary requests use the same profile owner as JSON, through real HTTP handlers."""

import asyncio
import hashlib
import io
import threading
import time
import urllib.error
import urllib.parse
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from gateway import hosted_room_discussion as discussion
from gateway import hosted_room_driver as driver
from gateway import hosted_rooms
from gateway.config import PlatformConfig
from gateway.hosted_room_artifacts import (
    RoomArtifactOutbox,
    RoomArtifactScope,
    terminal_artifact_manifest,
)
from gateway.hosted_room_peer import (
    GatewayRoomCatalog,
    HostedMemberDispatch,
    attachment_manifest_digest,
    catalog_mapping,
    decode_room_grant,
    issue_room_grant,
)
from gateway.platforms import api_server, api_server_room_attachments as uploads
from hermes_cli import profiles
from tui_gateway import hosted_room_peer_http as http
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
from tui_gateway.hosted_room_service import HostedRoomService


class _Body:
    def __init__(self, data):
        self.data = data

    def set_read_chunk_size(self, size):
        pass

    async def readany(self):
        data, self.data = self.data, b""
        return data

    async def iter_chunked(self, size):
        while self.data:
            chunk, self.data = self.data[:size], self.data[size:]
            yield chunk


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    root = tmp_path / "target"
    root.mkdir()
    # Independent configured homes make selecting the wrong profile observable.
    named = tmp_path / "reviewer-home"
    named.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(
        hosted_rooms, "local_authority_gateway_id", lambda: "install:target"
    )
    monkeypatch.setattr(
        profiles,
        "profiles_to_serve",
        lambda **kwargs: [("default", root), ("reviewer", named)],
    )
    monkeypatch.setattr(
        profiles, "get_profile_dir", lambda name: named if name == "reviewer" else root
    )
    uploads._spool.cache_clear()
    adapter = api_server.APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "synthetic-profile-test-key"})
    )
    adapter.gateway_runner = SimpleNamespace(
        config=SimpleNamespace(
            multiplex_profiles=True,
            multiplex_profile_allowlist=["default", "reviewer"],
        )
    )
    adapter._derive_browser_control_principal = lambda *_args: None
    adapter._browser_control_transport_family = lambda *_args: "http"
    router = web.UrlDispatcher()
    for method, path, handler in adapter._http_route_table():
        if "/attachments" in path or "/artifacts/" in path:
            router.add_route(method, path, handler)
            router.add_route(method, "/p/{profile}" + path, handler)
    middleware = adapter._make_profile_prefix_middleware()
    observed = []

    async def invoke(request, raw):
        headers = dict(request.header_items())
        headers["Host"] = urllib.parse.urlsplit(request.full_url).netloc
        target = make_mocked_request(
            request.get_method(),
            urllib.parse.urlsplit(request.full_url).path,
            headers=headers,
            payload=_Body(raw),
        )
        target._match_info = await router.resolve(target)
        return await middleware(target, target.match_info.handler)

    def opened(request, **kwargs):
        assert kwargs["reject_redirects"] is True
        raw = request.data or b""
        if not isinstance(raw, bytes):
            raw = b"".join(raw)
        response = asyncio.run(invoke(request, raw))
        observed.append((request.get_method(), request.full_url, response.status))
        if response.status >= 400:
            raise urllib.error.HTTPError(
                request.full_url,
                response.status,
                "real handler",
                response.headers,
                io.BytesIO(response.body),
            )
        stream = io.BytesIO(response.body)
        stream.headers = response.headers
        return stream

    monkeypatch.setattr(http, "_open_roomlink_url", opened)
    yield SimpleNamespace(root=root, named=named, adapter=adapter, observed=observed)
    uploads._spool.cache_clear()
    adapter._response_store.close()
    adapter._run_idempotency_store.close()


def _grant(bridge, profile, **overrides):
    scope = dict(
        room_id="binary",
        home_install_id="install:target",
        authority_gateway_id="install:target",
        authority_epoch=1,
        member_id="remote",
        target_install_id="install:target",
        target_profile=profile,
    )
    grant = issue_room_grant(
        bridge.adapter._room_grant_secret(),
        grant_id="binary-test",
        **scope,
        **overrides,
    )
    claims = decode_room_grant(
        bridge.adapter._room_grant_secret(), grant, permission="status"
    )
    home = bridge.named if profile == "reviewer" else bridge.root
    for db in {bridge.root / "state.db", home / "state.db"}:
        hosted_rooms.reserve_peer_room(
            db, claims=claims, expires_at=claims["status_expires_at"]
        )
    return grant, claims, home


def _input(claims, data):
    manifest = [
        {
            "attachment_id": "att_" + "1" * 32,
            "kind": "file",
            "name": "input.bin",
            "mime": "application/octet-stream",
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    ]
    dispatch = HostedMemberDispatch(
        protocol_version=2,
        **{
            key: claims[key]
            for key in (
                "room_id",
                "home_install_id",
                "authority_gateway_id",
                "authority_epoch",
                "member_id",
                "target_install_id",
                "target_profile",
                "execution_policy_digest",
            )
        },
        task_id="input-task",
        execution_generation=1,
        source_event_seq=1,
        cancellation_scope_id="cancel",
        prompt="Read",
        prompt_digest=hashlib.sha256(b"Read").hexdigest(),
        capability_digest="a" * 64,
        trace_id="trace",
        attachment_manifest_digest=attachment_manifest_digest(manifest),
    )
    return dispatch, manifest


@pytest.mark.parametrize(
    "profile,size,scoped",
    [(None, 12, False), ("default", 12, False), ("reviewer", 12, False),
     ("reviewer", 10_000_001, False), ("reviewer", 12, True), ("default", 12, True)],
)
def test_binary_upload_reaches_real_selected_profile_and_retires_exact_batch(
    bridge, profile, size, scoped
):
    grant, claims, home = _grant(bridge, profile or "default")
    data = b"x" * size
    dispatch, manifest = _input(claims, data)
    client = http.PeerRunsHTTPClient(
        base_url="https://files.invalid" + ("/p/" + profile if scoped else ""),
        api_key="", target_profile=profile
    )
    result = client.stage_attachments(
        dispatch=dispatch.as_mapping(),
        attachments=[{**manifest[0], "data": data}],
        grant=grant,
    )
    assert result["complete"]
    spool = uploads.RoomAttachmentSpool(home / "state.db")
    rows = spool.materialize(dispatch)
    assert Path(rows[0]["path"]).read_bytes() == data
    sibling = replace(dispatch, task_id="sibling-input")
    spool.prepare(sibling, manifest)
    spool.put(
        claims=claims,
        task_id=sibling.task_id,
        execution_generation=1,
        attachment_id=manifest[0]["attachment_id"],
        data=data,
    )
    assert (
        client.discard_attachments(
            task_id=dispatch.task_id, execution_generation=1, grant=grant
        )["removed"]
        == 1
    )
    assert Path(spool.materialize(sibling)[0]["path"]).read_bytes() == data
    prefix = "/p/" + profile if profile else ""
    assert all(
        url.startswith("https://files.invalid" + prefix + "/v1/")
        for _, url, _ in bridge.observed
    )
    assert [method for method, _, _ in bridge.observed] == ["POST", "PUT", "DELETE"]


@pytest.mark.parametrize(
    "invalid",
    ["wrong_profile", "signature", "permission", "scope", "generation", "bytes"],
)
def test_named_profile_upload_rejects_bad_authority_or_bytes_without_completing(
    bridge, invalid
):
    grant, claims, home = _grant(bridge, "reviewer")
    data = b"valid input"
    dispatch, manifest = _input(claims, data)
    client = http.PeerRunsHTTPClient(
        base_url="https://files.invalid", api_key="", target_profile="reviewer"
    )
    client._request(
        "/v1/room-members/attachments",
        method="POST",
        room_grant=grant,
        body={"hosted_room_dispatch": dispatch.as_mapping(), "attachments": manifest},
        reject_redirects=True,
    )
    if invalid == "wrong_profile":
        client = http.PeerRunsHTTPClient(
            base_url="https://files.invalid", api_key="", target_profile="default"
        )
    elif invalid == "signature":
        grant = grant.partition(".")[0] + ".invalid"
    elif invalid in {"permission", "scope"}:
        grant = issue_room_grant(
            bridge.adapter._room_grant_secret(),
            grant_id="other-grant",
            **{
                key: claims[key]
                for key in (
                    "room_id",
                    "home_install_id",
                    "authority_gateway_id",
                    "authority_epoch",
                    "target_install_id",
                    "target_profile",
                    "execution_policy_digest",
                )
            },
            member_id="other" if invalid == "scope" else "remote",
            permissions=("status",)
            if invalid == "permission"
            else ("status", "attachment.stage"),
        )
    generation = 2 if invalid == "generation" else 1
    with pytest.raises(http.PeerRunsHTTPError):
        client._put_attachment(
            f"/v1/room-members/attachments/input-task/{generation}/{manifest[0]['attachment_id']}",
            data=b"wrong bytes" if invalid == "bytes" else data,
            grant=grant,
        )
    expected_profile = "default" if invalid == "wrong_profile" else "reviewer"
    assert "/p/" + expected_profile + "/v1/" in bridge.observed[-1][1]
    with pytest.raises(uploads.RoomAttachmentSpoolIncomplete):
        uploads.RoomAttachmentSpool(home / "state.db").materialize(dispatch)


@pytest.mark.parametrize("rehydrate", [False, True])
def test_cold_named_profile_artifact_publication_uses_real_middleware_and_bytes(
    bridge, rehydrate
):
    server = SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())
    db = bridge.root.parent / "authority" / "state.db"
    service = HostedRoomService(server, db_path=db)
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(
            installation_id="install:target",
            persistent_process=True,
            attachments=True,
            target_profile="reviewer",
        )
    )
    service.create_room(
        room_id="binary",
        name="Binary",
        members=[
            {"member_id": "local", "profile": "default", "handle": "local"},
            {
                "member_id": "remote",
                "profile": "reviewer",
                "handle": "remote",
                "target": {
                    "kind": "peer",
                    "peer_id": "install:target",
                    "installation_id": "install:target",
                    "profile": "reviewer",
                    "capability_digest": catalog.catalog_digest,
                },
            },
        ],
    )
    grant, claims, _home = _grant(
        bridge,
        "reviewer",
        execution_policy_digest=catalog.execution_policy.policy_digest,
    )
    service.register_peer_route(
        room_id="binary",
        member_id="remote",
        route=PeerMemberRoute(
            home_install_id=claims["home_install_id"],
            member_id="remote",
            target_install_id="install:target",
            target_profile="reviewer",
            grant=grant,
            capability_digest=catalog.catalog_digest,
            execution_policy_digest=catalog.execution_policy.policy_digest,
            cancellation_scope_id="cancel",
            trace_id="trace",
            attachments=True,
        ),
        client=http.PeerRunsHTTPClient(
            base_url="https://files.invalid",
            api_key="",
            target_profile="reviewer",
            receipt_db_path=db,
        ),
        target_url="https://files.invalid",
        catalog=catalog,
    )
    hosted_rooms.append_event(
        db,
        room_id="binary",
        event_id="source",
        kind="message.user",
        actor={"kind": "user", "id": "local"},
        authority_gateway_id="install:target",
        authority_epoch=1,
        payload={"text": "@remote produce files", "thread_id": "work"},
    )
    plan = discussion.plan_next_task(
        hosted_rooms.room_state(db, room_id="binary"),
        service._events("binary"),
        local_profiles=("default",),
    ).task
    task = driver.admit_task(db, plan.identity, payload=plan.payload, clock=time.time)
    lease = driver.acquire_lease(
        db,
        room_id="binary",
        gateway_id="install:target",
        authority_epoch=1,
        process_generation="binary-test",
        ttl_seconds=300,
        clock=time.time,
    )
    attempt = driver.start_task(
        db, plan.identity, lease, expected_cancel_generation=0, clock=time.time
    )
    scope = RoomArtifactScope.from_mapping({
        **{
            key: claims[key]
            for key in (
                "room_id",
                "home_install_id",
                "authority_gateway_id",
                "authority_epoch",
                "member_id",
                "target_install_id",
                "target_profile",
            )
        },
        "task_id": plan.identity.task_id,
        "execution_generation": attempt.execution_generation,
    })
    box = RoomArtifactOutbox(bridge.named / "state.db")
    for index in range(3):
        box.put_bytes(
            scope=scope,
            source_name=f"output-{index}.txt",
            data=f"output {index}".encode(),
        )
    result = {
        "text": "Files",
        "run_id": "output-run",
        "artifacts": terminal_artifact_manifest(box.db_path, scope),
    }
    driver.settle_task(
        db,
        attempt,
        settlement_id="output-result",
        status="settled",
        result=result,
        clock=time.time,
    )
    bridge.adapter._run_statuses["output-run"] = {
        "run_id": "output-run",
        "status": "completed",
        "room_artifact_scope": scope.as_mapping(),
        "artifacts": result["artifacts"],
    }
    owner_request = make_mocked_request(
        "GET",
        "/p/reviewer/v1/runs/output-run",
        headers={"Authorization": "HermesRoom " + grant},
    )
    bridge.adapter._run_owners["output-run"] = bridge.adapter._run_idempotency_scope(
        owner_request
    )
    cold = HostedRoomService(server, db_path=db)
    if rehydrate:
        cold.peer_routes.clear()
        cold.peer_clients.clear()
        cold._hydrate_persisted_peer_route("binary", "remote")
    cold.prepare_room(cold.bindings()[0])
    messages = [
        event for event in cold._events("binary") if event["kind"] == "message.member"
    ]
    assert len(messages) == 1
    assert len(messages[0]["payload"]["attachments"]) == 3
    for item in messages[0]["payload"]["attachments"]:
        assert (
            cold.read_attachment(
                room_id="binary",
                attachment_id=item["attachment_id"],
                recipient_member_id="remote",
            ).data
            == ("output " + item["name"][7]).encode()
        )
    assert not box.list(scope)
    assert driver.get_task(db, task["identity"])["result"] == result
    assert [method for method, _, _ in bridge.observed] == ["GET", "GET", "GET", "POST"]
    assert all(
        status == 200 and "/p/reviewer/v1/" in url for _, url, status in bridge.observed
    )
    assert bridge.observed[-1][1].endswith("/artifacts/ack")


@pytest.mark.parametrize(
    "profile,prefix,scoped",
    [(None, "", False), ("default", "/p/default", False),
     ("reviewer:west", "/p/reviewer%3Awest", False),
     ("reviewer:west", "/p/reviewer%3Awest", True), ("default", "/p/default", True)],
)
def test_all_request_builders_preserve_base_path_profile_and_escaped_ids(
    monkeypatch, profile, prefix, scoped
):
    requests = []

    def opened(request, **kwargs):
        requests.append(request)
        return io.BytesIO(b'{"complete":true}')

    monkeypatch.setattr(http, "_open_roomlink_url", opened)
    client = http.PeerRunsHTTPClient(
        base_url="https://files.invalid/gateway" + (prefix if scoped else "") + "/",
        api_key="", target_profile=profile
    )
    client.probe(grant="synthetic.grant")
    client._put_attachment("/v1/upload", data=b"bytes", grant="synthetic.grant")
    client.read_artifact(
        run_id="run/one", artifact_id="file one", grant="synthetic.grant"
    )
    assert [request.full_url for request in requests] == [
        "https://files.invalid/gateway" + prefix + path
        for path in (
            "/v1/room-members/capabilities",
            "/v1/upload",
            "/v1/runs/run%2Fone/artifacts/file%20one",
        )
    ]


@pytest.fixture
def binary_server():
    class Peer(BaseHTTPRequestHandler):
        observed = []
        status = 200
        body = b"bytes"
        drip = False

        def handle_binary(self):
            self.observed.append((self.command, self.path))
            if self.command == "PUT":
                self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(self.status)
            if self.status in {301, 302, 303, 307, 308}:
                self.send_header("Location", "/sink")
            self.send_header("Content-Length", str(len(self.body)))
            self.end_headers()
            try:
                if self.drip:
                    for byte in self.body:
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        time.sleep(0.02)
                else:
                    self.wfile.write(self.body)
            except OSError:
                pass

        do_GET = handle_binary
        do_PUT = handle_binary

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Peer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield (
            Peer,
            http.PeerRunsHTTPClient(
                base_url=f"http://127.0.0.1:{server.server_port}",
                api_key="",
                target_profile="reviewer",
                timeout_seconds=0.2,
            ),
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.mark.parametrize("operation", ["GET", "PUT"])
@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_named_binary_requests_reject_redirect_without_forwarding(
    binary_server, operation, status
):
    peer, client = binary_server
    peer.status = status
    with pytest.raises(http.PeerRunsHTTPError, match="refused an HTTP redirect"):
        if operation == "GET":
            client.read_artifact(
                run_id="run", artifact_id="artifact", grant="synthetic.grant"
            )
        else:
            client._put_attachment(
                "/v1/upload", data=b"private bytes", grant="synthetic.grant"
            )
    assert peer.observed == [
        (
            operation,
            "/p/reviewer/v1/runs/run/artifacts/artifact"
            if operation == "GET"
            else "/p/reviewer/v1/upload",
        )
    ]


@pytest.mark.parametrize(
    "failure,error",
    [
        ("empty", "bytes are invalid"),
        ("oversized", "size limit"),
        ("deadline", "time budget"),
    ],
)
def test_named_artifact_read_preserves_byte_and_whole_response_limits(
    binary_server, failure, error
):
    peer, client = binary_server
    peer.body = (
        b""
        if failure == "empty"
        else b"x" * (http.MAX_ATTACHMENT_BYTES + 1)
        if failure == "oversized"
        else b"x" * 200
    )
    peer.drip = failure == "deadline"
    if not peer.drip:
        client.timeout_seconds = 5
    started = time.monotonic()
    with pytest.raises(http.PeerRunsHTTPError, match=error) as caught:
        client.read_artifact(
            run_id="run", artifact_id="artifact", grant="synthetic.grant"
        )
    assert peer.observed == [("GET", "/p/reviewer/v1/runs/run/artifacts/artifact")]
    assert caught.value.retryable is peer.drip
    if peer.drip:
        assert time.monotonic() - started < 3
