"""Scoped grant UAT: home service to a real peer API adapter, no Desktop."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from gateway.config import PlatformConfig
from gateway.hosted_rooms import local_authority_gateway_id
from gateway.platforms.api_server import APIServerAdapter
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
from tui_gateway.hosted_room_service import HostedRoomService


class _LocalRPC:
    def resolve_exact(self, **kwargs):
        return None

    def create(self, **kwargs):
        return {"session_id": "local-session"}

    def resume(self, **kwargs):
        return {"session_id": kwargs["session_id"]}

    def submit(self, **kwargs):
        kwargs["on_terminal"]({"status": "settled", "text": "local reply"})
        return {"accepted": True}

    def history(self, **kwargs):
        return []

    def info(self, **kwargs):
        return {"active": False, "task_id": None}

    def interrupt(self, **kwargs):
        return {"interrupted": True}


def _server_module():
    return SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())


def _target_app(adapter):
    app = web.Application()
    app.router.add_post(
        "/v1/room-members/invitations",
        adapter._handle_room_member_invitation,
    )
    app.router.add_get(
        "/v1/room-members/capabilities",
        adapter._handle_room_member_capabilities,
    )
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    return app


@pytest.mark.asyncio
async def test_in_process_scoped_transport_contract_finishes_headlessly(
    tmp_path: Path,
):
    target = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "target-peer-key-1234567890"})
    )
    target._run_idempotency_store.close()
    from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore

    target._run_idempotency_store = RunIdempotencyStore(
        str(tmp_path / "target-runs.db")
    )
    server = TestServer(_target_app(target))
    await server.start_server()
    client = PeerRunsHTTPClient(
        base_url=str(server.make_url("")).rstrip("/"),
        api_key="target-peer-key-1234567890",
    )
    home_install_id = local_authority_gateway_id()
    invitation = await asyncio.to_thread(
        client.issue_invitation,
        room_id="room-1",
        home_install_id=home_install_id,
        authority_gateway_id=home_install_id,
        authority_epoch=1,
        member_id="member-peer",
        grant_id="grant-room-1",
    )
    catalog = invitation["catalog"]
    probe = await asyncio.to_thread(
        client.probe,
        grant=invitation["grant"],
    )
    assert probe["catalog"] == catalog
    route = PeerMemberRoute(
        home_install_id=home_install_id,
        member_id="member-peer",
        target_install_id=catalog["installation_id"],
        target_profile="default",
        capability_digest=catalog["catalog_digest"],
        execution_policy_digest=catalog["execution_policy"]["policy_digest"],
        cancellation_scope_id="cancel-room-1",
        trace_id="trace-room-1",
        grant=invitation["grant"],
    )
    home = HostedRoomService(
        _server_module(),
        db_path=tmp_path / "home-state.db",
        peer_routes={("room-1", "member-peer"): route},
        peer_clients={catalog["installation_id"]: client},
    )
    home.rpc = _LocalRPC()
    home.runtime.rpc = home.rpc
    home.local_profiles = lambda: ("local",)
    home.create_room(
        room_id="room-1",
        name="Scoped room",
        members=[
            {"member_id": "local", "profile": "local", "handle": "local"},
            {
                "member_id": "member-peer",
                "profile": "default",
                "handle": "reviewer",
                "target": {
                    "kind": "peer",
                    "peer_id": "peer-target",
                    "installation_id": catalog["installation_id"],
                    "profile": "default",
                    "capability_digest": catalog["catalog_digest"],
                },
            },
        ],
    )

    agent = MagicMock()
    agent.run_conversation.return_value = {
        "final_response": "Scoped peer response."
    }
    agent.session_prompt_tokens = agent.session_completion_tokens = (
        agent.session_total_tokens
    ) = 0
    with patch.object(target, "_create_agent", return_value=agent):
        home.start()
        home.send(
            room_id="room-1",
            event_id="user-1",
            payload={"text": "@reviewer inspect", "thread_id": "thread-1"},
        )
        deadline = asyncio.get_running_loop().time() + 5
        while asyncio.get_running_loop().time() < deadline:
            if any(
                event["kind"] == "message.member"
                for event in home._events("room-1")
            ):
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError(
                "peer reply was not published: "
                f"status={home.runtime.status()} events={home._events('room-1')}"
            )
        assert home.stop(timeout=1.0)

    reply = next(
        event
        for event in home._events("room-1")
        if event["kind"] == "message.member"
    )
    assert reply["payload"]["text"] == "Scoped peer response."
    assert reply["actor"]["connection_id"] == "peer-target"
    await server.close()
    target._run_idempotency_store.close()
