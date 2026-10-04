"""Scoped grant UAT: home service to a real peer API adapter, no Desktop."""

from __future__ import annotations

import asyncio
import errno
import threading
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

import hermes_cli.urllib_security as urllib_security
from gateway import hosted_room_driver as driver
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


async def _linked_home(tmp_path: Path):
    """A real target API adapter on loopback and a home room with one member on it."""
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
    return target, server, home


def _agent():
    agent = MagicMock()
    agent.run_conversation.return_value = {
        "final_response": "Scoped peer response."
    }
    agent.session_prompt_tokens = agent.session_completion_tokens = (
        agent.session_total_tokens
    ) = 0
    return agent


@pytest.mark.asyncio
async def test_in_process_scoped_transport_contract_finishes_headlessly(
    tmp_path: Path,
):
    target, server, home = await _linked_home(tmp_path)
    agent = _agent()
    with patch.object(target, "_create_agent", return_value=agent):
        home.start()
        home.send(
            room_id="room-1",
            event_id="user-1",
            payload={"text": "@reviewer inspect", "thread_id": "thread-1"},
        )
        deadline = asyncio.get_running_loop().time() + 20
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
        assert home.stop(timeout=5.0)

    reply = next(
        event
        for event in home._events("room-1")
        if event["kind"] == "message.member"
    )
    assert reply["payload"]["text"] == "Scoped peer response."
    assert reply["actor"]["connection_id"] == "peer-target"
    await server.close()
    target._run_idempotency_store.close()


async def _settled_peer_turn(home, *, timeout: float = 20.0):
    """Wait for the peer member's room turn to reach a terminal state."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        for status in driver.TERMINAL_STATUSES:
            for task in driver.list_tasks(home.db_path, room_id="room-1", status=status):
                if task["payload"].get("target_member_id") == "member-peer":
                    return task
        await asyncio.sleep(0.02)
    raise AssertionError(f"peer turn did not settle: status={home.runtime.status()}")


@pytest.mark.asyncio
async def test_lost_admission_reply_and_refused_replay_run_the_turn_once(
    tmp_path: Path, monkeypatch,
):
    """The target admits the turn but its reply is lost, and the identical replay cannot
    connect. That refusal says nothing about the first request, so the home must recover the
    same attempt rather than requeue the turn under a new idempotency key."""
    target, server, home = await _linked_home(tmp_path)
    home.runtime.lease_ttl_seconds = 1.0  # uncertain work is recovered once the lease expires
    home.runtime.poll_interval_seconds = 0.05
    real_open = urllib_security.open_credentialed_url
    keys = []

    def lose_reply_then_refuse_replay(request, timeout):
        if request.get_method() == "POST" and request.full_url.endswith("/v1/runs"):
            keys.append(request.get_header("Idempotency-key"))
            if len(keys) == 1:
                with real_open(request, timeout=timeout) as response:
                    response.read()
                raise urllib.error.URLError(ConnectionResetError(errno.ECONNRESET, "reply lost"))
            if len(keys) == 2:
                raise urllib.error.URLError(ConnectionRefusedError(errno.ECONNREFUSED, "refused"))
        return real_open(request, timeout=timeout)

    monkeypatch.setattr(urllib_security, "open_credentialed_url", lose_reply_then_refuse_replay)
    agent = _agent()
    try:
        with patch.object(target, "_create_agent", return_value=agent):
            home.start()
            home.send(
                room_id="room-1",
                event_id="user-1",
                payload={"text": "@reviewer inspect", "thread_id": "thread-1"},
            )
            task = await _settled_peer_turn(home)
            assert home.stop(timeout=5.0)
    finally:
        await server.close()
        target._run_idempotency_store.close()

    assert task["status"] == "settled"
    assert agent.run_conversation.call_count == 1
    assert set(keys) == {f"room:{task['identity'].task_id}:1"}
