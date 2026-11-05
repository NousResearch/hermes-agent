"""Peer admission recovery across a real member API adapter, no Desktop: lost replies, Retry."""

from __future__ import annotations

import asyncio
import errno
import threading
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle
from tests.waiting import async_wait_until, wait_until
from tests.tui_gateway.test_hosted_room_service import _FakeRPC, _server

from hermes_cli import urllib_security
from gateway import hosted_room_driver as driver
from gateway.config import PlatformConfig
from gateway.hosted_rooms import local_authority_gateway_id
from gateway.platforms.api_server import APIServerAdapter
from persistence.gateway.hosted_rooms import TERMINAL_STATUSES
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
from tui_gateway.hosted_room_service import HostedRoomService


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
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    return app


async def _linked_home(tmp_path: Path, **runtime):
    """A real target API adapter on loopback and a home room with one member on it; ``runtime``
    overrides the home driver's timings."""
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
        _server(),
        store=storage_handle(tmp_path / "home"),
        placement="profile",
        peer_routes={("room-1", "member-peer"): route},
        peer_clients={catalog["installation_id"]: client},
    )
    home.rpc = _FakeRPC()
    home.runtime.rpc = home.rpc
    for name, value in runtime.items():
        setattr(home.runtime, name, value)
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


def _lose_first_reply(monkeypatch, *, arrives: bool = True, down: threading.Event | None = None):
    """Lose the reply to the first admission POST (after it reached the target, if ``arrives``),
    then refuse admissions while ``down`` is set (by default only the immediate replay). Returns
    the idempotency keys of every admission sent."""
    real_open, keys = urllib_security.open_credentialed_url, []

    def lossy(request, timeout):
        if request.get_method() == "POST" and request.full_url.endswith("/v1/runs"):
            keys.append(request.get_header("Idempotency-key"))
            if len(keys) == 1:
                if arrives:
                    with real_open(request, timeout=timeout) as response:
                        response.read()
                raise urllib.error.URLError(ConnectionResetError(errno.ECONNRESET, "reply lost"))
            if down.is_set() if down is not None else len(keys) == 2:
                raise urllib.error.URLError(ConnectionRefusedError(errno.ECONNREFUSED, "refused"))
        return real_open(request, timeout=timeout)

    monkeypatch.setattr(urllib_security, "open_credentialed_url", lossy)
    return keys


async def _peer_task(home, statuses=TERMINAL_STATUSES, *, timeout: float = 20.0):
    """Wait for the peer member's room turn to reach one of ``statuses`` (by default, settle)."""
    return await async_wait_until(
        lambda: next((task for status in statuses
                      for task in driver.list_tasks(home.store, placement=home.placement, room_id="room-1",
                                                    status=status)
                      if task["payload"].get("target_member_id") == "member-peer"), None),
        timeout=timeout, what=f"peer turn reaching {statuses}")


@pytest.mark.asyncio
async def test_lost_admission_reply_and_refused_replay_run_the_turn_once(
    tmp_path: Path, monkeypatch,
):
    """The target admits the turn but its reply is lost, and the identical replay cannot
    connect. That refusal says nothing about the first request, so the home must recover the
    same attempt rather than requeue the turn under a new idempotency key."""
    # Keep the lease valid through the refused replay; only then may expiry trigger recovery.
    target, server, home = await _linked_home(tmp_path, lease_ttl_seconds=30.0, poll_interval_seconds=0.05)
    keys, agent = _lose_first_reply(monkeypatch), _agent()
    try:
        with patch.object(target, "_create_agent", return_value=agent):
            home.start()
            home.send(room_id="room-1", event_id="user-1",
                      payload={"text": "@reviewer inspect", "thread_id": "thread-1"})
            task = await _peer_task(home, timeout=60)
            assert home.stop(timeout=5.0)
    finally:
        await server.close()
        target._run_idempotency_store.close()

    assert task["status"] == "settled"
    assert agent.run_conversation.call_count == 1
    assert set(keys) == {f"room:{task['identity'].task_id}:1"}


async def _member_replies(home, *, timeout: float = 20.0):
    """Wait until the settled turn's reply is published to the room (the next room cycle)."""
    return await async_wait_until(
        lambda: [event for event in home._events("room-1") if event["kind"] == "message.member"],
        timeout=timeout, what="peer reply publication")


async def _retry_past_backoff(home, task_id: str):
    """Retry once the client's short recovery backoff has passed."""
    def retry():
        try:
            return home.retry_room_task("room-1", task_id=task_id)
        except Exception as exc:
            if "backing off" not in str(exc):
                raise
        return None
    return await asyncio.to_thread(wait_until, retry, timeout=4, what="peer recovery leaving backoff")


@pytest.mark.asyncio
@pytest.mark.parametrize("first_attempt_arrived", [False, True])
async def test_retry_of_a_deferred_turn_follows_its_running_attempt_to_the_reply(
    tmp_path: Path, monkeypatch, first_attempt_arrived,
):
    """The peer is unreachable past the deferral window, then comes back while the turn's run
    is going: either the first attempt arrived, or Retry's same-key replay starts it now. Retry
    must not report failure or leave the turn deferred; it follows that one run to its reply."""
    target, server, home = await _linked_home(
        tmp_path, lease_ttl_seconds=1.0, poll_interval_seconds=0.05, indeterminate_defer_seconds=0.5)
    peer_down = threading.Event()
    peer_down.set()
    keys = _lose_first_reply(monkeypatch, arrives=first_attempt_arrived, down=peer_down)
    agent, release = _agent(), threading.Event()
    reply = agent.run_conversation.return_value

    def run_until_released(*args, **kwargs):
        assert release.wait(20)
        return reply

    agent.run_conversation.side_effect = run_until_released
    try:
        with patch.object(target, "_create_agent", return_value=agent):
            home.start()
            home.send(room_id="room-1", event_id="user-1",
                      payload={"text": "@reviewer inspect", "thread_id": "thread-1"})
            deferred = await _peer_task(home, ("deferred",))
            home.runtime.lease_ttl_seconds = 30  # Only the initial ambiguity needs a short lease.
            peer_down.clear()
            retried = await _retry_past_backoff(home, deferred["identity"].task_id)
            assert (retried["status"], retried["execution_generation"]) == ("indeterminate", 1)
            release.set()
            task = await _peer_task(home)
            replies = await _member_replies(home)
            assert home.stop(timeout=5.0)
    finally:
        release.set()
        await server.close()
        target._run_idempotency_store.close()

    assert (task["status"], task["execution_generation"]) == ("settled", 1)
    assert agent.run_conversation.call_count == 1
    assert set(keys) == {f"room:{task['identity'].task_id}:1"}
    assert [event["payload"]["text"] for event in replies] == ["Scoped peer response."]


@pytest.mark.asyncio
async def test_retry_of_deferred_task_does_not_steal_newer_turn_observation(tmp_path, monkeypatch):
    """Retry may resolve the older run while a retained transport still watches the newer one."""
    from tools.bot_relay import TurnBusyError, acquire_turn_lock

    target, server, home = await _linked_home(
        tmp_path, lease_ttl_seconds=1.0, poll_interval_seconds=0.05, active_poll_interval_seconds=0.05,
        indeterminate_defer_seconds=0.5, turn_timeout_seconds=60)
    home.runtime.turn_lock = lambda profile: acquire_turn_lock(home.root, profile, timeout_seconds=0)
    peer_down = threading.Event()
    peer_down.set()
    keys = _lose_first_reply(monkeypatch, down=peer_down)
    agent, started_b, release_b, finished_b = _agent(), threading.Event(), threading.Event(), threading.Event()

    def conversations(*args, **kwargs):
        if agent.run_conversation.call_count == 1:
            return {"final_response": "A completed"}
        started_b.set()
        assert release_b.wait(20)
        finished_b.set()
        return {"final_response": "B completed"}

    agent.run_conversation.side_effect = conversations
    try:
        with patch.object(target, "_create_agent", return_value=agent):
            home.start()
            home.send(room_id="room-1", event_id="user-A",
                      payload={"text": "@reviewer task A", "thread_id": "thread-A"})
            a = await _peer_task(home, ("deferred",))
            home.runtime.lease_ttl_seconds = 30  # Keep the healthy attempt leased while Retry probes A.
            peer_down.clear()
            home.send(room_id="room-1", event_id="user-B",
                      payload={"text": "@reviewer task B", "thread_id": "thread-B"})
            assert await asyncio.to_thread(started_b.wait, 10)
            b = await _peer_task(home, ("running",))
            assert b["identity"] != a["identity"]
            try:
                await asyncio.to_thread(home.retry_room_task, "room-1", task_id=a["identity"].task_id)
            except TurnBusyError:
                assert driver.get_task(home.store, a["identity"], placement=home.placement)["status"] == "deferred"
            release_b.set()
            assert await asyncio.to_thread(finished_b.wait, 5)
            def terminal_b():
                current = driver.get_task(home.store, b["identity"], placement=home.placement)
                return current if current["status"] in TERMINAL_STATUSES else None
            current = await async_wait_until(terminal_b, timeout=5, what="newer peer turn settling")
            assert current["status"] == "settled", (current["status"], home.runtime.status(), keys)
            assert current["result"]["text"] == "B completed"
            assert agent.run_conversation.call_count == 2
            assert set(keys) == {f"room:{task['identity'].task_id}:1" for task in (a, b)}
    finally:
        release_b.set()
        home.stop(timeout=5)
        await asyncio.gather(*target._active_run_tasks.values(), return_exceptions=True)
        await server.close()
        target._run_idempotency_store.close()
