"""Scoped grant UAT: home service to a real peer API adapter, no Desktop.

Regression for #99960: cancelling an unseen admission must never admit it.
"""

from __future__ import annotations

import asyncio
import errno
import json
import sqlite3
import threading
import urllib.error
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

import hermes_cli.urllib_security as urllib_security
from gateway import hosted_room_driver as driver
from gateway.config import PlatformConfig
from gateway.hosted_room_peer import GatewayRoomCatalog
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
    app.router.add_post("/v1/runs/stop", adapter._handle_stop_run)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    app.router.add_post(
        "/v1/room-members/grants/revoke", adapter._handle_room_member_grant_revoke)
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
    )
    home.register_peer_route(
        room_id="room-1", member_id="member-peer", route=route, client=client,
        target_url=client.base_url, catalog=GatewayRoomCatalog.from_mapping(catalog))
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


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["stop", "disband"])
async def test_cancel_of_absent_deferred_peer_fences_late_admission_after_restart(
    tmp_path: Path, monkeypatch, operation,
):
    """Stop/Disband proves absence without admitting work, even across target restart."""
    import tui_gateway.server as rpc_server
    from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore

    target, server, home = await _linked_home(tmp_path)
    home.runtime.lease_ttl_seconds = 1.0
    home.runtime.poll_interval_seconds = 0.05
    home.runtime.indeterminate_defer_seconds = 0.5
    real_open = urllib_security.open_credentialed_url
    original_requests, delivered_admissions, cancellation_requests = [], [], []
    peer_down, cancellation_down, cancellation_reply_lost = (
        threading.Event(), threading.Event(), threading.Event())
    peer_down.set()
    cancellation_down.set()

    def drop_before_target_then_stay_down(request, timeout):
        if request.get_method() == "POST" and request.full_url.endswith("/v1/runs"):
            original_requests.append(request)
            if len(original_requests) == 1:
                raise urllib.error.URLError(ConnectionResetError(errno.ECONNRESET, "request lost"))
            if not peer_down.is_set():
                delivered_admissions.append(request)
        if peer_down.is_set():
            raise urllib.error.URLError(ConnectionRefusedError(errno.ECONNREFUSED, "refused"))
        if request.get_method() == "POST" and request.full_url.endswith("/v1/runs/stop"):
            cancellation_requests.append(request)
            if len(cancellation_requests) == 1:
                with real_open(request, timeout=timeout) as response:
                    response.read()
                cancellation_reply_lost.set()
                raise urllib.error.URLError(ConnectionResetError(errno.ECONNRESET, "cancellation reply lost"))
        if cancellation_reply_lost.is_set() and cancellation_down.is_set():
            raise urllib.error.URLError(ConnectionRefusedError(errno.ECONNREFUSED, "refused"))
        return real_open(request, timeout=timeout)

    monkeypatch.setattr(urllib_security, "open_credentialed_url", drop_before_target_then_stay_down)
    monkeypatch.setattr(rpc_server, "get_hosted_room_service", lambda: home)
    agent, release = _agent(), threading.Event()

    def run_until_stopped(*args, **kwargs):
        assert release.wait(20)
        return {"final_response": "", "interrupted": True}

    agent.run_conversation.side_effect = run_until_stopped
    agent.interrupt.side_effect = lambda *a, **kw: release.set()
    admissions, factories = [], []
    try:
        with ExitStack() as patches:
            def instrument(adapter):
                admissions.append(patches.enter_context(patch.object(
                    adapter, "_activate_admitted_request", wraps=adapter._activate_admitted_request)))
                factories.append(patches.enter_context(patch.object(
                    adapter, "_create_agent", return_value=agent)))

            instrument(target)
            home.start()
            home.send(
                room_id="room-1", event_id="user-1",
                payload={"text": "@reviewer inspect", "thread_id": "thread-1"},
            )
            deferred = await _peer_task_in(home, ("deferred",))
            assert deferred["execution_generation"] == 1
            assert sum(admission.call_count for admission in admissions) == 0
            assert home.stop(timeout=5.0)
            home = HostedRoomService(_server_module(), db_path=home.db_path)
            home.rpc = home.runtime.rpc = _LocalRPC()
            home.local_profiles = lambda: ("local",)
            home.runtime.poll_interval_seconds = 0.05
            home.start()
            peer_down.clear()
            handler = rpc_server._methods[f"groups.{operation}"]
            params = {"room_id": "room-1", "cancel_id": "cancel-absent"}
            response = await asyncio.to_thread(handler, 1, params)
            if "error" in response:
                assert operation == "disband" and "still stopping" in response["error"]["message"], response
            assert sum(admission.call_count for admission in admissions) == 0
            assert cancellation_reply_lost.is_set()
            pending = driver.get_task(home.db_path, deferred["identity"])
            assert (pending["status"], pending["execution_generation"]) == ("stopping", 1)
            assert not delivered_admissions and agent.run_conversation.call_count == 0
            key = f"room:{deferred['identity'].task_id}:1"
            with sqlite3.connect(tmp_path / "target-runs.db") as conn:
                [stored] = conn.execute(
                    "SELECT status_json FROM run_idempotency WHERE idempotency_key=?", (key,)).fetchall()
            proof = json.loads(stored[0])
            assert proof["status"] == "cancelled" and proof["admission_cancelled"] is True
            assert home.stop(timeout=5.0)
            home = HostedRoomService(_server_module(), db_path=home.db_path)
            home.rpc = home.runtime.rpc = _LocalRPC()
            home.local_profiles = lambda: ("local",)
            home.runtime.poll_interval_seconds = 0.05
            home.start()
            cancellation_down.clear()
            response = await asyncio.to_thread(handler, 2, params)
            if "error" in response:
                assert operation == "disband" and "still stopping" in response["error"]["message"], response
            task = await _settled_peer_turn(home)
            if operation == "disband" and "error" in response:
                response = await asyncio.to_thread(handler, 3, params)
            assert "error" not in response, response
            assert (task["status"], task["execution_generation"]) == ("cancelled", 1)
            assert sum(admission.call_count for admission in admissions) == 0
            assert sum(factory.call_count for factory in factories) == 0
            assert agent.run_conversation.call_count == 0
            assert not delivered_admissions
            assert not target._active_run_tasks and not target._run_streams
            assert {request.get_header("Idempotency-key") for request in original_requests} == {key}
            assert len(cancellation_requests) >= 2
            assert {request.get_header("Idempotency-key") for request in cancellation_requests} == {key}
            assert home.stop(timeout=5.0)

            async def deliver_original_and_check_absence():
                def deliver():
                    try:
                        with real_open(original_requests[0], timeout=5) as reply:
                            return reply.status, json.loads(reply.read())
                    except urllib.error.HTTPError as exc:
                        return exc.code, json.loads(exc.read())

                status, body = await asyncio.to_thread(deliver)
                if operation == "stop":
                    assert status == 202 and body["status"] == "cancelled", (status, body)
                    assert body["replayed"] is True
                else:
                    assert status == 403, (status, body)
                with sqlite3.connect(tmp_path / "target-runs.db") as conn:
                    rows = conn.execute(
                        "SELECT status_json FROM run_idempotency WHERE idempotency_key=?", (key,)).fetchall()
                if operation == "stop":
                    assert len(rows) == 1
                    proof = json.loads(rows[0][0])
                    assert proof["status"] == "cancelled" and proof["admission_cancelled"] is True
                else:
                    assert rows == []
                    with sqlite3.connect(tmp_path / "target-runs.db") as conn:
                        assert conn.execute("SELECT retired_through FROM run_room_authorities").fetchall() == [(1,)]
                assert sum(admission.call_count for admission in admissions) == 0
                assert sum(factory.call_count for factory in factories) == 0
                assert agent.run_conversation.call_count == 0
                assert not target._active_run_tasks and not target._run_streams

            await deliver_original_and_check_absence()
            peer_port = server.port
            await server.close()
            target._run_idempotency_store.close()
            target = APIServerAdapter(
                PlatformConfig(enabled=True, extra={"key": "target-peer-key-1234567890"}))
            target._run_idempotency_store.close()
            target._run_idempotency_store = RunIdempotencyStore(str(tmp_path / "target-runs.db"))
            instrument(target)
            server = TestServer(_target_app(target), port=peer_port)
            await server.start_server()
            await deliver_original_and_check_absence()
    finally:
        release.set()
        home.stop(timeout=5.0)
        await asyncio.gather(*target._active_run_tasks.values(), return_exceptions=True)
        await server.close()
        target._run_idempotency_store.close()


async def _peer_task_in(home, statuses, *, timeout: float = 20.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        for status in statuses:
            for task in driver.list_tasks(home.db_path, room_id="room-1", status=status):
                if task["payload"].get("target_member_id") == "member-peer":
                    return task
        await asyncio.sleep(0.02)
    raise AssertionError(f"peer turn never reached {statuses}: status={home.runtime.status()}")


async def _member_replies(home, *, timeout: float = 20.0):
    """Wait until the settled turn's reply is published to the room (the next room cycle)."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        replies = [event for event in home._events("room-1") if event["kind"] == "message.member"]
        if replies:
            return replies
        await asyncio.sleep(0.02)
    raise AssertionError(f"peer reply was not published: events={home._events('room-1')}")


async def _retry_past_backoff(home, task_id: str):
    """Retry once the client's short recovery backoff (at most poll_max_seconds) has passed."""
    for _ in range(40):
        try:
            return await asyncio.to_thread(home.retry_room_task, "room-1", task_id=task_id)
        except Exception as exc:
            if "backing off" not in str(exc):
                raise
        await asyncio.sleep(0.1)
    raise AssertionError("recovery stayed in backoff")


@pytest.mark.asyncio
@pytest.mark.parametrize("first_attempt_arrived", [False, True])
async def test_retry_of_a_deferred_turn_follows_its_running_attempt_to_the_reply(
    tmp_path: Path, monkeypatch, first_attempt_arrived,
):
    """The peer is unreachable past the deferral window, then comes back while the turn's run
    is going: either the first attempt arrived, or Retry's same-key replay starts it now. Retry
    must not report failure or leave the turn deferred; it follows that one run to its reply."""
    target, server, home = await _linked_home(tmp_path)
    home.runtime.lease_ttl_seconds = 5.0
    home.runtime.poll_interval_seconds = 0.05
    home.runtime.indeterminate_defer_seconds = 0.5
    real_open = urllib_security.open_credentialed_url
    keys, peer_down = [], threading.Event()
    peer_down.set()

    def lose_first_reply_then_stay_down(request, timeout):
        if request.get_method() == "POST" and request.full_url.endswith("/v1/runs"):
            keys.append(request.get_header("Idempotency-key"))
            if len(keys) == 1:
                if first_attempt_arrived:
                    with real_open(request, timeout=timeout) as response:
                        response.read()
                raise urllib.error.URLError(ConnectionResetError(errno.ECONNRESET, "reply lost"))
            if peer_down.is_set():
                raise urllib.error.URLError(ConnectionRefusedError(errno.ECONNREFUSED, "refused"))
        return real_open(request, timeout=timeout)

    monkeypatch.setattr(urllib_security, "open_credentialed_url", lose_first_reply_then_stay_down)
    agent, release = _agent(), threading.Event()
    reply = agent.run_conversation.return_value

    def run_until_released(*args, **kwargs):
        assert release.wait(20)
        return reply

    agent.run_conversation.side_effect = run_until_released
    try:
        with patch.object(target, "_create_agent", return_value=agent):
            home.start()
            home.send(
                room_id="room-1",
                event_id="user-1",
                payload={"text": "@reviewer inspect", "thread_id": "thread-1"},
            )
            deferred = await _peer_task_in(home, ("deferred",))
            peer_down.clear()
            retried = await _retry_past_backoff(home, deferred["identity"].task_id)
            assert (retried["status"], retried["execution_generation"]) == ("indeterminate", 1)
            release.set()
            task = await _settled_peer_turn(home)
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
async def test_stop_of_deferred_peer_waits_for_exact_remote_acknowledgement(
    tmp_path: Path, monkeypatch,
):
    """Losing the admission reply must not let Stop claim the unseen run has stopped."""
    target, server, home = await _linked_home(tmp_path)
    home.runtime.lease_ttl_seconds = 1.0
    home.runtime.poll_interval_seconds = 0.05
    home.runtime.indeterminate_defer_seconds = 0.5
    real_open = urllib_security.open_credentialed_url
    keys, peer_down = [], threading.Event()
    peer_down.set()

    def lose_reply_then_stay_down(request, timeout):
        if request.get_method() == "POST" and request.full_url.endswith("/v1/runs"):
            keys.append(request.get_header("Idempotency-key"))
            if len(keys) == 1:
                with real_open(request, timeout=timeout) as response:
                    response.read()
                raise urllib.error.URLError(ConnectionResetError(errno.ECONNRESET, "reply lost"))
        if peer_down.is_set():
            raise urllib.error.URLError(ConnectionRefusedError(errno.ECONNREFUSED, "refused"))
        return real_open(request, timeout=timeout)

    monkeypatch.setattr(urllib_security, "open_credentialed_url", lose_reply_then_stay_down)
    agent, release = _agent(), threading.Event()

    def run_until_stopped(*args, **kwargs):
        assert release.wait(20)
        return {"final_response": "", "interrupted": True}

    agent.run_conversation.side_effect = run_until_stopped
    agent.interrupt.side_effect = lambda *a, **kw: release.set()
    try:
        with patch.object(target, "_create_agent", return_value=agent):
            home.start()
            home.send(
                room_id="room-1", event_id="user-1",
                payload={"text": "@reviewer inspect", "thread_id": "thread-1"},
            )
            deferred = await _peer_task_in(home, ("deferred",))
            with pytest.raises(RuntimeError, match="still stopping"):
                await asyncio.to_thread(
                    home.stop_room, "room-1", cancel_id="stop-deferred", require_acknowledged=True)
            pending = driver.get_task(home.db_path, deferred["identity"])
            assert (pending["status"], pending["execution_generation"]) == ("stopping", 1)
            assert not release.is_set()
            assert home.stop(timeout=5.0)
            home = HostedRoomService(_server_module(), db_path=home.db_path)
            home.rpc = home.runtime.rpc = _LocalRPC()
            home.local_profiles = lambda: ("local",)
            home.runtime.poll_interval_seconds = 0.05
            peer_down.clear()
            home.start()
            task = await _settled_peer_turn(home)
            assert (task["status"], task["execution_generation"]) == ("cancelled", 1)
            assert release.is_set()
            assert home.stop(timeout=5.0)
    finally:
        release.set()
        home.stop(timeout=5.0)
        await asyncio.gather(*target._active_run_tasks.values(), return_exceptions=True)
        await server.close()
        target._run_idempotency_store.close()

    assert agent.run_conversation.call_count == 1
    assert set(keys) == {f"room:{task['identity'].task_id}:1"}
    assert agent.interrupt.call_count == 1


@pytest.mark.asyncio
async def test_retry_of_deferred_task_does_not_steal_newer_turn_observation(tmp_path, monkeypatch):
    """Retry may resolve the older run while a retained transport still watches the newer one."""
    from tools.bot_relay import TurnBusyError, acquire_turn_lock

    target, server, home = await _linked_home(tmp_path)
    home.runtime.turn_lock = lambda profile: acquire_turn_lock(home.root, profile, timeout_seconds=0)
    home.runtime.lease_ttl_seconds = 1.0
    home.runtime.poll_interval_seconds = 0.05
    home.runtime.active_poll_interval_seconds = 0.05
    home.runtime.indeterminate_defer_seconds = 0.5
    home.runtime.turn_timeout_seconds = 60
    real_open = urllib_security.open_credentialed_url
    keys, peer_down = [], threading.Event()
    peer_down.set()

    def lose_reply_then_stay_down(request, timeout):
        if request.get_method() == "POST" and request.full_url.endswith("/v1/runs"):
            keys.append(request.get_header("Idempotency-key"))
            if len(keys) == 1:
                with real_open(request, timeout=timeout) as response:
                    response.read()
                raise urllib.error.URLError(ConnectionResetError(errno.ECONNRESET, "reply lost"))
            if peer_down.is_set():
                raise urllib.error.URLError(ConnectionRefusedError(errno.ECONNREFUSED, "refused"))
        return real_open(request, timeout=timeout)

    monkeypatch.setattr(urllib_security, "open_credentialed_url", lose_reply_then_stay_down)
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
            a = await _peer_task_in(home, ("deferred",))
            peer_down.clear()
            home.send(room_id="room-1", event_id="user-B",
                      payload={"text": "@reviewer task B", "thread_id": "thread-B"})
            assert await asyncio.to_thread(started_b.wait, 10)
            b = await _peer_task_in(home, ("running",))
            assert b["identity"] != a["identity"]
            try:
                await asyncio.to_thread(home.retry_room_task, "room-1", task_id=a["identity"].task_id)
            except TurnBusyError:
                assert driver.get_task(home.db_path, a["identity"])["status"] == "deferred"
            release_b.set()
            assert await asyncio.to_thread(finished_b.wait, 5)
            deadline = asyncio.get_running_loop().time() + 5
            while asyncio.get_running_loop().time() < deadline:
                current = driver.get_task(home.db_path, b["identity"])
                if current["status"] in driver.TERMINAL_STATUSES:
                    break
                await asyncio.sleep(.05)
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
