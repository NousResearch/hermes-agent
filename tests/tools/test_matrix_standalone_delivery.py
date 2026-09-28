"""HTTP fallback refuses unsafe targets before sending plaintext."""

import asyncio
from types import SimpleNamespace

from aiohttp import web
import pytest

from gateway.config import PlatformConfig
from plugins.platforms.matrix.standalone import standalone_send


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["fake", "http"])
@pytest.mark.parametrize("interruption", ["cancel", "deadline"])
@pytest.mark.parametrize("phase", ["before_put", "after_put"])
async def test_http_delivery_receipt_survives_only_post_acceptance_interruption(
    monkeypatch, transport, interruption, phase
):
    from plugins.platforms.matrix.standalone import _HTTPDelivery, _MatrixAPIError

    room = "!receipt:remote.test"
    blocked = asyncio.Event()
    release = asyncio.Event()
    sent = []
    members_read = 0

    async def exchange(method, path, payload=None):
        nonlocal members_read
        if path.endswith("m.room.encryption"):
            raise _MatrixAPIError(404, {"errcode": "M_NOT_FOUND"})
        if path.endswith("joined_members"):
            members_read += 1
            if members_read == 2:
                blocked.set()
                await release.wait()
            return {
                "joined": {
                    "@bot:remote.test": {},
                    "@alice:remote.test": {},
                    "@bob:remote.test": {},
                }
            }
        assert method == "PUT" and "/send/" in path, (method, path)
        if phase == "before_put":
            blocked.set()
            await release.wait()
            raise _MatrixAPIError(503, {"error": "send unavailable"})
        sent.append(payload)
        return {"event_id": "$accepted"}

    async def request(self, method, path, **kwargs):
        return await exchange(method, path, kwargs.get("json"))

    async def respond(request):
        try:
            data = await exchange(
                request.method,
                request.path,
                await request.json() if request.method == "PUT" else None,
            )
        except _MatrixAPIError as exc:
            return web.json_response(exc.response, status=exc.status)
        return web.json_response(data)

    runner = None
    homeserver = "http://matrix.test"
    if transport == "fake":
        monkeypatch.setattr(_HTTPDelivery, "request", request)
    else:
        app = web.Application()
        app.router.add_route("*", "/{path:.*}", respond)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        server = site._server
        assert isinstance(server, asyncio.Server)
        homeserver = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    original_wait_for = asyncio.wait_for

    async def deadline(awaitable, timeout):
        task = asyncio.ensure_future(awaitable)
        await blocked.wait()
        return await original_wait_for(task, timeout=0)

    if interruption == "deadline":
        monkeypatch.setattr(asyncio, "wait_for", deadline)
    task = asyncio.create_task(
        standalone_send(
            PlatformConfig(
                token="token", extra={"homeserver": homeserver, "e2ee_mode": "off"}
            ),
            room,
            "Accepted brief",
            thread_id="$root",
        )
    )
    try:
        if interruption == "cancel":
            await blocked.wait()
            task.cancel()
        if phase == "before_put" and interruption == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await task
            result = None
        else:
            result = await task
    finally:
        release.set()
        if runner is not None:
            await runner.cleanup()
    if phase == "before_put":
        assert sent == []
        if result is not None:
            assert result == {"error": f"Matrix target '{room}': API timeout (45s)"}
        return
    assert result == {
        "success": True,
        "platform": "matrix",
        "chat_id": room,
        "message_id": "$accepted",
        "thread_id": "$root",
        "chat_type": "unknown",
    }
    assert [payload["m.relates_to"] for payload in sent] == [
        {
            "rel_type": "m.thread",
            "event_id": "$root",
            "is_falling_back": True,
            "m.in_reply_to": {"event_id": "$root"},
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "plain",
        "plain_dm",
        "membership_error",
        "identity_error",
        "encrypted",
        "state_error",
        "alias_error",
        "join_error",
        "wrong_join",
        "mxid",
        "required",
    ],
)
async def test_standalone_checks_destination_and_encryption(case):
    room = "!destination:remote.test"
    calls = []
    sent = []

    async def respond(request):
        calls.append((
            request.method,
            request.path,
            request.query.getall("server_name", []),
        ))
        if "/directory/room/" in request.path:
            if case == "alias_error":
                return web.json_response(
                    {"errcode": "M_NOT_FOUND", "error": "alias absent"}, status=404
                )
            return web.json_response({"room_id": room, "servers": ["route.test"]})
        if "/join/" in request.path:
            if case == "join_error":
                return web.json_response(
                    {"errcode": "M_FORBIDDEN", "error": "join forbidden"}, status=403
                )
            return web.json_response({
                "room_id": "!other:remote.test" if case == "wrong_join" else room
            })
        if "/state/m.room.encryption" in request.path:
            if case == "encrypted":
                return web.json_response({"algorithm": "m.megolm.v1.aes-sha2"})
            if case == "state_error":
                return web.json_response(
                    {"errcode": "M_FORBIDDEN", "error": "state forbidden"}, status=403
                )
            return web.json_response({"errcode": "M_NOT_FOUND"}, status=404)
        if request.path.endswith("joined_members"):
            if case == "membership_error":
                return web.json_response({"errcode": "M_FORBIDDEN"}, status=403)
            members = {"@bot:remote.test": {}, "@alice:remote.test": {}}
            if case not in {"plain_dm", "identity_error"}:
                members["@bob:remote.test"] = {}
            return web.json_response({"joined": members})
        if request.path.endswith("account/whoami"):
            return web.json_response({
                "user_id": "@other:remote.test"
                if case == "identity_error"
                else "@bot:remote.test"
            })
        if "/send/" in request.path:
            sent.append(await request.json())
            return web.json_response({"event_id": "$sent"})
        raise AssertionError(request.path)

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", respond)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    server = site._server
    assert isinstance(server, asyncio.Server)
    port = server.sockets[0].getsockname()[1]
    config = PlatformConfig(
        token="token",
        extra={
            "homeserver": f"http://127.0.0.1:{port}",
            "e2ee_mode": "required" if case == "required" else "off",
        },
    )
    target = "@person:remote.test" if case == "mxid" else "#destination:remote.test"
    try:
        result = await standalone_send(config, target, "report", thread_id="$root")
    finally:
        await runner.cleanup()

    if case in {"plain", "plain_dm", "membership_error", "identity_error"}:
        assert result == {
            "success": True,
            "platform": "matrix",
            "chat_id": room,
            "message_id": "$sent",
            "thread_id": "$root",
            "chat_type": {"plain": "group", "plain_dm": "dm"}.get(case, "unknown"),
        }
        assert [
            (method, routes) for method, path, routes in calls if "/join/" in path
        ] == [("POST", ["route.test"])]
        assert [payload["m.relates_to"] for payload in sent] == [
            {
                "rel_type": "m.thread",
                "event_id": "$root",
                "is_falling_back": True,
                "m.in_reply_to": {"event_id": "$root"},
            }
        ]
        return

    assert sent == []
    expected = {
        "encrypted": "encrypted",
        "state_error": "state forbidden",
        "alias_error": "alias absent",
        "join_error": "join forbidden",
        "wrong_join": "!other:remote.test",
        "mxid": "MXID",
        "required": "required",
    }
    assert expected[case] in result["error"]
