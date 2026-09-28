"""HTTP fallback refuses unsafe targets before sending plaintext."""

from types import SimpleNamespace

from aiohttp import web
import pytest

from plugins.platforms.matrix.standalone import standalone_send


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "plain",
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
    port = site._server.sockets[0].getsockname()[1]
    config = SimpleNamespace(
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

    if case == "plain":
        assert result == {
            "success": True,
            "platform": "matrix",
            "chat_id": room,
            "message_id": "$sent",
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
