"""Ephemeral delivery uses native authenticated alias resolution."""

import asyncio
import json
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from plugins.platforms.matrix import adapter as matrix
from tools import send_message_senders as senders
from tools.send_message_senders import _send_matrix_via_adapter
from tools.send_message_tool import _send_to_platform


@pytest.mark.asyncio
@pytest.mark.parametrize("token", ["access-token", ""])
async def test_alias_reaches_connected_native_adapter(monkeypatch, token):
    alias = "#scheduled:remote.test"
    adapter = SimpleNamespace(
        connect=AsyncMock(return_value=True),
        disconnect=AsyncMock(),
        send=AsyncMock(return_value=SendResult(success=True, message_id="$sent")),
    )
    monkeypatch.setattr(senders, "_live_adapter", lambda *a, **kw: (None, None))
    monkeypatch.setattr(matrix, "MatrixAdapter", lambda config: adapter)
    config = SimpleNamespace(
        token=token, extra={"homeserver": "https://remote.test", "password": "password"}
    )

    result = await senders._send_matrix_via_adapter(
        config, alias, "report", thread_id="$root"
    )

    assert result == {
        "success": True,
        "platform": "matrix",
        "chat_id": alias,
        "message_id": "$sent",
        "thread_id": "$root",
        "chat_type": "unknown",
    }
    adapter.send.assert_awaited_once_with(
        alias,
        "report",
        metadata={"thread_id": "$root", "_original_target": f"{alias}/$root"},
    )
    adapter.disconnect.assert_awaited_once()


@pytest.mark.parametrize("separator", ["/", ":"])
@pytest.mark.parametrize(
    "destination", ["!scheduled:remote.test", "#scheduled:remote.test"]
)
@pytest.mark.parametrize("rejected", [False, True])
def test_bare_platform_send_uses_configured_home_thread(
    tmp_path, monkeypatch, separator, destination, rejected
):
    from gateway.config import PlatformConfig
    from tools.send_message_tool import send_message_tool
    from tools import send_message_tool as messaging
    import model_tools

    target = f"{destination}{separator}$root"
    (tmp_path / "config.yaml").write_text(
        json.dumps({
            "platforms": {
                "matrix": {
                    "enabled": True,
                    "token": "token",
                    "home_channel": {"platform": "matrix", "chat_id": target},
                    "extra": {"homeserver": "https://remote.test", "e2ee_mode": "off"},
                }
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = matrix.MatrixAdapter(
        PlatformConfig(enabled=True, extra={"e2ee_mode": "off"})
    )
    adapter._client = SimpleNamespace(
        resolve_room_alias=AsyncMock(
            return_value=SimpleNamespace(
                room_id="!scheduled:remote.test", servers=["route.test"]
            )
        ),
        join_room=AsyncMock(return_value="!scheduled:remote.test"),
        send_message_event=AsyncMock(return_value="$sent"),
        get_state_event=AsyncMock(side_effect=_MissingEncryption()),
        crypto=None,
    )
    adapter._refresh_dm_cache = AsyncMock()
    if rejected:
        adapter._client.get_state_event.side_effect = ValueError("state forbidden")
    monkeypatch.setattr(senders, "_live_adapter", lambda *a, **kw: (None, adapter))
    monkeypatch.setattr(model_tools, "_run_async", asyncio.run)
    monkeypatch.setattr(messaging, "_mirror_sent_message", lambda *a, **kw: False)

    result = json.loads(send_message_tool({"target": "matrix", "message": "report"}))

    if rejected:
        assert target in result["error"]
        adapter._client.send_message_event.assert_not_awaited()
        return
    assert result.get("success"), result
    args = adapter._client.send_message_event.await_args.args
    assert (str(args[0]), args[2].get("m.relates_to")) == (
        "!scheduled:remote.test",
        {
            "rel_type": "m.thread",
            "event_id": "$root",
            "is_falling_back": True,
            "m.in_reply_to": {"event_id": "$root"},
        },
    )


class _MissingEncryption(Exception):
    errcode = "M_NOT_FOUND"


def _finish_gateway_loop(
    loop: asyncio.AbstractEventLoop, gateway: threading.Thread
) -> None:
    async def drain():
        current = asyncio.current_task()
        tasks = [task for task in asyncio.all_tasks() if task is not current]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await loop.shutdown_asyncgens()

    if gateway.is_alive():
        asyncio.run_coroutine_threadsafe(drain(), loop).result(timeout=5)
        loop.call_soon_threadsafe(loop.stop)
        gateway.join(5)
    else:
        loop.run_until_complete(drain())
    assert not gateway.is_alive()
    loop.close()


@pytest.mark.parametrize("gateway_loop", [False, True])
@pytest.mark.parametrize("phase", ["resolution", "send", "revalidation"])
def test_native_send_keeps_an_accepted_receipt_through_cancellation(
    monkeypatch, phase, gateway_loop
):
    """Caller cancellation stops a Matrix send before it starts. Once the send starts, the
    send and its post-send check finish, so an accepted event is reported as sent. This holds
    when the caller awaits the send directly and when the send runs on the gateway loop."""
    from gateway.session_identity import replace_source

    started = threading.Event()
    release = threading.Event()

    async def wait_for(event):
        while not event.is_set():
            await asyncio.sleep(0.01)

    class FakeAdapter:
        gateway_runner = None

        def __init__(self):
            self.resolutions = 0
            self.sent = []

        async def resolve_delivery_target(self, source, *, refresh=False):
            self.resolutions += 1
            if (phase, self.resolutions) in {("resolution", 1), ("revalidation", 2)}:
                started.set()
                await wait_for(release)
            return replace_source(source, chat_type="group")

        async def send(self, chat_id, content, metadata=None):
            if phase == "send":
                started.set()
                await wait_for(release)
            self.sent.append(chat_id)
            return SendResult(success=True, message_id="$accepted")

    adapter = FakeAdapter()
    loop = asyncio.new_event_loop() if gateway_loop else None
    gateway = None
    if loop is not None:
        gateway = threading.Thread(target=loop.run_forever, daemon=True)
        gateway.start()
    runner = SimpleNamespace(_gateway_loop=loop)
    monkeypatch.setattr(senders, "_live_adapter", lambda *a, **kw: (runner, adapter))

    async def scenario():
        task = asyncio.ensure_future(
            _send_matrix_via_adapter(SimpleNamespace(), "!room:example.org", "hello")
        )
        await wait_for(started)
        task.cancel()
        # A cancelled resolution never needs releasing. Releasing it could let it finish before
        # the cancellation reaches the gateway loop.
        if phase != "resolution":
            release.set()
        try:
            return await task
        except asyncio.CancelledError:
            return "cancelled"

    try:
        result = asyncio.run(asyncio.wait_for(scenario(), timeout=5))
    finally:
        release.set()
        if loop is not None and gateway is not None:
            _finish_gateway_loop(loop, gateway)
    accepted = {
        "success": True,
        "platform": "matrix",
        "chat_id": "!room:example.org",
        "message_id": "$accepted",
        "chat_type": "group",
    }
    assert (result, adapter.sent) == (
        ("cancelled", [])
        if phase == "resolution"
        else (accepted, ["!room:example.org"])
    )


def test_send_matrix_via_adapter_forwards_alias_after_connect():
    calls = []

    class FakeAdapter:
        def __init__(self, _config):
            pass

        async def connect(self):
            calls.append(("connect",))
            return True

        async def send(self, chat_id, message, metadata=None):
            calls.append(("send", chat_id, message, metadata))
            return SimpleNamespace(success=True, message_id="$text")

        async def disconnect(self):
            calls.append(("disconnect",))

    fake_module = SimpleNamespace(MatrixAdapter=FakeAdapter)

    with (
        patch.dict(sys.modules, {"plugins.platforms.matrix.adapter": fake_module}),
        patch("tools.send_message_senders._live_adapter", return_value=(None, None)),
    ):
        result = asyncio.run(
            _send_matrix_via_adapter(
                SimpleNamespace(
                    enabled=True,
                    token="tok",
                    extra={"homeserver": "https://matrix.example.com"},
                ),
                "#general:example.com",
                "hello",
            )
        )

    assert result == {
        "success": True,
        "platform": "matrix",
        "chat_id": "#general:example.com",
        "message_id": "$text",
        "chat_type": "unknown",
    }
    assert calls == [
        ("connect",),
        (
            "send",
            "#general:example.com",
            "hello",
            {"_original_target": "#general:example.com"},
        ),
        ("disconnect",),
    ]


def test_matrix_text_send_forwards_thread_id():
    """Text-only Matrix delivery forwards the target's thread ID."""
    captured = {}

    async def fake_send_matrix(
        pconfig, chat_id, message, media_files=None, thread_id=None
    ):
        captured["chat_id"] = chat_id
        captured["thread_id"] = thread_id
        captured["message"] = message
        return {
            "success": True,
            "platform": "matrix",
            "chat_id": chat_id,
            "message_id": "$evt",
        }

    with patch("tools.send_message_tool._send_matrix_via_adapter", fake_send_matrix):
        result = asyncio.run(
            _send_to_platform(
                Platform.MATRIX,
                SimpleNamespace(
                    enabled=True,
                    token="tok",
                    extra={"homeserver": "https://matrix.example.com"},
                ),
                "!room:example.com",
                "threaded reply",
                thread_id="$thread-root",
            )
        )

    assert result == {
        "success": True,
        "platform": "matrix",
        "chat_id": "!room:example.com",
        "message_id": "$evt",
    }
    assert captured == {
        "chat_id": "!room:example.com",
        "thread_id": "$thread-root",
        "message": "threaded reply",
    }
