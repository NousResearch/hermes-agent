"""Ephemeral delivery uses native authenticated alias resolution."""

import asyncio
import json
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
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
        alias, "report", metadata={"thread_id": "$root", "_original_target": f"{alias}/$root"}
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


@pytest.mark.parametrize("gateway_loop", [False, True])
@pytest.mark.parametrize("phase", ["before_start", "resolution", "send", "revalidation"])
def test_native_send_keeps_an_accepted_receipt_through_cancellation(monkeypatch, phase, gateway_loop):
    """Caller cancellation stops a Matrix send before it starts. Once the send starts, the
    send and its post-send check finish, so an accepted event is reported as sent. This holds
    when the caller awaits the send directly and when the send runs on the gateway loop."""
    from gateway.session_identity import replace_source

    started = threading.Event()
    release = threading.Event()
    unblock = threading.Event()

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
    if loop is not None:
        threading.Thread(target=loop.run_forever, daemon=True).start()
    runner = SimpleNamespace(_gateway_loop=loop)
    monkeypatch.setattr(senders, "_live_adapter", lambda *a, **kw: (runner, adapter))
    if phase == "before_start" and loop is not None:
        loop.call_soon_threadsafe(unblock.wait)

    async def scenario():
        task = asyncio.ensure_future(
            _send_matrix_via_adapter(SimpleNamespace(), "!room:example.org", "hello")
        )
        if phase != "before_start":
            await wait_for(started)
        elif loop is not None:
            # Let the dispatch queue up behind the blocked gateway loop.
            await asyncio.sleep(0.1)
        task.cancel()
        if phase == "before_start":
            await asyncio.sleep(0.1)
            unblock.set()
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
        unblock.set()
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
    accepted = {
        "success": True, "platform": "matrix", "chat_id": "!room:example.org",
        "message_id": "$accepted", "chat_type": "group",
    }
    expected = {
        "before_start": ("cancelled", 0, []),
        "resolution": ("cancelled", 1, []),
        "send": (accepted, 2, ["!room:example.org"]),
        "revalidation": (accepted, 2, ["!room:example.org"]),
    }
    assert (result, adapter.resolutions, adapter.sent) == expected[phase]


@pytest.mark.parametrize("gateway_state", ["blocked", "stopped", "closed"])
def test_cron_send_bound_releases_a_stuck_gateway_dispatch(monkeypatch, gateway_state):
    """Cron bounds a standalone send with wait_for (#115469). A Matrix send dispatched to a
    gateway loop that cannot finish it must still end with TimeoutError at that bound."""
    from gateway.session_identity import replace_source
    from tools import send_message_tool

    monkeypatch.setattr(send_message_tool, "_CANCELLED_SEND_GRACE_SECONDS", 0.2)
    loop = asyncio.new_event_loop()
    gateway = threading.Thread(target=loop.run_forever, daemon=True)
    gateway.start()
    started, hold, unblock = threading.Event(), threading.Event(), threading.Event()

    class HeldAdapter:
        gateway_runner = None

        async def resolve_delivery_target(self, source, *, refresh=False):
            return replace_source(source, chat_type="group")

        async def send(self, chat_id, content, metadata=None):
            started.set()
            while not hold.is_set():
                await asyncio.sleep(0.01)
            return SendResult(success=True, message_id="$late")

    runner = SimpleNamespace(_gateway_loop=loop)
    monkeypatch.setattr(senders, "_live_adapter", lambda *a, **kw: (runner, HeldAdapter()))
    if gateway_state == "blocked":
        loop.call_soon_threadsafe(unblock.wait)
    outcome = []

    def cron_send():
        async def send():
            return await asyncio.wait_for(
                _send_to_platform(Platform.MATRIX, PlatformConfig(enabled=True), "!room:example.org", "hi"),
                timeout=0.5,
            )

        try:
            outcome.append(asyncio.run(send()))
        except BaseException as exc:  # noqa: BLE001
            outcome.append(type(exc).__name__)

    worker = threading.Thread(target=cron_send, daemon=True)
    worker.start()
    try:
        if gateway_state != "blocked":
            assert started.wait(2)
            loop.call_soon_threadsafe(loop.stop)
            gateway.join(2)
            if gateway_state == "closed":
                loop.close()
        worker.join(3)
    finally:
        unblock.set()
        hold.set()
        if not loop.is_closed():
            loop.call_soon_threadsafe(loop.stop)
    assert (worker.is_alive(), outcome) == (False, ["TimeoutError"])


@pytest.mark.parametrize("gateway_loop", [False, True])
def test_cancelled_caller_sends_no_further_chunks(monkeypatch, gateway_loop):
    """A caller cancelled while the first chunk is being sent gets that chunk's receipt with an
    error, and the chunks that had not started are not sent."""
    from gateway.platforms.base import BasePlatformAdapter
    from gateway.session_identity import replace_source
    from tools import send_message_tool

    started, hold = threading.Event(), threading.Event()
    sent = []

    class ChunkAdapter:
        gateway_runner = None

        def __init__(self, _config=None):
            pass

        async def connect(self):
            return True

        async def disconnect(self):
            return None

        async def resolve_delivery_target(self, source, *, refresh=False):
            return replace_source(source, chat_type="group")

        async def send(self, chat_id, content, metadata=None):
            started.set()
            while not hold.is_set():
                await asyncio.sleep(0.01)
            sent.append(content)
            return SendResult(success=True, message_id=f"$chunk{len(sent)}")

    loop = None
    live = (None, None)
    if gateway_loop:
        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, daemon=True).start()
        live = (SimpleNamespace(_gateway_loop=loop), ChunkAdapter())
    monkeypatch.setattr(matrix, "MatrixAdapter", ChunkAdapter)
    monkeypatch.setattr(senders, "_live_adapter", lambda *a, **kw: live)
    monkeypatch.setattr(send_message_tool, "_platform_max_length", lambda platform: 40)
    message = "word " * 30
    chunks = len(BasePlatformAdapter.truncate_message(message, 40))

    async def scenario():
        task = asyncio.ensure_future(
            _send_to_platform(Platform.MATRIX, PlatformConfig(enabled=True), "!room:example.org", message)
        )
        while not started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        hold.set()
        try:
            return await task
        except asyncio.CancelledError:
            return "cancelled"

    try:
        result = asyncio.run(asyncio.wait_for(scenario(), timeout=5))
    finally:
        hold.set()
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
    assert chunks > 1
    assert (result, len(sent)) == (
        {"error": f"send cancelled after 1 of {chunks} chunks were delivered", "message_id": "$chunk1"},
        1,
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

    with patch.dict(sys.modules, {"plugins.platforms.matrix.adapter": fake_module}), \
         patch("tools.send_message_senders._live_adapter", return_value=(None, None)):
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
        ("send", "#general:example.com", "hello", {"_original_target": "#general:example.com"}),
        ("disconnect",),
    ]


def test_matrix_text_send_forwards_thread_id():
    """Text-only Matrix delivery forwards the target's thread ID."""
    captured = {}

    async def fake_send_matrix(pconfig, chat_id, message, media_files=None, thread_id=None):
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
                SimpleNamespace(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.com"}),
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
