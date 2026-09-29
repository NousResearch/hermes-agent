"""Explicit Matrix threads use the requesting session's live delivery owner."""

import asyncio
import importlib
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agent.secret_scope import (
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from gateway.config import PlatformConfig
from gateway.session_context import (
    clear_session_vars,
    get_session_env,
    set_session_vars,
)
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from hermes_cli.tools_config import _get_platform_tools
from plugins.platforms.matrix.adapter import MatrixAdapter
from tools.registry import registry


ROOM = "!room:server"
USER = "@alice:server"


def _adapter():
    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "homeserver": "https://server",
                "user_id": "@bot:server",
                "allowed_users": USER,
                "e2ee_mode": "off",
            },
        )
    )
    adapter._joined_rooms.add(ROOM)
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter.set_authorization_check(
        lambda user, chat_type, chat_id: user in adapter._allowed_user_ids
    )
    return adapter


async def _dispatch(adapter, arguments):
    tokens = set_session_vars(
        platform="matrix",
        chat_id=ROOM,
        user_id=USER,
        thread_id="$original",
        transport_adapter=adapter,
        transport_loop=asyncio.get_running_loop(),
    )
    try:
        response = await asyncio.to_thread(
            registry.dispatch, "matrix_thread_create", arguments
        )
        assert isinstance(response, str)
        result = json.loads(response)
        assert get_session_env("HERMES_SESSION_THREAD_ID") == "$original"
        return result
    finally:
        clear_session_vars(tokens)


@pytest.mark.asyncio
@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize(
    "existing,encrypted", [(False, False), (True, False), (False, True), (True, True)]
)
async def test_registry_thread_delivery_and_profile_isolation(
    tmp_path, monkeypatch, existing, encrypted, routed
):
    from mautrix.client import Client
    from mautrix.client.state_store.memory import MemoryStateStore
    from mautrix.types import JSON, EncryptedEvent, EventType, MessageEvent

    importlib.import_module("model_tools")
    from toolsets import _HERMES_CORE_TOOLS, resolve_toolset

    assert (
        "matrix_threads" in _get_platform_tools({}, "matrix"),
        "matrix_threads" in _get_platform_tools({}, "telegram"),
    ) == (True, False)
    assert "matrix_thread_create" in resolve_toolset("matrix_threads")
    assert "matrix_thread_create" not in _HERMES_CORE_TOOLS
    adapters = {}
    deliveries = {"A": [], "B": []}
    set_multiplex_active(True)
    try:
        for profile in ("A", "B", "A"):
            home = tmp_path / profile
            home.mkdir(exist_ok=True)
            scope = set_secret_scope({}, profile_home=str(home))
            home_token = set_hermes_home_override(str(home))
            try:
                owner = "A" if routed else profile
                owner_home = tmp_path / owner
                adapter = adapters.get(owner)
                if adapter is None:
                    adapter = _adapter()
                    adapter.set_owner_profile(owner)
                    adapter._resolve_store_dir()
                    store = MemoryStateStore()
                    monkeypatch.setattr(
                        store, "is_encrypted", AsyncMock(return_value=encrypted)
                    )
                    client = Client(
                        mxid="@bot:server", base_url="https://server", state_store=store
                    )
                    monkeypatch.setattr(client.api, "request", AsyncMock())
                    adapter._client = client
                    adapters[owner] = adapter
                client = adapter._client
                sent = deliveries[profile]
                start = len(sent)
                root = f"$root-{profile}-{start}"
                raw: dict[str, Any] = {
                    "event_id": root,
                    "room_id": ROOM,
                    "sender": USER,
                    "type": "m.room.message",
                    "origin_server_ts": 1,
                    "content": {"msgtype": "m.text", "body": "Visible root"},
                }
                fetch = AsyncMock(return_value=MessageEvent.deserialize(JSON(raw)))
                monkeypatch.setattr(client, "get_event", fetch)
                if encrypted:
                    fetch.return_value = EncryptedEvent.deserialize(
                        JSON({
                            **raw,
                            "type": "m.room.encrypted",
                            "content": {
                                "algorithm": "m.megolm.v1.aes-sha2",
                                "ciphertext": "cipher",
                                "session_id": "session",
                                "sender_key": "sender",
                                "device_id": "device",
                            },
                        })
                    )
                    crypto = SimpleNamespace(
                        decrypt_megolm_event=AsyncMock(
                            return_value=MessageEvent.deserialize(JSON(raw))
                        ),
                        encrypt_megolm_event=AsyncMock(
                            side_effect=lambda room, kind, content: {
                                "encrypted": content
                            }
                        ),
                    )
                    monkeypatch.setattr(client, "crypto", crypto)

                async def request(method, path, content, **kwargs):
                    assert asyncio.get_running_loop() is owner_loop
                    assert get_hermes_home() == home
                    body = content["encrypted"] if encrypted else content
                    event_id = (
                        root
                        if not existing and len(sent) == start
                        else f"$sent-{profile}-{len(sent)}"
                    )
                    sent.append((event_id, body, str(path)))
                    return {"event_id": event_id}

                owner_loop = asyncio.get_running_loop()
                monkeypatch.setattr(
                    client.api, "request", AsyncMock(side_effect=request)
                )
                adapter.max_message_length = 120
                arguments = {"message": "Initial reply " * 30}
                arguments.update(
                    {"root_event_id": root}
                    if existing
                    else {"root_text": "Visible root"}
                )
                if existing:
                    adapter._thread_fallbacks.remember(ROOM, root, "$earlier-reply")
                result = await _dispatch(adapter, arguments)
                replies = sent[start + (not existing) :]
                assert result == {
                    "success": True,
                    "room_id": ROOM,
                    "root_event_id": root,
                    "initial_reply_event_id": replies[0][0],
                }
                if not existing:
                    assert "m.relates_to" not in sent[start][1]
                previous = root
                for event_id, content, path in replies:
                    assert content["m.relates_to"] == {
                        "rel_type": "m.thread",
                        "event_id": root,
                        "is_falling_back": True,
                        "m.in_reply_to": {"event_id": previous},
                    }
                    assert (
                        str(
                            EventType.ROOM_ENCRYPTED
                            if encrypted
                            else EventType.ROOM_MESSAGE
                        )
                        in path
                    )
                    previous = event_id
                assert root in adapter._threads
                assert adapter._thread_fallbacks.latest(ROOM, root) == previous
                assert root in json.loads(
                    (owner_home / "matrix_threads.json").read_text(encoding="utf-8")
                )
                assert adapter._store_dir.is_relative_to(owner_home)
                if routed and profile != owner:
                    assert not (home / "matrix_threads.json").exists()
                fetch.assert_awaited_once_with(
                    ROOM, root
                ) if existing else fetch.assert_not_awaited()
                if encrypted and existing:
                    crypto.decrypt_megolm_event.assert_awaited_once()
            finally:
                reset_hermes_home_override(home_token)
                reset_secret_scope(scope)
    finally:
        set_multiplex_active(False)
        for adapter in adapters.values():
            await adapter._client.api.session.close()


_FAILURE_ERRORS: dict[str, str] = {
    "room": "Matrix room is not allowed or joined",
    "requester": "Matrix requester is not authorized for this room",
    "disconnected": "Matrix client is disconnected",
    "empty": "message must contain non-whitespace text",
    "conflicting": "Supply either root_text or root_event_id",
    "withdrawn": "Matrix root was withdrawn",
    "unrelated": "Matrix root is missing or belongs to another room",
    "decrypt_room": "Matrix root is missing or belongs to another room",
    "keys": "missing decryption keys",
    "encryption_keys": "missing encryption keys",
    "missing": "RuntimeError: M_NOT_FOUND",
    "permission": "MForbidden: Cannot send",
    "reply_permission": "MForbidden: Cannot send",
    "interrupt": "Matrix thread creation interrupted",
    "interrupt_after_root": "Matrix thread creation interrupted",
    "cancel": "Matrix thread creation cancelled",
    "cancel_reply": "Matrix thread creation cancelled",
    "unknown_requester": "Matrix requester is not authorized for this room",
    "reply_requester": "Matrix requester is not authorized for this room",
    "wrong_surface": "Matrix thread creation requires a live Matrix session",
    "stopped_loop": "Matrix gateway loop is unavailable",
    "invalid_root": "root_event_id must be a Matrix event ID",
    "blank_root": "root_text must contain non-whitespace text",
    "encrypt_failure": "RuntimeError: encryption key failure",
    "encrypt_interrupt": "Matrix thread creation interrupted",
    "interrupt_after_reply": "Matrix thread creation interrupted",
    "chunk_permission": "MForbidden: Cannot send",
}
for name in ("thread", "edit", "annotation"):
    _FAILURE_ERRORS[name] = "Matrix root must be an eligible main-timeline message"
for name in (
    "policy_owner",
    "fetch_owner",
    "decrypt_owner",
    "send_owner",
    "crypto_owner",
    "profile_owner",
    "user_owner",
    "store_owner",
    "client_user_owner",
    "token_owner",
    "server_owner",
    "device_owner",
    "state_store_owner",
    "crypto_store_owner",
    "encrypt_owner",
    "read_owner",
):
    _FAILURE_ERRORS[name] = "Matrix client ownership changed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "room",
        "requester",
        "disconnected",
        "missing",
        "withdrawn",
        "unrelated",
        "thread",
        "edit",
        "annotation",
        "keys",
        "policy_owner",
        "fetch_owner",
        "decrypt_owner",
        "send_owner",
        "permission",
        "reply_permission",
        "interrupt",
        "interrupt_after_root",
        "cancel",
        "cancel_reply",
        "empty",
        "conflicting",
        "encryption_keys",
        "crypto_owner",
        "profile_owner",
        "user_owner",
        "store_owner",
        "client_user_owner",
        "token_owner",
        "server_owner",
        "device_owner",
        "state_store_owner",
        "crypto_store_owner",
        "reply_requester",
        "encrypt_owner",
        "encrypt_failure",
        "encrypt_interrupt",
        "interrupt_after_reply",
        "chunk_permission",
        "decrypt_room",
        "read_owner",
        "unknown_requester",
        "wrong_surface",
        "stopped_loop",
        "invalid_root",
        "blank_root",
    ],
)
async def test_registry_thread_refusals_and_partial_delivery(
    tmp_path, monkeypatch, failure
):
    importlib.import_module("model_tools")
    from mautrix.errors import MForbidden
    from mautrix.types import JSON, MessageEvent
    from tools.interrupt import set_interrupt

    adapter = _adapter()
    root_content: dict[str, Any] = {"msgtype": "m.text", "body": "Existing"}
    raw: dict[str, Any] = {
        "event_id": "$existing",
        "room_id": ROOM,
        "sender": USER,
        "type": "m.room.message",
        "origin_server_ts": 1,
        "content": root_content,
    }
    client = SimpleNamespace(
        crypto=None,
        api=SimpleNamespace(token="test-token"),
        get_event=AsyncMock(return_value=raw),
        send_message_event=AsyncMock(),
        state_store=SimpleNamespace(is_encrypted=AsyncMock(return_value=False)),
    )
    adapter._client = client
    sent = []
    arguments = {"root_text": "New root", "message": "First reply"}
    if failure in {
        "missing",
        "withdrawn",
        "unrelated",
        "thread",
        "edit",
        "annotation",
        "keys",
        "fetch_owner",
        "decrypt_owner",
        "decrypt_room",
    }:
        arguments = {"root_event_id": "$existing", "message": "First reply"}
    if failure == "room":
        adapter._joined_rooms.clear()
    if failure == "requester":
        adapter._allowed_user_ids = {"@other:server"}
    if failure == "unknown_requester":
        adapter.set_authorization_check(lambda *args: None)
    if failure == "disconnected":
        adapter._client = None
    if failure == "missing":
        client.get_event.side_effect = RuntimeError("M_NOT_FOUND")
    if failure == "withdrawn":
        raw["unsigned"] = {"redacted_because": {}}
    if failure == "unrelated":
        raw["room_id"] = "!else:server"
    if failure in {"thread", "edit", "annotation"}:
        root_content["m.relates_to"] = {
            "rel_type": {
                "thread": "m.thread",
                "edit": "m.replace",
                "annotation": "m.annotation",
            }[failure]
        }
    if failure in {"keys", "decrypt_owner", "decrypt_room"}:
        client.get_event.return_value = {
            **raw,
            "type": "m.room.encrypted",
            "content": {
                "algorithm": "m.megolm.v1.aes-sha2",
                "ciphertext": "cipher",
                "session_id": "session",
                "sender_key": "sender",
                "device_id": "device",
            },
        }
    if failure == "empty":
        arguments["message"] = " "
    if failure == "conflicting":
        arguments["root_event_id"] = "$existing"
    if failure == "invalid_root":
        arguments = {"root_event_id": "bad", "message": "First reply"}
    if failure == "blank_root":
        arguments["root_text"] = " "
    if failure == "encryption_keys":
        client.state_store.is_encrypted.return_value = True
    if failure == "crypto_store_owner":
        crypto_store = SimpleNamespace(crypto_store=object())
        client.crypto = crypto_store
    if failure in {"interrupt_after_reply", "chunk_permission"}:
        adapter.max_message_length = 120
        arguments["message"] = "Initial thread reply " * 30

    replacement = None

    def replace_owner():
        nonlocal replacement
        replacement = SimpleNamespace(crypto=None, send_message_event=AsyncMock())
        adapter._client = replacement

    async def policy(room):
        nonlocal operation_task
        operation_task = asyncio.current_task()
        if failure in {"policy_owner", "read_owner"}:
            replace_owner()
        if failure == "crypto_owner":
            client.crypto = object()
        if failure == "profile_owner":
            adapter.set_owner_profile("other")
        if failure == "user_owner":
            adapter._user_id = "@other:server"
        if failure == "store_owner":
            adapter._store_dir = tmp_path / "other-store"
        if failure == "client_user_owner":
            client.mxid = "@other:server"
        if failure == "token_owner":
            client.api.token = "other-token"
        if failure == "server_owner":
            client.api.base_url = "https://other-server"
        if failure == "device_owner":
            client.device_id = "other-device"
        if failure == "state_store_owner":
            client.state_store = object()
        if failure == "crypto_store_owner":
            crypto_store.crypto_store = object()
        return True

    adapter._is_allowed_matrix_room_event = policy

    async def fetch(room, event):
        replace_owner()
        return raw

    if failure == "fetch_owner":
        client.get_event.side_effect = fetch

    async def decrypt(event):
        if failure == "decrypt_owner":
            replace_owner()
        if failure == "decrypt_room":
            raw["room_id"] = "!else:server"
        return MessageEvent.deserialize(JSON(raw))

    if failure in {"decrypt_owner", "decrypt_room"}:
        client.crypto = SimpleNamespace(decrypt_megolm_event=decrypt)

    async def send(room, kind, content, **kwargs):
        if (
            failure == "permission"
            or failure == "reply_permission"
            and sent
            or failure == "chunk_permission"
            and len(sent) == 2
        ):
            raise MForbidden(403, "Cannot send")
        if failure == "cancel" or failure == "cancel_reply" and sent:
            reached.set()
            await blocked.wait()
        event_id = "$root" if not sent else "$reply"
        sent.append((event_id, content))
        if failure == "send_owner":
            replace_owner()
        if failure == "interrupt_after_root":
            set_interrupt(True, worker_tid)
        if failure == "interrupt_after_reply" and len(sent) == 2:
            set_interrupt(True, worker_tid)
        if failure == "reply_requester":
            adapter.set_authorization_check(lambda *args: False)
        return event_id

    client.send_message_event.side_effect = send
    worker_tid = None
    operation_task = None
    reached = asyncio.Event()
    blocked = asyncio.Event()
    native_client = None
    if failure in {"encrypt_owner", "encrypt_failure", "encrypt_interrupt"}:
        from mautrix.client import Client
        from mautrix.client.state_store.memory import MemoryStateStore

        native_store = MemoryStateStore()
        monkeypatch.setattr(native_store, "is_encrypted", AsyncMock(return_value=True))
        native_client = Client(
            mxid="@bot:server", base_url="https://server", state_store=native_store
        )
        wire_request = AsyncMock()
        monkeypatch.setattr(native_client.api, "request", wire_request)

        async def encrypt(*args):
            if failure == "encrypt_owner":
                replace_owner()
            if failure == "encrypt_failure":
                raise RuntimeError("encryption key failure")
            if failure == "encrypt_interrupt":
                set_interrupt(True, worker_tid)
            return {"ciphertext": "test-ciphertext"}

        monkeypatch.setattr(
            native_client, "crypto", SimpleNamespace(encrypt_megolm_event=encrypt)
        )
        adapter._client = native_client

    def dispatch():
        nonlocal worker_tid
        import threading

        worker_tid = threading.get_ident()
        set_interrupt(failure == "interrupt")
        try:
            return (
                registry.dispatch("matrix_read", {"kind": "room"})
                if failure == "read_owner"
                else registry.dispatch("matrix_thread_create", arguments)
            )
        finally:
            set_interrupt(False)

    stopped_loop = asyncio.new_event_loop() if failure == "stopped_loop" else None
    tokens = set_session_vars(
        platform="cli" if failure == "wrong_surface" else "matrix",
        chat_id=ROOM,
        user_id=USER,
        transport_adapter=adapter,
        transport_loop=stopped_loop or asyncio.get_running_loop(),
    )
    try:
        pending = asyncio.create_task(asyncio.to_thread(dispatch))
        if failure in {"cancel", "cancel_reply"}:
            barrier = asyncio.create_task(reached.wait())
            done, _ = await asyncio.wait(
                {pending, barrier}, return_when=asyncio.FIRST_COMPLETED
            )
            if barrier in done:
                assert operation_task is not None
                operation_task.cancel()
            barrier.cancel()
            await asyncio.gather(barrier, return_exceptions=True)
        response = await pending
        assert isinstance(response, str)
        result = json.loads(response)
    finally:
        if stopped_loop is not None:
            stopped_loop.close()
        clear_session_vars(tokens)
        if native_client is not None:
            await native_client.api.session.close()
    expected: dict[str, Any] = {"success": False, "error": _FAILURE_ERRORS[failure]}
    if failure == "read_owner":
        expected = {"error": _FAILURE_ERRORS[failure]}
    if sent:
        expected.update(room_id=ROOM, root_event_id="$root", partial=True)
    visible = failure in {"interrupt_after_reply", "chunk_permission"}
    if visible:
        expected["initial_reply_event_id"] = "$reply"
    if failure in {"cancel", "cancel_reply"}:
        expected["delivery_uncertain"] = True
    assert result == expected
    assert ("$root" in adapter._threads) is visible
    if native_client is not None:
        wire_request.assert_not_awaited()
    if replacement is not None:
        replacement.send_message_event.assert_not_awaited()
    assert len(sent) <= (2 if visible else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("keys", [False, True])
async def test_sdk_admission_never_sends_plaintext_after_learning_encryption(
    monkeypatch, keys
):
    from mautrix.client import Client
    from mautrix.client.state_store.memory import MemoryStateStore
    from mautrix.errors import MNotFound
    from mautrix.types import EventType, RoomEncryptionStateEventContent
    from plugins.platforms.matrix.read_context import (
        MatrixSessionAccess,
        MatrixSessionError,
    )

    adapter = _adapter()
    store = MemoryStateStore()
    client = Client(mxid="@bot:server", base_url="https://server", state_store=store)
    adapter._client = client
    monkeypatch.setattr(
        client, "get_state_event", AsyncMock(side_effect=MNotFound(404, "Unencrypted"))
    )
    encrypt = AsyncMock(return_value={"ciphertext": "encrypted"})
    if keys:
        monkeypatch.setattr(
            client, "crypto", SimpleNamespace(encrypt_megolm_event=encrypt)
        )
    wire = AsyncMock(return_value={"event_id": "$confirmed"})
    monkeypatch.setattr(client.api, "request", wire)

    async def identify(room):
        await store.set_encryption_info(room, RoomEncryptionStateEventContent())
        return False

    adapter._is_dm_room = identify
    try:
        access = MatrixSessionAccess.capture(adapter, ROOM, USER)
        if keys:
            assert (
                await access.send_message({"msgtype": "m.text", "body": "Secret"})
                == "$confirmed"
            )
            assert str(EventType.ROOM_ENCRYPTED) in str(wire.call_args.args[1])
            assert wire.call_args.args[2] == {"ciphertext": "encrypted"}
        else:
            with pytest.raises(MatrixSessionError, match="missing encryption keys"):
                await access.send_message({"msgtype": "m.text", "body": "Secret"})
            wire.assert_not_awaited()
    finally:
        await client.api.session.close()


_WIRE_FAILURES = {"connection", 502, 503, 504}


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_send", [0, 1])
@pytest.mark.parametrize(
    "failure",
    ["connection", 502, 503, 504, "state_lookup", "group_session", "admission_timeout"],
)
async def test_sdk_send_failure_reports_uncertainty_only_after_the_request(
    monkeypatch, failure, failed_send
):
    from aiohttp import ServerDisconnectedError
    from mautrix.client import Client
    from mautrix.client.state_store.memory import MemoryStateStore
    from mautrix.errors import (
        EncryptionError,
        MatrixConnectionError,
        make_request_error,
    )

    importlib.import_module("model_tools")
    adapter = _adapter()
    store = MemoryStateStore()
    client = Client(mxid="@bot:server", base_url="https://server", state_store=store)
    client.api.default_retry_count = 0
    adapter._client = client
    accepted = []
    preparing = []

    def failing():
        return len(accepted) == failed_send

    async def is_encrypted(room):
        preparing.append(len(accepted))
        if failure == "state_lookup" and failing():
            return None
        return failure == "group_session" and failing()

    async def identify(room):
        if failure == "admission_timeout" and failing() and failed_send in preparing:
            raise TimeoutError("room identity lookup timed out")
        return False

    async def transport(*args):
        event_id = "$root" if not accepted else "$reply"
        accepted.append(event_id)
        if len(accepted) - 1 != failed_send:
            return {"event_id": event_id}, SimpleNamespace(status=200)
        if failure == "connection":
            raise ServerDisconnectedError("Response lost after acceptance")
        raise make_request_error(failure, "<html>Gateway error</html>", None, None)

    monkeypatch.setattr(store, "is_encrypted", is_encrypted)
    monkeypatch.setattr(
        client,
        "get_state_event",
        AsyncMock(side_effect=MatrixConnectionError("state lookup connection reset")),
    )
    monkeypatch.setattr(
        client,
        "crypto",
        SimpleNamespace(
            encrypt_megolm_event=AsyncMock(
                side_effect=EncryptionError("no outbound session")
            )
        ),
    )
    monkeypatch.setattr(
        client,
        "share_group_session",
        AsyncMock(side_effect=MatrixConnectionError("key share connection reset")),
    )
    adapter._is_dm_room = identify
    monkeypatch.setattr(client.api, "_send", transport)
    try:
        result = await _dispatch(adapter, {"root_text": "Root", "message": "Reply"})
        expected: dict[str, Any] = {
            "success": False,
            "error": {
                "connection": "MatrixConnectionError: Response lost after acceptance",
                "state_lookup": "MatrixConnectionError: state lookup connection reset",
                "group_session": "MatrixConnectionError: key share connection reset",
                "admission_timeout": "TimeoutError: room identity lookup timed out",
            }.get(
                failure,
                f"MatrixUnknownRequestError: {failure}: <html>Gateway error</html>",
            ),
        }
        if failure in _WIRE_FAILURES:
            expected["delivery_uncertain"] = True
        if failed_send:
            expected.update(room_id=ROOM, root_event_id="$root", partial=True)
        assert result == expected
        assert (
            accepted == ["$root", "$reply"][: failed_send + (failure in _WIRE_FAILURES)]
        )
        assert "$root" not in adapter._threads
    finally:
        await client.api.session.close()


@pytest.mark.asyncio
async def test_reply_delivered_before_a_reconnect_is_recorded():
    from plugins.platforms.matrix.reply_context import MatrixEventContext

    importlib.import_module("model_tools")
    adapter = _adapter()

    def client():
        return SimpleNamespace(
            crypto=None,
            api=SimpleNamespace(token="token"),
            state_store=SimpleNamespace(is_encrypted=AsyncMock(return_value=False)),
        )

    async def send(room, kind, content, **kwargs):
        if "m.relates_to" not in content:
            return "$root"
        adapter._client = client()
        adapter._user_id = "@other:server"
        return "$reply"

    adapter._client = client()
    adapter._client.send_message_event = send
    result = await _dispatch(adapter, {"root_text": "Root", "message": "Reply"})

    observed = {
        "result": result,
        "latest": adapter._thread_fallbacks.latest(ROOM, "$root"),
        "cached": await adapter._event_context_cache.resolve(None, ROOM, "$reply"),
    }
    assert observed == {
        "result": {
            "success": False,
            "error": "Matrix client ownership changed",
            "room_id": ROOM,
            "root_event_id": "$root",
            "initial_reply_event_id": "$reply",
            "partial": True,
        },
        "latest": "$reply",
        "cached": MatrixEventContext("@bot:server", "Reply"),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "confirmed,existing",
    [(0, False), (1, False), (2, False), pytest.param(0, True, id="existing-root")],
)
@pytest.mark.parametrize("cancelled", [False, True])
async def test_owner_loop_stop_bounds_waits_and_preserves_delivery_progress(
    monkeypatch, confirmed, existing, cancelled
):
    import threading

    importlib.import_module("model_tools")
    adapter = _adapter()
    adapter.max_message_length = 120
    adapter._client = SimpleNamespace(
        crypto=None,
        api=SimpleNamespace(token="token"),
        state_store=SimpleNamespace(is_encrypted=AsyncMock(return_value=False)),
        get_event=AsyncMock(
            return_value={
                "event_id": "$existing",
                "room_id": ROOM,
                "type": "m.room.message",
                "content": {"msgtype": "m.text", "body": "Existing root"},
            }
        ),
    )
    owner = asyncio.new_event_loop()
    reached = threading.Event()
    sent = []
    tasks = []

    async def send(room, kind, content, **kwargs):
        tasks.append(asyncio.current_task())
        if len(sent) == confirmed:
            owner.call_soon(owner.stop)
            reached.set()
            await asyncio.Event().wait()
        event_id = "$root" if not sent else "$reply"
        sent.append(event_id)
        return event_id

    adapter._client.send_message_event = send
    real_wait = asyncio.wait_for
    waits = []

    async def expire(awaitable, timeout):
        if timeout not in {300.0, 5.0}:
            return await real_wait(awaitable, timeout)
        waits.append(timeout)
        assert await asyncio.to_thread(reached.wait, 15), (
            "Send did not reach the stopped owner"
        )
        if len(waits) == 1 and cancelled:
            raise asyncio.CancelledError
        raise TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", expire)
    thread = threading.Thread(target=owner.run_forever)
    thread.start()
    tokens = set_session_vars(
        platform="matrix",
        chat_id=ROOM,
        user_id=USER,
        transport_adapter=adapter,
        transport_loop=owner,
    )
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                registry.dispatch,
                "matrix_thread_create",
                {
                    "message": "Reply " * 60,
                    **(
                        {"root_event_id": "$existing"}
                        if existing
                        else {"root_text": "Root"}
                    ),
                },
            ),
            timeout=15,
        )
        expected = {
            "success": False,
            "error": "Matrix thread creation cancelled"
            if cancelled
            else "Matrix thread creation timed out",
            "delivery_uncertain": True,
        }
        if existing:
            expected.update(room_id=ROOM, root_event_id="$existing")
        elif confirmed:
            expected.update(room_id=ROOM, root_event_id="$root", partial=True)
        if confirmed == 2:
            expected["initial_reply_event_id"] = "$reply"
        assert isinstance(result, str)
        assert json.loads(result) == expected
        assert waits == [300.0, 5.0]
        assert sent == ["$root", "$reply"][:confirmed]
    finally:
        clear_session_vars(tokens)
        thread.join(timeout=15)
        assert not thread.is_alive()
        for task in set(tasks):
            task.cancel()

        def drain():
            async def finish():
                await asyncio.gather(*set(tasks), return_exceptions=True)

            owner.run_until_complete(finish())
            owner.close()

        await asyncio.to_thread(drain)
