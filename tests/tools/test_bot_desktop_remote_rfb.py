"""Remote Screen transport keeps the local lease and wire contracts on a real TCP connection."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import struct

import pytest

from hermes_cli.web_routers import display
from tools.bot_desktop import lease, runtime
from tools.bot_desktop.rfb_auth import vnc_response

_VERSION = b"RFB 003.008\n"
_KEY = b"\x04\x01\0\0\0\0\0a"
_POINTER = b"\x05\x01\0\x02\0\x03"
_UPDATE = b"\x03\0" + b"\0" * 8
_CHALLENGE = bytes(range(16))
# Independently computed with OpenSSL des-ecb, key 0e86ceceeef64e26.
_RESPONSE = bytes.fromhex("b866924125c8eebb9debc1db61c538e2")


def _config(home, **values):
    from hermes_cli.config import atomic_config_write
    home.mkdir(parents=True, exist_ok=True)
    atomic_config_write(home / "config.yaml", {"bot_desktop": values})


@pytest.fixture(autouse=True)
def _isolation():
    lease._reset_for_tests()
    yield
    lease._reset_for_tests()


class _Ws:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()
        self.closes = []

    async def receive(self):
        return await self.incoming.get()

    async def send_bytes(self, data):
        await self.outgoing.put(data)

    async def close(self, code=1000, reason=""):
        self.closes.append((code, reason))

    def send(self, data):
        self.incoming.put_nowait({"type": "websocket.receive", "bytes": data})

    def disconnect(self, code):
        self.incoming.put_nowait({"type": "websocket.disconnect", "code": code})

    async def handshake(self):
        assert await asyncio.wait_for(self.outgoing.get(), 5) == _VERSION + b"\x01\x01"
        # Exercise arbitrary WS fragmentation, including coalesced ClientInit and a request.
        for byte in _VERSION + b"\x01":
            self.send(bytes([byte]))
        assert await asyncio.wait_for(self.outgoing.get(), 5) == b"\0" * 4
        self.send(b"\0" + _UPDATE)  # Exclusive is rewritten to shared.


@asynccontextmanager
async def _rfb(home, *, password="", fail=False, reason=b"", version=8, thumbnail=False, configured=None):
    received = asyncio.Queue()
    handlers = set()
    errors = []

    async def serve(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        try:
            banner = f"RFB 003.{version:03d}\n".encode()
            for byte in banner:
                writer.write(bytes([byte]))
                await writer.drain()
            expected_version = _VERSION if version == 889 else banner
            assert await reader.readexactly(12) == expected_version
            security = 2 if password else 1
            writer.write(security.to_bytes(4, "big") if version == 3 else bytes([1, security]))
            await writer.drain()
            if version != 3:
                assert await reader.readexactly(1) == bytes([security])
            if password:
                for byte in _CHALLENGE:
                    writer.write(bytes([byte]))
                    await writer.drain()
                response = await reader.readexactly(16)
                await received.put(response)
                assert response == _RESPONSE if not fail else response != _RESPONSE
            if password or version >= 8:
                writer.write((1 if fail else 0).to_bytes(4, "big"))
                if reason:
                    writer.write(len(reason).to_bytes(4, "big") + reason)
                await writer.drain()
            if fail:
                assert await reader.read() == b""  # Failure must not wait for our EOF.
                return
            assert await reader.readexactly(1) == b"\x01"
            if thumbnail:
                writer.write(struct.pack(">HH", 2, 1) + b"\0" * 16 + b"\0" * 4)
                await writer.drain()
                assert await reader.readexactly(20) == bytes.fromhex(
                    "00000000 20180001 00ff00ff00ff 100800 000000")
                assert await reader.readexactly(8) == b"\x02\0\0\x01\0\0\0\0"
                assert await reader.readexactly(10) == struct.pack(">BBHHHH", 3, 0, 0, 0, 2, 1)
                writer.write(b"\0\0\0\x01" + struct.pack(">HHHHi", 0, 0, 2, 1, 0)
                             + bytes.fromhex("0000ff0000ff0000"))
                await writer.drain()
            else:
                assert await reader.readexactly(10) == _UPDATE
                await received.put("ready")
            while data := await reader.read(1024):
                await received.put(data)
        except Exception as exc:
            errors.append(exc)
        finally:
            writer.close()
            await writer.wait_closed()
            handlers.discard(task)

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    _config(home, remote_endpoint=f"127.0.0.1:{port}", remote_allow_loopback=True,
            remote_password=password if configured is None else configured)
    try:
        yield received
    finally:
        server.close()
        await server.wait_closed()
        if handlers:
            await asyncio.wait_for(asyncio.gather(*handlers), 5)
        assert not errors, errors


async def _next(queue):
    return await asyncio.wait_for(queue.get(), 5)


def test_remote_transport_input_lease_eviction_and_activity(tmp_path, monkeypatch):
    async def forbidden(*args, **kwargs):
        pytest.fail("remote transport attempted a Unix socket")
    monkeypatch.setattr(asyncio, "open_unix_connection", forbidden)

    async def run():
        async with _rfb(tmp_path) as received:
            ws = _Ws()
            task = asyncio.create_task(display._bridge(ws, {"hermes_home": str(tmp_path), "viewer_id": "one"}))
            try:
                await ws.handshake()
                assert await _next(received) == "ready"
                # A following update is a barrier proving that both input messages were parsed/dropped.
                ws.send(_KEY + _POINTER + _UPDATE)
                assert await _next(received) == _UPDATE
                lease.acquire("one", profile_key=str(tmp_path))
                await asyncio.sleep(0)  # Deliver the bridge's on_change callback.
                ws.send(_KEY + _POINTER)
                assert await _next(received) == _KEY + _POINTER
                with pytest.raises(lease.HumanHasControl):
                    lease.assert_agent_may_act(profile_key=str(tmp_path))
                lease.release("one", profile_key=str(tmp_path))
                lease.assert_agent_may_act(profile_key=str(tmp_path))
                await asyncio.sleep(0)
                ws.send(_KEY + _UPDATE)
                assert await _next(received) == _UPDATE
                lease.acquire("one", profile_key=str(tmp_path))
                lease.acquire("two", profile_key=str(tmp_path))
                await asyncio.wait_for(task, 5)
                assert (4000, "control-taken") in ws.closes
                assert lease.get(profile_key=str(tmp_path)).viewer_id == "two"
                assert (tmp_path / "bot-desktop" / "activity").exists()
            finally:
                ws.disconnect(1000)
                await asyncio.wait_for(task, 5)
    asyncio.run(run())


@pytest.mark.parametrize("code,keeps", [(1006, True), (1005, True), (1000, False), (1001, False)])
def test_remote_close_matrix(tmp_path, code, keeps):
    async def run():
        async with _rfb(tmp_path) as received:
            lease.acquire("one", profile_key=str(tmp_path))
            ws = _Ws()
            task = asyncio.create_task(display._bridge(ws, {"hermes_home": str(tmp_path), "viewer_id": "one"}))
            await ws.handshake()
            assert await _next(received) == "ready"
            ws.disconnect(code)
            await asyncio.wait_for(task, 5)
            assert (lease.get(profile_key=str(tmp_path)).holder == lease.HUMAN) is keeps
    asyncio.run(run())


@pytest.mark.parametrize("version", [3, 7, 8, 889])
@pytest.mark.parametrize("password", ["", "password"])
def test_remote_auth_and_none_are_hidden_from_viewer(tmp_path, version, password):
    async def run():
        async with _rfb(tmp_path, password=password, version=version) as received:
            ws = _Ws()
            task = asyncio.create_task(display._bridge(ws, {"hermes_home": str(tmp_path)}))
            await ws.handshake()
            if password:
                assert await _next(received) == _RESPONSE
            assert await _next(received) == "ready"
            ws.disconnect(1000)
            await asyncio.wait_for(task, 5)
    asyncio.run(run())


@pytest.mark.parametrize("reason", [b"", b"incorrect-test-secret", b"x" * 5000],
                         ids=["absent", "echoed", "oversized"])
def test_wrong_password_fails_without_hanging_or_leaking(tmp_path, caplog, reason):
    async def check():
        async with _rfb(tmp_path, password="password", fail=True, reason=reason,
                        configured="incorrect-test-secret"):
            ws = _Ws()
            await asyncio.wait_for(display._bridge(ws, {"hermes_home": str(tmp_path)}), 5)
            assert ws.closes[0] == (4001, "remote screen authentication failed")
            assert "incorrect-test-secret" not in repr(ws.closes) + caplog.text
    asyncio.run(check())


@pytest.mark.parametrize("password,expected", [
    ("password-longer-than-eight", _RESPONSE),
    ("password", _RESPONSE),
    (bytes.fromhex("8040c020a060e010").decode("latin-1"),
     bytes.fromhex("6c5e94dcadd39f1d6c5e94dcadd39f1d")),
])
def test_vnc_des_known_vectors(password, expected):
    challenge = _CHALLENGE if password.startswith("password") else bytes.fromhex("1122334455667788" * 2)
    assert vnc_response(password, challenge) == expected
    assert vnc_response("a", challenge) == vnc_response("a" + "\0" * 7, challenge)


@pytest.mark.parametrize("endpoint", ["host", ":5900", "host:no", "host:0", "host:65536",
                                      "127.0.0.1:5900", "localhost:5900", "[::1]:5900",
                                      "[::ffff:127.0.0.1]:5900"])
def test_invalid_or_loopback_config_is_refused(tmp_path, monkeypatch, endpoint):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _config(tmp_path, remote_endpoint=endpoint)
    with pytest.raises(ValueError):
        runtime.remote_endpoint()


def test_quoted_false_does_not_allow_loopback(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _config(tmp_path, remote_endpoint="localhost:5900", remote_allow_loopback="false")
    with pytest.raises(ValueError, match="loopback"):
        runtime.remote_endpoint()
    with pytest.raises(ValueError, match="loopback"):
        runtime.validate_remote_peer("127.0.0.1")


def test_observe_status_and_management_for_remote_only(tmp_path, monkeypatch):
    import tui_gateway.server as server
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _config(tmp_path, remote_endpoint="100.65.115.112:5900", remote_password="test-status-secret")
    def call(name):
        return server.handle_request({"jsonrpc": "2.0", "id": 1, "method": name, "params": {}})
    result = call("display.observe")
    assert result["result"]["ticket"]
    assert result["result"]["supported"] is True
    assert result["result"]["remote"] == "100.65.115.112:5900"
    status = call("display.status")["result"]
    assert status["remote"] == result["result"]["remote"]
    assert not status["verified"] and not status["running"]
    assert "test-status-secret" not in json.dumps(result)
    lease.acquire("one")
    for operation in ("start", "stop", "install"):
        assert "remote screen" in call(f"display.{operation}")["error"]["message"]
        assert lease.get().viewer_id == "one"
    _config(tmp_path)
    assert "not running" in call("display.observe")["error"]["message"]


def test_screen_cli_reports_invalid_remote_config_without_traceback(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    from hermes_cli.subcommands import computer_use_screen
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _config(tmp_path, remote_endpoint="host:no-port", remote_password="cli-test-secret")
    for action in ("status", "start", "stop", "install"):
        assert getattr(computer_use_screen, f"_screen_{action}")(SimpleNamespace()) == 1
        output = capsys.readouterr().out
        assert "remote_endpoint" in output
        assert "cli-test-secret" not in output


def test_remote_thumbnail_uses_rfb_and_respects_human_control(tmp_path, monkeypatch):
    import tui_gateway.server as server
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    async def run():
        async with _rfb(tmp_path, password="password", thumbnail=True) as received:
            def call():
                return server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "display.thumbnail", "params": {}})
            result = await asyncio.to_thread(call)
            assert result["result"]["data_url"].startswith("data:image/jpeg;base64,")
            assert await _next(received) == _RESPONSE
            lease.acquire("one", profile_key=str(tmp_path))
            assert (await asyncio.to_thread(call))["result"]["suppressed"] == "human_has_control"
    asyncio.run(run())


def test_profile_config_and_secret_precedence_a_b_a(tmp_path, monkeypatch):
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from tui_gateway.server import _session_profile_runtime_scope
    a, b = tmp_path / "a", tmp_path / "b"
    for home, host in ((a, "100.64.0.1"), (b, "100.64.0.2")):
        _config(home, remote_endpoint=f"{host}:5900", remote_password="config-test-value")
    (b / ".env").write_text("HERMES_BOT_DESKTOP_REMOTE_PASSWORD=profile-b-test-value\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_BOT_DESKTOP_REMOTE_PASSWORD", "launch-test-value")
    was_active = is_multiplex_active()
    set_multiplex_active(True)
    try:
        for home, host, secret in ((a, "100.64.0.1", "config-test-value"),
                                   (b, "100.64.0.2", "profile-b-test-value"),
                                   (a, "100.64.0.1", "config-test-value")):
            with _session_profile_runtime_scope({"profile_home": str(home)}):
                assert runtime.remote_endpoint() == (host, 5900)
                assert runtime.remote_password() == secret
                assert secret not in json.dumps(runtime.status().as_dict())
    finally:
        set_multiplex_active(was_active)


@pytest.mark.parametrize("password", ["password", ""])
def test_bridge_launch_env_secret_precedence_under_multiplex(tmp_path, monkeypatch, password):
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from tui_gateway import launch_profile_policy
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("HERMES_HOME", str(a))
    monkeypatch.setenv("HERMES_BOT_DESKTOP_REMOTE_PASSWORD", password)
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    was_active = is_multiplex_active()
    launch_profile_policy.activate_multi_profile_hosting()
    async def run():
        async with _rfb(a, password=password, configured="config-must-not-win") as qa:
            async with _rfb(b, password="password") as qb:
                for home, queue, expected in ((a, qa, password), (b, qb, "password"), (a, qa, password)):
                    ws = _Ws()
                    task = asyncio.create_task(display._bridge(ws, {"hermes_home": str(home)}))
                    await ws.handshake()
                    if expected:
                        assert await _next(queue) == _RESPONSE
                    assert await _next(queue) == "ready"
                    ws.disconnect(1000)
                    await asyncio.wait_for(task, 5)
    try:
        asyncio.run(run())
    finally:
        set_multiplex_active(was_active)


@pytest.mark.parametrize("remote", [False, True])
def test_dial_selection_and_unreachable_codes(tmp_path, monkeypatch, remote):
    calls = []
    async def tcp(*args):
        calls.append("tcp")
        raise OSError("unreachable")
    async def unix(*args):
        calls.append("unix")
        raise OSError("unreachable")
    monkeypatch.setattr(asyncio, "open_connection", tcp)
    monkeypatch.setattr(asyncio, "open_unix_connection", unix)
    _config(tmp_path, remote_endpoint="100.64.0.1:5900" if remote else "",
            remote_password="dial-test-secret")
    sd = tmp_path / "bot-desktop"
    sd.mkdir()
    (sd / "rfb.sock").touch()
    ws = _Ws()
    asyncio.run(display._bridge(ws, {"hermes_home": str(tmp_path)}))
    assert calls == ["tcp" if remote else "unix"]
    assert ws.closes == [(4001, "remote screen unreachable" if remote else "Bot Desktop socket unreachable")]
    assert "dial-test-secret" not in repr(ws.closes)


@pytest.mark.parametrize("password,failed", [("", False), ("password", False), ("wrong", True)])
def test_auth_fragmented_stream_without_socket(password, failed):
    """Protocol coverage also runs where the test host prohibits listening sockets."""
    from tools.bot_desktop.rfb_auth import authenticate
    async def run():
        reader = asyncio.StreamReader()
        class Writer:
            def __init__(self):
                self.sent = bytearray()
            def write(self, data):
                self.sent.extend(data)
            async def drain(self):
                pass
        writer = Writer()
        security = 2 if password else 1
        data = (_VERSION + bytes([1, security]) + (_CHALLENGE if password else b"")
                + (1 if failed else 0).to_bytes(4, "big"))
        async def supply():
            for byte in data:
                reader.feed_data(bytes([byte]))
                await asyncio.sleep(0)
        task = asyncio.create_task(supply())
        try:
            if failed:
                with pytest.raises(ValueError, match="authentication failed"):
                    await asyncio.wait_for(authenticate(reader, writer, password), 5)
            else:
                await asyncio.wait_for(authenticate(reader, writer, password), 5)
                assert writer.sent == _VERSION + bytes([security]) + (_RESPONSE if password else b"")
        finally:
            await task
    asyncio.run(run())


def test_public_warning_and_loopback_peer_check(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _config(tmp_path)
    with pytest.raises(ValueError, match="loopback"):
        runtime.validate_remote_peer("127.0.0.1")
    with pytest.raises(ValueError, match="loopback"):
        runtime.validate_remote_peer("0.0.0.0")
    runtime.validate_remote_peer("100.65.115.112")
    runtime.validate_remote_peer("192.168.1.5")
    assert not caplog.records
    runtime.validate_remote_peer("8.8.8.8")
    runtime.validate_remote_peer("8.8.8.8")
    assert len(caplog.records) == 1
    caplog.clear()
    _config(tmp_path, remote_warn_public=False)
    runtime.validate_remote_peer("8.8.4.4")
    assert not caplog.records


@pytest.mark.parametrize("code,keeps", [(1000, False), (1005, True), (1006, True),
                                      (4000, True), (1003, True)])
def test_bridge_protocol_and_lease_without_listening_socket(tmp_path, monkeypatch, code, keeps):
    """Exercise both pumps in sandboxes too; the TCP tests above remain the transport proof."""
    async def run():
        reader = asyncio.StreamReader()
        reader.feed_data(_VERSION + b"\x01\x02" + _CHALLENGE + b"\0" * 4)
        writes = asyncio.Queue()
        class Writer:
            def write(self, data):
                writes.put_nowait(data)
            async def drain(self):
                pass
            def get_extra_info(self, name):
                return ("100.64.0.1", 5900)
            def close(self):
                reader.feed_eof()
            async def wait_closed(self):
                pass
        async def connect(host, port):
            assert (host, port) == ("100.64.0.1", 5900)
            return reader, Writer()
        monkeypatch.setattr(asyncio, "open_connection", connect)
        _config(tmp_path, remote_endpoint="100.64.0.1:5900", remote_password="password")
        ws = _Ws()
        task = asyncio.create_task(display._bridge(ws, {"hermes_home": str(tmp_path), "viewer_id": "one"}))
        await ws.handshake()
        assert await _next(writes) == _VERSION
        assert await _next(writes) == b"\x02"
        assert await _next(writes) == _RESPONSE
        assert await _next(writes) == b"\x01" + _UPDATE
        ws.send(_KEY + _POINTER + _UPDATE)
        assert await _next(writes) == _UPDATE
        lease.acquire("one", profile_key=str(tmp_path))
        await asyncio.sleep(0)
        ws.send(_KEY + _POINTER)
        assert await _next(writes) == _KEY + _POINTER
        reader.feed_data(b"server framebuffer")
        assert await _next(ws.outgoing) == b"server framebuffer"
        if code == 4000:
            lease.acquire("two", profile_key=str(tmp_path))
        elif code == 1003:
            ws.incoming.put_nowait({"type": "websocket.receive", "text": "invalid"})
        else:
            ws.disconnect(code)
        await asyncio.wait_for(task, 5)
        assert (lease.get(profile_key=str(tmp_path)).holder == lease.HUMAN) is keeps
        if code in (4000, 1003):
            assert ws.closes[0][0] == code
    asyncio.run(run())


def test_remote_thumbnail_raw_pixels_without_listening_socket(tmp_path, monkeypatch):
    from tools.bot_desktop.rfb_thumbnail import grab
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _config(tmp_path, remote_endpoint="100.64.0.1:5900")
    async def run():
        reader = asyncio.StreamReader()
        reader.feed_data(_VERSION + b"\x01\x01" + b"\0" * 4
                         + struct.pack(">HH", 2, 1) + b"\0" * 20
                         + b"\0\0\0\x01" + struct.pack(">HHHHi", 0, 0, 2, 1, 0)
                         + bytes.fromhex("0000ff0000ff0000"))
        class Writer:
            def __init__(self):
                self.sent = bytearray()
                self.closed = False
            def write(self, data):
                self.sent.extend(data)
            async def drain(self):
                pass
            def get_extra_info(self, name):
                return ("100.64.0.1", 5900)
            def close(self):
                self.closed = True
            async def wait_closed(self):
                pass
        writer = Writer()
        async def connect(*args):
            return reader, writer
        monkeypatch.setattr(asyncio, "open_connection", connect)
        image = await grab(("100.64.0.1", 5900), "")
        assert image.size == (2, 1)
        assert image.getpixel((0, 0)) == (255, 0, 0)
        assert image.getpixel((1, 0)) == (0, 255, 0)
        assert writer.closed
        assert writer.sent[:14] == _VERSION + b"\x01\x01"
    asyncio.run(run())


@pytest.mark.parametrize("offered", [[30, 33, 36, 35], [1]])
def test_unsupported_auth_is_actionable_without_selecting_or_leaking(tmp_path, monkeypatch, offered):
    from tools.bot_desktop.rfb_auth import authenticate
    import tui_gateway.server as gateway
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    secret = "never-echo-this-password"

    async def run():
        exchanges = asyncio.Queue()

        async def serve(reader, writer):
            writer.write(b"RFB 003.889\n")
            await writer.drain()
            banner = await reader.readexactly(12)
            writer.write(bytes([len(offered), *offered]))
            await writer.drain()
            remaining = await reader.read()
            await exchanges.put((banner, remaining))
            writer.close()
            await writer.wait_closed()

        async with await asyncio.start_server(serve, "127.0.0.1", 0) as listener:
            endpoint = ("127.0.0.1", listener.sockets[0].getsockname()[1])
            _config(tmp_path, remote_endpoint=f"{endpoint[0]}:{endpoint[1]}",
                    remote_allow_loopback=True, remote_password=secret)
            reader, writer = await asyncio.open_connection(*endpoint)
            try:
                with pytest.raises(ValueError) as exc:
                    await authenticate(reader, writer, secret)
                message = str(exc.value)
                assert str(offered) in message
                assert "30/33/35/36" in message and "security review" in message
                assert "classic VNC password" in message and "security type 2" in message
                assert "REMOTE host" in message and "Hermes does not change" in message
                assert len(message) < 2048
            finally:
                writer.close()
                await writer.wait_closed()
            ws = _Ws()
            await asyncio.wait_for(display._bridge(ws, {"hermes_home": str(tmp_path)}), 5)
            code, reason = ws.closes[0]
            assert code == 4001 and len(reason.encode()) <= 123
            assert all(str(security_type) in reason for security_type in offered)
            assert "remote host" in reason and "type 2" in reason and "unsupported" in reason
            assert "Screen Sharing" in reason
            result = await asyncio.to_thread(gateway.handle_request, {
                "jsonrpc": "2.0", "id": 1, "method": "display.thumbnail", "params": {}})
            assert result["error"]["message"] == message
            for _ in range(3):
                # A refusal sends only the negotiated banner, never a selection or credential.
                assert await _next(exchanges) == (_VERSION, b"")
            for material in (secret, _CHALLENGE.hex(), vnc_response(secret, _CHALLENGE).hex()):
                assert material not in message + repr(ws.closes) + json.dumps(result)
    asyncio.run(run())


def test_remote_screen_refuses_local_driver_after_lease_but_does_not_fence_mcp(tmp_path, monkeypatch):
    from tools.computer_use.tool import handle_computer_use
    from tools import mcp_tool_handlers
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _config(tmp_path, remote_endpoint="100.64.0.1:5900")
    lease.acquire("watching-human")
    assert json.loads(handle_computer_use({"action": "click", "coordinate": [1, 1]}))["code"] == "human_has_control"

    reached = []
    def acquire(name, timeout):
        reached.append(name)
        return None, "inert transport boundary"
    monkeypatch.setattr(mcp_tool_handlers, "_acquire_call_server", acquire)
    # Neither an independently configured desktop server nor an unrelated server has a lease mapping.
    for name in ("independent-desktop", "unrelated-search"):
        handler = mcp_tool_handlers._make_tool_handler(name, "test", 1)
        assert handler({}) == "inert transport boundary"
    assert reached == ["independent-desktop", "unrelated-search"]

    lease.release("watching-human")
    for action in ("click", "capture"):
        result = json.loads(handle_computer_use({"action": action, "coordinate": [1, 1]}))
        assert result["code"] == "remote_screen_not_drivable"
        assert "gateway host" in result["error"] and "different display" in result["error"]
