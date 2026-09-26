"""Edge TTS shares Hermes' platform TLS authority."""

import asyncio
import ssl
import sys
import types

import pm
import pytest

from agent.ssl_verify import _shared_context
from tools.tts_tool import _import_edge_tts


def test_edge_uses_the_platform_context_for_both_request_paths(monkeypatch, tmp_path):
    retired_ca = tmp_path / "retired-ca.pem"
    retired_ca.write_text("not a PEM bundle")
    monkeypatch.setenv("HERMES_CA_BUNDLE", str(retired_ca))
    package = types.ModuleType("edge_tts")
    for name in ("communicate", "voices"):
        module = types.ModuleType(f"edge_tts.{name}")
        module._SSL_CTX = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        setattr(package, name, module)
    monkeypatch.setitem(sys.modules, "edge_tts", package)
    monkeypatch.setattr(pm, "ensure_import", lambda extra: None)

    assert _import_edge_tts() is package
    platform_context = _shared_context(None)
    assert package.communicate._SSL_CTX is platform_context
    assert package.voices._SSL_CTX is platform_context
    assert platform_context.verify_mode == ssl.CERT_REQUIRED
    assert platform_context.check_hostname


def test_edge_requests_use_the_platform_context(monkeypatch):
    edge_tts = pytest.importorskip("edge_tts")
    monkeypatch.setattr(pm, "ensure_import", lambda extra: None)
    captured = []

    class RequestReached(Exception):
        pass

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def get(self, *args, ssl, **kwargs):
            captured.append(("http", ssl))
            raise RequestReached

        def ws_connect(self, *args, ssl, **kwargs):
            captured.append(("websocket", ssl))
            raise RequestReached

    monkeypatch.setattr(edge_tts.communicate.aiohttp, "ClientSession", Session)
    _import_edge_tts()

    with pytest.raises(RequestReached):
        asyncio.run(edge_tts.list_voices())

    async def stream():
        async for _ in edge_tts.Communicate("hello").stream():
            pass

    with pytest.raises(RequestReached):
        asyncio.run(stream())

    platform_context = _shared_context(None)
    assert captured == [("http", platform_context), ("websocket", platform_context)]
    assert platform_context.verify_mode == ssl.CERT_REQUIRED
    assert platform_context.check_hostname
