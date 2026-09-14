"""Native SSRF transport and current web dispatch with injected I/O only."""
import asyncio
import json
import socket
from unittest.mock import Mock
import httpx
import pytest
from plugins.web.local import provider as local
from tools import url_safety as safety


def test_native_transport_blocks_dns_rebinding(monkeypatch):
    answers = iter(["93.184.216.34", "127.0.0.1"])
    monkeypatch.setattr(safety, "_getaddrinfo", lambda *a: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (next(answers), 80))])
    monkeypatch.setattr(safety, "_global_allow_private_urls", lambda: False)
    monkeypatch.setattr(local, "check_website_access", lambda u: None)
    dial = Mock(side_effect=AssertionError("must not dial private IP"))
    from httpcore._backends.sync import SyncBackend
    monkeypatch.setattr(SyncBackend, "connect_tcp", dial)
    with pytest.raises(safety.SSRFConnectionBlocked):
        local._fetch_readable("http://example.com/article")
    dial.assert_not_called()


def test_native_transport_pins_public_ip(monkeypatch):
    monkeypatch.setattr(safety, "_getaddrinfo", lambda *a: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))])
    monkeypatch.setattr(safety, "_global_allow_private_urls", lambda: False)
    monkeypatch.setattr(local, "check_website_access", lambda u: None)
    dial = Mock(side_effect=RuntimeError("injected stop before socket"))
    from httpcore._backends.sync import SyncBackend
    monkeypatch.setattr(SyncBackend, "connect_tcp", dial)
    with pytest.raises(RuntimeError, match="injected stop"):
        local._fetch_readable("http://example.com/article")
    assert dial.call_args.args[0] == "93.184.216.34"


def test_current_dispatch_local_and_cache_policy(monkeypatch, tmp_path):
    from tools import web_tools as web
    from tools import website_policy
    from agent import web_search_registry as registry
    registry._reset_for_tests()
    registry.register_provider(local.LocalWebSearchProvider())
    monkeypatch.setattr(local.LocalWebSearchProvider, "is_available", lambda self: True)
    monkeypatch.setattr(web, "_load_web_config", lambda: {"extract_backend": "local"})
    monkeypatch.setattr(web, "_ensure_web_plugins_loaded", lambda: None)
    async def safe(url): return True
    monkeypatch.setattr(web, "async_is_safe_url", safe)
    monkeypatch.setattr(local, "is_safe_url", lambda u: True)
    monkeypatch.setattr(local, "check_website_access", lambda u: None)
    monkeypatch.setattr(website_policy, "check_website_access", lambda u: None)
    from tools import web_result_cache as cache
    monkeypatch.setattr(cache, "extract_cache_get", lambda *a, **kw: None)
    calls = []
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="integration content")
    monkeypatch.setattr(local, "_make_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        result = json.loads(asyncio.run(web.web_extract_tool(["https://example.com/integration"])))
        assert "integration content" in json.dumps(result)
        assert calls == ["https://example.com/integration"]
        cached = {"url": "https://example.com/integration", "title": "", "content": "cached text"}
        monkeypatch.setattr(cache, "extract_cache_get", lambda *a, **kw: cached)
        result = json.loads(asyncio.run(web.web_extract_tool(["https://example.com/integration"])))
        assert "cached text" in json.dumps(result)
        assert len(calls) == 1
        blocked = {"message": "blocked by fixture policy", "host": "example.com", "rule": "deny", "source": "fixture"}
        monkeypatch.setattr(website_policy, "check_website_access", lambda u: blocked)
        monkeypatch.setattr(local, "check_website_access", lambda u: blocked)
        result = json.loads(asyncio.run(web.web_extract_tool(["https://example.com/integration"])))
        assert "blocked by fixture policy" in json.dumps(result)
        assert "cached text" not in json.dumps(result)
        assert len(calls) == 1
    finally:
        registry._reset_for_tests()
