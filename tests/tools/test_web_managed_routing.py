"""Real config, dispatch and local HTTP must preserve web billing boundaries."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
import httpx


@pytest.fixture
def local_gateway(monkeypatch):
    from tools import managed_tool_gateway

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, dict(self.headers), body))
            if self.path == "/v2/search":
                status = 200
                payload = {"success": True, "data": {"web": [{
                    "title": "Local fixture", "url": "https://example.test", "description": "fixture",
                }]}}
            else:
                status, payload = 503, {"error": "local test outage"}
            content = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "test-nous-token")
    monkeypatch.setenv("PERPLEXITY_GATEWAY_URL", base + "/perplexity")
    monkeypatch.setenv("FIRECRAWL_GATEWAY_URL", base)
    monkeypatch.setenv("PERPLEXITY_BASE_URL", base + "/direct")
    # Only entitlement is stubbed: route and credential resolution remain real.
    monkeypatch.setattr(managed_tool_gateway, "managed_nous_tools_enabled", lambda **kw: True)

    # CI omits the firecrawl extra; stand in for the SDK, still over real HTTP.
    class FirecrawlSDK:
        def __init__(self, api_key, api_url):
            self.api_key, self.api_url = api_key, api_url

        def search(self, query, limit):
            response = httpx.post(
                f"{self.api_url}/v2/search", json={"query": query, "limit": limit},
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            return response.json()

    monkeypatch.setattr("plugins.web.firecrawl.provider.Firecrawl", FirecrawlSDK)
    monkeypatch.setattr("tools.web_tools._firecrawl_client", None, raising=False)
    try:
        yield requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize(
    "selection,direct_key,expected_paths",
    [
        ({"backend": "nous"}, False, ["/perplexity/search", "/v2/search"]),
        ({"backend": "nous"}, True, ["/direct/search"]),
        ({"backend": "nous", "search_backend": "perplexity"}, True, ["/direct/search"]),
        ({"search_backend": "perplexity"}, False, []),
        ({"backend": "nous", "search_backend": "perplexity"}, False, []),
        ({"backend": "perplexity"}, False, []),
        ({"backend": "firecrawl", "search_backend": "perplexity"}, False, []),
        ({}, False, ["/perplexity/search", "/v2/search"]),
        ({"extract_backend": "exa"}, False, ["/perplexity/search", "/v2/search"]),
    ],
)
def test_only_managed_search_may_use_billed_fallback(
    monkeypatch, tmp_path, local_gateway, selection, direct_key, expected_paths,
):
    from hermes_cli.config import atomic_config_write
    from tools import web_tools
    from tests.tools.conftest import register_all_web_providers

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_write(tmp_path / "config.yaml", {
        "web": {**selection, "keyless_rescue": False, "cache_enabled": True},
    })
    if direct_key:
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-direct-key")
    else:
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    register_all_web_providers()
    expected = [
        (path, "Bearer test-direct-key" if path == "/direct/search" else "Bearer test-nous-token")
        for path in expected_paths
    ]
    for attempt in (1, 2):
        result = json.loads(web_tools.web_search_tool("local fixture", limit=3))
        # A fallback must not be cached: the identical query retries Perplexity.
        assert [(path, headers["Authorization"]) for path, headers, _ in local_gateway] == expected * attempt
        assert result["success"] == bool(expected_paths and expected_paths[-1] == "/v2/search")
        if result["success"]:
            assert result["data"]["fallback_from"] == "managed_primary"
            assert "local test outage" in result["data"]["backend_error"]
            assert web_tools._get_extract_backend() == selection.get("extract_backend", "firecrawl")
    for path, headers, body in local_gateway:
        if path != "/v2/search":
            managed = path == "/perplexity/search"
            assert body.get("search_type") == ("fast" if managed else None)
            assert body["search_context_size"] == "low"
            assert headers.get("X-Pplx-Integration") == (None if managed else "hermes-agent")


def test_unentitled_managed_search_names_the_gateway(monkeypatch, tmp_path, local_gateway):
    from hermes_cli.config import atomic_config_write
    from tools import managed_tool_gateway, web_tools
    from tests.tools.conftest import register_all_web_providers

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_write(tmp_path / "config.yaml", {"web": {"backend": "nous", "keyless_rescue": False}})
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.setattr(managed_tool_gateway, "managed_nous_tools_enabled", lambda **kw: False)
    register_all_web_providers()

    error = json.loads(web_tools.web_search_tool("local fixture", limit=3))["error"]
    assert "Nous Tool Gateway" in error and "hermes tools" in error
    assert "PERPLEXITY_API_KEY" not in error
    assert local_gateway == []
