"""Gateway advertised-tools regressions (PR #135861 follow-on):

* transport: ``tools_list`` speaks the stateless dialect — ``Mcp-Method``
  mirrors the body method, ``Mcp-Name`` only on tools/call, namespaced
  ``_meta`` keys, SSE + plain-JSON frames both parse, error envelopes raise
* auth chain: BROWSER_MCP_API_KEY > CAMOFOX_API_KEY (generic lane credential
  wins over the community-compat one)
* plugin: every advertised non-browser tool registers with its server-carried
  schema; browser verbs are excluded (routed natively); dispatch extras never
  cross the wire as tool arguments; discovery failure falls back to the disk
  cache; a non-gateway backend registers nothing
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture(autouse=True)
def _isolate_lane_env(monkeypatch):
    for name in ("BROWSER_MCP_URL", "BROWSER_MCP_API_KEY", "BROWSER_CDP_URL",
                 "CAMOFOX_URL", "CAMOFOX_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BROWSER_MCP_URL", "https://api.browser-gw.example")
    monkeypatch.setenv("BROWSER_MCP_API_KEY", "k" * 8)


class _Resp:
    status_code = 200
    text = ""

    def __init__(self, payload=None, sse=False, status=200):
        self.status_code = status
        if payload is not None:
            body = json.dumps(payload)
            self.text = f"data: {body}\n\n" if sse else body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _instrument(monkeypatch, reply, calls):
    """Fake requests.post; record (url, headers, body) per call."""
    import tools.browser_mcp_transport as transport

    def _fake_post(url, **kw):
        calls.append({"url": url, "headers": dict(kw.get("headers") or {}),
                      "body": kw.get("json") or {}})
        return reply() if callable(reply) else reply

    monkeypatch.setattr(transport.requests, "post", _fake_post)


def test_tools_list_speaks_the_dialect(monkeypatch):
    import tools.browser_mcp_transport as transport

    calls = []
    payload = {"result": {"tools": [
        {"name": "extract", "inputSchema": {"type": "object", "properties": {"url": {"type": "string"}}}},
        {"name": "browser_navigate", "inputSchema": {"type": "object", "properties": {}}},
    ]}}
    _instrument(monkeypatch, _Resp(payload, sse=True), calls)

    tools = transport.tools_list()

    assert [t["name"] for t in tools] == ["extract", "browser_navigate"]
    req = calls[0]
    assert req["url"] == "https://api.browser-gw.example/mcp"
    assert req["headers"]["Mcp-Method"] == "tools/list"
    assert "Mcp-Name" not in req["headers"]  # tools/call only
    meta = req["body"]["params"]["_meta"]
    assert "io.modelcontextprotocol/protocolVersion" in meta
    assert "io.modelcontextprotocol/clientCapabilities" in meta
    assert req["body"]["method"] == "tools/list"


def test_tools_list_plain_json_frame(monkeypatch):
    import tools.browser_mcp_transport as transport

    _instrument(monkeypatch, _Resp({"result": {"tools": [{"name": "crawl"}]}}), [])
    assert [t["name"] for t in transport.tools_list()] == ["crawl"]


def test_rpc_error_and_auth_rejection(monkeypatch):
    import tools.browser_mcp_transport as transport

    _instrument(monkeypatch, _Resp({"error": {"code": -32602, "message": "Invalid params"}}, sse=True), [])
    with pytest.raises(RuntimeError, match="Invalid params"):
        transport.tools_list()

    _instrument(monkeypatch, _Resp(status=401), [])
    with pytest.raises(PermissionError):
        transport.tools_list()


def test_api_key_chain_generic_lane_first(monkeypatch):
    import tools.browser_mcp_transport as transport

    calls = []
    _instrument(monkeypatch, _Resp({"result": {"tools": []}}), calls)

    monkeypatch.setenv("BROWSER_MCP_API_KEY", "bk")
    monkeypatch.setenv("CAMOFOX_API_KEY", "ck")
    transport.tools_list()
    assert calls[-1]["headers"]["Authorization"] == "Bearer bk"

    monkeypatch.delenv("BROWSER_MCP_API_KEY")
    transport.tools_list()
    assert calls[-1]["headers"]["Authorization"] == "Bearer ck"


class _Ctx:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name, toolset=None, schema=None, handler=None,
                      check_fn=None, description=None, **_):
        self.tools[name] = {"toolset": toolset, "schema": schema,
                            "handler": handler, "description": description}


def _advertised():
    return [
        {"name": "extract",
         "description": "Extract page content",
         "inputSchema": {"type": "object",
                         "properties": {"url": {"type": "string"},
                                        "format": {"type": "string"}},
                         "required": ["url"]}},
        {"name": "crawl", "description": "Crawl a site",
         "inputSchema": {"type": "object", "properties": {"url": {"type": "string"}}}},
        {"name": "browser_navigate", "description": "routed natively",
         "inputSchema": {"type": "object", "properties": {}}},
    ]


def test_plugin_registers_advertised_non_browser_tools(monkeypatch, tmp_path):
    import tools.browser_mcp_transport as transport
    from plugins.gateway_tools import register

    monkeypatch.setattr(transport, "tools_list", lambda timeout=None: _advertised())
    monkeypatch.setattr(transport, "_is_gateway_backend", lambda: True)
    monkeypatch.setenv("PLUGIN_DATA", str(tmp_path))
    # bind _call BEFORE register(): register() from-imports it at call time
    seen = {}

    def _fake_call(name, arguments):
        seen["name"], seen["args"] = name, arguments
        return {"content": [{"text": "{\"title\": \"T\"}"}], "isError": False}

    monkeypatch.setattr(transport, "_call", _fake_call)

    ctx = _Ctx()
    register(ctx)

    assert sorted(ctx.tools) == ["crawl", "extract"]  # browser_* excluded
    schema = ctx.tools["extract"]["schema"]
    assert schema["parameters"]["required"] == ["url"]  # schema carried verbatim
    assert ctx.tools["extract"]["description"] == "Extract page content"

    # Handler: dispatch extras (task_id) filtered out; text returned as str
    out = ctx.tools["extract"]["handler"]({"url": "https://x", "format": "markdown", "task_id": "t1"})
    assert seen["name"] == "extract"
    assert seen["args"] == {"url": "https://x", "format": "markdown"}  # task_id never crosses
    assert isinstance(out, str) and "T" in out


def test_plugin_iserror_yields_error_payload(monkeypatch, tmp_path):
    import tools.browser_mcp_transport as transport
    from plugins.gateway_tools import register

    monkeypatch.setattr(transport, "tools_list", lambda timeout=None: _advertised())
    monkeypatch.setattr(transport, "_is_gateway_backend", lambda: True)
    monkeypatch.setenv("PLUGIN_DATA", str(tmp_path))
    monkeypatch.setattr(transport, "_call",
                       lambda name, arguments: {"content": [{"text": "boom"}], "isError": True})

    ctx = _Ctx()
    register(ctx)
    out = ctx.tools["extract"]["handler"]({"url": "https://x"})
    assert "boom" in out


def test_discovery_failure_falls_back_to_cache(monkeypatch, tmp_path):
    import tools.browser_mcp_transport as transport
    from plugins.gateway_tools import register

    monkeypatch.setattr(transport, "_is_gateway_backend", lambda: True)
    monkeypatch.setenv("PLUGIN_DATA", str(tmp_path))
    tmp_path.joinpath("advertised_tools.json").write_text(json.dumps({"tools": _advertised()}))

    def _boom(timeout=None):
        raise RuntimeError("warming up")

    monkeypatch.setattr(transport, "tools_list", _boom)

    ctx = _Ctx()
    register(ctx)
    assert sorted(ctx.tools) == ["crawl", "extract"]


def test_cache_first_never_hits_network_when_cache_exists(monkeypatch, tmp_path):
    import tools.browser_mcp_transport as transport
    from plugins.gateway_tools import register

    monkeypatch.setattr(transport, "_is_gateway_backend", lambda: True)
    monkeypatch.setenv("PLUGIN_DATA", str(tmp_path))
    tmp_path.joinpath("advertised_tools.json").write_text(json.dumps({"tools": _advertised()}))

    calls = []

    def _must_not_run(timeout=None):
        calls.append(1)
        raise AssertionError("tools_list must not run when a cache exists")

    monkeypatch.setattr(transport, "tools_list", _must_not_run)

    ctx = _Ctx()
    register(ctx)
    assert not calls
    assert sorted(ctx.tools) == ["crawl", "extract"]


def test_no_gateway_registers_nothing(monkeypatch, tmp_path):
    import tools.browser_mcp_transport as transport
    from plugins.gateway_tools import register

    monkeypatch.setattr(transport, "_is_gateway_backend", lambda: False)
    monkeypatch.setenv("PLUGIN_DATA", str(tmp_path))
    ctx = _Ctx()
    register(ctx)
    assert ctx.tools == {}