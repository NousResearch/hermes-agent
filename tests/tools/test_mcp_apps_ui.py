"""L1 tests for MCP Apps host-side rendering (io.modelcontextprotocol/ui).

Covers the Python host core added for desktop MCP Apps rendering:

* MCP client extracts an interactive UI card from a tool result's ``_meta.ui``
  and stashes it out-of-band (keyed by tool_call_id) instead of leaking the big
  HTML into the model-facing result.
* ``call_mcp_app_request`` proxies a card's bridged JSON-RPC frame to the
  session and serializes the result in the wire shape the iframe expects.
* The gateway ``mcp.app.request`` method wraps that as a JSON-RPC response, and
  ``_on_tool_complete`` attaches the stashed card to the ``tool.complete`` event.

The ``_meta.ui`` shape used here mirrors the real utp MCP Apps server:
``_meta.ui.resource = {uri, mimeType, text}`` + ``_meta.ui.csp = {...}``.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import approval, mcp_tool


class _FakeContentBlock:
    def __init__(self, text: str, block_type: str = "text"):
        self.text = text
        self.type = block_type

    def model_dump(self, exclude_none=False, by_alias=False):
        return {"type": self.type, "text": self.text}


class _FakeCallToolResult:
    def __init__(self, content, is_error=False, structuredContent=None, meta=None):
        self.content = content
        self.isError = is_error
        self.structuredContent = structuredContent
        self.meta = meta


def _ui_meta(uri="ui://utp/catalog-search"):
    return {
        "ui": {
            "resource": {
                "uri": uri,
                "mimeType": "text/html;profile=mcp-app",
                "text": "<!DOCTYPE html><html><head></head><body>card</body></html>",
            },
            "csp": {
                "scriptSrc": "'unsafe-inline' 'unsafe-eval' *.alicdn.com",
                "connectDomains": ["*.alicdn.com"],
                "resourceDomains": ["*.alicdn.com"],
                "allowUnsafeEval": True,
            },
        }
    }


def _fake_run_on_mcp_loop(coro_or_factory, timeout=30):
    coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
    loop = asyncio.new_event_loop()
    try:
        async def _install_lock_and_run():
            for srv in list(mcp_tool._servers.values()):
                if getattr(srv, "_rpc_lock", None) is None:
                    srv._rpc_lock = asyncio.Lock()
            return await coro
        return loop.run_until_complete(_install_lock_and_run())
    finally:
        loop.close()


@pytest.fixture
def _patch_mcp_server():
    fake_session = MagicMock()
    fake_server = SimpleNamespace(session=fake_session, _rpc_lock=None)
    with patch.dict(mcp_tool._servers, {"test-server": fake_server}, clear=False), \
            patch("tools.mcp_tool._run_on_mcp_loop", side_effect=_fake_run_on_mcp_loop):
        yield fake_session


# --------------------------------------------------------------------------
# _extract_mcp_ui / stash / pop
# --------------------------------------------------------------------------

class TestExtractAndStash:
    def test_extracts_ui_card_from_meta(self):
        payload = mcp_tool._extract_mcp_ui(
            _FakeCallToolResult([], meta=_ui_meta()), "utp")
        assert payload is not None
        assert payload["server"] == "utp"
        assert payload["uri"] == "ui://utp/catalog-search"
        assert payload["html"].startswith("<!DOCTYPE html>")
        assert payload["mimeType"] == "text/html;profile=mcp-app"
        assert payload["csp"]["connectDomains"] == ["*.alicdn.com"]

    def test_no_ui_meta_returns_none(self):
        assert mcp_tool._extract_mcp_ui(
            _FakeCallToolResult([], meta={}), "utp") is None
        assert mcp_tool._extract_mcp_ui(
            _FakeCallToolResult([], meta=None), "utp") is None
        # resource missing uri/text -> None
        bad = {"ui": {"resource": {"uri": "ui://x"}}}
        assert mcp_tool._extract_mcp_ui(
            _FakeCallToolResult([], meta=bad), "utp") is None

    def test_stash_and_pop_roundtrip(self):
        payload = {"server": "utp", "uri": "ui://x",
                   "html": "<html>", "csp": None}
        mcp_tool._stash_mcp_ui_payload("tc-42", payload)
        assert mcp_tool.pop_mcp_ui_payload("tc-42") == payload
        # single-use: second pop is empty
        assert mcp_tool.pop_mcp_ui_payload("tc-42") is None

    def test_pop_empty_key(self):
        assert mcp_tool.pop_mcp_ui_payload("") is None


class TestHandlerStashesUi:
    def test_ui_result_stashed_not_leaked_to_model(self, _patch_mcp_server):
        session = _patch_mcp_server
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeContentBlock("已打开搜品卡片")],
                structuredContent={"products": []},
                meta=_ui_meta(),
            )
        )
        tokens = approval.set_current_observability_context(
            tool_call_id="tc-ui-1")
        try:
            handler = mcp_tool._make_tool_handler(
                "test-server", "utp_catalog_search", 30.0)
            raw = handler({})
        finally:
            approval.reset_current_observability_context(tokens)

        data = json.loads(raw)
        # Model-facing result carries the short text + structuredContent only.
        assert data["result"] == "已打开搜品卡片"
        assert data["structuredContent"] == {"products": []}
        # The large HTML never enters the model-facing payload.
        assert "<!DOCTYPE html>" not in raw
        # ...but is available out-of-band keyed by tool_call_id.
        stashed = mcp_tool.pop_mcp_ui_payload("tc-ui-1")
        assert stashed is not None
        assert stashed["uri"] == "ui://utp/catalog-search"

    def test_non_ui_result_stashes_nothing(self, _patch_mcp_server):
        session = _patch_mcp_server
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeContentBlock("plain")])
        )
        tokens = approval.set_current_observability_context(
            tool_call_id="tc-plain")
        try:
            handler = mcp_tool._make_tool_handler(
                "test-server", "some_tool", 30.0)
            handler({})
        finally:
            approval.reset_current_observability_context(tokens)
        assert mcp_tool.pop_mcp_ui_payload("tc-plain") is None


# --------------------------------------------------------------------------
# Referenced-form cards (tool-def _meta.ui.resourceUri -> resources/read)
#
# utp gates tool-def _meta on the client declaring the io.modelcontextprotocol/ui
# extension at initialize. catalog/product/cart tools carry only a resourceUri
# (no inline HTML in the result), so the host must resolve the ui:// resource.
# --------------------------------------------------------------------------


class _FakeTool:
    def __init__(self, name, meta=None):
        self.name = name
        self.meta = meta


class _FakeResourceContents:
    def __init__(self, text, mime="text/html;profile=mcp-app"):
        self.text = text
        self.mimeType = mime

    def model_dump(self, exclude_none=False, by_alias=False):
        return {"text": self.text, "mimeType": self.mimeType, "uri": "db://internal/schema"}


class _FakeReadResourceResult:
    def __init__(self, contents):
        self.contents = contents


@pytest.fixture(autouse=True)
def _clear_ui_resources():
    # Keep the module-level referenced-form registries isolated per test.
    with mcp_tool._mcp_ui_resources_lock:
        mcp_tool._mcp_ui_tool_resources.clear()
        mcp_tool._mcp_ui_resource_html_cache.clear()
    yield
    with mcp_tool._mcp_ui_resources_lock:
        mcp_tool._mcp_ui_tool_resources.clear()
        mcp_tool._mcp_ui_resource_html_cache.clear()


def _ref_ui_meta(uri="ui://utp/catalog-search"):
    return {"ui": {"resourceUri": uri, "csp": {"connectDomains": ["*.alicdn.com"]}}}


class TestReferencedFormCapture:
    def test_captures_resource_uri_from_tool_def(self):
        tools = [
            _FakeTool("utp_catalog_search", meta=_ref_ui_meta()),
            _FakeTool("utp_cart_add", meta=None),  # non-UI tool
        ]
        mcp_tool._capture_ui_tool_resources("utp", tools)
        with mcp_tool._mcp_ui_resources_lock:
            mapping = mcp_tool._mcp_ui_tool_resources.get("utp")
        assert mapping is not None
        assert mapping["utp_catalog_search"]["resourceUri"] == "ui://utp/catalog-search"
        assert mapping["utp_catalog_search"]["csp"] == {
            "connectDomains": ["*.alicdn.com"]}
        assert "utp_cart_add" not in mapping

    def test_no_ui_tools_clears_server_entry(self):
        mcp_tool._capture_ui_tool_resources(
            "utp", [_FakeTool("utp_catalog_search", meta=_ref_ui_meta())])
        mcp_tool._capture_ui_tool_resources(
            "utp", [_FakeTool("plain", meta=None)])
        with mcp_tool._mcp_ui_resources_lock:
            assert "utp" not in mcp_tool._mcp_ui_tool_resources


class TestResolveReferencedUi:
    def test_resolves_and_caches_html(self):
        mcp_tool._capture_ui_tool_resources(
            "utp", [_FakeTool("utp_catalog_search", meta=_ref_ui_meta())])
        session = MagicMock()
        session.read_resource = AsyncMock(
            return_value=_FakeReadResourceResult(
                [_FakeResourceContents("<!DOCTYPE html><body>card</body>")])
        )
        server = SimpleNamespace(session=session)

        async def _run():
            first = await mcp_tool._resolve_referenced_mcp_ui(server, "utp", "utp_catalog_search")
            second = await mcp_tool._resolve_referenced_mcp_ui(server, "utp", "utp_catalog_search")
            return first, second

        first, second = asyncio.new_event_loop().run_until_complete(_run())
        assert first is not None
        assert first["server"] == "utp"
        assert first["uri"] == "ui://utp/catalog-search"
        assert first["html"].startswith("<!DOCTYPE html>")
        assert first["mimeType"] == "text/html;profile=mcp-app"
        assert first["csp"] == {"connectDomains": ["*.alicdn.com"]}
        # Static HTML is fetched at most once, then cache-served.
        assert session.read_resource.await_count == 1
        assert second["html"] == first["html"]

    def test_tool_without_resource_uri_returns_none(self):
        server = SimpleNamespace(session=MagicMock())

        async def _run():
            return await mcp_tool._resolve_referenced_mcp_ui(server, "utp", "unknown_tool")

        assert asyncio.new_event_loop().run_until_complete(_run()) is None

    def test_handler_stashes_referenced_card(self, _patch_mcp_server):
        # A tool whose result carries NO inline _meta but whose definition
        # advertised a resourceUri must still surface a card via resources/read.
        session = _patch_mcp_server
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeContentBlock("Found 20 products")],
                structuredContent={"products": []},
                meta=None,
            )
        )
        session.read_resource = AsyncMock(
            return_value=_FakeReadResourceResult(
                [_FakeResourceContents("<!DOCTYPE html><body>catalog</body>")])
        )
        mcp_tool._capture_ui_tool_resources(
            "test-server", [_FakeTool("utp_catalog_search", meta=_ref_ui_meta())])
        tokens = approval.set_current_observability_context(
            tool_call_id="tc-ref-1")
        try:
            handler = mcp_tool._make_tool_handler(
                "test-server", "utp_catalog_search", 30.0)
            raw = handler({"keyword": "x", "search_type": "KEYWORD_SEARCH"})
        finally:
            approval.reset_current_observability_context(tokens)
        # Model-facing result still short; big HTML stays out-of-band.
        assert "<!DOCTYPE html>" not in raw
        stashed = mcp_tool.pop_mcp_ui_payload("tc-ref-1")
        assert stashed is not None
        assert stashed["uri"] == "ui://utp/catalog-search"
        assert stashed["html"].startswith("<!DOCTYPE html>")


# --------------------------------------------------------------------------
# call_mcp_app_request (bridge proxy)
# --------------------------------------------------------------------------

class TestBridgeProxy:
    def test_tools_call_serialized(self, _patch_mcp_server):
        session = _patch_mcp_server
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeContentBlock("detail")],
                structuredContent={"id": "p1"},
            )
        )
        out = mcp_tool.call_mcp_app_request(
            "test-server", "tools/call", {"name": "utp_catalog_product",
                                          "arguments": {"product_id": "p1"}}
        )
        assert out["isError"] is False
        assert out["structuredContent"] == {"id": "p1"}
        assert out["content"] == [{"type": "text", "text": "detail"}]

    def test_unknown_server_errors(self):
        out = mcp_tool.call_mcp_app_request(
            "nope", "tools/call", {"name": "x"})
        assert set(out.keys()) == {"error"}
        assert out["error"]["code"] == -32000

    def test_unsupported_method_errors(self, _patch_mcp_server):
        out = mcp_tool.call_mcp_app_request(
            "test-server", "resources/subscribe", {})
        assert out["error"]["code"] == -32601

    def test_initialize_is_noop(self, _patch_mcp_server):
        assert mcp_tool.call_mcp_app_request(
            "test-server", "initialize", {}) == {}


# --------------------------------------------------------------------------
# Security: bridge whitelist validation
# --------------------------------------------------------------------------

class TestBridgeSecurity:
    """Verify that bridge requests are validated per the security model."""

    def test_tools_call_blocks_unregistered_tool(self, _patch_mcp_server):
        """P0 (fixed): a card cannot call a tool the server did not advertise."""
        session = _patch_mcp_server
        session.call_tool = AsyncMock()
        # Simulate a server that only exposes catalog_search.
        server = mcp_tool._servers["test-server"]
        server._registered_tool_names = ["utp_catalog_search"]

        # A sandboxed card tries to call a tool the server never declared.
        out = mcp_tool.call_mcp_app_request(
            "test-server", "tools/call",
            {"name": "utp_checkout_complete", "arguments": {}}
        )

        assert set(out.keys()) == {"error"}
        assert out["error"]["code"] == -32601
        assert "not registered" in out["error"]["message"]
        # The real tool was never invoked — the bridge blocked the call.
        session.call_tool.assert_not_called()

    def test_tools_call_permits_registered_tool(self, _patch_mcp_server):
        """Sanity: a card CAN call tools the server advertised."""
        session = _patch_mcp_server
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeContentBlock("added")]))
        server = mcp_tool._servers["test-server"]
        server._registered_tool_names = ["utp_cart_add"]

        out = mcp_tool.call_mcp_app_request(
            "test-server", "tools/call",
            {"name": "utp_cart_add", "arguments": {"product_id": "p1"}}
        )

        assert "error" not in out
        session.call_tool.assert_called_once_with(
            "utp_cart_add", arguments={"product_id": "p1"})

    def test_resources_read_blocks_non_ui_uri(self, _patch_mcp_server):
        """P0 (fixed): resources/read rejects non-ui:// URIs through the bridge.

        A sandboxed card should only access ui:// resources.  Non-ui:// URIs
        (db://, file://, etc.) are for the model — the bridge must block them.
        """
        session = _patch_mcp_server
        session.read_resource = AsyncMock()

        out = mcp_tool.call_mcp_app_request(
            "test-server", "resources/read",
            {"uri": "db://internal/schema"}
        )

        # PROOF: the bridge REJECTED the non-ui:// URI before forwarding.
        assert set(out.keys()) == {"error"}
        assert out["error"]["code"] == -32601
        session.read_resource.assert_not_called()

    def test_resources_read_allows_ui_uri(self, _patch_mcp_server):
        """Sanity: legitimate ui:// resource reads still work."""
        session = _patch_mcp_server
        session.read_resource = AsyncMock(
            return_value=_FakeReadResourceResult(
                [_FakeResourceContents("<!DOCTYPE html><body>catalog</body>")])
        )

        out = mcp_tool.call_mcp_app_request(
            "test-server", "resources/read",
            {"uri": "ui://utp/catalog-search"}
        )

        assert "contents" in out
        assert "error" not in out
        session.read_resource.assert_called_once_with("ui://utp/catalog-search")


# --------------------------------------------------------------------------
# gateway: mcp.app.request method + tool.complete ui attach
# --------------------------------------------------------------------------

class TestGatewayBridge:
    def _server(self):
        import tui_gateway.server as srv
        return srv

    def test_method_wraps_response_and_preserves_id(self):
        srv = self._server()
        handler = srv._methods["mcp.app.request"]
        with patch("tools.mcp_tool.call_mcp_app_request",
                   return_value={"isError": False, "content": [], "structuredContent": {"ok": 1}}):
            resp = handler(7, {"server": "utp", "message": {
                           "jsonrpc": "2.0", "id": 99, "method": "tools/call", "params": {"name": "t"}}})
        assert resp["id"] == 7
        inner = resp["result"]["response"]
        assert inner["jsonrpc"] == "2.0"
        assert inner["id"] == 99
        assert inner["result"]["structuredContent"] == {"ok": 1}

    def test_method_maps_error_dict(self):
        srv = self._server()
        handler = srv._methods["mcp.app.request"]
        with patch("tools.mcp_tool.call_mcp_app_request",
                   return_value={"error": {"code": -32601, "message": "nope"}}):
            resp = handler(1, {"server": "utp", "message": {
                           "jsonrpc": "2.0", "id": 5, "method": "x"}})
        inner = resp["result"]["response"]
        assert inner["error"]["code"] == -32601
        assert inner["id"] == 5

    def test_method_validates_params(self):
        srv = self._server()
        handler = srv._methods["mcp.app.request"]
        # missing server
        resp = handler(1, {"message": {"jsonrpc": "2.0", "method": "x"}})
        assert resp["error"]["code"] == -32602

    def test_tool_complete_attaches_ui_and_emits(self):
        srv = self._server()
        emitted = []
        card = {"server": "utp", "uri": "ui://utp/catalog-search",
                "html": "<html>", "csp": None}
        with patch.object(srv, "_emit", side_effect=lambda ev, sid, payload=None: emitted.append((ev, payload))), \
                patch("tools.mcp_tool.pop_mcp_ui_payload", return_value=card):
            srv._on_tool_complete(
                "sid-1", "tc-1", "utp_catalog_search", {}, json.dumps({"result": "ok"}))
        assert emitted, "tool.complete should emit when a UI card is present"
        ev, payload = emitted[-1]
        assert ev == "tool.complete"
        assert payload["ui"] == card

    def test_tool_complete_without_ui_unaffected(self):
        srv = self._server()
        emitted = []
        with patch.object(srv, "_emit", side_effect=lambda ev, sid, payload=None: emitted.append((ev, payload))), \
                patch("tools.mcp_tool.pop_mcp_ui_payload", return_value=None), \
                patch.object(srv, "_tool_progress_enabled", return_value=False):
            srv._on_tool_complete(
                "sid-2", "tc-2", "read_file", {}, json.dumps({"result": "x"}))
        # No UI, progress disabled -> no emit (existing behaviour preserved).
        assert emitted == []
