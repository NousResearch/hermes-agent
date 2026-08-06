"""End-to-end MCP Apps host tests against a real stdio server subprocess.

Complements the fake-based L1 tests in ``test_mcp_apps_ui.py``: those pin the
host's internal logic, these drive the actual wire protocol through
``_connect_server`` -> stdio transport -> ``ClientSession``, against
``tests/fixtures/mcp_apps_server.py``.

That gap matters because the referenced-card form is a *negotiated* feature:
real MCP Apps servers only emit tool-definition ``_meta.ui`` when the client
declared ``io.modelcontextprotocol/ui`` at initialize time. A fake session can
never catch the host dropping that declaration -- the fixture gates on it, so
these tests fail loudly if it regresses.
"""

import asyncio
import sys
from pathlib import Path

import pytest

from tools import mcp_tool

FIXTURE_SERVER = Path(__file__).resolve().parents[1] / \
    "fixtures" / "mcp_apps_server.py"

REFERENCED_URI = "ui://fixture/referenced-card"
INLINE_URI = "ui://fixture/inline-card"


@pytest.fixture(autouse=True)
def _isolate_ui_registries(monkeypatch):
    """Keep the module-global UI maps from leaking across tests."""
    monkeypatch.setattr(mcp_tool, "_mcp_ui_tool_resources", {})
    monkeypatch.setattr(mcp_tool, "_mcp_ui_resource_html_cache", {})
    # The OSV preflight is a network call on the spawn path; the fixture is a
    # local file, so skip it rather than make every test wait on a timeout.
    monkeypatch.setattr(
        "tools.osv_check.check_package_for_malware", lambda *_a, **_kw: None
    )


def _drive(coro_factory):
    """Connect to the fixture server, run ``coro_factory(server)``, tear down."""

    async def _run():
        server = await mcp_tool._connect_server(
            "fixture",
            {"command": sys.executable, "args": [str(FIXTURE_SERVER)]},
        )
        try:
            return await coro_factory(server)
        finally:
            await server.shutdown()

    return asyncio.run(_run())


@pytest.mark.skipif(not FIXTURE_SERVER.exists(), reason="fixture server missing")
def test_host_declares_ui_extension_so_referenced_meta_survives():
    """The fixture only emits referenced ``_meta`` to a UI-declaring client."""

    async def _check(server):
        return dict(mcp_tool._mcp_ui_tool_resources.get("fixture") or {})

    mapping = _drive(_check)

    assert "referenced_card" in mapping, (
        "no referenced-form descriptor captured -- the host likely stopped "
        "declaring the io.modelcontextprotocol/ui capability at initialize"
    )
    assert mapping["referenced_card"]["resourceUri"] == REFERENCED_URI
    assert mapping["referenced_card"]["csp"]["allowUnsafeEval"] is True
    # Tools without a card must not acquire one.
    assert "inline_card" not in mapping
    assert "plain_tool" not in mapping


@pytest.mark.skipif(not FIXTURE_SERVER.exists(), reason="fixture server missing")
def test_inline_card_is_extracted_from_a_real_tool_result():
    async def _check(server):
        async with server._rpc_lock:
            result = await server.session.call_tool("inline_card", arguments={})
        return mcp_tool._extract_mcp_ui(result, "fixture"), result

    payload, result = _drive(_check)

    assert payload is not None
    assert payload["uri"] == INLINE_URI
    assert payload["mimeType"] == "text/html;profile=mcp-app"
    assert "inline fixture card" in payload["html"]
    assert payload["csp"]["connectDomains"] == ["fixture.invalid"]
    # The card rides the side channel; the model-facing text stays short.
    model_text = "".join(getattr(b, "text", "") or "" for b in result.content)
    assert model_text == "inline card attached"
    assert len(payload["html"]) > 10 * len(model_text)


@pytest.mark.skipif(not FIXTURE_SERVER.exists(), reason="fixture server missing")
def test_referenced_card_is_resolved_over_resources_read():
    async def _check(server):
        async with server._rpc_lock:
            result = await server.session.call_tool(
                "referenced_card", arguments={})
        # The result itself carries no card -- it must come from the resource.
        inline = mcp_tool._extract_mcp_ui(result, "fixture")
        async with server._rpc_lock:
            resolved = await mcp_tool._resolve_referenced_mcp_ui(
                server, "fixture", "referenced_card"
            )
            # Second resolve must be served from cache, not a new read.
            again = await mcp_tool._resolve_referenced_mcp_ui(
                server, "fixture", "referenced_card"
            )
        return inline, resolved, again, dict(mcp_tool._mcp_ui_resource_html_cache)

    inline, resolved, again, cache = _drive(_check)

    assert inline is None
    assert resolved is not None
    assert resolved["uri"] == REFERENCED_URI
    assert "referenced fixture card" in resolved["html"]
    assert resolved["csp"]["scriptSrc"].startswith("'unsafe-inline'")
    assert again["html"] == resolved["html"]
    assert list(cache) == [("fixture", REFERENCED_URI)]


@pytest.mark.skipif(not FIXTURE_SERVER.exists(), reason="fixture server missing")
def test_plain_tool_yields_no_card_in_either_form():
    async def _check(server):
        async with server._rpc_lock:
            result = await server.session.call_tool("plain_tool", arguments={})
            resolved = await mcp_tool._resolve_referenced_mcp_ui(
                server, "fixture", "plain_tool"
            )
        return mcp_tool._extract_mcp_ui(result, "fixture"), resolved

    inline, resolved = _drive(_check)

    assert inline is None
    assert resolved is None
