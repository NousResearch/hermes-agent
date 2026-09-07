"""Independent production-path probes using synthetic credentials and peer data only."""

import asyncio
import logging
import os
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest
from mcp import types
from hermes_cli.config import _quote_env_value
from tools import mcp_tool as core, mcp_tool_config as config
from tools import mcp_tool_discovery as discovery, mcp_tool_registration as registration
from tools import mcp_tool_loop as loop
from tools import registry as registry_module


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in (
        "_servers",
        "_server_connect_errors",
        "_server_trust_levels",
        "_tool_read_only_hints",
        "_server_error_counts",
        "_server_breaker_opened_at",
        "_lazy_server_configs",
        "_server_scope_keys",
        "_lazy_server_tool_names",
        "_lazy_server_fingerprints",
        "_mcp_tool_server_names",
    ):
        monkeypatch.setattr(core, name, {})
    for name in ("_server_connecting", "_parallel_safe_servers"):
        monkeypatch.setattr(core, name, set())
    monkeypatch.setattr(registry_module, "registry", registry_module.ToolRegistry())
    monkeypatch.setattr(loop, "_ensure_mcp_loop", lambda: None)
    monkeypatch.setattr(
        loop,
        "_run_on_mcp_loop",
        lambda fn, **kw: asyncio.run(fn() if callable(fn) else fn),
    )
    core._ensure_mcp_sdk()


def generations(tmp_path, value, **extra):
    f = tmp_path / "server.env"
    raw = {
        "url": "https://example.invalid/mcp",
        "env_file": str(f),
        "headers": {"X-Review": "${REVIEW_SECRET}"},
        "skip_preflight": True,
        **extra,
    }
    f.write_text("REVIEW_SECRET=" + _quote_env_value(value) + "\n", encoding="utf-8")
    a = config._resolve_mcp_server_config(raw)
    f.write_text("REVIEW_SECRET=ReviewNextGeneration821\n")
    b = config._resolve_mcp_server_config(raw)
    f.unlink()
    assert value in config._mcp_redaction_values(a)
    assert value not in config._mcp_redaction_values(b)
    assert "REVIEW_SECRET" not in os.environ
    return a, b


def task(cfg):
    t = core.MCPServerTask("final-review")
    assert asyncio.run(t._prepare_run(cfg))
    t.initialize_result = NS(capabilities=NS(tools=NS(), prompts=None, resources=None))
    return t


def test_content_type_redacts_before_lowercase(tmp_path, monkeypatch, caplog):
    secret = "ReviewOpaqueMiXeD493Tail"
    cfg, _ = generations(tmp_path, secret, skip_preflight=False)
    requests = []

    def respond(req):
        requests.append({"method": req.method, "header": req.headers["X-Review"]})
        return httpx.Response(
            200,
            headers={"Content-Type": "application/" + req.headers["X-Review"]},
            content=b"",
            request=req,
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: original(transport=httpx.MockTransport(respond), **kw),
    )
    caplog.set_level(logging.DEBUG, logger="tools.mcp_tool")
    t = core.MCPServerTask("final-review")
    assert not asyncio.run(t._prepare_run(cfg))
    from tools.mcp_tool_errors import _format_connect_error

    assert t._error is not None
    output = _format_connect_error(t._error, t._redaction_values)
    assert requests and all(r["header"] == secret for r in requests)
    assert "Content-Type" in output
    assert secret.lower() not in (caplog.text + output).lower()


@pytest.mark.parametrize("variant", ["plain", "normalized"])
def test_lazy_phantom_removal_log_keeps_config_generation(
    tmp_path, monkeypatch, caplog, variant
):
    secret = "ReviewPhantomOpaque384Tail" + (
        "-suffix" if variant == "normalized" else ""
    )
    cfg, _ = generations(tmp_path, secret, lazy=True)
    entry = {
        "tools": [
            {
                "name": secret,
                "description": "ordinary",
                "inputSchema": {"type": "object"},
            }
        ]
    }
    caplog.set_level(logging.DEBUG, logger="tools.mcp_tool")
    registered = registration._register_from_cache_sync("final-review", cfg, entry)
    assert registered and all(
        registry_module.registry.get_schema(n) for n in registered
    )
    live = task(cfg)
    live._tools = []
    live.session = NS()
    monkeypatch.setattr(discovery, "_connect_server", AsyncMock(return_value=live))
    assert discovery._ensure_lazy_server_connected("final-review")
    assert all(
        registry_module.registry.get_toolset_for_tool(n) is None for n in registered
    )
    assert "phantom cached tool" in caplog.text
    assert "ReviewPhantomOpaque384Tail" not in caplog.text


def test_repr_quote_choice_does_not_split_secret(tmp_path, monkeypatch, caplog):
    secret = "ReviewQuotedOpaque'valueTail"
    cfg, _ = generations(tmp_path, secret)
    t = task(cfg)
    peer_name = 'outside"' + secret
    peer = types.Tool(
        name=peer_name, description="ordinary", inputSchema={"type": "object"}
    )
    t._tools = [peer, peer]
    caplog.set_level(logging.DEBUG, logger="tools.mcp_tool")
    names = registration._register_server_tools("final-review", t, cfg)
    assert len(names) == 1
    assert "duplicate registration candidate" in caplog.text
    assert "ReviewQuotedOpaque" not in caplog.text
