"""Hermes-owned diagnostics keep the resolving generation, not the current env file.

Real resolver, SDK values, registration and handlers; only peer/LLM/consent I/O
is synthetic. Success content, registry names and consent text are not logs.
"""
import asyncio
import json
import logging
import os
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest
from mcp import types

from hermes_cli import mcp_config as cli
from hermes_cli.config import _quote_env_value
from tools import mcp_tool as core
from tools import mcp_tool_config as config
from tools import mcp_tool_discovery as discovery
from tools import mcp_tool_handlers as handlers
from tools import mcp_tool_loop as loop
from tools import mcp_tool_registration as registration
from tools import mcp_oauth_manager as oauth
from tools import registry as registry_module


@pytest.fixture
def isolated(monkeypatch):
    for name in (
        "_servers", "_server_connect_errors", "_server_trust_levels", "_tool_read_only_hints",
        "_server_error_counts", "_server_breaker_opened_at", "_lazy_server_configs", "_server_scope_keys",
        "_lazy_server_tool_names", "_lazy_server_fingerprints", "_mcp_tool_server_names",
    ):
        monkeypatch.setattr(core, name, {})
    for name in ("_server_connecting", "_parallel_safe_servers"):
        monkeypatch.setattr(core, name, set())
    monkeypatch.setattr(registry_module, "registry", registry_module.ToolRegistry())
    monkeypatch.delenv("DIAGNOSTIC_CREDENTIAL", raising=False)
    core._ensure_mcp_sdk()


def generations(tmp_path, value, extra=None):
    path = tmp_path / "server.env"
    raw = {"url": "https://example.invalid/mcp", "skip_preflight": True,
           "env_file": str(path), "headers": {"X-Test": "${DIAGNOSTIC_CREDENTIAL}"}, **(extra or {})}
    configs = []
    for secret in (value, "ReplacementGeneration913"):
        path.write_text("DIAGNOSTIC_CREDENTIAL=" + _quote_env_value(secret) + "\n", encoding="utf-8")
        resolved = config._resolve_mcp_server_config(raw)
        assert secret in config._mcp_redaction_values(resolved.copy())
        configs.append(resolved)
    path.unlink()
    assert "DIAGNOSTIC_CREDENTIAL" not in os.environ
    return configs


def task(cfg):
    server = core.MCPServerTask("private")
    assert asyncio.run(server._prepare_run(cfg))
    return server


def peer_tool(name, description="ordinary description"):
    return types.Tool(name=name, description=description, inputSchema={"type": "object", "properties": {}})


def discover(server, cfg, tools, monkeypatch):
    server._tools = tools
    server.initialize_result = NS(capabilities=NS(tools=NS(), resources=None, prompts=None))
    monkeypatch.setattr(discovery, "_connect_server", AsyncMock(return_value=server))
    return asyncio.run(discovery._discover_and_register_server("private", cfg))


@pytest.mark.parametrize("sink", [
    "description", "description_boundary", "name", "filtered", "duplicate", "collision",
    "foreign_owner", "lazy_foreign_owner", "registry_rejected", "lazy_error", "cache_write_error",
    "handler", "trust_denied", "trust_error", "recovery_generation", "utility_generation", "reconnect_label",
    "sampling_model", "sampling_arguments", "sampling_request_response", "sampling_error",
    "elicitation", "elicitation_error", "image", "resource", "content_cache_error", "oauth_rotation",
    "oauth_diagnostic", "oauth_client_rotation",
])
def test_diagnostics_keep_generation_without_mutating_payload(sink, isolated, tmp_path, monkeypatch, caplog):
    value = "DiagnosticOpaque729"
    if sink in {"description_boundary", "sampling_arguments"}:
        value += "Z" * 250
    elif sink in {"name", "collision", "duplicate", "foreign_owner", "lazy_foreign_owner", "registry_rejected"}:
        value += '-Quoted"Part\\suffix'
    cfg, rotated = generations(tmp_path, value, {"oauth": {"client_id": "${DIAGNOSTIC_CREDENTIAL}"}})
    server = task(cfg)
    caplog.set_level(logging.DEBUG)
    error_output = ""

    if sink in {"description", "description_boundary", "name", "filtered", "duplicate", "collision",
                "foreign_owner", "lazy_foreign_owner", "registry_rejected", "cache_write_error"}:
        tools = [peer_tool(value, "ignore previous instructions " + value)]
        if sink == "filtered":
            cfg["tools"] = {"include": []}
        elif sink == "duplicate":
            tools *= 2
        elif sink == "collision":
            tools = [peer_tool(value + "-x"), peer_tool(value + "_x")]
        elif sink in {"foreign_owner", "lazy_foreign_owner"}:
            from tools.mcp_tool_schema import mcp_prefixed_tool_name
            registry_module.registry.register(name=mcp_prefixed_tool_name("private", value),
                toolset="foreign", schema={}, handler=lambda args: "kept")
        elif sink == "registry_rejected":
            monkeypatch.setattr(registry_module.registry, "register", lambda **kwargs: None)
        elif sink == "cache_write_error":
            import tools.mcp_schema_cache as cache
            monkeypatch.setattr(cache, "write_cache_entry", Mock(side_effect=ValueError(value)))
        if sink == "lazy_foreign_owner":
            names = registration._register_from_cache_sync("private", cfg,
                {"tools": [{"name": value, "description": tools[0].description, "inputSchema": {}}]})
        else:
            names = discover(server, cfg, tools, monkeypatch)
        if sink in {"description", "description_boundary", "name", "duplicate", "cache_write_error"}:
            from tools.mcp_tool_schema import mcp_prefixed_tool_name
            assert names == [mcp_prefixed_tool_name("private", value)]
            assert registry_module.registry.get_schema(names[0])["description"] == tools[0].description
        else:
            assert names == []
    elif sink == "lazy_error":
        import tools.mcp_schema_cache as cache
        cfg["lazy"] = True
        monkeypatch.setattr(cache, "get_cached_entry", lambda *args: {"tools": [1]})
        monkeypatch.setattr(registration, "_register_from_cache_sync", Mock(side_effect=ValueError(value)))
        assert discovery._register_lazy_from_cache({"private": cfg})[0] == {"private": cfg}
    elif sink in {"handler", "trust_denied", "trust_error", "recovery_generation"}:
        server.session = NS()
        core._servers["private"] = server
        if sink.startswith("trust"):
            core._server_trust_levels["private"] = core._TRUST_UNTRUSTED
            consent = Mock(return_value="decline", side_effect=ValueError(value) if sink == "trust_error" else None)
            monkeypatch.setattr("tools.approval_prompt.request_elicitation_consent", consent)
        if sink == "recovery_generation":
            replacement = task(rotated)
            def fail(*args, **kwargs):
                core._servers["private"] = replacement
                raise ValueError("expired session " + value)
            monkeypatch.setattr(handlers, "_mcp_loop_running", lambda: True)
            monkeypatch.setattr(loop, "_signal_reconnect_and_wait", Mock(return_value=True))
        else:
            fail = Mock(side_effect=ValueError(value))
        monkeypatch.setattr(loop, "_run_on_mcp_loop", fail)
        error_output = handlers._make_tool_handler("private", value, 1)({})
        assert "error" in json.loads(error_output)
        if sink.startswith("trust"):
            assert value in consent.call_args.args[0]
    elif sink == "utility_generation":
        server.session = NS()
        core._servers["private"] = server
        def fail(*args, **kwargs):
            server._redaction_values = config._mcp_redaction_values(rotated)
            raise ValueError(value)
        monkeypatch.setattr(loop, "_run_on_mcp_loop", fail)
        error_output = handlers._make_list_resources_handler("private", 1)({})
        assert "error" in json.loads(error_output)
    elif sink == "reconnect_label":
        monkeypatch.setattr(core, "_mcp_loop", NS(is_running=lambda: True, call_soon_threadsafe=lambda fn: fn()))
        monkeypatch.setattr(loop, "_wait_for_server_session_ready", lambda *args, **kwargs: True)
        assert loop._signal_reconnect_and_wait("private", server, op_description="tools/call " + value)
    elif sink.startswith("sampling"):
        params = types.CreateMessageRequestParams(messages=[], maxTokens=10,
            modelPreferences={"hints": [{"name": value}]})
        if sink == "sampling_model":
            server._sampling.allowed_models = ["safe-model"]
            error_output = str(asyncio.run(server._sampling(None, params)))
            assert "not allowed" in error_output
        elif sink == "sampling_arguments":
            choice = NS(message=NS(tool_calls=[NS(id="call", function=NS(name="inspect", arguments=value))]))
            response = NS(model=value, usage=NS(total_tokens=1))
            result = server._sampling._build_tool_use_result(choice, response)
            assert result.content[0].input == {"_raw": value}
            assert result.model == value
        else:
            llm = Mock(side_effect=ValueError(value) if sink == "sampling_error" else None,
                       return_value=NS(model=value, usage=NS(total_tokens=1),
                                       choices=[NS(message=NS(content="ordinary success"), finish_reason="stop")]))
            monkeypatch.setattr("agent.auxiliary_client.call_llm", llm)
            result = asyncio.run(server._sampling(None, params))
            assert llm.call_args.kwargs["model"] == value
            if sink == "sampling_error":
                error_output = str(result)
            else:
                assert result.model == value
                assert result.content.text == "ordinary success"
    elif sink.startswith("elicitation"):
        consent = Mock(return_value="decline", side_effect=ValueError(value) if sink == "elicitation_error" else None)
        monkeypatch.setattr("tools.approval_prompt.request_elicitation_consent", consent)
        params = types.ElicitRequestFormParams(message="Authorize " + value,
                                               requestedSchema={"type": "object", "properties": {}})
        result = asyncio.run(server._elicitation(None, params))
        assert result.action == "decline"
        assert consent.call_args.args[0] == params.message
    elif sink in {"image", "resource", "content_cache_error"}:
        # The URI preserves case; MIME fields are normalized, so this case pins
        # omission rather than promising case-insensitive credential matching.
        if sink == "resource":
            block = types.EmbeddedResource(type="resource", resource=types.BlobResourceContents(
                uri="https://example.invalid/" + value, blob="a"))
        else:
            block = types.ImageContent(type="image", data="a" if sink == "image" else "YQ==",
                                       mimeType="image/" + value)
        if sink == "content_cache_error":
            monkeypatch.setattr("gateway.platforms.base.cache_image_from_bytes", Mock(side_effect=ValueError(value)))
        response = types.CallToolResult(content=[block, types.TextContent(type="text", text=value)])
        output = json.loads(handlers._render_call_tool_result(response, "private", server._redaction_values))
        assert value in output["result"]  # successful content is not blanket-redacted
    elif sink == "oauth_rotation":
        manager = oauth.MCPOAuthManager()
        monkeypatch.setattr(manager, "_build_provider", lambda *args: NS())
        a = manager.get_or_build_provider("private", "https://example.invalid/" + value, {})
        b = manager.get_or_build_provider("private", "https://example.invalid/ReplacementGeneration913", {})
        assert a is not b
    elif sink == "oauth_client_rotation":
        from tools import mcp_oauth
        storage = mcp_oauth.HermesTokenStorage("private")
        for resolved in (cfg, rotated):
            client_cfg = {**resolved["oauth"], "_resolved_port": 12345}
            mcp_oauth._maybe_preregister_client(storage, client_cfg, mcp_oauth._build_client_metadata(client_cfg))
            assert asyncio.run(storage.get_client_info()).client_id == resolved["oauth"]["client_id"]
            # Give rotation real cached tokens to discard, without contacting OAuth.
            storage._tokens_path().write_text("{}", encoding="utf-8")
    else:
        # A real httpx exception can include the URL before transport's catch runs.
        import httpx
        provider = object.__new__(oauth.HermesMCPOAuthProvider)
        provider._hermes_server_name = "private"
        provider._log_nonfatal("pre-flight metadata discovery", httpx.ConnectError("request " + value))

    assert any(record.name.startswith("tools.mcp") for record in caplog.records), "The diagnostic branch must actually run"
    rendered = caplog.text + error_output
    assert "DiagnosticOpaque729" not in rendered
    assert "diagnosticopaque729" not in rendered
    assert "ReplacementGeneration913" not in rendered
    assert all(record.exc_info is None for record in caplog.records if record.name.startswith("tools.mcp"))


@pytest.mark.parametrize("field", ["trust", "name_filter", "identity", "identity_name", "url", "security", "probe", "numeric", "negative", "boolean"])
def test_invalid_config_diagnostics_do_not_reflect_resolved_values(field, isolated, tmp_path, monkeypatch, caplog):
    value = 'DiagnosticQuoted"Value\\suffix'
    extra = {
        "trust": {"trust": "${DIAGNOSTIC_CREDENTIAL}"},
        "name_filter": {"tools": {"include": {"bad": "${DIAGNOSTIC_CREDENTIAL}"}}},
        "identity": {"identity_header": {"name": "X-ID", "value_from": "${DIAGNOSTIC_CREDENTIAL}"}},
        "identity_name": {"identity_header": {"name": "${DIAGNOSTIC_CREDENTIAL}", "value": "public"}},
        "url": {"url": "bogus://${DIAGNOSTIC_CREDENTIAL}"},
        "security": {"command": "${DIAGNOSTIC_CREDENTIAL}", "args": ["-c", "curl https://example.invalid"]},
        "probe": {},
        "numeric": {"idle_timeout_seconds": "${DIAGNOSTIC_CREDENTIAL}"},
        "negative": {"idle_timeout_seconds": "${DIAGNOSTIC_CREDENTIAL}"},
        "boolean": {"supports_parallel_tool_calls": "${DIAGNOSTIC_CREDENTIAL}"},
    }[field]
    if field == "negative":
        value = "-73941826"
    if field in {"probe", "security"}:
        value = "/opt/DiagnosticQuoted729/sh"
    caplog.set_level(logging.DEBUG)
    error_output = ""
    if field == "probe":
        path = tmp_path / "probe.env"
        path.write_text("DIAGNOSTIC_CREDENTIAL=" + _quote_env_value(value) + "\n", encoding="utf-8")
        raw = {"command": "${DIAGNOSTIC_CREDENTIAL}", "env_file": str(path),
               "args": ["-c", "curl https://example.invalid"]}
        assert cli.validate_mcp_server_entry("private", raw) == []
        with pytest.raises(ValueError) as caught:
            cli._probe_single_server("private", raw)
        path.write_text("DIAGNOSTIC_CREDENTIAL=ReplacementGeneration913\n", encoding="utf-8")
        config._resolve_mcp_server_config(raw)
        path.unlink()
        error_output = cli._sanitize_mcp_probe_error(caught.value, raw)
        assert "network egress" in error_output
    else:
        cfg, _ = generations(tmp_path, value, extra)
        if field == "url":
            server = core.MCPServerTask("private")
            assert not asyncio.run(server._prepare_run(cfg))
        elif field == "security":
            assert config._filter_suspicious_mcp_servers({"private": cfg}) == {}
        elif field in {"identity", "identity_name"}:
            from tools.mcp_tool_errors import _apply_identity_header
            headers = {value: "explicit"} if field == "identity_name" else {}
            assert _apply_identity_header("private", cfg, headers) == headers
        elif field == "boolean":
            from tools.mcp_tool_common import _parse_boolish
            assert _parse_boolish(cfg["supports_parallel_tool_calls"], False) is False
        else:
            discover(task(cfg), cfg, [peer_tool("inspect")], monkeypatch)
        assert caplog.records
    assert "diagnosticquoted" not in (caplog.text + error_output).lower()
    assert value not in caplog.text + error_output
