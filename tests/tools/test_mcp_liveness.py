from __future__ import annotations

import json
import logging
import os
import sys
from types import SimpleNamespace

import pytest

from hermes_platform import declaration
from hermes_platform.resolver.availability import Availability
from tools.mcp_liveness import describe, parse_liveness


def _decl(tmp_path, *, min_version=None):
    executable = tmp_path / "example-app"
    executable.write_text("fixture", encoding="utf-8")
    raw = {sys.platform: {"presence": "executable", "location": str(executable)}}
    requires = {"app": True}
    if min_version is not None:
        raw[sys.platform]["version"] = {"kind": "plist" if sys.platform == "darwin" else "none"}
        requires["min_version"] = min_version
    return declaration.parse_declaration("Example App", raw, requires, where="test")


def test_parse_liveness_contract():
    assert parse_liveness({"kind": "static"}).kind == "static"
    assert parse_liveness({"kind": "interactive_session"}).kind == "interactive_session"
    live = parse_liveness({
        "kind": "server_json",
        "path": "/tmp/example.json",
        "fields": {"url": "endpoint", "token": "secret", "pid": "process"},
    })
    assert (live.kind, live.path, live.url_field, live.token_field, live.pid_field) == (
        "server_json", "/tmp/example.json", "endpoint", "secret", "process"
    )
    with pytest.raises(ValueError, match="unknown liveness kind"):
        parse_liveness({"kind": "unknown"})
    defaulted = parse_liveness({"kind": "server_json", "path": "/tmp/example.json"})
    assert (defaulted.url_field, defaulted.token_field, defaulted.pid_field) == ("http", "token", "pid")
    partial = parse_liveness({"kind": "server_json", "path": "/tmp/example.json", "fields": {"url": "endpoint"}})
    assert (partial.url_field, partial.token_field, partial.pid_field) == ("endpoint", "token", "pid")
    with pytest.raises(ValueError, match="may only override"):
        parse_liveness({"kind": "server_json", "path": "/tmp/example.json", "fields": {"port": "p"}})


def test_invalid_registered_liveness_degrades_to_static(monkeypatch, caplog):
    import hermes_cli.agent_plugins as agent_plugins
    from tools.mcp_liveness import liveness_for

    monkeypatch.setattr(agent_plugins, "liveness_for", lambda name: {"kind": "server_json"}, raising=False)
    caplog.set_level(logging.WARNING)
    assert liveness_for("example-server").kind == "static"
    assert any("invalid liveness declaration" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    ("state", "available", "fragment"),
    [
        ("app_not_running", Availability("available"), "is not running"),
        ("endpoint_unavailable", Availability("available"), "local endpoint is unavailable"),
        ("no_interactive_session", Availability("available"), "interactive desktop session"),
        ("version_too_old", Availability("version_too_old", version="1.2", min_version="2.0"), "version 1.2 is too old"),
        ("missing_app", Availability("missing_app"), "is not installed"),
    ],
)
def test_describe_has_one_state_sentence_and_one_action(tmp_path, state, available, fragment):
    sentence = describe(_decl(tmp_path), available, state)
    assert sentence.startswith("Example App")
    assert fragment in sentence
    assert sentence.count("try again") <= 1


def test_live_endpoint_reloads_file_and_registers_token_before_use(tmp_path, monkeypatch, caplog):
    import hermes_cli.agent_plugins as agent_plugins
    from agent import redact
    from tools.mcp_tool_transport import _live_endpoint

    runtime = tmp_path / "server.json"
    decl = _decl(tmp_path)
    declaration.register("example-server", decl)
    raw = {
        "kind": "server_json",
        "path": str(runtime),
        "fields": {"url": "http", "token": "token", "pid": "pid"},
    }
    monkeypatch.setattr(agent_plugins, "liveness_for", lambda name: raw, raising=False)
    calls = []
    monkeypatch.setattr(redact, "register_vault_redaction_value", calls.append)
    caplog.set_level(logging.DEBUG)
    try:
        runtime.write_text(json.dumps({"http": "http://127.0.0.1:1111", "token": "first-secret", "pid": os.getpid()}))
        first = _live_endpoint("example-server")
        runtime.write_text(json.dumps({"http": "http://127.0.0.1:2222", "token": "second-secret", "pid": os.getpid()}))
        second = _live_endpoint("example-server")
    finally:
        declaration.unregister("example-server")
    assert first == ("http://127.0.0.1:1111/mcp", {"Authorization": "Bearer first-secret"})
    assert second == ("http://127.0.0.1:2222/mcp", {"Authorization": "Bearer second-secret"})
    assert calls == ["first-secret", "second-secret"]
    assert all(secret not in record.getMessage() for record in caplog.records for secret in calls)


def test_runtime_file_without_token_connects_without_authorization(tmp_path, monkeypatch):
    import hermes_cli.agent_plugins as agent_plugins
    from agent import redact
    from tools.mcp_tool_transport import _live_endpoint

    runtime = tmp_path / "server.json"
    declaration.register("example-server", _decl(tmp_path))
    monkeypatch.setattr(agent_plugins, "liveness_for", lambda name: {
        "kind": "server_json",
        "path": str(runtime),
        "fields": {"url": "http", "token": "token", "pid": "pid"},
    }, raising=False)
    calls = []
    monkeypatch.setattr(redact, "register_vault_redaction_value", calls.append)
    try:
        runtime.write_text(json.dumps({"http": "http://127.0.0.1:3333", "pid": os.getpid()}))
        result = _live_endpoint("example-server")
    finally:
        declaration.unregister("example-server")
    assert result is not None
    url, headers = result
    assert url == "http://127.0.0.1:3333/mcp"
    assert "Authorization" not in headers
    assert calls == []


def test_missing_runtime_file_never_falls_back(tmp_path, monkeypatch):
    import hermes_cli.agent_plugins as agent_plugins
    from tools.mcp_tool_transport import LiveEndpointUnavailable, _live_endpoint

    decl = _decl(tmp_path)
    declaration.register("example-server", decl)
    monkeypatch.setattr(agent_plugins, "liveness_for", lambda name: {
        "kind": "server_json",
        "path": str(tmp_path / "missing.json"),
        "fields": {"url": "http", "token": "token", "pid": "pid"},
    }, raising=False)
    try:
        with pytest.raises(LiveEndpointUnavailable):
            _live_endpoint("example-server")
    finally:
        declaration.unregister("example-server")


def test_hydrated_error_shape_for_registered_declaration(tmp_path, monkeypatch):
    import hermes_cli.agent_plugins as agent_plugins
    from tools import mcp_tool_handlers
    from tools import mcp_tool_discovery

    decl = _decl(tmp_path)
    declaration.register("example-server", decl)
    monkeypatch.setattr(agent_plugins, "liveness_for", lambda name: {"kind": "static"}, raising=False)
    monkeypatch.setattr(mcp_tool_discovery, "_get_connected_server_for_call", lambda name: None)
    monkeypatch.setattr(mcp_tool_handlers._core, "_bump_server_error", lambda name: None)
    try:
        server, error = mcp_tool_handlers._acquire_call_server("example-server", 0)
    finally:
        declaration.unregister("example-server")
    payload = json.loads(error)
    assert server is None
    assert payload["server"] == "example-server"
    assert payload["state"] == "app_not_running"
    assert payload["app"]["name"] == "Example App"
    assert payload["user_action"]
    assert payload["retry"] == "after_user_action"


def test_connected_interactive_session_server_is_offerable_from_a_service_session(tmp_path, monkeypatch):
    import hermes_cli.agent_plugins as agent_plugins
    from hermes_platform.host import facts
    from tools import mcp_tool_handlers

    declaration.register("example-server", _decl(tmp_path))
    monkeypatch.setattr(agent_plugins, "liveness_for", lambda name: {"kind": "interactive_session"}, raising=False)
    monkeypatch.setattr(facts, "interactive_session", lambda: False)
    try:
        assert mcp_tool_handlers._declared_app_offerable("example-server") is True
    finally:
        declaration.unregister("example-server")


def test_undeclared_error_text_is_unchanged(monkeypatch):
    from tools import mcp_tool_discovery, mcp_tool_handlers

    monkeypatch.setattr(mcp_tool_discovery, "_get_connected_server_for_call", lambda name: None)
    monkeypatch.setattr(mcp_tool_handlers._core, "_bump_server_error", lambda name: None)
    _server, error = mcp_tool_handlers._acquire_call_server("plain-server", 0)
    assert json.loads(error)["error"] == "MCP server 'plain-server' is not connected"
