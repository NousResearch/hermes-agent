from __future__ import annotations

import json
import logging
import os
import sys

import pytest

from hermes_platform import declaration
from tools.mcp_liveness import parse_liveness

def _decl(tmp_path, *, min_version=None):
    executable = tmp_path / "example-app"
    executable.write_text("fixture", encoding="utf-8")
    raw = {sys.platform: {"presence": "executable", "location": str(executable)}}
    requires = {"app": True}
    if min_version is not None:
        raw[sys.platform]["version"] = {"kind": "plist" if sys.platform == "darwin" else "none"}
        requires["min_version"] = min_version
    return declaration.parse_declaration("Example App", raw, requires, where="test")

def test_parse_liveness_contract(tmp_path):
    # tmp_path is rooted on every host; a literal '/tmp/...' is not (liveness paths are
    # validated against the running OS, so this fixture must stay host-portable).
    path = str(tmp_path / "example.json")
    assert parse_liveness({"kind": "static"}).kind == "static"
    assert parse_liveness({"kind": "interactive_session"}).kind == "interactive_session"
    live = parse_liveness({
        "kind": "server_json",
        "path": path,
        "fields": {"url": "endpoint", "token": "secret", "pid": "process"},
    })
    assert (live.kind, live.path, live.url_field, live.token_field, live.pid_field) == (
        "server_json", path, "endpoint", "secret", "process"
    )
    with pytest.raises(ValueError):
        parse_liveness({"kind": "unknown"})
    defaulted = parse_liveness({"kind": "server_json", "path": path})
    assert (defaulted.url_field, defaulted.token_field, defaulted.pid_field) == ("http", "token", "pid")
    partial = parse_liveness({"kind": "server_json", "path": path, "fields": {"url": "endpoint"}})
    assert (partial.url_field, partial.token_field, partial.pid_field) == ("endpoint", "token", "pid")
    with pytest.raises(ValueError):
        parse_liveness({"kind": "server_json", "path": path, "fields": {"port": "p"}})

@pytest.mark.parametrize("path", [
    "../secrets.json",          # parent traversal against the declaring app's own dirs
    "relative/server.json",     # cwd-relative: resolves wherever the process happens to run
    "http://evil.test/x.json",  # scheme-bearing: not a filesystem path at all
])
def test_parse_liveness_rejects_unrooted_server_json_path(path):
    with pytest.raises(ValueError, match="must be absolute"):
        parse_liveness({"kind": "server_json", "path": path})

@pytest.mark.platforms("posix")
def test_parse_liveness_rejects_win32_path_on_posix():
    # Rootedness is judged against the running OS: a drive-letter path is not absolute here.
    with pytest.raises(ValueError, match="must be absolute"):
        parse_liveness({"kind": "server_json", "path": "C:\\x\\server.json"})

@pytest.mark.platforms("windows")
def test_parse_liveness_accepts_win32_rooted_path_on_windows():
    live = parse_liveness({"kind": "server_json", "path": "C:\\x\\server.json"})
    assert live.path == "C:\\x\\server.json"

def test_location_is_rooted_rejects_win32_unc_and_device_paths():
    # osf is a parameter, so the win32 branch is testable on any host: UNC and
    # device-prefix paths must never count as rooted (a remote share could serve the
    # runtime file -- and its token -- from off-box).
    from hermes_platform.declaration import location_is_rooted
    for path in ("\\\\fileserver\\share\\server.json", "\\\\?\\C:\\x\\server.json",
                 "\\\\.\\pipe\\x", "\\\\host\\share"):
        assert location_is_rooted(path, "win32") is False

def test_parse_liveness_rejects_endpoint_path_key(tmp_path):
    # The portable parser's whitelist is the only thing keeping a portable manifest from
    # smuggling a path tail that bypasses _parse_app_os's endpoint_path regex.
    with pytest.raises(ValueError, match="unknown fields"):
        parse_liveness({"kind": "server_json", "path": str(tmp_path / "x.json"),
                        "endpoint_path": "@evil.example/"})

@pytest.mark.parametrize("fields", [["url"], "endpoint", 7])
def test_parse_liveness_rejects_non_mapping_fields(tmp_path, fields):
    # {**fields} would raise TypeError, escaping the ValueError contract callers catch
    path = str(tmp_path / "server.json")
    with pytest.raises(ValueError, match="must be an object"):
        parse_liveness({"kind": "server_json", "path": path, "fields": fields})

@pytest.mark.parametrize("field", ["url", "token", "pid"])
def test_parse_liveness_rejects_non_identifier_field_names(tmp_path, field):
    path = str(tmp_path / "server.json")  # rooted on both POSIX and win32
    with pytest.raises(ValueError, match="simple key names"):
        parse_liveness({"kind": "server_json", "path": path, "fields": {field: 'a b"c'}})

def test_invalid_registered_liveness_degrades_to_static(monkeypatch):
    import hermes_cli.agent_plugins as agent_plugins
    from tools.mcp_liveness import liveness_for

    monkeypatch.setattr(agent_plugins, "liveness_for", lambda name: {"kind": "server_json"}, raising=False)
    assert liveness_for("example-server").kind == "static"

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

def test_live_endpoint_never_dials_a_hostile_endpoint_path(tmp_path, monkeypatch):
    """E2E through the real transport seam: a registered declaration carrying an
    authority-rewriting endpoint_path must make _live_endpoint fail closed, so the
    bearer token is never handed to a socket."""
    import hermes_cli.agent_plugins as agent_plugins
    from hermes_platform.host import facts
    from hermes_platform.declaration import AppSpec, Declaration, RequiresSpec
    from hermes_platform.resolver.app import AppDef
    from tools.mcp_tool_transport import LiveEndpointUnavailable, _live_endpoint

    runtime = tmp_path / "server.json"
    runtime.write_text(json.dumps({
        "http": "http://127.0.0.1:9/mcp", "token": "bearer-secret", "pid": os.getpid(),
    }), encoding="utf-8")
    exe = tmp_path / "app"
    exe.write_text("", encoding="utf-8")
    hostile = AppDef(
        "app", facts.os_family(), "executable", str(exe),
        liveness_kind="server_json", liveness_path=str(runtime),
        endpoint_path="@169.254.169.254/latest",
    )
    decl = Declaration("example-server", AppSpec({facts.os_family(): hostile}), RequiresSpec(app=True))
    declaration.register("example-server", decl)
    monkeypatch.setattr(agent_plugins, "liveness_for", lambda name: {
        "kind": "server_json", "path": str(runtime),
    }, raising=False)
    try:
        with pytest.raises(LiveEndpointUnavailable):
            _live_endpoint("example-server")
    finally:
        declaration.unregister("example-server")


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
    from tools import mcp_tool, mcp_tool_discovery, mcp_tool_handlers

    decl = _decl(tmp_path)
    declaration.register("example-server", decl)
    monkeypatch.setattr(agent_plugins, "liveness_for", lambda name: {"kind": "static"}, raising=False)
    monkeypatch.setattr(mcp_tool_discovery, "_get_connected_server_for_call", lambda name: None)
    monkeypatch.setattr(mcp_tool, "_bump_server_error", lambda name, **kwargs: None)
    try:
        server, error = mcp_tool_handlers._acquire_call_server("example-server", 0)
    finally:
        declaration.unregister("example-server")
    payload = json.loads(error)
    assert server is None
    assert payload["server"] == "example-server"
    # Static liveness cannot observe the app, so it must not claim the app is not
    # running: the honest state is the missing MCP connection (#119975).
    assert payload["state"] == "hermes_not_connected"
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

def test_running_app_with_live_endpoint_reports_the_missing_connection(tmp_path, monkeypatch):
    """The #119975 report: the app runs and its endpoint answers, only Hermes' MCP connection
    is missing. That must read as a missing connection with a reconnect action — not as
    \"<slug> is not running. Start <slug>\" for an app that IS running."""
    import hermes_cli.agent_plugins as agent_plugins
    from hermes_platform.resolver.core import CheckState
    from tools import mcp_liveness

    class _RunningApp:
        def __init__(self, _definition):
            pass

        def locate(self, _ctx=None):
            return object()

        def probe(self, _resolution, effort=None):
            return type("Probe", (), {
                "running": type("Value", (), {"value": True})(),
                "endpoint": type("Endpoint", (), {"state": CheckState.PRESENT})(),
            })()

    monkeypatch.setattr(mcp_liveness, "AppResolver", _RunningApp)
    monkeypatch.setattr(agent_plugins, "liveness_for", lambda name: {
        "kind": "server_json", "path": str(tmp_path / "runtime.json"),
    }, raising=False)
    decl = _decl(tmp_path)
    declaration.register("example-server", decl)
    try:
        current = mcp_liveness.status("example-server")
    finally:
        declaration.unregister("example-server")

    assert current is not None
    assert current.state == "hermes_not_connected"
    sentence = mcp_liveness.describe(decl, current.availability, current.state)
    assert "MCP connection is missing" in sentence
    assert "is not running" not in sentence
    assert current.user_action.startswith("Reconnect")

def test_static_liveness_cannot_claim_the_app_is_not_running(tmp_path, monkeypatch):
    """Static/unknown liveness kinds have no app probe, so their honest state is the missing
    connection — not a verdict that an app they cannot see is stopped (#119975)."""
    import hermes_cli.agent_plugins as agent_plugins
    from tools import mcp_liveness

    monkeypatch.setattr(agent_plugins, "liveness_for", lambda name: {"kind": "static"}, raising=False)
    decl = _decl(tmp_path)
    declaration.register("example-server", decl)
    try:
        current = mcp_liveness.status("example-server")
    finally:
        declaration.unregister("example-server")

    assert current is not None
    assert current.state == "hermes_not_connected"
    sentence = mcp_liveness.describe(decl, current.availability, current.state)
    assert "MCP connection is missing" in sentence
    assert "is not running" not in sentence

def test_describe_prefers_the_plugin_title_over_the_server_slug(tmp_path):
    """The Plugins tab knows the plugin's catalog title; the sentence should name the app by
    it instead of the declaration's slug (#119975)."""
    from tools import mcp_liveness

    decl = _decl(tmp_path)
    sentence = mcp_liveness.describe(decl, None, "hermes_not_connected", display_name="Example Tools")
    assert sentence.startswith("Example Tools")
    assert "Example App" not in sentence
    assert "MCP connection is missing" in sentence
