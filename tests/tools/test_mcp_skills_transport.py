from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

import pytest


async def _async_result(value):
    return value


def _entry():
    from tools.mcp_skills_protocol import SkillEntry
    body = b"---\nname: remote-demo\ndescription: Remote demo\n---\n"
    uri = "skill://fixture/remote-demo/SKILL.md"
    return SkillEntry.model_validate({
        "uri": uri,
        "frontmatter": {"name": "remote-demo", "description": "Remote demo"},
        "resources": [{"uri": uri, "digest": "sha256:" + hashlib.sha256(body).hexdigest(), "size": len(body)}],
    })


def test_client_requests_skills_extension_only_when_opted_in():
    from tools.mcp_tool import MCPServerTask
    from tools.mcp_skills_protocol import SKILLS_EXTENSION

    server = MCPServerTask("fixture")
    server._config = {"skills": {"enabled": True}}
    assert server._session_kwargs()["extensions"] == {SKILLS_EXTENSION: {}}
    server._config = {"skills": {"enabled": False}}
    assert "extensions" not in server._session_kwargs()


def test_skills_opt_in_disables_schema_only_lazy_start():
    from tools.mcp_tool_discovery import _resolve_server_lazy
    assert _resolve_server_lazy("fixture", {"lazy": True, "skills": {"enabled": True}}) is False
    assert _resolve_server_lazy("fixture", {"lazy": True}) is True


def test_startup_discovery_publishes_metadata_without_resource_read(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_tool import MCPServerTask
    from tools.mcp_skills_protocol import SkillsListResult
    from tools.mcp_skills_registry import clear_runtime_state, pin_session

    class Session:
        def __init__(self):
            self.methods = []

        async def send_request(self, request, adapter):
            self.methods.append(request.method)
            return SkillsListResult(skills=[_entry()])

        async def read_resource(self, uri):
            raise AssertionError("startup discovery fetched a body")

    async def run():
        server = MCPServerTask("fixture")
        server._config = {"skills": {"enabled": True}}
        server.initialize_result = SimpleNamespace(capabilities=SimpleNamespace(
            extensions={"io.modelcontextprotocol/skills": {}}))
        server.session = Session()
        await server._discover_skills()
        return server

    clear_runtime_state()
    server = asyncio.run(run())
    assert server.session is not None
    assert server.session.methods == ["skills/list"]
    rows = pin_session(tmp_path, "session-a")
    assert rows[0]["frontmatter"]["description"] == "Remote demo"
    clear_runtime_state()


def test_nonadvertising_server_skips_extension_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_tool import MCPServerTask

    class Session:
        async def send_request(self, request, adapter):
            raise AssertionError("nonadvertising server received extension request")

    async def run():
        server = MCPServerTask("fixture")
        server._config = {"skills": {"enabled": True}}
        server.initialize_result = SimpleNamespace(capabilities=SimpleNamespace(extensions={}))
        server.session = Session()
        await server._discover_skills()
        return server

    server = asyncio.run(run())
    assert server._skills_catalog == ()
    assert server._skills_diagnostic is not None
    assert "did not advertise" in server._skills_diagnostic


def test_successful_empty_list_retains_point_get_readiness(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_tool import MCPServerTask
    from tools.mcp_skills_protocol import SkillsListResult

    class Session:
        async def send_request(self, request, adapter):
            assert request.method == "skills/list"
            return SkillsListResult(skills=[])

    async def run():
        server = MCPServerTask("fixture")
        server._config = {"skills": {"enabled": True}}
        server.initialize_result = SimpleNamespace(capabilities=SimpleNamespace(
            extensions={"io.modelcontextprotocol/skills": {}}))
        server.session = Session()
        await server._discover_skills()
        return server

    server = asyncio.run(run())
    assert server._skills_catalog == ()
    assert server._skills_local_opt_in is True
    assert server._skills_advertised is True
    assert server._skills_list_completed is True
    assert server._skills_connected is True
    assert server._skills_ready_session is server.session
    assert server._skills_ready_epoch == server._skills_epoch
    server._deregister_tools()
    assert server._skills_connected is False
    assert server._skills_list_completed is False
    assert server._skills_ready_session is None
    assert server._skills_ready_epoch is None


def test_server_deregistration_removes_live_catalog_for_new_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_tool import MCPServerTask
    from tools.mcp_skills_registry import clear_runtime_state, pin_session, publish_live_catalog

    clear_runtime_state()
    publish_live_catalog(tmp_path, "fixture", "config-a", [_entry()])
    server = MCPServerTask("fixture")
    server._skills_home = str(tmp_path)
    server._deregister_tools()
    assert pin_session(tmp_path, "new-session") == ()
    clear_runtime_state()


def test_clean_rediscovery_removes_stale_live_catalog_for_every_nonpublish_outcome(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_tool import MCPServerTask
    from tools.mcp_skills_registry import clear_runtime_state, pin_session, publish_live_catalog

    class FailingSession:
        async def send_request(self, request, adapter):
            raise RuntimeError("refresh failed")

    async def refresh(mode):
        server = MCPServerTask("fixture")
        server._skills_home = str(tmp_path)
        server._skills_config_fingerprint = "config-a"
        server._config = {"skills": {"enabled": mode != "disabled"}}
        extensions = {"io.modelcontextprotocol/skills": {}} if mode == "failed" else {}
        server.initialize_result = SimpleNamespace(capabilities=SimpleNamespace(extensions=extensions))
        server.session = FailingSession()
        await server._discover_skills()
        return server

    for index, mode in enumerate(("unadvertised", "failed", "disabled")):
        clear_runtime_state()
        publish_live_catalog(tmp_path, "fixture", "config-a", [_entry()])
        server = asyncio.run(refresh(mode))
        assert server._skills_catalog == ()
        assert pin_session(tmp_path, f"new-session-{index}") == ()
    clear_runtime_state()


def test_reconnect_without_skills_extension_revokes_prior_generation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_skills_cache as cache
    from tools.mcp_skills_registry import clear_runtime_state, pin_session, publish_live_catalog
    from tools.mcp_tool import MCPServerTask

    class NewSession:
        async def send_request(self, request, adapter):
            raise AssertionError("replacement without Skills extension received skills/list")

    async def run():
        server = MCPServerTask("fixture")
        server._config = {"skills": {"enabled": True}}
        server.session = object()
        server._skills_home = str(tmp_path)
        server._skills_config_fingerprint = "config-a"
        server._skills_local_opt_in = True
        server._skills_advertised = True
        server._skills_list_completed = True
        server._skills_connected = True
        server._skills_epoch = 8
        server._negotiate_session = lambda *_args: _async_result(
            SimpleNamespace(capabilities=SimpleNamespace(extensions={})))
        server._discover_tools = lambda: _async_result(None)
        server._wait_for_lifecycle_event = lambda: _async_result("shutdown")
        await server._serve_session(NewSession(), 1)
        return server

    clear_runtime_state()
    publish_live_catalog(tmp_path, "fixture", "config-a", [_entry()])
    server = asyncio.run(run())
    assert pin_session(tmp_path, "post-drop") == ()
    assert server._skills_local_opt_in is True
    assert server._skills_advertised is False
    assert server._skills_list_completed is False
    assert server._skills_connected is False
    assert server._skills_ready_session is None
    assert server._skills_ready_epoch is None
    with pytest.raises(RuntimeError, match="extension-ready"):
        cache._capture_source("fixture", server, tmp_path)
    clear_runtime_state()


def test_reconnect_skills_list_failure_revokes_prior_generation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_skills_cache as cache
    from tools.mcp_skills_protocol import SKILLS_EXTENSION
    from tools.mcp_skills_registry import clear_runtime_state, pin_session, publish_live_catalog
    from tools.mcp_tool import MCPServerTask

    class NewSession:
        def __init__(self):
            self.methods = []

        async def send_request(self, request, adapter):
            self.methods.append(request.method)
            raise RuntimeError("replacement list failed")

    async def run():
        server = MCPServerTask("fixture")
        server._config = {"skills": {"enabled": True}}
        server.session = object()
        server._skills_home = str(tmp_path)
        server._skills_config_fingerprint = "config-a"
        server._skills_local_opt_in = True
        server._skills_advertised = True
        server._skills_list_completed = True
        server._skills_connected = True
        server._skills_epoch = 11
        server._negotiate_session = lambda *_args: _async_result(SimpleNamespace(
            capabilities=SimpleNamespace(extensions={SKILLS_EXTENSION: {}})))
        server._discover_tools = lambda: _async_result(None)
        server._wait_for_lifecycle_event = lambda: _async_result("shutdown")
        session = NewSession()
        await server._serve_session(session, 1)
        return server, session

    clear_runtime_state()
    publish_live_catalog(tmp_path, "fixture", "config-a", [_entry()])
    server, session = asyncio.run(run())
    assert session.methods == ["skills/list"]
    assert pin_session(tmp_path, "post-failure") == ()
    assert server._skills_local_opt_in is True
    assert server._skills_advertised is True
    assert server._skills_list_completed is False
    assert server._skills_connected is False
    assert server._skills_ready_session is None
    assert server._skills_ready_epoch is None
    with pytest.raises(RuntimeError, match="extension-ready"):
        cache._capture_source("fixture", server, tmp_path)
    clear_runtime_state()
