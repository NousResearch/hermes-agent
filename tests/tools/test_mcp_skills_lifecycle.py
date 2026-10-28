"""Catalog lifecycle regressions: reconnects, notifications and late discovery results."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.skills_contracts import SkillError, SkillErrorCode
from agent.skills_index_delta import stage_skills_index_note
from tests.fakes.fake_mcp_skills import FakeSkill, FakeSkillsServer, FakeSkillsSender
from tools import mcp_skills_manager as catalogs
from tools.mcp_tool import MCPServerTask


@pytest.fixture
def catalog(monkeypatch):
    remote = FakeSkillsServer([
        FakeSkill(uri="skill://team/removed/SKILL.md", name="removed"),
        FakeSkill(uri="skill://team/kept/SKILL.md", name="kept"),
    ], ttl_ms=600_000)
    manager = catalogs.SkillsOverMcpManager(FakeSkillsSender(remote))
    monkeypatch.setattr(catalogs, "_MANAGER", manager)
    monkeypatch.setattr(catalogs, "current_scope_id", lambda: "")
    server = MCPServerTask("catalog")
    server._config = {"skills": {"enabled": True}}
    server.initialize_result = remote.initialize_result()
    server.session = SimpleNamespace(_negotiated_version="")
    return remote, manager, server


@pytest.mark.asyncio
async def test_reconnect_updates_the_current_index_without_rewriting_the_prompt(catalog, monkeypatch, tmp_path):
    from agent import system_prompt
    from agent.prompt_builder import build_skills_system_prompt

    remote, manager, server = catalog
    await catalogs.publish_catalog_after_connect(server.name, server._config, server)
    def render():
        return build_skills_system_prompt(skills_dir_override=tmp_path)
    prompt = render()
    assert "catalog::team/removed" in prompt
    monkeypatch.setattr(system_prompt, "_skills_prompt", lambda agent: render())
    server._negotiate_session = AsyncMock(return_value=remote.initialize_result())
    server._discover_tools = AsyncMock()
    connected, stop = asyncio.Event(), asyncio.Event()

    async def wait_for_stop():
        connected.set()
        await stop.wait()
        return "shutdown"

    server._wait_for_lifecycle_event = wait_for_stop
    remote.skills = remote.skills[1:]
    connection = asyncio.create_task(server._serve_session(SimpleNamespace(_negotiated_version=""), 5))
    await connected.wait()
    assert {entry.name for entry in manager.entries("")} == {"kept"}
    current = render()
    assert "catalog::team/removed" not in current
    agent = SimpleNamespace(provider="test", api_mode="chat_completions")
    assert stage_skills_index_note(agent, prompt, [])
    assert "No longer available: catalog::team/removed" in agent._skills_index_note
    assert "catalog::team/removed" in prompt
    stop.set()
    await connection
    assert manager.entries("") == []


@pytest.mark.asyncio
async def test_a_new_generation_cannot_reuse_the_previous_rendered_index(catalog, tmp_path):
    from agent.prompt_builder import build_skills_system_prompt

    remote, manager, server = catalog
    await catalogs.publish_catalog_after_connect(server.name, server._config, server)
    first = build_skills_system_prompt(skills_dir_override=tmp_path)
    remote.skills = remote.skills[1:]
    server.session = SimpleNamespace(_negotiated_version="")
    await catalogs.publish_catalog_after_connect(server.name, server._config, server)
    second = build_skills_system_prompt(skills_dir_override=tmp_path)
    assert "catalog::team/removed" in first
    assert "catalog::team/removed" not in second
    assert manager.snapshot(server.name, "").snapshot_revision == 1


@pytest.mark.asyncio
async def test_shared_transport_refreshes_only_its_opted_in_scopes(catalog, monkeypatch):
    from tools.mcp_skills_lifecycle import publish_transport_catalogs, retire_transport_catalogs

    remote, manager, server = catalog
    await catalogs.publish_catalog_after_connect(server.name, server._config, server, scope_id="owner")
    await catalogs.publish_catalog_after_connect(server.name, server._config, server, scope_id="adopter")
    other = MCPServerTask(server.name)
    other._config = server._config
    other.initialize_result = remote.initialize_result()
    other.session = SimpleNamespace(_negotiated_version="")
    await catalogs.publish_catalog_after_connect(other.name, other._config, other, scope_id="other")
    monkeypatch.setattr(catalogs, "current_scope_id", lambda: "unrelated")
    remote.skills = remote.skills[1:]
    server.session = SimpleNamespace(_negotiated_version="")
    await publish_transport_catalogs(server)
    assert len(manager.entries("owner")) == len(manager.entries("adopter")) == 1
    assert len(manager.entries("other")) == 2
    assert manager.entries("unrelated") == []
    retire_transport_catalogs(server)
    assert manager.entries("owner") == manager.entries("adopter") == []
    assert len(manager.entries("other")) == 2


@pytest.mark.asyncio
async def test_disabled_server_does_not_list_skills_on_connect_or_notification(catalog):
    from tools.mcp_skills_lifecycle import publish_transport_catalogs, refresh_transport_catalogs

    remote, manager, server = catalog
    server._config = {}
    await publish_transport_catalogs(server)
    refresh_transport_catalogs(server)
    assert remote.calls == []
    assert manager.published("") == []


@pytest.mark.asyncio
async def test_confirmed_disconnect_fences_an_inflight_catalog(catalog):
    _remote, manager, server = catalog
    await catalogs.publish_catalog_after_connect(server.name, server._config, server)
    original = manager._list_all
    started, release = asyncio.Event(), asyncio.Event()

    async def list_after_disconnect(inst):
        result = await original(inst)
        started.set()
        await release.wait()
        return result

    manager._list_all = list_after_disconnect
    refresh = asyncio.create_task(manager.discover(server.name, ""))
    await started.wait()
    manager.mark_disconnected(server.name, "", "transport closed")
    release.set()
    with pytest.raises(SkillError) as error:
        await refresh
    assert error.value.code is SkillErrorCode.VERSION_CHANGED
    assert manager.published("") == []


@pytest.mark.asyncio
async def test_resource_notification_eventually_refreshes_inside_the_debounce_window(catalog, monkeypatch):
    from tools.mcp_tool_common import _core
    from tests.waiting import async_wait_until

    _core._ensure_mcp_sdk()
    if not _core._MCP_NOTIFICATION_TYPES:
        pytest.skip("MCP notification types unavailable")
    remote, manager, server = catalog
    await catalogs.publish_catalog_after_connect(server.name, server._config, server)
    monkeypatch.setattr(catalogs, "_mcp_loop", asyncio.get_running_loop)
    monkeypatch.setattr(catalogs, "MIN_REFRESH_INTERVAL_SECONDS", 0.01)
    remote.skills = remote.skills[1:]
    notification = _core.ResourceListChangedNotification(method="notifications/resources/list_changed")
    message = notification if isinstance(notification, _core.ServerNotification) else _core.ServerNotification(notification)
    await server._make_message_handler()(message)
    await async_wait_until(lambda: len(manager.entries("")) == 1, timeout=5)
    assert {entry.name for entry in manager.entries("")} == {"kept"}
