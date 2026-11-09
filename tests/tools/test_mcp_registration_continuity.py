"""Unchanged MCP catalog refresh must not revoke an active run's registry attestation."""
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from tools.registry import ToolRegistry
from tools import mcp_tool_registration as registration
from tools import mcp_tool_handlers as handlers


def test_alias_rediscovery_is_not_a_registry_mutation():
    registry = ToolRegistry()
    registry.register_toolset_alias("test", "mcp-test")
    generation = registry._generation
    registry.register_toolset_alias("test", "mcp-test")
    assert registry._generation == generation


def test_same_live_or_lazy_mcp_catalog_keeps_attestation(tmp_path, monkeypatch):
    registry = ToolRegistry()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    tool = SimpleNamespace(name="ping", description="Read-only fixture", inputSchema={"type":"object","properties":{}})
    def refresh(timeout=10):
        candidates = registration._tool_candidates("continuity_fixture", [tool], lambda _: True, timeout)
        return registration._register_candidates("continuity_fixture", candidates, check_fn=handlers._make_check_fn("continuity_fixture"), scope=lambda: str(tmp_path), lazy=True)
    with patch("tools.registry.registry", registry):
        assert len(refresh()) == 1
        generation = registry._generation
        entry = registry.get_entry("mcp__continuity_fixture__ping")
        assert entry is not None
        assert len(refresh()) == 1
        assert registry._generation == generation
        assert registry.get_entry(entry.name) is entry
        refresh(timeout=11)
        assert registry._generation > generation


def test_changed_schema_or_handler_still_revokes_attestation():
    registry = ToolRegistry()
    handler = lambda _: "ok"
    args: dict[str, Any] = dict(name="mcp__fixture__ping", toolset="mcp-fixture", schema={"name":"mcp__fixture__ping"}, handler=handler)
    registry.register(**args)
    generation = registry._generation
    registry.register(**{**args, "handler": lambda _: "different"})
    assert registry._generation > generation
    generation = registry._generation
    registry.register(**{**args, "schema":{"name":"mcp__fixture__ping","description":"changed"}})
    assert registry._generation > generation


def test_generated_resource_and_prompt_handlers_keep_identity():
    for factory in registration._UTILITY_HANDLER_FACTORIES.values():
        assert factory("fixture", 10) is factory("fixture", 10)
        assert factory("fixture", 10) is not factory("fixture", 11)
    assert handlers._make_check_fn("fixture") is handlers._make_check_fn("fixture")
    assert handlers._make_check_fn("fixture") is not handlers._make_check_fn("other")


def test_noop_refresh_rechecks_availability_without_new_entry():
    from tools.registry import _check_fn_cached
    ready = {"value": False}
    check = lambda: ready["value"]
    registry = ToolRegistry()
    handler = lambda _: None
    kwargs: dict[str, Any] = dict(name="mcp__availability__ping", toolset="mcp-availability", schema={}, handler=handler, check_fn=check)
    registry.register(**kwargs)
    assert _check_fn_cached(check) is False
    generation = registry._generation
    ready["value"] = True
    registry.register(**kwargs)
    assert registry._generation == generation
    assert _check_fn_cached(check) is True


def test_same_catalog_republication_is_atomic_across_sessions():
    from concurrent.futures import ThreadPoolExecutor
    registry = ToolRegistry()
    handler = handlers._make_tool_handler("shared", "ping", 10, native_images=True)
    check = handlers._make_check_fn("shared")
    def republish(_):
        registry.register(name="mcp__shared__ping", toolset="mcp-shared", schema={}, handler=handler, check_fn=check)
        registry.register_toolset_alias("shared", "mcp-shared")
    republish(0)
    generation = registry._generation
    entry = registry.get_entry("mcp__shared__ping")
    assert entry is not None
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(republish, range(32)))
    assert registry._generation == generation
    assert registry.get_entry(entry.name) is entry


def test_non_mcp_registration_retains_replacement_semantics():
    registry = ToolRegistry()
    handler = lambda _: None
    def register():
        registry.register(name="ordinary", toolset="builtin", schema={}, handler=handler)
    register()
    generation = registry._generation
    register()
    assert registry._generation > generation


def test_same_catalog_is_not_confused_with_another_profile(tmp_path):
    registry = ToolRegistry()
    handler = handlers._make_tool_handler("shared_scope", "ping", 10, native_images=True)
    kwargs: dict[str, Any] = dict(name="mcp__shared_scope__ping", toolset="mcp-shared_scope", schema={}, handler=handler)
    registry.register(**kwargs, scope=str(tmp_path / "one"))
    generation = registry._generation
    registry.register(**kwargs, scope=str(tmp_path / "two"))
    assert registry._generation > generation
    assert registry.snapshot_registration(kwargs["name"], scope=str(tmp_path / "one")) is not None
    assert registry.snapshot_registration(kwargs["name"], scope=str(tmp_path / "two")) is not None


async def _unchanged_live_refresh():
    from unittest.mock import AsyncMock
    from tools.mcp_tool import MCPServerTask
    registry = ToolRegistry()
    server = MCPServerTask("refresh_fixture")
    server._config = {}
    server.initialize_result = SimpleNamespace(capabilities=SimpleNamespace(tools=SimpleNamespace(), resources=SimpleNamespace(), prompts=SimpleNamespace()))
    server._tools = [SimpleNamespace(name="ping", description="fixture", inputSchema={})]
    server.session = SimpleNamespace(list_tools=AsyncMock(return_value=SimpleNamespace(tools=server._tools)))
    with patch("tools.registry.registry", registry), patch.object(registration, "_write_schema_cache"):
        server._registered_tool_names = registration._register_server_tools(server.name, server, {})
        assert len(server._registered_tool_names) > len(server._tools)
        generation = registry._generation
        entries = {n: registry.get_entry(n) for n in server._registered_tool_names}
        await server._refresh_tools()
        assert registry._generation == generation
        assert all(registry.get_entry(name) is entry for name, entry in entries.items())


def test_unchanged_tools_list_changed_preserves_native_and_utility_entries():
    import asyncio
    asyncio.run(_unchanged_live_refresh())


def test_handler_factory_normalizes_defaults_and_keyword_order():
    first = handlers._make_tool_handler("canonical", "ping", 10)
    assert first is handlers._make_tool_handler(tool_timeout=10, tool_name="ping", server_name="canonical", native_images=False)


def test_handler_factory_does_not_retain_unloaded_catalogs():
    import gc
    import weakref
    handler = handlers._make_tool_handler("unloaded_fixture", "ping", 10)
    reference = weakref.ref(handler)
    del handler
    gc.collect()
    assert reference() is None


def test_caller_mutating_schema_or_requirements_is_a_real_change():
    registry = ToolRegistry()
    schema = {"name": "mcp__mutable__ping", "parameters": {"type": "object", "properties": {}}}
    requirements = ["FIRST_FIXTURE_KEY"]
    handler = lambda _: None
    kwargs: dict[str, Any] = dict(name=schema["name"], toolset="mcp-mutable", schema=schema, requires_env=requirements, handler=handler)
    registry.register(**kwargs)
    generation = registry._generation
    schema["parameters"]["properties"]["added"] = {"type": "string"}
    registry.register(**kwargs)
    assert registry._generation > generation
    generation = registry._generation
    requirements.append("SECOND_FIXTURE_KEY")
    registry.register(**kwargs)
    assert registry._generation > generation


def test_each_registration_policy_field_still_invalidates_on_change():
    variants = [dict(is_async=True), dict(emoji="new"), dict(description="new"),
                dict(requires_env=["FIXTURE_KEY"]), dict(max_result_size_chars=123),
                dict(execution_timeout=12), dict(check_fn=lambda: False),
                dict(dynamic_schema_overrides=lambda _: {})]
    for changed in variants:
        registry = ToolRegistry()
        kwargs: dict[str, Any] = dict(name="mcp__policy__ping", toolset="mcp-policy", schema={}, handler=lambda _: None)
        registry.register(**kwargs)
        generation = registry._generation
        registry.register(**{**kwargs, **changed})
        assert registry._generation > generation, changed


def test_schema_boolean_numeric_contract_changes_are_not_noops():
    for first, second in [(True, 1), (False, 0), (1, True), (0, False)]:
        for constraint in ("enum", "const"):
            registry = ToolRegistry()
            handler = lambda _: None
            def register(value):
                v = [value] if constraint == "enum" else value
                registry.register(name="mcp__json_types__ping", toolset="mcp-json_types", handler=handler,
                                  schema={"parameters": {"type": "object", "properties": {"nested": {constraint: v}}}})
            register(first)
            generation = registry._generation
            register(second)
            assert registry._generation > generation, (first, second, constraint)
