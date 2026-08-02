"""Profile-isolation regressions for the central tool registry."""

from __future__ import annotations

import contextlib
import contextvars
import json
import sys
import threading
import types
from unittest.mock import patch

import pytest

from tools.registry import ToolRegistry, invalidate_check_fn_cache

_HANDLER_MARKER = ""


def _schema(name: str, marker: str) -> dict:
    return {
        "name": name,
        "description": marker,
        "parameters": {"type": "object", "properties": {}},
    }


def _handler_template(args, **kwargs):
    return json.dumps({"marker": _HANDLER_MARKER})


def _handler(module_name: str, marker: str):
    return types.FunctionType(
        _handler_template.__code__,
        {"__name__": module_name, "json": json, "_HANDLER_MARKER": marker},
        name=_handler_template.__name__,
    )


@pytest.fixture
def profile_scope(monkeypatch):
    current = contextvars.ContextVar("test_plugin_profile", default="launch")
    module = types.ModuleType("agent.plugin_profile_scope")

    def canonical(key=None):
        return str(current.get() if key is None else key)

    def current_profile_key():
        return canonical()

    def freeze_profile_key(profile_key=None):
        return canonical(profile_key)

    @contextlib.contextmanager
    def bind_profile_key(profile_key):
        key = canonical(profile_key)
        token = current.set(key)
        try:
            yield key
        finally:
            current.reset(token)

    for name, value in {
        "current_profile_key": current_profile_key,
        "freeze_profile_key": freeze_profile_key,
        "bind_profile_key": bind_profile_key,
    }.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, "agent.plugin_profile_scope", module)
    invalidate_check_fn_cache(all_profiles=True)
    yield module
    invalidate_check_fn_cache(all_profiles=True)


def _register_profile_override(registry, scope, key: str, marker: str, *, check_fn=None):
    module_name = f"hermes_plugins.{key}.tool"
    with scope.bind_profile_key(key):
        registry.register_plugin_override_policy(f"hermes_plugins.{key}", True)
        registry.register(
            name="shared_tool",
            toolset=f"plugin-{key}",
            schema=_schema("shared_tool", marker),
            handler=_handler(module_name, marker),
            check_fn=check_fn,
            override=True,
        )


def _generation(registry, key):
    getter = getattr(registry, "generation", None)
    return getter(key) if getter is not None else registry._generation


def test_builtin_base_and_profile_overlays_isolate_schema_handler_policy_and_generation(profile_scope):
    reg = ToolRegistry()
    reg.register(
        name="shared_tool",
        toolset="core",
        schema=_schema("shared_tool", "builtin"),
        handler=_handler("tools.test_builtin", "builtin"),
    )
    base_generation = _generation(reg, "launch")

    _register_profile_override(reg, profile_scope, "alpha", "alpha")
    alpha_generation = _generation(reg, "alpha")
    _register_profile_override(reg, profile_scope, "beta", "beta")
    beta_generation = _generation(reg, "beta")

    with profile_scope.bind_profile_key("launch"):
        assert reg.get_definitions({"shared_tool"})[0]["function"]["description"] == "builtin"
        assert json.loads(reg.dispatch("shared_tool", {})) == {"marker": "builtin"}
    with profile_scope.bind_profile_key("alpha"):
        assert reg.get_definitions({"shared_tool"})[0]["function"]["description"] == "alpha"
        assert json.loads(reg.dispatch("shared_tool", {})) == {"marker": "alpha"}
    with profile_scope.bind_profile_key("beta"):
        assert reg.get_definitions({"shared_tool"})[0]["function"]["description"] == "beta"
        assert json.loads(reg.dispatch("shared_tool", {})) == {"marker": "beta"}

    assert _generation(reg, "launch") == base_generation
    assert _generation(reg, "alpha") == alpha_generation

    assert hasattr(reg, "clear_profile")
    reg.clear_profile("alpha")
    assert _generation(reg, "beta") == beta_generation
    with profile_scope.bind_profile_key("alpha"):
        assert json.loads(reg.dispatch("shared_tool", {})) == {"marker": "builtin"}
    with profile_scope.bind_profile_key("beta"):
        assert json.loads(reg.dispatch("shared_tool", {})) == {"marker": "beta"}


def test_same_check_callable_is_cached_per_profile_and_invalidated_locally(profile_scope):
    reg = ToolRegistry()
    allowed = {"alpha": True, "beta": False}
    calls = {"alpha": 0, "beta": 0}

    def shared_check():
        key = profile_scope.current_profile_key()
        calls[key] += 1
        return allowed[key]

    _register_profile_override(reg, profile_scope, "alpha", "alpha", check_fn=shared_check)
    _register_profile_override(reg, profile_scope, "beta", "beta", check_fn=shared_check)

    with profile_scope.bind_profile_key("alpha"):
        assert reg.get_definitions({"shared_tool"})
        assert reg.get_definitions({"shared_tool"})
    with profile_scope.bind_profile_key("beta"):
        assert reg.get_definitions({"shared_tool"}) == []
        assert reg.get_definitions({"shared_tool"}) == []
    assert calls == {"alpha": 1, "beta": 1}

    allowed["alpha"] = False
    with profile_scope.bind_profile_key("alpha"):
        invalidate_check_fn_cache()
        assert reg.get_definitions({"shared_tool"}) == []
    with profile_scope.bind_profile_key("beta"):
        assert reg.get_definitions({"shared_tool"}) == []
    assert calls == {"alpha": 2, "beta": 1}


def test_override_authorization_cannot_cross_profile_boundaries(profile_scope):
    reg = ToolRegistry()
    reg.register(
        name="shared_tool",
        toolset="core",
        schema=_schema("shared_tool", "builtin"),
        handler=_handler("tools.test_builtin", "builtin"),
    )
    with profile_scope.bind_profile_key("alpha"):
        reg.register_plugin_override_policy("hermes_plugins.same", True)
        reg.register(
            name="shared_tool", toolset="plugin-alpha",
            schema=_schema("shared_tool", "alpha"),
            handler=_handler("hermes_plugins.same.tool", "alpha"),
            override=True,
        )
    with profile_scope.bind_profile_key("beta"):
        with pytest.raises(PermissionError, match="allow_tool_override"):
            reg.register(
                name="shared_tool", toolset="plugin-beta",
                schema=_schema("shared_tool", "beta"),
                handler=_handler("hermes_plugins.same.tool", "beta"),
                override=True,
            )
        assert json.loads(reg.dispatch("shared_tool", {})) == {"marker": "builtin"}
    with profile_scope.bind_profile_key("alpha"):
        assert json.loads(reg.dispatch("shared_tool", {})) == {"marker": "alpha"}


@pytest.mark.parametrize(
    "namespace",
    [
        "hermes_plugins",
        "hermes_plugins.profile_0123456789abcdef",
    ],
)
@pytest.mark.parametrize("policy_order", [("allowed", "denied"), ("denied", "allowed")])
def test_override_authorization_is_plugin_specific_in_any_policy_load_order(
    profile_scope, namespace, policy_order
):
    reg = ToolRegistry()
    reg.register(
        name="shared_tool",
        toolset="core",
        schema=_schema("shared_tool", "builtin"),
        handler=_handler("tools.test_builtin", "builtin"),
    )
    owners = {
        slug: f"{namespace}.{slug}"
        for slug in ("allowed", "denied")
    }

    with profile_scope.bind_profile_key("alpha"):
        for slug in policy_order:
            reg.register_plugin_override_policy(owners[slug], slug == "allowed")

        with pytest.raises(PermissionError, match="allow_tool_override"):
            reg.register(
                name="shared_tool",
                toolset="plugin-denied",
                schema=_schema("shared_tool", "denied"),
                handler=_handler(f"{owners['denied']}.tool", "denied"),
                override=True,
            )

        reg.register(
            name="shared_tool",
            toolset="plugin-allowed",
            schema=_schema("shared_tool", "allowed"),
            handler=_handler(f"{owners['allowed']}.tool", "allowed"),
            override=True,
        )
        assert json.loads(reg.dispatch("shared_tool", {})) == {"marker": "allowed"}


def test_profile_qualified_denied_plugin_cannot_inherit_deregister_grant(profile_scope):
    reg = ToolRegistry()
    reg.register(
        name="protected",
        toolset="core",
        schema=_schema("protected", "builtin"),
        handler=_handler("tools.test_builtin", "builtin"),
    )
    namespace = "hermes_plugins.profile_0123456789abcdef"

    with profile_scope.bind_profile_key("alpha"):
        reg.register_plugin_override_policy(f"{namespace}.denied", False)
        reg.register_plugin_override_policy(f"{namespace}.allowed", True)
        with patch.object(
            ToolRegistry,
            "_caller_module",
            return_value=f"{namespace}.denied.cleanup",
        ):
            with pytest.raises(PermissionError, match="allow_tool_override"):
                reg.deregister("protected")

        assert json.loads(reg.dispatch("protected", {})) == {"marker": "builtin"}


def test_concurrent_same_name_dispatch_never_crosses_profiles(profile_scope):
    reg = ToolRegistry()
    _register_profile_override(reg, profile_scope, "alpha", "alpha")
    _register_profile_override(reg, profile_scope, "beta", "beta")
    barrier = threading.Barrier(3)
    failures = []

    def worker(key):
        with profile_scope.bind_profile_key(key):
            barrier.wait(timeout=5)
            for _ in range(200):
                marker = json.loads(reg.dispatch("shared_tool", {}))["marker"]
                if marker != key:
                    failures.append((key, marker))

    threads = [threading.Thread(target=worker, args=(key,)) for key in ("alpha", "beta")]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert failures == []


def test_failed_transaction_rollback_does_not_clobber_concurrent_same_object_tool_writer():
    from agent.plugin_profile_scope import (
        bind_profile_key,
        plugin_registration_transaction,
    )

    reg = ToolRegistry()
    shared_handler = _handler("hermes_plugins.alpha.tool", "shared")
    temporary_schema = _schema("shared_tool", "temporary")
    concurrent_schema = _schema("shared_tool", "concurrent")

    with pytest.raises(RuntimeError, match="plugin load failed"):
        with plugin_registration_transaction("alpha"):
            reg.register_plugin_override_policy("hermes_plugins.alpha", True)
            reg.register(
                name="shared_tool",
                toolset="plugin-alpha",
                schema=temporary_schema,
                handler=shared_handler,
            )

            def concurrent_writer():
                with bind_profile_key("alpha"):
                    reg.register(
                        name="shared_tool",
                        toolset="plugin-alpha",
                        schema=concurrent_schema,
                        handler=shared_handler,
                    )

            worker = threading.Thread(target=concurrent_writer)
            worker.start()
            worker.join(timeout=5)
            assert not worker.is_alive()
            raise RuntimeError("plugin load failed")

    with bind_profile_key("alpha"):
        entry = reg.get_entry("shared_tool")
        assert entry is not None
        assert entry.handler is shared_handler
        assert entry.schema["description"] == "concurrent"


def test_bound_snapshot_freezes_schema_check_handler_policy_toolset_and_alias(profile_scope):
    reg = ToolRegistry()

    def old_check():
        return True

    def new_check():
        return False

    with profile_scope.bind_profile_key("alpha"):
        reg.register_plugin_override_policy("hermes_plugins.alpha", True)
        reg.register_toolset_alias("probe", "plugin-old")
        reg.register(
            name="shared_tool",
            toolset="plugin-old",
            schema=_schema("shared_tool", "old"),
            handler=_handler("hermes_plugins.alpha.tool", "old"),
            check_fn=old_check,
        )
        snapshot = reg.capture_profile_snapshot()

        reg.register(
            name="shared_tool",
            toolset="plugin-new",
            schema=_schema("shared_tool", "new"),
            handler=_handler("hermes_plugins.alpha.tool", "new"),
            check_fn=new_check,
            override=True,
        )
        reg.register_toolset_alias("probe", "plugin-new")
        reg.register_plugin_override_policy("hermes_plugins.alpha", False)

        token = reg.bind_profile_snapshot(snapshot)
        try:
            entry = reg.get_entry("shared_tool")
            assert entry is not None
            assert entry.toolset == "plugin-old"
            assert entry.schema["description"] == "old"
            assert entry.check_fn is old_check
            assert json.loads(entry.handler({})) == {"marker": "old"}
            assert reg.get_toolset_alias_target("probe") == "plugin-old"
            assert reg._profile_policy("alpha")["hermes_plugins.alpha"] is True
            assert reg.get_definitions({"shared_tool"})[0]["function"][
                "description"
            ] == "old"
        finally:
            reg.reset_profile_snapshot(token)

        live = reg.get_entry("shared_tool")
        assert live is not None
        assert live.toolset == "plugin-new"
        assert reg.get_toolset_alias_target("probe") == "plugin-new"
        assert reg._profile_policy("alpha")["hermes_plugins.alpha"] is False
        assert reg.get_definitions({"shared_tool"}) == []
