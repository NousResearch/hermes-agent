"""Trusted context for plugin commands dispatched through TUI gateway RPCs."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import plugins
from hermes_cli.plugin_invocation import PluginInvocationContextUnavailable
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        module = importlib.import_module("tui_gateway.server")

    methods = dict(module._methods)
    yield module
    module._methods.clear()
    module._methods.update(methods)
    module._sessions.clear()


def test_catalog_and_dispatch_use_live_session_context(server, monkeypatch):
    manager = PluginManager(scope_key="/tmp/hermes-tui-plugin-context-test")
    context = PluginContext(PluginManifest(name="neutral-consumer", source="user"), manager)
    seen = []
    availability_contexts = []

    def available(invocation):
        availability_contexts.append(invocation)
        return (
            invocation.platform == "tui"
            and invocation.session_id == "tui-session"
            and invocation.execution_kind == "root"
        )

    def handler(raw_args):
        invocation = context.invocation
        seen.append(
            (
                raw_args,
                invocation,
                {
                    "profile": invocation.profile,
                    "actor": invocation.authenticated_actor,
                    "target": invocation.target,
                    "origin": invocation.origin,
                },
            )
        )
        return "tui handled"

    context.register_command(
        "context-probe",
        handler,
        availability=available,
    )
    monkeypatch.setattr(plugins, "_ensure_plugins_discovered", lambda: manager)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    server._sessions["rpc-session"] = {
        "session_key": "tui-session",
        "profile_home": None,
    }

    hidden = server._methods["commands.catalog"](1, {})
    visible = server._methods["commands.catalog"](2, {"session_id": "rpc-session"})
    dispatched = server._methods["command.dispatch"](
        3,
        {"session_id": "rpc-session", "name": "context-probe", "arg": "MiXeD"},
    )

    assert "/context-probe" not in dict(hidden["result"]["pairs"])
    assert "/context-probe" in dict(visible["result"]["pairs"])
    assert dispatched["result"] == {"type": "plugin", "output": "tui handled"}
    raw_args, invocation, snapshot = seen.pop()
    assert raw_args == "MiXeD"
    assert snapshot == {
        "profile": "default",
        "actor": None,
        "target": "tui-session",
        "origin": None,
    }
    with pytest.raises(PluginInvocationContextUnavailable, match="expired"):
        _ = invocation.profile
    with pytest.raises(PluginInvocationContextUnavailable):
        _ = context.invocation
    for discovered_context in availability_contexts:
        with pytest.raises(PluginInvocationContextUnavailable, match="expired"):
            _ = discovered_context.session_id


def test_unavailable_registered_command_does_not_dispatch(server, monkeypatch):
    manager = PluginManager(scope_key="/tmp/hermes-tui-unavailable-command-test")
    context = PluginContext(PluginManifest(name="neutral-consumer", source="user"), manager)
    handler = MagicMock()
    context.register_command(
        "context-probe",
        handler,
        availability=lambda invocation: invocation.authenticated_actor is not None,
    )
    monkeypatch.setattr(plugins, "_ensure_plugins_discovered", lambda: manager)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    server._sessions["rpc-session"] = {"session_key": "tui-session", "profile_home": None}

    response = server._methods["command.dispatch"](
        1, {"session_id": "rpc-session", "name": "context-probe", "arg": "do it"}
    )

    assert response["error"]["code"] == 4018
    handler.assert_not_called()
