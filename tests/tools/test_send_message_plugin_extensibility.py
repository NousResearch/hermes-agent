"""Cross-surface regressions for standalone platform send extensibility (#64900)."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from gateway.platform_registry import PlatformEntry, platform_registry
from tools.send_message_tool import resolve_send_target, send_message_tool


@pytest.fixture
def plugin_platform():
    name = "fmsg-ext-test"
    seen: list[dict] = []

    def parser(ref: str):
        normalized = ref.strip().lower()
        if normalized.startswith("@") and "@" in normalized[1:]:
            return normalized, None
        return None

    def validator(ref: str):
        return not ref.endswith("@blocked.example")

    async def handler(args, chat_id, platform_name, pconfig):
        seen.append({"args": dict(args), "chat_id": chat_id, "platform": platform_name})
        return {"success": True, "platform": platform_name, "chat_id": chat_id}

    entry = PlatformEntry(
        name=name,
        label="Fixture Message",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
        parse_target_ref_fn=parser,
        validate_target_ref_fn=validator,
        send_message_handler=handler,
    )
    platform_registry.register(entry)
    try:
        yield name, entry, seen
    finally:
        platform_registry.unregister(name)


def _config_for(name: str):
    platform = Platform(name)
    pconfig = SimpleNamespace(enabled=True, token=None, extra={})
    return platform, pconfig, SimpleNamespace(
        platforms={platform: pconfig},
        get_home_channel=lambda _platform: None,
    )


def test_platform_parser_normalizes_and_validator_rejects(plugin_platform):
    name, _entry, _seen = plugin_platform
    assert resolve_send_target(name, "  @Alice@Example.COM  ") == (
        "@alice@example.com",
        None,
        None,
    )
    chat_id, thread_id, error = resolve_send_target(
        name, "@alice@blocked.example"
    )
    assert chat_id is None
    assert thread_id is None
    assert error == f"Invalid target '@alice@blocked.example' on {name}"


def test_registered_plugin_rejects_unrecognized_opaque_target(plugin_platform):
    name, _entry, seen = plugin_platform

    with patch("gateway.channel_directory.resolve_channel_name", return_value=None):
        chat_id, thread_id, error = resolve_send_target(
            name, "dm:opaque-recipient"
        )

    assert chat_id is None
    assert thread_id is None
    assert "plugin parser did not recognize it" in error
    assert seen == []


def test_plugin_parser_failures_are_diagnosable_without_leaking_exception(plugin_platform):
    name, entry, _seen = plugin_platform

    def broken_parser(_ref):
        raise RuntimeError("credential-shaped plugin detail")

    entry.parse_target_ref_fn = broken_parser
    assert resolve_send_target(name, "recipient") == (
        None,
        None,
        f"Target parser failed for platform '{name}'",
    )

    entry.parse_target_ref_fn = lambda _ref: {"chat_id": "wrong-shape"}
    assert resolve_send_target(name, "recipient") == (
        None,
        None,
        f"Target parser for platform '{name}' returned an invalid result",
    )


def test_plugin_validator_custom_diagnostic_blocks_delivery(plugin_platform):
    name, entry, seen = plugin_platform
    entry.validate_target_ref_fn = lambda _chat_id: "recipient is outside the allowlist"

    with patch("gateway.channel_directory.resolve_channel_name", return_value=None):
        chat_id, thread_id, error = resolve_send_target(name, "@alice@example.com")

    assert chat_id is None
    assert thread_id is None
    assert error == (
        f"Invalid target '@alice@example.com' on {name}: "
        "recipient is outside the allowlist"
    )
    assert seen == []


@pytest.mark.parametrize("async_handler", [False, True])
def test_host_send_honors_sync_and_async_plugin_handlers(plugin_platform, async_handler):
    name, entry, seen = plugin_platform
    platform, pconfig, config = _config_for(name)

    if not async_handler:
        def sync_handler(args, chat_id, platform_name, pconfig):
            seen.append({"args": dict(args), "chat_id": chat_id, "platform": platform_name})
            return {"success": True, "platform": platform_name, "chat_id": chat_id}
        entry.send_message_handler = sync_handler

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        result = json.loads(send_message_tool({
            "target": f"{name}:@Alice@Example.COM",
            "message": "hello",
            "subject": "greeting",
        }))

    assert result["success"] is True
    assert result["platform"] == name
    assert result["chat_id"] == "@alice@example.com"
    assert seen[-1]["args"]["subject"] == "greeting"


def test_cli_and_cron_share_plugin_target_normalization(plugin_platform, monkeypatch, capsys):
    from cron.scheduler_delivery import _resolve_single_delivery_target
    from hermes_cli.send_cmd import cmd_send

    name, _entry, _seen = plugin_platform
    _platform, _pconfig, config = _config_for(name)
    args = SimpleNamespace(
        list_targets=False,
        to=f"{name}:@Alice@Example.COM",
        message="hello",
        file=None,
        subject=None,
        json=True,
        quiet=False,
    )

    monkeypatch.setattr("hermes_cli.send_cmd._load_hermes_env", lambda: None)
    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("gateway.mirror.mirror_to_session", return_value=True), \
         pytest.raises(SystemExit) as exc:
        cmd_send(args)
    assert exc.value.code == 0
    assert json.loads(capsys.readouterr().out)["chat_id"] == "@alice@example.com"

    cron_target = _resolve_single_delivery_target(
        {"name": "fixture"}, f"{name}:@Alice@Example.COM"
    )
    assert cron_target == {
        "platform": name,
        "chat_id": "@alice@example.com",
        "thread_id": None,
        "_resolved_from": "explicit",
    }


def test_send_message_remains_host_only(plugin_platform):
    from tools.registry import registry

    assert registry.get_entry("send_message") is None


def test_force_reload_unregisters_profile_owned_platform(plugin_platform, monkeypatch):
    from hermes_cli.plugins import PluginManager

    name, _entry, _seen = plugin_platform
    manager = PluginManager()
    manager._plugin_platform_names.add(name)
    manager._discovered = True
    monkeypatch.setattr(manager, "_discover_and_load_inner", lambda: None)

    manager.discover_and_load(force=True)

    assert platform_registry.get(name) is None
    assert name not in manager._plugin_platform_names


def test_fresh_process_real_plugin_fixture_covers_host_send_and_cron(tmp_path):
    """A standalone directory plugin is visible to host-driven send paths."""
    home = tmp_path / "home"
    plugin = home / "plugins" / "fmsg-fixture"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: fmsg-fixture\nversion: 0.1.0\ndescription: fixture\nkind: platform\n"
    )
    (home / "config.yaml").write_text("plugins:\n  enabled:\n    - fmsg-fixture\n")
    (plugin / "__init__.py").write_text(
        "async def _send(args, chat_id, platform_name, pconfig):\n"
        "    return {'success': True, 'platform': platform_name, 'chat_id': chat_id}\n"
        "def _parse(ref):\n"
        "    ref = ref.strip().lower()\n"
        "    return (ref, None) if ref.startswith('@') and '@' in ref[1:] else None\n"
        "def register(ctx):\n"
        "    ctx.register_platform(name='fmsg', label='Fmsg', "
        "adapter_factory=lambda cfg: None, check_fn=lambda: True, "
        "parse_target_ref_fn=_parse, send_message_handler=_send)\n"
    )
    script = r'''
import json
from types import SimpleNamespace
from unittest.mock import patch
from hermes_cli.plugins import discover_plugins
from gateway.config import Platform
from tools.registry import registry
from tools.send_message_tool import send_message_tool

discover_plugins()
platform = Platform("fmsg")
pconfig = SimpleNamespace(enabled=True, token=None, extra={})
config = SimpleNamespace(platforms={platform: pconfig}, get_home_channel=lambda p: None)
with patch("gateway.config.load_gateway_config", return_value=config), \
     patch("tools.interrupt.is_interrupted", return_value=False), \
     patch("gateway.mirror.mirror_to_session", return_value=True):
    host_send = json.loads(send_message_tool({"target": "fmsg:@Alice@Example.COM",
                                              "message": "hello", "subject": "hi"}))
from cron.scheduler_delivery import _resolve_single_delivery_target
cron = _resolve_single_delivery_target({}, "fmsg:@Alice@Example.COM")
print(json.dumps({"host_send": host_send, "cron": cron,
                  "model_registered": registry.get_entry("send_message") is not None}))
'''
    env = dict(os.environ)
    env.update({
        "HERMES_HOME": str(home),
        "HERMES_KANBAN_TASK": "fixture",
        "PYTHONPATH": os.getcwd(),
    })
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["host_send"]["chat_id"] == "@alice@example.com"
    assert payload["cron"]["chat_id"] == "@alice@example.com"
    assert payload["model_registered"] is False
