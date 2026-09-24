"""Plugin media routing follows the registration contract, not a platform-name list."""

import asyncio
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform
from gateway.platform_registry import PlatformEntry, platform_registry
from tools.send_message_tool import _send_to_platform


def _entry(name, sender, *, standalone_media=False):
    return PlatformEntry(
        name=name, label="Media plugin", adapter_factory=lambda cfg: None,
        check_fn=lambda: True, standalone_sender_fn=sender,
        standalone_media=standalone_media, max_message_length=5,
    )


def test_declared_media_routes_to_standalone_for_captioned_and_media_only(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    sender = AsyncMock(return_value={"success": True, "media_delivered": True})
    platform_registry.register(_entry("test_media_plugin", sender, standalone_media=True))
    try:
        platform = Platform("test_media_plugin")
        config = SimpleNamespace(extra={})
        files = [("/tmp/report.md", False)]
        result = asyncio.run(_send_to_platform(platform, config, "room", "hello world", media_files=files,
                                               force_document=True))
        assert result["success"] and "warnings" not in result
        media_per_chunk = [call.kwargs["media_files"] for call in sender.await_args_list]
        assert media_per_chunk[:-1] == [None] * (len(media_per_chunk) - 1)
        assert media_per_chunk[-1] == files
        assert sender.await_args_list[-1].kwargs["force_document"] is True

        sender.reset_mock()
        result = asyncio.run(_send_to_platform(platform, config, "room", "", media_files=files))
        assert result["success"] and "warnings" not in result
        sender.assert_awaited_once()
        assert sender.await_args.kwargs["media_files"] == files
    finally:
        platform_registry.unregister("test_media_plugin")


def test_undeclared_plugin_keeps_legacy_media_only_refusal(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    sender = AsyncMock(return_value={"success": True})
    platform_registry.register(_entry("test_text_plugin", sender))
    try:
        result = asyncio.run(_send_to_platform(Platform("test_text_plugin"), SimpleNamespace(extra={}),
                                                "room", "", media_files=[("/tmp/report.md", False)]))
        assert "only media attachments" in result["error"]
        sender.assert_not_awaited()
    finally:
        platform_registry.unregister("test_text_plugin")


def test_declared_media_without_sender_fails_closed(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    platform_registry.register(_entry("test_missing_sender", None, standalone_media=True))
    try:
        result = asyncio.run(_send_to_platform(Platform("test_missing_sender"), SimpleNamespace(extra={}),
                                                "room", "", media_files=[("/tmp/report.md", False)]))
        assert "missing standalone_sender_fn" in result["error"]
    finally:
        platform_registry.unregister("test_missing_sender")


def test_discovered_plugin_declaration_delivers_media_through_host_send(tmp_path):
    """A plugin's public registration flag reaches the real host send path."""
    home = tmp_path / "home"
    plugin = home / "plugins" / "media-fixture"
    plugin.mkdir(parents=True)
    (home / "config.yaml").write_text("plugins:\n  enabled:\n    - media-fixture\n")
    (plugin / "plugin.yaml").write_text(
        "name: media-fixture\nversion: 0.1.0\ndescription: fixture\nkind: platform\n"
    )
    (plugin / "__init__.py").write_text(
        "calls = []\n"
        "async def send(pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False):\n"
        "    calls.append({'chat_id': chat_id, 'message': message, 'media_files': media_files})\n"
        "    return {'success': True, 'media_delivered': bool(media_files)}\n"
        "def register(ctx):\n"
        "    ctx.register_platform(name='media_fixture', label='Media fixture', "
        "adapter_factory=lambda cfg: None, check_fn=lambda: True, "
        "parse_target_ref_fn=lambda ref: (ref, None), "
        "standalone_sender_fn=send, standalone_media=True)\n"
    )
    media = tmp_path / "report.md"
    media.write_text("report")
    script = r'''
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch
from hermes_cli.plugins import discover_plugins
from gateway.config import Platform
from gateway.platform_registry import platform_registry
from tools.send_message_tool import send_message_tool

discover_plugins()
entry = platform_registry.get("media_fixture")
platform = Platform("media_fixture")
pconfig = SimpleNamespace(enabled=True, token=None, extra={})
config = SimpleNamespace(platforms={platform: pconfig}, get_home_channel=lambda p: None)
with patch("gateway.config.load_gateway_config", return_value=config), \
     patch("tools.interrupt.is_interrupted", return_value=False), \
     patch("gateway.mirror.mirror_to_session", return_value=True):
    results = [json.loads(send_message_tool({"target": "media_fixture:room",
                "message": text + "\nMEDIA:" + sys.argv[1]})) for text in ("caption", "")]
print(json.dumps({"results": results, "calls": entry.standalone_sender_fn.__globals__["calls"]}))
'''
    env = dict(os.environ, HERMES_HOME=str(home), PYTHONPATH=os.getcwd())
    completed = subprocess.run(
        [sys.executable, "-c", script, str(media)], cwd=os.getcwd(), env=env,
        text=True, capture_output=True, check=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert len(payload["calls"]) == 2
    assert [call["message"] for call in payload["calls"]] == ["caption", ""]
    assert all(call["chat_id"] == "room" for call in payload["calls"])
    assert all(call["media_files"] == [[str(media.resolve()), False]] for call in payload["calls"])
    assert all(result["success"] and result["media_delivered"] and not result.get("warnings")
               for result in payload["results"])
