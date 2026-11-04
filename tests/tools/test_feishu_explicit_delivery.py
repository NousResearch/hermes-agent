"""Explicit delivery reaches the raw API despite a streaming send wrapper (#49364)."""

import asyncio
from importlib import import_module
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("content, message_type", [("Explicit hello", "text"), ("**Explicit report**", "post")])
def test_explicit_plugin_delivery_bypasses_only_the_public_send_hook(monkeypatch, content, message_type):
    from gateway.config import PlatformConfig
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import discover_plugins
    from tools.send_message_tool import _send_to_platform

    discover_plugins()
    sender = platform_registry.get("feishu").standalone_sender_fn
    adapter = import_module(sender.__module__).FeishuAdapter
    requests = []
    hooks = []

    def create(request):
        requests.append(request)
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_raw"))

    async def suppressed(self, *args, **kwargs):
        hooks.append(True)
        return SimpleNamespace(success=True, message_id="om_suppressed")

    client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(create=create))))
    monkeypatch.setattr(adapter, "_build_lark_client", lambda self, domain: client)
    monkeypatch.setattr(adapter, "send", suppressed)
    result = asyncio.run(_send_to_platform("feishu", PlatformConfig(), "oc_chat", content))

    assert result.get("success"), result
    assert result["message_id"] == "om_raw"
    assert not hooks and len(requests) == 1
    assert requests[0].request_body.receive_id == "oc_chat"
    assert requests[0].request_body.msg_type == message_type
