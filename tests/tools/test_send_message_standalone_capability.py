from unittest.mock import MagicMock, patch

from tools.send_message_tool import supports_standalone_send


def test_native_direct_platforms_are_standalone_capable():
    for platform_name in ("signal", "weixin", "bluebubbles", "qqbot"):
        assert supports_standalone_send(platform_name) is True


def test_registered_standalone_sender_is_capable():
    entry = MagicMock()
    entry.standalone_sender_fn = object()
    with patch("tools.send_message_tool.prepare_send_message_platforms"), patch(
        "gateway.platform_registry.platform_registry.get", return_value=entry
    ):
        assert supports_standalone_send("custom-platform") is True


def test_registered_live_only_platform_is_not_standalone_capable():
    entry = MagicMock()
    entry.standalone_sender_fn = None
    with patch("tools.send_message_tool.prepare_send_message_platforms"), patch(
        "gateway.platform_registry.platform_registry.get", return_value=entry
    ):
        assert supports_standalone_send("live-only") is False


def test_unregistered_platform_is_not_standalone_capable():
    with patch("tools.send_message_tool.prepare_send_message_platforms"), patch(
        "gateway.platform_registry.platform_registry.get", return_value=None
    ):
        assert supports_standalone_send("missing-platform") is False


def test_live_adapter_only_native_platform_is_not_whitelisted():
    with patch("tools.send_message_tool.prepare_send_message_platforms"), patch(
        "gateway.platform_registry.platform_registry.get", return_value=None
    ):
        assert supports_standalone_send("yuanbao") is False
        assert supports_standalone_send("relay") is False


def test_blank_platform_is_not_standalone_capable():
    assert supports_standalone_send("") is False
