"""Registry-declared standalone media routing for plugin platforms (#121864).

A plugin platform that provides ``standalone_sender_fn`` must be able to declare
that its media sends route through that sender — without a core edit naming it.
The declaration lives on ``PlatformEntry.standalone_media`` (filled from
``ctx.register_platform(..., standalone_media=True)``) and ``_send_to_platform``
consults the registry first, keeping ``_PLUGIN_STANDALONE_MEDIA`` as the
backward-compatible default for the four in-tree entries.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.platform_registry import PlatformEntry, platform_registry


class _FakePlatform:
    """Stand-in for the gateway.config.Platform enum; only ``.value`` is read."""

    def __init__(self, value):
        self.value = value


@pytest.fixture
def no_discover(monkeypatch):
    """The routing default re-runs plugin discovery; the entry is registered
    directly here, so the scan is a no-op."""
    import hermes_cli.plugins

    monkeypatch.setattr(hermes_cli.plugins, "discover_plugins", lambda *a, **k: None)


def _register_media_platform(monkeypatch, name="vkfixture"):
    calls: list[dict] = []

    async def fake_sender(pconfig, chat_id, message, *, thread_id=None,
                          media_files=None, force_document=False, **kwargs):
        calls.append({"chat_id": chat_id, "message": message,
                      "media_files": media_files, "kwargs": kwargs})
        return {"success": True, "message_id": "fixture-1"}

    entry = PlatformEntry(
        name=name,
        label="Fixture VK",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
        standalone_sender_fn=fake_sender,
    )
    # Attribute form: predates the constructor kwarg on unpatched trees, so the
    # behaviour tests below fail (RED) there instead of erroring at setup.
    entry.standalone_media = True
    platform_registry.register(entry)
    monkeypatch.setattr(
        platform_registry, "get",
        lambda n, _orig=platform_registry.get: entry if n == name else _orig(n),
    )
    return entry, calls


def test_platform_entry_declares_standalone_media_routing():
    """The registration surface carries the flag ``register_platform`` forwards."""
    async def _sender(*a, **k):
        return {"success": True}

    entry = PlatformEntry(
        name="flagprobe",
        label="Flag Probe",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
        standalone_sender_fn=_sender,
        standalone_media=True,
    )
    assert entry.standalone_media is True
    assert PlatformEntry(
        name="flagprobe-off",
        label="Flag Probe Off",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
    ).standalone_media is False


@pytest.mark.asyncio
async def test_declared_flag_without_sender_falls_back_to_generic_path(monkeypatch, no_discover):
    """``standalone_media=True`` with no ``standalone_sender_fn`` never crashes the
    send path; it falls back to the generic text-only route."""
    from tools.send_message_tool import _send_to_platform

    entry = PlatformEntry(
        name="nosenderfix",
        label="No Sender",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
        standalone_media=True,
    )
    platform_registry.register(entry)
    try:
        result = await _send_to_platform(
            _FakePlatform(entry.name),
            SimpleNamespace(enabled=True, token=None, extra={}),
            "chat-1",
            "",
            media_files=[("/tmp/report.md", False)],
        )
    finally:
        platform_registry.unregister(entry.name)
    assert "only media attachments" in result.get("error", "")


@pytest.mark.asyncio
async def test_registry_declared_platform_delivers_media_only_send(monkeypatch, no_discover):
    """Media-only send reaches the plugin sender instead of failing with
    'had only media attachments'."""
    from tools.send_message_tool import _send_to_platform

    entry, calls = _register_media_platform(monkeypatch)
    try:
        result = await _send_to_platform(
            _FakePlatform(entry.name),
            SimpleNamespace(enabled=True, token=None, extra={}),
            "chat-1",
            "",
            media_files=[("/tmp/report.md", False)],
        )
    finally:
        platform_registry.unregister(entry.name)
    assert result.get("success") is True
    assert calls and calls[0]["media_files"] == [("/tmp/report.md", False)]


@pytest.mark.asyncio
async def test_registry_declared_platform_keeps_captioned_attachment(monkeypatch, no_discover):
    """Captioned send delivers the attachment with no 'omitted' warning."""
    from tools.send_message_tool import _send_to_platform

    entry, calls = _register_media_platform(monkeypatch, name="vkfixture2")
    try:
        result = await _send_to_platform(
            _FakePlatform(entry.name),
            SimpleNamespace(enabled=True, token=None, extra={}),
            "chat-1",
            "weekly report",
            media_files=[("/tmp/report.md", False)],
        )
    finally:
        platform_registry.unregister(entry.name)
    assert result.get("success") is True
    assert "warnings" not in result
    assert calls and calls[0]["media_files"] == [("/tmp/report.md", False)]
