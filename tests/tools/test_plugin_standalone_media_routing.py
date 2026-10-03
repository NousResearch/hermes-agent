"""Contract tests for plugin-declared standalone media routing (#121864).

A platform that ships as a plugin used to reach the standalone media path only if core named it
in ``_PLUGIN_STANDALONE_MEDIA``. These tests pin the declared contract instead: the plugin states
the routing facts on ``PlatformEntry``, core honours them without naming the platform, and the
in-tree seed dict keeps working as the fallback.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from gateway.platform_registry import PlatformEntry, platform_registry
from tools.send_message_tool import (
    _PLUGIN_STANDALONE_MEDIA,
    _send_to_platform,
    _standalone_media_route,
)

MEDIA = [("/tmp/report.md", False)]
LONG_TEXT = "chunk " * 4000  # over any platform max length -> several chunks


async def _noop_sender(*args, **kwargs):  # pragma: no cover - replaced in every test
    return {"success": True}


def _entry(name, **kwargs):
    """A PlatformEntry with the minimum a registry lookup needs."""
    return PlatformEntry(name=name, label=name.title(), adapter_factory=lambda cfg: None,
                         check_fn=lambda: True, source="plugin",
                         standalone_sender_fn=kwargs.pop("standalone_sender_fn", _noop_sender), **kwargs)


def _recording_sender(calls):
    async def sender(*args, **kwargs):
        calls.append((args, kwargs))
        return {"success": True}
    return sender


def _send(entry, message, media=(), with_limit=True, **kwargs):
    """Run the real ``_send_to_platform`` against *entry*, recording what its sender was handed."""
    calls = []
    entry = _entry(entry.name, standalone_sender_fn=_recording_sender(calls),
                   standalone_media=entry.standalone_media,
                   standalone_captionable=entry.standalone_captionable,
                   standalone_media_sentinel=entry.standalone_media_sentinel,
                   standalone_pass_force_document=entry.standalone_pass_force_document)
    limit = (lambda n: 200) if with_limit else (lambda n: None)
    get = ((lambda n: entry) if with_limit
           else (lambda n: entry if n == "probe" else None))
    with patch.object(platform_registry, "get", get), \
         patch("tools.send_message_tool._platform_max_length", limit), \
         patch("tools.send_message_tool._plugin_standalone_sender",
               lambda name, label=None, discover=False: (entry.standalone_sender_fn, None)):
        res = asyncio.run(_send_to_platform("probe", SimpleNamespace(), "1", message,
                                            media_files=list(media), **kwargs))
    return res, calls


# ---- the declaration itself -------------------------------------------------------------

def test_platform_entry_defaults_leave_media_routing_off():
    entry = _entry("someplug")
    assert (entry.standalone_media, entry.standalone_captionable,
            entry.standalone_media_sentinel, entry.standalone_pass_force_document) == (False, False, None, False)


def test_register_platform_forwards_the_declaration():
    """``register_platform(**entry_kwargs)`` reaches PlatformEntry without a signature change."""
    from hermes_cli.plugins import PluginContext

    captured = {}

    class _Manager:
        scope_key = None

        def _track_scoped_registration(self, *a, **kw):
            return None

    ctx = PluginContext.__new__(PluginContext)
    ctx.manifest = SimpleNamespace(name="probe-plugin")
    ctx._manager = _Manager()

    with patch.object(platform_registry, "register", lambda entry, **kw: captured.update(entry=entry)), \
         patch.object(platform_registry, "snapshot_registration", lambda *a, **kw: (None, None)):
        ctx.register_platform(
            name="probe", label="Probe", adapter_factory=lambda cfg: None, check_fn=lambda: True,
            standalone_sender_fn=_noop_sender, standalone_media=True, standalone_captionable=True,
            standalone_media_sentinel=[], standalone_pass_force_document=True,
        )
    entry = captured["entry"]
    assert (entry.standalone_media, entry.standalone_captionable,
            entry.standalone_media_sentinel, entry.standalone_pass_force_document) == (True, True, [], True)


# ---- route resolution -------------------------------------------------------------------

def test_seed_dict_still_resolves_for_in_tree_platforms():
    """Nothing registered behind the name: the in-tree seed row is the answer, not None."""
    for name, seeded in _PLUGIN_STANDALONE_MEDIA.items():
        with patch.object(platform_registry, "get", lambda n, _name=name: None):
            assert _standalone_media_route(name) == seeded


def test_unregistered_platform_outside_the_seed_dict_has_no_route():
    with patch.object(platform_registry, "get", lambda n: None):
        assert _standalone_media_route("not-a-platform") is None


def test_declaration_wins_over_the_seed_dict():
    entry = _entry("slack", standalone_media=True, standalone_captionable=True,
                   standalone_media_sentinel=[], standalone_pass_force_document=True)
    with patch.object(platform_registry, "get", lambda n: entry):
        assert _standalone_media_route("slack") == ("Slack", False, True, [], True)


def test_entry_without_the_declaration_does_not_get_the_route():
    """A plugin that merely provides standalone_sender_fn keeps the old behaviour."""
    with patch.object(platform_registry, "get", lambda n: _entry("probe")):
        assert _standalone_media_route("probe") is None


# ---- the send path ----------------------------------------------------------------------

def test_media_send_reaches_the_plugin_sender():
    res, calls = _send(_entry("probe", standalone_media=True), "report", media=MEDIA)
    assert res == {"success": True}
    assert calls[0][1]["media_files"] == MEDIA


def test_media_only_send_is_not_rejected_as_unsupported():
    """Regression: the generic path answered ``... had only media attachments`` for plugins."""
    res, calls = _send(_entry("probe", standalone_media=True), "", media=MEDIA)
    assert "error" not in res
    assert calls[0][1]["media_files"] == MEDIA


def test_text_only_send_does_not_take_the_standalone_route():
    """The declaration routes MEDIA; plain text keeps the generic/registry path."""
    res, calls = _send(_entry("probe", standalone_media=True), "hello")
    assert len(calls) == 1  # the generic registry sender, not the standalone one
    assert calls[0][1]["media_files"] == []


def test_captionable_declaration_sends_one_captioned_media_message():
    res, calls = _send(_entry("probe", standalone_media=True, standalone_captionable=True),
                       "report", media=MEDIA)
    assert len(calls) == 1
    assert calls[0][1]["caption"] == "report"
    assert calls[0][1]["media_files"] == MEDIA


def test_declared_list_sentinel_rides_the_first_chunk():
    """A declared sentinel selects the media-carrying chunk — no platform name in the decision."""
    res, calls = _send(_entry("probe", standalone_media=True, standalone_media_sentinel=[]),
                       LONG_TEXT, media=MEDIA)
    assert len(calls) > 1
    assert calls[0][1]["media_files"] == MEDIA
    assert all(c[1]["media_files"] == [] for c in calls[1:])


def test_declared_none_sentinel_also_rides_the_first_chunk():
    res, calls = _send(_entry("probe", standalone_media=True, standalone_media_sentinel=None),
                       LONG_TEXT, media=MEDIA)
    assert len(calls) > 1
    assert calls[0][1]["media_files"] == MEDIA


def test_pass_force_document_reaches_the_sender_only_when_declared():
    """Undeclared: the kwarg is not invented. Declared: force_document is forwarded as asked."""
    res, calls = _send(_entry("probe", standalone_media=True), "report", media=MEDIA)
    assert "force_document" not in calls[0][1]
    res, calls = _send(_entry("probe", standalone_media=True, standalone_pass_force_document=True),
                       "report", media=MEDIA, force_document=True)
    assert calls[0][1]["force_document"] is True


def test_sender_error_is_returned_unchanged():
    entry = _entry("probe", standalone_media=True)
    with patch.object(platform_registry, "get", lambda n: entry), \
         patch("tools.send_message_tool._plugin_standalone_sender",
               lambda name, label=None, discover=False: (None, {"error": "nope"})):
        res = asyncio.run(_send_to_platform("probe", SimpleNamespace(), "1", "report", media_files=MEDIA))
    assert res == {"error": "nope"}


def test_declared_route_skips_the_plugin_discovery_scan():
    """The registry resolved the entry, so the route never asks the resolver for a fresh scan."""
    seen = {}

    def fake_resolver(platform_name, *, label=None, discover=True):
        seen.update(platform_name=platform_name, label=label, discover=discover)
        return _recording_sender([]), None

    entry = _entry("probe", standalone_media=True)
    with patch.object(platform_registry, "get", lambda n: entry), \
         patch("tools.send_message_tool._plugin_standalone_sender", fake_resolver):
        asyncio.run(_send_to_platform("probe", SimpleNamespace(), "1", "report", media_files=MEDIA))
    assert seen == {"platform_name": "probe", "label": "Probe", "discover": False}


def test_seed_platform_keeps_its_own_chunk_shape():
    """An in-tree seed row keeps its historical shape when nothing declares otherwise.

    ``slack``'s seed row carries ``[]``, whose historical meaning is "media rides the final
    chunk" — the declaration must not repurpose a row the plugin did not write.
    """
    res, calls = _send(_entry("slack"), LONG_TEXT, media=MEDIA)
    assert len(calls) > 1
    assert calls[-1][1]["media_files"] == MEDIA
    assert all(c[1]["media_files"] == [] for c in calls[:-1])


def test_handle_send_end_to_end_honours_the_declaration():
    """The real entry point (``send_message`` action='send') reaches the declared sender.

    Pins the contract the issue reports as broken: a media-only send must not come back as
    ``... had only media attachments``.
    """
    import json
    import tempfile
    from pathlib import Path as _Path
    from tools.send_message_tool import _handle_send

    report = _Path(tempfile.mkdtemp()) / "report.md"
    report.write_text("numbers\n", encoding="utf-8")

    calls = []
    entry = _entry("probe", standalone_media=True, standalone_sender_fn=_recording_sender(calls))
    sent = {}

    def fake_run_async(coro):
        sent["result"] = asyncio.run(coro)
        return sent["result"]

    with patch.object(platform_registry, "get", lambda n: entry), \
         patch("tools.send_message_tool._plugin_standalone_sender",
               lambda name, label=None, discover=False: (entry.standalone_sender_fn, None)), \
         patch("gateway.config.load_gateway_config", lambda: SimpleNamespace()), \
         patch("tools.send_message_tool._resolve_platform_config",
               lambda name, config: ("probe", SimpleNamespace(), entry, None)), \
         patch("tools.send_message_tool._maybe_skip_cron_duplicate_send", lambda *a, **kw: None), \
         patch("tools.send_message_tool._authorize_relay_target", lambda *a, **kw: None), \
         patch("model_tools._run_async", fake_run_async):
        out = json.loads(_handle_send({"target": "probe:1", "message": f"MEDIA:{report}"}))
    assert "error" not in out
    assert calls and calls[0][1]["media_files"] == [(str(report), False)]
    assert sent["result"] == {"success": True}
