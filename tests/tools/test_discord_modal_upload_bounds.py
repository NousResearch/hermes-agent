# tests/tools/test_discord_modal_upload_bounds.py
"""Resource-bounding tests for the Discord rich-clarify modal upload path.

Covers issue #10: ``InteractivePromptModal.on_submit`` must reject oversized
per-file or aggregate submissions *before* reading their full contents, enforce
the field's ``file_policy`` file-count, leave no partial cache files on
rejection, and never resolve the prompt as successful for a rejected upload.

The tests drive the real modal submission path (``on_submit``) against an
isolated ``HERMES_HOME`` (the autouse ``_isolate_hermes_home`` fixture in
``tests/conftest.py`` redirects ``HERMES_HOME`` to a per-test temp dir).

Skipped when the *real* discord.py is not installed (the view/modal classes
inherit from ``discord.ui.View`` / ``discord.ui.Modal`` at class-definition
time and cannot be constructed under a test stub).
"""

from __future__ import annotations

import os
import sys
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Detect whether real discord.py is available (mirrors the sibling module).
# ---------------------------------------------------------------------------
def _real_discord_available() -> bool:
    mod = sys.modules.get("discord")
    if mod is not None and hasattr(mod, "__file__"):
        return True
    try:
        import discord  # noqa: F401
        return hasattr(discord, "__file__")
    except ImportError:
        return False


from tools.discord_interactive_views import unwrap_modal_children  # noqa: E402

discord = pytest.importorskip("discord")
if not _real_discord_available():
    pytest.skip("discord.py is stubbed by another test module", allow_module_level=True)

_ui = discord.ui

from tools.discord_interactive_views import (  # noqa: E402
    InteractivePromptModal,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _FakeAttachment:
    """Minimal stand-in for ``discord.Attachment`` used by ``on_submit``.

    Exposes the attributes the modal reads (``size``, ``filename``, ``id``,
    ``content_type``) and an async ``read()`` that returns the payload or
    raises to simulate a failed download.
    """

    def __init__(
        self,
        *,
        size: int,
        data: Optional[bytes] = None,
        filename: str = "report.bin",
        content_type: str = "application/octet-stream",
        att_id: int = 1,
        fail_read: bool = False,
    ) -> None:
        self.size = size
        self._data = b"x" * size if data is None else data
        self.filename = filename
        self.content_type = content_type
        self.id = att_id
        self._fail_read = fail_read

    async def read(self) -> bytes:
        if self._fail_read:
            raise OSError("simulated attachment read failure")
        return self._data


def _make_interaction() -> MagicMock:
    """A mock ``discord.Interaction`` for ``on_submit``."""
    interaction = MagicMock()
    interaction.user.id = 42
    interaction.user.display_name = "Tester"
    interaction.response.send_message = AsyncMock()
    interaction.followup.send = AsyncMock()
    return interaction


def _make_modal(
    fields: Optional[List[dict]] = None,
    *,
    prompt_id: str = "test-prompt",
) -> InteractivePromptModal:
    flds = fields or [{"key": "upload", "label": "Upload", "type": "file_upload"}]
    return InteractivePromptModal(
        prompt_id=prompt_id,
        option_index=0,
        modal_spec={"title": "T", "fields": flds},
        original_view=None,
    )


def _file_component(modal: InteractivePromptModal):
    """Return the inner ``FileUpload`` component of ``modal``."""
    for comp in unwrap_modal_children(modal.children):
        if isinstance(comp, _ui.FileUpload):
            return comp
    raise AssertionError("modal has no FileUpload field")


def _set_attachments(modal: InteractivePromptModal, attachments: List[_FakeAttachment]):
    _file_component(modal)._values = list(attachments)


def _uploads_dir() -> str:
    from hermes_constants import get_hermes_home
    return os.path.join(str(get_hermes_home()), "cache", "uploads")


def _cached_files() -> List[str]:
    d = _uploads_dir()
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in os.listdir(d)]


async def _submit(modal: InteractivePromptModal) -> MagicMock:
    """Run ``on_submit`` with a fresh interaction and a spied resolver.

    Returns the ``MagicMock`` standing in for ``resolve_gateway_clarify`` so
    each test can assert whether the prompt was resolved (call count + args).
    """
    interaction = _make_interaction()
    with patch("tools.clarify_gateway.resolve_gateway_clarify") as resolver:
        await modal.on_submit(interaction)
    return resolver


# ===========================================================================
# Valid uploads
# ===========================================================================

class TestValidUploads:
    def test_single_valid_file_cached_and_resolved(self):
        modal = _make_modal()
        _set_attachments(modal, [_FakeAttachment(size=256)])
        resolver = _run_sync(_submit(modal))

        # The prompt was resolved exactly once with a JSON payload.
        assert resolver.call_count == 1
        import json
        payload = json.loads(resolver.call_args.args[1])
        assert payload["status"] == "answered"
        files = payload.get("files") or []
        assert len(files) == 1
        meta = files[0]
        # Established response metadata shape.
        assert meta["field_key"] == "upload"
        assert meta["filename"] == "report.bin"
        assert meta["size"] == 256
        assert meta["content_type"] == "application/octet-stream"
        assert meta["attachment_id"] == "1"
        assert meta["cached_path"]

    def test_valid_file_written_under_active_hermes_home(self):
        modal = _make_modal()
        _set_attachments(modal, [_FakeAttachment(size=128)])
        resolver = _run_sync(_submit(modal))

        import json
        payload = json.loads(resolver.call_args.args[1])
        cached_path = payload["files"][0]["cached_path"]
        assert cached_path.startswith(_uploads_dir())
        assert os.path.isfile(cached_path)
        assert os.path.getsize(cached_path) == 128

    def test_multiple_valid_files_within_limits(self):
        modal = _make_modal(fields=[{
            "key": "docs", "label": "Docs", "type": "file_upload",
            "file_policy": {"max_files": 5, "min_files": 1},
        }])
        _set_attachments(modal, [
            _FakeAttachment(size=64, filename="a.bin", att_id=1),
            _FakeAttachment(size=64, filename="b.bin", att_id=2),
        ])
        resolver = _run_sync(_submit(modal))

        import json
        payload = json.loads(resolver.call_args.args[1])
        assert len(payload["files"]) == 2
        names = sorted(f["filename"] for f in payload["files"])
        assert names == ["a.bin", "b.bin"]
        for f in payload["files"]:
            assert os.path.isfile(f["cached_path"])


# ===========================================================================
# Per-file byte limit
# ===========================================================================

class TestPerFileLimit:
    def test_oversized_per_file_rejected_via_policy(self):
        modal = _make_modal(fields=[{
            "key": "upload", "label": "Upload", "type": "file_upload",
            "file_policy": {"max_bytes": 50},
        }])
        _set_attachments(modal, [_FakeAttachment(size=100)])
        resolver = _run_sync(_submit(modal))

        # Not resolved; nothing cached.
        assert resolver.call_count == 0
        assert _cached_files() == []

    def test_default_per_file_limit_rejects_large_file(self):
        modal = _make_modal()
        # 11 MiB > the 10 MiB default per-file cap.
        _set_attachments(modal, [_FakeAttachment(size=11 * 1024 * 1024)])
        resolver = _run_sync(_submit(modal))

        assert resolver.call_count == 0
        assert _cached_files() == []

    def test_valid_file_under_policy_limit_accepted(self):
        modal = _make_modal(fields=[{
            "key": "upload", "label": "Upload", "type": "file_upload",
            "file_policy": {"max_bytes": 200},
        }])
        _set_attachments(modal, [_FakeAttachment(size=150)])
        resolver = _run_sync(_submit(modal))

        assert resolver.call_count == 1
        assert len(_cached_files()) == 1


# ===========================================================================
# Aggregate byte limit
# ===========================================================================

class TestAggregateLimit:
    def test_aggregate_exceeds_default_rejected(self):
        # Per-file cap raised so each file passes individually; the default
        # aggregate cap (25 MiB) is what trips.
        modal = _make_modal(fields=[{
            "key": "upload", "label": "Upload", "type": "file_upload",
            "file_policy": {"max_files": 5, "max_bytes": 30 * 1024 * 1024},
        }])
        _set_attachments(modal, [
            _FakeAttachment(size=15 * 1024 * 1024, att_id=1),
            _FakeAttachment(size=15 * 1024 * 1024, att_id=2),
        ])
        resolver = _run_sync(_submit(modal))

        assert resolver.call_count == 0
        assert _cached_files() == []

    def test_aggregate_via_max_total_bytes_rejected(self):
        modal = _make_modal(fields=[{
            "key": "upload", "label": "Upload", "type": "file_upload",
            "file_policy": {
                "max_files": 5,
                "max_bytes": 50 * 1024 * 1024,
                "max_total_bytes": 5 * 1024 * 1024,
            },
        }])
        _set_attachments(modal, [_FakeAttachment(size=6 * 1024 * 1024)])
        resolver = _run_sync(_submit(modal))

        assert resolver.call_count == 0
        assert _cached_files() == []


# ===========================================================================
# File-count policy
# ===========================================================================

class TestFileCountPolicy:
    def test_too_many_files_rejected(self):
        modal = _make_modal(fields=[{
            "key": "upload", "label": "Upload", "type": "file_upload",
            "file_policy": {"max_files": 2, "min_files": 0},
        }])
        _set_attachments(modal, [
            _FakeAttachment(size=10, att_id=i) for i in range(3)
        ])
        resolver = _run_sync(_submit(modal))

        assert resolver.call_count == 0
        assert _cached_files() == []

    def test_min_files_not_met_rejected(self):
        modal = _make_modal(fields=[{
            "key": "upload", "label": "Upload", "type": "file_upload",
            "required": True,
            "file_policy": {"max_files": 3, "min_files": 1},
        }])
        # No attachments at all.
        _set_attachments(modal, [])
        resolver = _run_sync(_submit(modal))

        assert resolver.call_count == 0
        assert _cached_files() == []


# ===========================================================================
# Failed reads + partial-cache safety
# ===========================================================================

class TestFailedReadsAndPartialCache:
    def test_failed_read_rejected_and_no_partial_cache(self):
        modal = _make_modal(fields=[{
            "key": "upload", "label": "Upload", "type": "file_upload",
            "file_policy": {"max_files": 5},
        }])
        _set_attachments(modal, [
            _FakeAttachment(size=64, att_id=1),                       # reads OK
            _FakeAttachment(size=64, att_id=2, fail_read=True),       # blows up
        ])
        resolver = _run_sync(_submit(modal))

        # Not resolved; the first (successfully read) file must be purged so
        # no partial cache file survives the rejection.
        assert resolver.call_count == 0
        assert _cached_files() == []

    def test_rejection_leaves_no_cache_files(self):
        modal = _make_modal()
        _set_attachments(modal, [_FakeAttachment(size=20 * 1024 * 1024)])
        resolver = _run_sync(_submit(modal))

        assert resolver.call_count == 0
        assert _cached_files() == []


# ===========================================================================
# Safe, actionable rejection messages
# ===========================================================================

class TestRejectionMessages:
    def _capture(self, modal) -> str:
        interaction = _make_interaction()
        with patch("tools.clarify_gateway.resolve_gateway_clarify"):
            _run_sync(modal.on_submit(interaction))
        # The modal acks the submission (success or rejection) through
        # interaction.response.send_message.
        interaction.response.send_message.assert_called_once()
        return str(interaction.response.send_message.call_args.args[0])

    def test_rejection_message_is_safe_no_paths_or_exceptions(self):
        modal = _make_modal(fields=[{
            "key": "upload", "label": "Upload", "type": "file_upload",
            "file_policy": {"max_files": 1},
        }])
        _set_attachments(modal, [
            _FakeAttachment(size=10, att_id=1),
            _FakeAttachment(size=10, att_id=2),
        ])
        msg = self._capture(modal)

        uploads_dir = _uploads_dir()
        # Must not leak internal cache paths, HERMES_HOME, or exception text.
        assert uploads_dir not in msg
        assert "Traceback" not in msg
        assert "OSError" not in msg
        # Must be actionable: tells the user something was rejected.
        assert "reject" in msg.lower() or "too many" in msg.lower()

    def test_failed_read_rejection_message_is_safe(self):
        modal = _make_modal(fields=[{
            "key": "upload", "label": "Upload", "type": "file_upload",
            "file_policy": {"max_files": 5},
        }])
        _set_attachments(modal, [_FakeAttachment(size=64, fail_read=True)])
        msg = self._capture(modal)

        assert _uploads_dir() not in msg
        assert "Traceback" not in msg
        assert "OSError" not in msg
        assert "read" in msg.lower() or "try again" in msg.lower()

    def test_success_acked_with_confirmation(self):
        modal = _make_modal()
        _set_attachments(modal, [_FakeAttachment(size=32)])
        interaction = _make_interaction()
        with patch("tools.clarify_gateway.resolve_gateway_clarify"):
            _run_sync(modal.on_submit(interaction))
        interaction.response.send_message.assert_called_once()
        msg = str(interaction.response.send_message.call_args.args[0])
        assert "submitted" in msg.lower()


# ===========================================================================
# Config-driven limits + multi-field aggregate + lying-size guard
# ===========================================================================

def _set_field_attachments(modal: InteractivePromptModal, position: int, attachments):
    """Attach files to the Nth FileUpload field (0-indexed among file fields)."""
    fus = [c for c in unwrap_modal_children(modal.children) if isinstance(c, _ui.FileUpload)]
    fus[position]._values = list(attachments)


class TestConfigDrivenLimits:
    def test_lower_configured_per_file_limit_rejects_default_sized_file(self):
        # A 5 MiB file is under the 10 MiB built-in default but over a
        # configured 1 MiB cap → must be rejected, proving the bound comes
        # from config.yaml and not a hardcoded constant.
        modal = _make_modal()
        _set_attachments(modal, [_FakeAttachment(size=5 * 1024 * 1024)])
        with patch(
            "tools.discord_interactive_views._get_modal_upload_limits",
            return_value=(1 * 1024 * 1024, 25 * 1024 * 1024),
        ):
            interaction = _make_interaction()
            with patch("tools.clarify_gateway.resolve_gateway_clarify") as resolver:
                _run_sync(modal.on_submit(interaction))

        assert resolver.call_count == 0
        assert _cached_files() == []

    def test_configured_aggregate_limit_rejects_submission(self):
        modal = _make_modal(fields=[{
            "key": "upload", "label": "Upload", "type": "file_upload",
            "file_policy": {"max_files": 5, "max_bytes": 30 * 1024 * 1024},
        }])
        _set_attachments(modal, [
            _FakeAttachment(size=8 * 1024 * 1024, att_id=1),
            _FakeAttachment(size=8 * 1024 * 1024, att_id=2),
        ])
        # Tighten the aggregate cap to 10 MiB via config; 16 MiB combined
        # passes the per-file (30 MiB) cap but must trip the aggregate bound.
        with patch(
            "tools.discord_interactive_views._get_modal_upload_limits",
            return_value=(30 * 1024 * 1024, 10 * 1024 * 1024),
        ):
            interaction = _make_interaction()
            with patch("tools.clarify_gateway.resolve_gateway_clarify") as resolver:
                _run_sync(modal.on_submit(interaction))

        assert resolver.call_count == 0
        assert _cached_files() == []


class TestMultiFieldAggregate:
    def test_two_file_fields_combined_exceed_aggregate_rejected(self):
        modal = _make_modal(fields=[
            {"key": "a", "label": "A", "type": "file_upload",
             "file_policy": {"max_files": 1, "max_bytes": 30 * 1024 * 1024}},
            {"key": "b", "label": "B", "type": "file_upload",
             "file_policy": {"max_files": 1, "max_bytes": 30 * 1024 * 1024}},
        ])
        # Each file passes its 30 MiB per-file cap; combined 30 MiB exceeds
        # the 25 MiB default aggregate cap → rejected, nothing cached.
        _set_field_attachments(modal, 0, [_FakeAttachment(size=15 * 1024 * 1024, att_id=1)])
        _set_field_attachments(modal, 1, [_FakeAttachment(size=15 * 1024 * 1024, att_id=2)])
        resolver = _run_sync(_submit(modal))

        assert resolver.call_count == 0
        assert _cached_files() == []

    def test_two_file_fields_within_aggregate_both_cached(self):
        modal = _make_modal(fields=[
            {"key": "a", "label": "A", "type": "file_upload",
             "file_policy": {"max_files": 1}},
            {"key": "b", "label": "B", "type": "file_upload",
             "file_policy": {"max_files": 1}},
        ])
        _set_field_attachments(modal, 0, [_FakeAttachment(size=64, filename="a.bin", att_id=1)])
        _set_field_attachments(modal, 1, [_FakeAttachment(size=64, filename="b.bin", att_id=2)])
        resolver = _run_sync(_submit(modal))

        import json
        assert resolver.call_count == 1
        payload = json.loads(resolver.call_args.args[1])
        assert len(payload["files"]) == 2
        assert sorted(f["field_key"] for f in payload["files"]) == ["a", "b"]
        assert len(_cached_files()) == 2


class TestLyingSizeGuard:
    def test_payload_larger_than_reported_size_rejected_after_read(self):
        # Reported size passes Phase 2, but read() returns a payload over the
        # per-file cap → rejected by the post-read guard, no cache file, and
        # the prompt is not resolved.
        modal = _make_modal(fields=[{
            "key": "upload", "label": "Upload", "type": "file_upload",
            "file_policy": {"max_bytes": 200},
        }])
        _set_attachments(modal, [
            _FakeAttachment(size=64, data=b"x" * 4096),
        ])
        resolver = _run_sync(_submit(modal))

        assert resolver.call_count == 0
        assert _cached_files() == []


# ===========================================================================
# Helpers — drive the async on_submit from a sync test context
# ===========================================================================

def _run_sync(coro):
    """Await a coroutine to completion in a fresh event loop."""
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
