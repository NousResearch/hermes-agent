"""Cron/media delivery for LINE must not drop attachments.

The cron delivery layer parses a ``MEDIA:<path>`` tag out of a job's output and
routes the file to the platform's native media API. Which platforms support
that is decided by ``tools.send_message_tool._CHUNKED_ROUTES`` — a platform
missing from that table falls through to the generic text-only path, where the
attachment is silently discarded with a warning::

    "MEDIA attachments were omitted for line; native send_message media delivery
     is currently only supported for telegram, discord, matrix, weixin, signal,
     yuanbao, feishu, whatsapp and slack"

The LINE adapter implements ``send_document`` (a flex message with a download
button), so LINE belongs in that table. This test pins the registration so a
refactor cannot quietly drop it again.
"""

import pytest

from tools.send_message_tool import _CHUNKED_ROUTES, _MEDIA_PLATFORMS_NOTE, _via_adapter_route


def test_line_is_registered_for_media_delivery():
    assert "line" in _CHUNKED_ROUTES, (
        "LINE is missing from _CHUNKED_ROUTES — cron MEDIA attachments are "
        "dropped for LINE (the platform note lists only the registered ones)"
    )


def test_line_media_routes_through_the_adapter():
    """LINE's media path must go through the live adapter (send_document)."""
    assert _CHUNKED_ROUTES["line"][2] is _via_adapter_route


def test_line_route_does_not_require_media():
    """The first tuple element is "media required".

    LINE must stay False so a plain-text cron message (no attachment) still
    goes out through the same route instead of being skipped.
    """
    assert _CHUNKED_ROUTES["line"][0] is False


def test_media_platforms_note_mentions_line():
    """The user-facing warning must list LINE, or it misleads on failure."""
    assert "line" in _MEDIA_PLATFORMS_NOTE


def test_line_adapter_implements_send_document():
    """A registered route is useless without a native document sender."""
    from gateway.platforms.base import BasePlatformAdapter
    from plugins.platforms.line.adapter import LineAdapter

    assert LineAdapter.send_document is not BasePlatformAdapter.send_document, (
        "LineAdapter must override send_document — the base implementation "
        "only reports 'unsupported'"
    )


def test_line_media_files_use_the_file_capability():
    """The adapter must accept the "file" media kind (documents)."""
    from plugins.platforms.line.adapter import _OUTBOUND_MEDIA

    assert "file" in _OUTBOUND_MEDIA, (
        "_OUTBOUND_MEDIA has no 'file' entry — _check_media_file('file', …) "
        "raises KeyError and every document send fails"
    )


def test_media_tokens_do_not_expire_on_a_clock():
    """A daily briefing link must survive being read hours later.

    A 30-minute TTL meant the user opened the link in the evening and got a
    404 with no way to tell "expired" from "never existed". Tokens now live as
    long as the file they point at.
    """
    from plugins.platforms.line.adapter import MEDIA_TOKEN_TTL_SECONDS

    assert MEDIA_TOKEN_TTL_SECONDS == float("inf"), (
        "media token TTL should be permanent (inf) — the file's presence is "
        "the real signal"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
