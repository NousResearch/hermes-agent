"""The standalone Telegram send retries a chunk only when the failure is transient.

``_telegram_retry_delay`` used to look for a bare status digit anywhere in the exception text, so a
Markdown/HTML parse error that happened to quote ``502`` — or a byte offset that read like ``429`` —
bought two backoff sleeps plus two more identical attempts before the plain-text fallback could run
(#146922). The classification the platform adapter already owns decides now: PTB's ``BadRequest`` is
permanent (it subclasses ``NetworkError``) whatever its message says.

PTB is not installed in this environment and the adapter's classifier falls back to matching the PTB
error classes by name when it cannot import them, so these tests carry the class names PTB uses.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools import send_message_senders as senders


class BadRequest(Exception):
    """python-telegram-bot's ``BadRequest``: a Markdown/HTML parse failure raises this."""


class NetworkError(Exception):
    """python-telegram-bot's ``NetworkError``: 5xx and transport failures raise this."""

    retry_after: float | None = None


class TimedOut(NetworkError):
    """python-telegram-bot's ``TimedOut`` — a ``NetworkError`` subclass, as in PTB."""


_PARSE_FAILURE = (
    'Bad Request: can\'t parse entities: Unsupported start tag "b" at byte offset 502 '
    "(https://core.telegram.org/bots/api#formatting-options/429)"
)


def test_a_parse_error_quoting_a_status_digit_is_final():
    """The reported shape: digits in a parse error must not buy a retry (#146922)."""
    assert senders._telegram_retry_delay(BadRequest(_PARSE_FAILURE), 0) is None
    assert senders._telegram_retry_delay(BadRequest(_PARSE_FAILURE), 1) is None


def test_a_parse_error_quoting_flood_control_wording_is_final():
    assert senders._telegram_retry_delay(BadRequest("can't parse entities: too many requests"), 0) is None


@pytest.mark.parametrize("exc", [
    NetworkError("Bad Gateway"),
    NetworkError("502 Bad Gateway"),
    NetworkError("httpx.ConnectError: [Errno 111] Connection refused"),
])
def test_a_transient_transport_failure_still_backs_off(exc):
    assert senders._telegram_retry_delay(exc, 0) == 1.0
    assert senders._telegram_retry_delay(exc, 1) == 2.0


def test_a_raw_socket_error_is_not_retried():
    """It may have failed mid-write, where a resend would duplicate the message."""
    assert senders._telegram_retry_delay(ConnectionResetError("connection reset by peer"), 0) is None


def test_flood_control_still_honours_retry_after():
    exc = NetworkError("Too Many Requests: retry after 7")
    exc.retry_after = 7
    assert senders._telegram_retry_delay(exc, 0) == 7.0


def test_a_timeout_is_never_retried():
    """A timeout may already have reached Telegram, so a retry would duplicate the message."""
    assert senders._telegram_retry_delay(TimedOut("Timed out"), 0) is None
    assert senders._telegram_retry_delay(NetworkError("Request timeout"), 0) is None


class _ParseFailingBot:
    """Bot whose rich-text send never parses; the plain send succeeds."""

    def __init__(self):
        self.modes = []

    async def send_message(self, **kwargs):
        self.modes.append(kwargs.get("parse_mode"))
        if kwargs.get("parse_mode") is not None:
            raise BadRequest(_PARSE_FAILURE)
        return "sent"


@pytest.mark.asyncio
async def test_the_plain_text_fallback_runs_without_burning_the_backoff(monkeypatch):
    """The user-visible half: one rich attempt, then plain text — no 1s+2s of sleeping in between."""
    sleeps = []

    async def _sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(senders, "asyncio", SimpleNamespace(sleep=_sleep))
    bot = _ParseFailingBot()
    assert await senders._telegram_send_text_chunk(bot, 42, "chunk", "MarkdownV2", False, {}) == "sent"
    assert bot.modes == ["MarkdownV2", None]
    assert sleeps == []
