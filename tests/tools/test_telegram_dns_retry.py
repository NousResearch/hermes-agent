"""Standalone Telegram sends retry DNS/connect-phase failures but never timeouts."""

import asyncio

import pytest

from tools import send_message_senders as sms


class ConnectError(Exception):
    pass


@pytest.mark.parametrize("exc", [
    ConnectError("[Errno 11001] getaddrinfo failed"),
    RuntimeError("httpx.ConnectError: All connection attempts failed"),
    RuntimeError("Temporary failure in name resolution"),
])
def test_connect_phase_failures_get_capped_backoff(exc):
    assert sms._telegram_retry_delay(exc, 0) == 1.0
    assert sms._telegram_retry_delay(exc, 5) == 5.0


def test_timeouts_still_never_retried():
    assert sms._telegram_retry_delay(RuntimeError("Timed out"), 0) is None
    assert sms._telegram_retry_delay(RuntimeError("ConnectTimeout: timed out"), 0) is None


def test_existing_5xx_backoff_unchanged():
    assert sms._telegram_retry_delay(RuntimeError("502 Bad Gateway"), 2) == 4.0
    assert sms._telegram_retry_delay(RuntimeError("chat not found"), 0) is None


def test_send_retries_dns_blip_then_succeeds(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr(sms.asyncio, "sleep", no_sleep)

    class Bot:
        calls = 0

        async def send_message(self, **kw):
            Bot.calls += 1
            if Bot.calls < 3:
                raise ConnectError("[Errno 11001] getaddrinfo failed")
            return "sent"

    assert asyncio.run(sms._send_telegram_message_with_retry(Bot(), chat_id=1, text="x")) == "sent"
    assert Bot.calls == 3
