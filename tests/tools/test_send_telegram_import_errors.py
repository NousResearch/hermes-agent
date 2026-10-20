"""``_send_telegram`` must not blame python-telegram-bot for a nested import failure.

Live incident: a cron delivery reported
``python-telegram-bot not installed. Run: pip install python-telegram-bot`` while
``import telegram`` returned 22.8 in the very same venv. The body was wrapped in a
blanket ``except ImportError``, so ANY nested optional import failure (the lazily
imported telegram id helpers, ``gateway.platforms.base``) was misreported as a
missing PTB — and the real exception never reached the log.

The three cases below pin the whole contract: a nested failure surfaces itself, a
genuinely absent PTB still reports the install hint, and a healthy PTB no longer
takes the ImportError branch at all.
"""

import asyncio
import builtins

import pytest


def _import_blocker(monkeypatch, blocked_prefixes):
    """Make ``import <prefix>...`` raise, leaving every other import untouched."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        for prefix in blocked_prefixes:
            if name == prefix or name.startswith(prefix + "."):
                raise ImportError(f"simulated missing module: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def _send(token="123:FAKE", chat_id="someone", message="probe"):
    from tools.send_message_senders import _send_telegram

    return asyncio.run(_send_telegram(token, chat_id, message))


def test_nested_import_failure_is_not_blamed_on_python_telegram_bot(monkeypatch):
    """The regression: a nested failure must report itself, not 'PTB not installed'."""
    _import_blocker(monkeypatch, ["plugins.platforms.telegram.telegram_ids"])

    result = _send()

    assert "python-telegram-bot not installed" not in str(result), (
        "a nested import failure was still misattributed to python-telegram-bot"
    )
    assert "simulated missing module" in str(result), (
        f"the real nested failure must reach the caller, got {result!r}"
    )


def test_gateway_base_import_failure_is_not_blamed_on_python_telegram_bot(monkeypatch):
    """Same contract for the other nested import the function performs."""
    _import_blocker(monkeypatch, ["gateway.platforms.base"])

    result = _send()

    assert "python-telegram-bot not installed" not in str(result)
    assert "simulated missing module" in str(result)


def test_missing_python_telegram_bot_still_reports_the_install_hint(monkeypatch):
    """The narrowed probe must keep diagnosing a genuinely absent PTB."""
    _import_blocker(monkeypatch, ["telegram"])

    result = _send()

    assert "python-telegram-bot not installed" in str(result), (
        f"an actually missing PTB must keep its install hint, got {result!r}"
    )


def test_healthy_python_telegram_bot_does_not_take_the_import_error_branch():
    """With PTB installed, an ordinary failure is reported as a send failure."""
    pytest.importorskip("telegram")

    result = _send()

    assert "python-telegram-bot not installed" not in str(result)
    assert "Telegram send failed" in str(result), (
        f"a real send failure must be reported as such, got {result!r}"
    )
