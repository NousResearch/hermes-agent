"""media_dropped feedback for send_message (#32644, supersedes PR #34178).

These tests live OUTSIDE test_send_message_tool.py on purpose: that module is
gated by ``pytest.importorskip("telegram")``, and python-telegram-bot is not in
the ``[dev]``/``[all]`` extras, so in CI the whole module is skipped. The
media_dropped behavior is platform-agnostic (it lives in ``_handle_send`` before
``_send_to_platform``, which we mock), so it needs no telegram and must run in
CI to actually protect the headline feature.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from tools.send_message_tool import send_message_tool


@pytest.fixture(autouse=True)
def _reset_signal_scheduler():
    from gateway.platforms.signal_rate_limit import _reset_scheduler
    _reset_scheduler()
    yield
    _reset_scheduler()


def _run_async_immediately(coro):
    return asyncio.run(coro)


def _make_config():
    telegram_cfg = SimpleNamespace(enabled=True, token="***", extra={})
    return SimpleNamespace(
        platforms={Platform.TELEGRAM: telegram_cfg},
        get_home_channel=lambda _platform: None,
    ), telegram_cfg


def _send_message(message):
    return json.loads(
        send_message_tool({"action": "send", "target": "telegram:12345", "message": message})
    )


def test_dropped_media_path_surfaced_on_success(tmp_path, monkeypatch):
    # A MEDIA path rejected by the delivery gate (strict mode, outside roots)
    # must be reported via media_dropped + a warning rather than silently
    # succeeding with text only (#32644).
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")
    monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "0")
    config, _cfg = _make_config()
    secret = tmp_path / "secret.pdf"
    secret.write_bytes(b"%PDF secret")

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        result = _send_message(f"hello\nMEDIA:{secret}")

    assert result["success"] is True
    # No media reached the platform; the dropped path is surfaced.
    _args, kwargs = send_mock.await_args
    assert kwargs.get("media_files") == []
    assert result["media_dropped"] == [str(secret)]
    assert any("dropped" in w for w in result.get("warnings", []))


def test_media_dropped_not_added_when_send_errors(tmp_path, monkeypatch):
    # When the underlying send fails, the error result is NOT annotated with
    # media_dropped/warnings (mirror PR #34178: warning only on success).
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")
    monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "0")
    config, _cfg = _make_config()
    secret = tmp_path / "secret.pdf"
    secret.write_bytes(b"%PDF secret")

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"error": "boom"})), \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        result = _send_message(f"hello\nMEDIA:{secret}")

    assert "error" in result
    assert "media_dropped" not in result
    assert "warnings" not in result


def test_windows_media_tag_stripped_and_dropped_end_to_end(monkeypatch):
    # End-to-end through the tool path: a Windows MEDIA tag is stripped from
    # the sent text (no raw C:\ leak) and reported via media_dropped, never
    # delivered (#28989, #24032). Exercises extract_media → partition →
    # _handle_send together, not just the regex/validator in isolation.
    config, _cfg = _make_config()
    win_path = r"C:\Users\foo\report.pdf"

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        result = _send_message(f"here you go\nMEDIA:{win_path}")

    assert result["success"] is True
    _args, kwargs = send_mock.await_args
    sent_text = kwargs.get("content") if kwargs.get("content") is not None else send_mock.await_args.args[3]
    assert "C:" not in sent_text and "MEDIA:" not in sent_text  # no raw-path leak
    assert kwargs.get("media_files") == []                       # not delivered
    assert result["media_dropped"] == [win_path]                 # reported instead
