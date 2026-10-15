"""Native single-attempt entry: real target/auth/routing, synthetic Telegram transport."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from tools import send_message_senders, send_message_tool


class ProbeBot:
    def __init__(self, error=None):
        self.calls = []
        self.error = error
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True

    async def send_message(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(message_id=17)


@pytest.fixture
def native_send(monkeypatch):
    pytest.importorskip("telegram", reason="native single-attempt SDK path needs messaging extra")
    monkeypatch.setenv("PTB_TIMEDELTA", "true")
    pconfig = SimpleNamespace(enabled=True, token="test-only-managed-token", extra={})
    config = SimpleNamespace(platforms={Platform.TELEGRAM: pconfig})
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
    monkeypatch.setattr("model_tools._run_async", asyncio.run)
    monkeypatch.setattr(send_message_tool, "_mirror_sent_message", lambda *a: False)
    bots = []

    def install(bot):
        def factory(token):
            assert token == pconfig.token
            bots.append(bot)
            return bot
        monkeypatch.setattr(send_message_senders, "_telegram_bot", factory)
        return lambda **kw: json.loads(send_message_tool.send_message_once(
            {"target": "telegram:123", "message": "Fixed test notification", **kw}))
    return install, bots


def test_public_entry_never_retries_502_and_does_not_echo_exception(native_send):
    install, bots = native_send
    bot = ProbeBot(RuntimeError("502 Bad Gateway PRIVATE_EXCEPTION_SENTINEL"))
    invoke = install(bot)
    result = invoke()
    assert len(bot.calls) == 1
    assert bot.closed
    assert result["delivery_status"] == "unknown"
    assert "PRIVATE_EXCEPTION_SENTINEL" not in json.dumps(result)
    assert bots == [bot]


def test_rate_limit_is_definitely_unsent_and_is_not_retried(native_send):
    from telegram.error import RetryAfter

    install, _ = native_send
    bot = ProbeBot(RetryAfter(0))
    result = install(bot)()
    assert len(bot.calls) == 1
    assert result["delivery_status"] == "not_sent"
    assert result["error"] == "TELEGRAM_RATE_LIMITED"
    assert result["retry_after"] == 0


@pytest.mark.parametrize("body", ["x" * 4096, "😀" * 2048])
def test_supported_boundary_is_one_plain_message(native_send, body):
    install, _ = native_send
    bot = ProbeBot()
    result = install(bot)(message=body)
    assert result["delivery_status"] == "accepted"
    assert result["message_id"] == "17"
    assert len(bot.calls) == 1
    assert bot.calls[0]["text"] == body
    assert bot.calls[0]["parse_mode"] is None
    assert bot.closed


@pytest.mark.parametrize("error", [
    RuntimeError("can't parse entities"), RuntimeError("thread not found"), TimeoutError("network timeout"),
])
def test_single_attempt_has_no_parse_thread_or_timeout_fallback(native_send, error):
    install, _ = native_send
    bot = ProbeBot(error)
    result = install(bot)()
    assert len(bot.calls) == 1
    assert result["delivery_status"] == "unknown"
    assert bot.closed


def test_existing_entry_keeps_its_default_retry_policy(native_send):
    from telegram.error import RetryAfter

    install, _ = native_send
    bot = ProbeBot(RetryAfter(0))
    install(bot)
    result = json.loads(send_message_tool.send_message_tool(
        {"target": "telegram:123", "message": "Fixed test notification"}))
    assert "error" in result
    assert len(bot.calls) == 3


def test_once_entry_still_runs_native_egress_authorization(native_send, monkeypatch):
    install, bots = native_send
    bot = ProbeBot()
    invoke = install(bot)
    decisions = []

    def deny(platform, chat_id, thread_id=None, **kwargs):
        decisions.append((platform, chat_id))
        return "controlled native egress denial"

    monkeypatch.setattr("gateway.relay.egress.authorize_relay_target", deny)
    result = invoke()
    assert decisions == [("telegram", "123")]
    assert result["error"] == "controlled native egress denial"
    assert not bots
    assert not bot.calls


@pytest.mark.parametrize("status, expected", [(200, "accepted"), (429, "not_sent"), (502, "unknown")])
def test_real_telegram_sdk_issues_one_send_request(native_send, status, expected):
    import httpx
    from telegram import Bot
    from telegram.request import HTTPXRequest

    install, _ = native_send
    methods = []

    def transport(request):
        method = request.url.path.rsplit("/", 1)[-1]
        methods.append(method)
        if method == "getMe":
            return httpx.Response(200, json={"ok": True, "result": {
                "id": 123, "is_bot": True, "first_name": "Fixture", "username": "fixture_bot"}})
        assert method == "sendMessage", method
        if status == 200:
            return httpx.Response(200, json={"ok": True, "result": {
                "message_id": 17, "date": 0, "chat": {"id": 123, "type": "private"}, "text": "fixed"}})
        return httpx.Response(status, json={"ok": False, "error_code": status,
            "description": "controlled fixture response", "parameters": {"retry_after": 1} if status == 429 else {}})

    def request():
        return HTTPXRequest(httpx_kwargs={"transport": httpx.MockTransport(transport)})

    bot = Bot(token="123:TEST", request=request(), get_updates_request=request())
    result = install(bot)()
    assert result["delivery_status"] == expected
    assert methods == ["getMe", "sendMessage"]


def test_cron_duplicate_skip_does_not_attempt_notification(native_send, monkeypatch):
    install, bots = native_send
    bot = ProbeBot()
    invoke = install(bot)
    for key, value in {"PLATFORM": "telegram", "CHAT_ID": "123", "THREAD_ID": ""}.items():
        monkeypatch.setenv("HERMES_CRON_AUTO_DELIVER_" + key, value)
    result = invoke()
    assert result["success"] is True
    assert result["skipped"] is True
    assert "delivery_status" not in result
    assert not bots
    assert not bot.calls


@pytest.mark.parametrize("args", [
    None, [], {},
    {"target": "discord:123", "message": "fixed"},
    {"target": "telegram:123", "message": "fixed", "action": "list"},
    {"target": "telegram:123", "message": "fixed", "attachments": []},
    {"target": "telegram:123", "message": "MEDIA:/not-a-real-file fixed"},
    {"target": "telegram:123", "message": "media:/not-a-real-file.png fixed"},
    {"target": "telegram:123", "message": "mEdIa:/not-a-real-file.png fixed"},
    {"target": "telegram:123", "message": "MEDİA:/not-a-real-file.png fixed"},
    {"target": "telegram:123", "message": "[[audio_as_voice]] fixed"},
    {"target": "telegram:123", "message": "[[as_document]] fixed"},
    {"target": "telegram:123", "message": "x" * 4097},
    {"target": "telegram:123", "message": "😀" * 2049},
    {"target": "telegram:123", "message": "   "},
    {"target": 123, "message": "fixed"},
    {"target": "telegram:123", "message": {"private": "not text"}},
])
def test_unsupported_once_requests_are_rejected_before_config_or_transport(monkeypatch, args):
    def forbidden_config():
        raise AssertionError("unsupported request reached config or sender")
    monkeypatch.setattr("gateway.config.load_gateway_config", forbidden_config)
    result = json.loads(send_message_tool.send_message_once(args))
    assert result["error"] == "SINGLE_ATTEMPT_UNSUPPORTED"
    assert result["delivery_status"] == "not_sent"
