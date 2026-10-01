"""Signal reply-prefix coverage for standalone send paths."""

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import utf16_len
from gateway.platforms.signal import MAX_MESSAGE_LENGTH
from tools.send_message_tool import _send_signal, _send_to_platform


@pytest.fixture(autouse=True)
def _reset_signal_scheduler():
    from gateway.platforms.signal_rate_limit import _reset_scheduler

    _reset_scheduler()
    yield
    _reset_scheduler()


class _FakeSignalHttp:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, *_args, **_kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def post(self, url, json=None):
        self.calls.append({"url": url, "payload": json})
        if not self.responses:
            raise AssertionError("Unexpected extra POST")
        data = self.responses.pop(0)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)


def _install_signal_http(monkeypatch, fake):
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", fake)


class _ImmediateScheduler:
    def __init__(self, estimated_wait=0.0):
        self.estimated_wait = estimated_wait

    async def acquire(self, _count):
        return 0.0

    async def report_rpc_duration(self, _duration, _count):
        return None

    def estimate_wait(self, _count):
        return self.estimated_wait

    def feedback(self, _retry_after, _count):
        return None

    def state(self):
        return "test"


def test_standalone_signal_send_applies_reply_prefix_with_native_formatting(monkeypatch):
    fake = _FakeSignalHttp([{"result": {"timestamp": 1}}])
    _install_signal_http(monkeypatch, fake)

    result = asyncio.run(
        _send_signal(
            {
                "http_url": "http://localhost:8080",
                "account": "+155****4567",
                "reply_prefix": "🤖 **Hermes**\\n",
            },
            "+155****4321",
            "Battery check passed",
        )
    )

    assert result["success"] is True
    params = fake.calls[0]["payload"]["params"]
    assert params["message"] == "🤖 Hermes\nBattery check passed"
    assert params["textStyle"] == "3:6:BOLD"


def test_standalone_signal_chunking_uses_final_utf16_wire_length(monkeypatch):
    fake = _FakeSignalHttp([{"result": {"timestamp": idx}} for idx in range(1, 10)])
    _install_signal_http(monkeypatch, fake)
    long_message = "😀" * 5000

    result = asyncio.run(
        _send_to_platform(
            Platform.SIGNAL,
            SimpleNamespace(
                enabled=True,
                token=None,
                extra={
                    "http_url": "http://localhost:8080",
                    "account": "+155****4567",
                    "reply_prefix": "🤖 **Hermes**\\n",
                },
            ),
            "+155****4321",
            long_message,
        )
    )
    messages = [call["payload"]["params"]["message"] for call in fake.calls]

    assert result["success"] is True
    assert len(messages) >= 2
    assert messages[0].startswith("🤖 Hermes\n")
    assert all(utf16_len(message) <= MAX_MESSAGE_LENGTH for message in messages)
    assert fake.calls[0]["payload"]["params"]["textStyle"] == "3:6:BOLD"


@pytest.mark.parametrize(
    ("reply_prefix", "message"),
    [
        ("| Name | Value |\\n| --- | --- |\\n| Hermes | Agent |\\n", "x" * 7990),
        ("x" * MAX_MESSAGE_LENGTH, "body"),
    ],
)
def test_standalone_signal_prefix_never_exceeds_wire_limit(monkeypatch, reply_prefix, message):
    fake = _FakeSignalHttp([{"result": {"timestamp": idx}} for idx in range(1, 20)])
    _install_signal_http(monkeypatch, fake)

    result = asyncio.run(
        _send_signal(
            {
                "http_url": "http://localhost:8080",
                "account": "+155****4567",
                "reply_prefix": reply_prefix,
            },
            "+155****4321",
            message,
        )
    )
    messages = [call["payload"]["params"]["message"] for call in fake.calls]

    assert result["success"] is True
    assert messages
    assert all(utf16_len(wire_message) <= MAX_MESSAGE_LENGTH for wire_message in messages)


def test_attachment_total_failure_is_not_hidden_by_successful_text_chunk(monkeypatch, tmp_path):
    attachment = tmp_path / "photo.png"
    attachment.write_bytes(b"\x89PNG\r\n")
    rate_limit = {"error": {"code": -5, "message": "RateLimitException"}}
    fake = _FakeSignalHttp([
        {"result": {"timestamp": 1}},
        rate_limit,
        rate_limit,
    ])
    _install_signal_http(monkeypatch, fake)
    monkeypatch.setattr(
        "gateway.platforms.signal_rate_limit.get_scheduler",
        lambda: _ImmediateScheduler(),
    )

    result = asyncio.run(
        _send_signal(
            {"http_url": "http://localhost:8080", "account": "+155****4567"},
            "+155****4321",
            "x" * 9000,
            media_files=[(str(attachment), False)],
        )
    )

    assert "error" in result
    assert "attachment" in result["error"].lower()


def test_pacing_notice_is_split_to_signal_wire_limit(monkeypatch, tmp_path):
    attachment = tmp_path / "photo.png"
    attachment.write_bytes(b"\x89PNG\r\n")
    fake = _FakeSignalHttp([{"result": {"timestamp": idx}} for idx in range(1, 20)])
    _install_signal_http(monkeypatch, fake)
    monkeypatch.setattr(
        "gateway.platforms.signal_rate_limit.get_scheduler",
        lambda: _ImmediateScheduler(estimated_wait=11.0),
    )

    result = asyncio.run(
        _send_signal(
            {
                "http_url": "http://localhost:8080",
                "account": "+155****4567",
                "reply_prefix": "x" * MAX_MESSAGE_LENGTH,
            },
            "+155****4321",
            "body",
            media_files=[(str(attachment), False)],
        )
    )
    messages = [call["payload"]["params"]["message"] for call in fake.calls]

    assert result["success"] is True
    assert len(messages) >= 4
    assert all(utf16_len(wire_message) <= MAX_MESSAGE_LENGTH for wire_message in messages)
