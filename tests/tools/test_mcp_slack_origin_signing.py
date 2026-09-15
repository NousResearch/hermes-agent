"""Slack-origin signing stays gateway-scoped and is only available to opted-in MCP calls."""

import base64
import hashlib
import hmac
import json

import pytest


def _decode_segment(segment: str) -> bytes:
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


def test_request_hook_injects_only_for_the_active_protected_call(monkeypatch):
    from tools.mcp_slack_origin import (
        build_slack_origin_header,
        slack_origin_context,
        slack_origin_header_context,
        slack_origin_request_hook,
    )

    class Request:
        headers = {}

    monkeypatch.setenv("NEXUS_SLACK_ORIGIN_SECRET", "test-secret")
    with slack_origin_context(platform="slack", chat_id="D123", thread_id=None):
        header = build_slack_origin_header({"secret_env": "NEXUS_SLACK_ORIGIN_SECRET"}, issued_at=1_712_340_000)
        with slack_origin_header_context(header):
            request = Request()
            slack_origin_request_hook({"secret_env": "NEXUS_SLACK_ORIGIN_SECRET"}, request)

    assert set(request.headers) == {"x-nexus-slack-origin"}
    assert request.headers["x-nexus-slack-origin"].count(".") == 1


def test_protected_server_builds_signed_header_from_trusted_gateway_context(monkeypatch):
    from tools.mcp_slack_origin import build_slack_origin_header, slack_origin_context

    monkeypatch.setenv("NEXUS_SLACK_ORIGIN_SECRET", "test-secret")
    policy = {"secret_env": "NEXUS_SLACK_ORIGIN_SECRET"}

    with slack_origin_context(platform="slack", chat_id="D123", thread_id="171234.000100"):
        header = build_slack_origin_header(policy, issued_at=1_712_340_000)

    payload_segment, signature_segment = header.split(".")
    payload = _decode_segment(payload_segment)
    assert json.loads(payload) == {
        "chat_id": "D123",
        "thread_id": "171234.000100",
        "issued_at": 1_712_340_000,
    }
    assert hmac.compare_digest(
        _decode_segment(signature_segment),
        hmac.new(b"test-secret", payload, hashlib.sha256).digest(),
    )


@pytest.mark.parametrize(
    "platform, chat_id, thread_id, secret_present",
    [
        ("telegram", "D123", "171234.000100", True),
        ("slack", "", "171234.000100", True),
        ("slack", "D123", None, False),
    ],
)
def test_protected_server_fails_closed_without_valid_trusted_slack_context(
    monkeypatch, platform, chat_id, thread_id, secret_present
):
    from tools.mcp_slack_origin import SlackOriginSigningError, build_slack_origin_header, slack_origin_context

    monkeypatch.delenv("NEXUS_SLACK_ORIGIN_SECRET", raising=False)
    if secret_present:
        monkeypatch.setenv("NEXUS_SLACK_ORIGIN_SECRET", "test-secret")

    with slack_origin_context(platform=platform, chat_id=chat_id, thread_id=thread_id):
        with pytest.raises(SlackOriginSigningError):
            build_slack_origin_header({"secret_env": "NEXUS_SLACK_ORIGIN_SECRET"}, issued_at=1_712_340_000)
