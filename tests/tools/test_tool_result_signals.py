"""Typed control signals carried by tool result envelopes."""

import json

from tools.tool_result_signals import (
    billing_blocker_tool_error,
    is_billing_blocker_signal,
    parse_tool_result_signal,
    provider_billing_blocker_from_exception,
)


def test_billing_blocker_round_trip_is_explicit_and_non_retryable():
    result = billing_blocker_tool_error(
        "Firecrawl returned HTTP 402 Payment Required",
        provider="firecrawl",
    )

    signal = parse_tool_result_signal(result)

    assert is_billing_blocker_signal(signal)
    assert signal is not None
    assert signal.provider == "firecrawl"
    assert signal.retryable is False


def test_plain_payment_required_text_is_not_a_control_signal():
    result = json.dumps({"error": "fixture contains HTTP 402 Payment Required"})

    assert parse_tool_result_signal(result) is None


def test_exception_classification_requires_explicit_status_metadata():
    plain = RuntimeError("HTTP 402 Payment Required")

    assert provider_billing_blocker_from_exception(plain, provider="example") is None

    typed = RuntimeError("Payment Required")
    typed.status_code = 402
    blocker = provider_billing_blocker_from_exception(typed, provider="example")
    assert blocker is not None
    assert blocker.provider == "example"


def test_browser_route_renders_provider_billing_blocker_as_tool_result():
    from unittest.mock import patch

    from tools.browser_use_cli import _route_exec_result
    from tools.tool_result_signals import ProviderBillingBlocker

    blocker = ProviderBillingBlocker(provider="browserbase", message="Payment Required")
    with patch("tools.browser_use_cli._route_exec_current", side_effect=blocker):
        _env, _sidecar, _generation, result = _route_exec_result(
            {}, "", "task", False, False, "route"
        )

    signal = parse_tool_result_signal(result)
    assert is_billing_blocker_signal(signal)
    assert signal is not None and signal.provider == "browserbase"

