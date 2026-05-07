"""Web provider billing blockers survive provider wrappers and fallback routing."""

import json
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from tools.tool_result_signals import ProviderBillingBlocker, billing_blocker_tool_error
from tools.web_tools import _memoized_search, web_extract_tool, web_search_tool


class _Provider:
    name = "paid-test"

    def supports_search(self):
        return True

    def search(self, _query, _limit):
        return json.loads(billing_blocker_tool_error("Payment Required", provider=self.name))


def test_billing_failure_envelope_does_not_enter_rescue_fallback():
    provider = _Provider()
    query = f"billing-{uuid.uuid4()}"

    with (
        patch("tools.web_tools._managed_search_fallback") as managed_fallback,
        patch("tools.web_tools._rescue_search") as keyless_fallback,
    ):
        result = _memoized_search(provider, query, 5)

    assert result["_hermes_tool_signal"]["code"] == "billing_blocker"
    managed_fallback.assert_not_called()
    keyless_fallback.assert_not_called()


def test_web_search_converts_typed_provider_exception_to_tool_signal():
    provider = _Provider()

    with (
        patch("tools.web_tools._ensure_web_plugins_loaded"),
        patch("tools.web_tools._get_search_backend", return_value=provider.name),
        patch("agent.web_search_registry.get_provider", return_value=provider),
        patch(
            "tools.web_tools._memoized_search",
            side_effect=ProviderBillingBlocker(provider=provider.name, message="Payment Required"),
        ),
    ):
        result = json.loads(web_search_tool("billing"))

    assert result["success"] is False
    assert result["provider"] == provider.name
    assert result["_hermes_tool_signal"]["code"] == "billing_blocker"


@pytest.mark.asyncio
async def test_web_extract_converts_typed_provider_exception_to_tool_signal():
    provider = _Provider()

    with (
        patch("tools.web_tools.async_is_safe_url", new=AsyncMock(return_value=True)),
        patch("tools.web_tools._get_extract_backend", return_value=provider.name),
        patch("tools.web_tools._ensure_web_plugins_loaded"),
        patch("tools.web_tools._resolve_extract_provider", return_value=(provider, None)),
        patch(
            "tools.web_tools._extract_safe_urls",
            new=AsyncMock(
                side_effect=ProviderBillingBlocker(provider=provider.name, message="Payment Required")
            ),
        ),
    ):
        result = json.loads(await web_extract_tool(["https://example.com"]))

    assert result["success"] is False
    assert result["provider"] == provider.name
    assert result["_hermes_tool_signal"]["code"] == "billing_blocker"



