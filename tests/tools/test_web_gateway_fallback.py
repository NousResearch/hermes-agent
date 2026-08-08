"""Regression tests for direct-credential fallback when the tool gateway is
unavailable (#79628).

When ``use_gateway: true`` is set but the Nous Tool Gateway cannot
authenticate (expired Portal session), the firecrawl client builder used to
raise a hard configuration error even though a valid direct
``FIRECRAWL_API_KEY`` was present. The fix falls back to the direct
credential (mirroring the Krea/FAL pattern) and logs that the gateway was
skipped.
"""

import os

import pytest
from unittest.mock import patch, MagicMock


class TestFirecrawlGatewayFallback:
    """The gateway-unavailable + direct-key-present matrix for firecrawl."""

    def _patch_gateway(self, gateway_value):
        """Patch the gateway resolver to return ``gateway_value``.

        Returns a patcher for ``tools.web_tools.resolve_managed_tool_gateway``
        (the canonical lookup used by the firecrawl provider).
        """
        return patch(
            "tools.web_tools.resolve_managed_tool_gateway",
            return_value=gateway_value,
        )

    def test_direct_key_fallback_when_gateway_unavailable(self):
        """Direct key + use_gateway:true + dead gateway → falls back to key."""
        import tools.web_tools

        # Reset the singleton cache
        tools.web_tools._firecrawl_client = None
        tools.web_tools._firecrawl_client_config = None

        # The reporter's exact setup: direct key present, gateway resolves None
        with patch.dict(
            os.environ,
            {
                "FIRECRAWL_API_KEY": "fc-test-key",
                "HERMES_HOME": "/tmp/hermes-test-home",
            },
        ):
            with self._patch_gateway(None):
                with patch(
                    "tools.web_tools.prefers_gateway",
                    return_value=True,  # use_gateway: true
                ):
                    with patch("tools.web_tools.Firecrawl") as mock_fc:
                        from tools.web_tools import _get_firecrawl_client
                        result = _get_firecrawl_client()
                        mock_fc.assert_called_once_with(
                            api_key="fc-test-key",
                        )
                        assert result is mock_fc.return_value

    def test_gateway_wins_when_resolvable(self):
        """Gateway resolvable → gateway token used (no direct fallback)."""
        import tools.web_tools

        tools.web_tools._firecrawl_client = None
        tools.web_tools._firecrawl_client_config = None

        gateway = MagicMock()
        gateway.nous_user_token = "nous-token"
        gateway.gateway_origin = "https://firecrawl-gateway.nousresearch.com"

        with patch.dict(
            os.environ,
            {"FIRECRAWL_API_KEY": "fc-test-key"},
        ):
            with self._patch_gateway(gateway):
                with patch(
                    "tools.web_tools.prefers_gateway",
                    return_value=True,
                ):
                    with patch("tools.web_tools.Firecrawl") as mock_fc:
                        from tools.web_tools import _get_firecrawl_client
                        _get_firecrawl_client()
                        mock_fc.assert_called_once_with(
                            api_key="nous-token",
                            api_url="https://firecrawl-gateway.nousresearch.com",
                        )

    def test_no_direct_key_still_raises(self):
        """No direct key + dead gateway → still raises (unchanged behavior)."""
        import tools.web_tools

        tools.web_tools._firecrawl_client = None
        tools.web_tools._firecrawl_client_config = None

        with patch.dict(os.environ, {}):
            with self._patch_gateway(None):
                with patch(
                    "tools.web_tools.prefers_gateway",
                    return_value=True,
                ):
                    with patch("tools.web_tools.Firecrawl"):
                        with patch(
                            "tools.web_tools._read_nous_access_token",
                            return_value=None,
                        ):
                            from tools.web_tools import _get_firecrawl_client
                            with pytest.raises(ValueError, match="FIRECRAWL_API_KEY"):
                                _get_firecrawl_client()
