"""Endpoint credentials stay out of MCP validation and deferred connection errors."""

import pytest

from tools.mcp_tool_common import _sanitize_error
from tools.mcp_tool_errors import InvalidMcpUrlError, _validate_remote_mcp_url


@pytest.mark.parametrize("name", ["code", "signature", "x-amz-signature", "x-goog-signature", "sig"])
def test_connection_error_masks_opaque_query_credentials(name):
    text = f"Connection failed at https://example.test/mcp?{name}=opaque-fixture&region=west#fragment"
    safe = _sanitize_error(text)
    assert "opaque-fixture" not in safe
    assert f"{name}=***" in safe
    assert "region=west#fragment" in safe


def test_invalid_url_preserves_diagnostics_without_credentials():
    with pytest.raises(InvalidMcpUrlError) as error:
        _validate_remote_mcp_url("fixture", "ftp://example.test/mcp?code=opaque-fixture")
    assert "opaque-fixture" not in str(error.value)
    assert "scheme must be http or https" in str(error.value)
    assert "example.test" in str(error.value)
