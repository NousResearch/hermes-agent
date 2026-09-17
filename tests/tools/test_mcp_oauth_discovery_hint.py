"""Registration-failure errors hide discovery failures; the probe must lead (#113771)."""

import urllib.error

import pytest

from tools.mcp_oauth import discovery_failure_summary


def _http_error(url, code=403):
    raise urllib.error.HTTPError(url, code, "Forbidden", {}, None)


def test_all_discovery_urls_failing_returns_summary():
    summary = discovery_failure_summary(
        "https://mcp.tradingview.com/mcp", _opener=lambda url, timeout: _http_error(url))
    assert summary is not None
    assert "could not read authorization-server metadata" in summary
    assert "403" in summary


def test_any_discovery_url_ok_returns_none():
    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    seen = []

    def _opener(url, timeout):
        seen.append(url)
        return _Resp()

    assert discovery_failure_summary("https://example.com/mcp", _opener=_opener) is None
    assert seen, "expected at least one discovery probe"


def test_unreachable_discovery_reports_error_kind_not_traceback():
    def _opener(url, timeout):
        raise OSError("dns down")

    summary = discovery_failure_summary("https://example.com/mcp", _opener=_opener)
    assert summary is not None
    assert "OSError" in summary
    assert "dns down" not in summary


def test_empty_url_returns_none():
    assert discovery_failure_summary(None) is None
    assert discovery_failure_summary("") is None
