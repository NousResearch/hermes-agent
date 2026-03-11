"""Tests for browser_navigate markdown fast-path.

Coverage:
  _try_fetch_markdown() — config-driven header probe and provider proxy
  browser_navigate()    — read_only flag, interactive bypass
"""

import json
import pytest
from unittest.mock import patch, MagicMock

import httpx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(status_code=200, content_type="text/html", text="<html></html>"):
    """Build a fake httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = {"content-type": content_type}
    resp.text = text
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


# ---------------------------------------------------------------------------
# _try_fetch_markdown
# ---------------------------------------------------------------------------

class TestTryFetchMarkdown:
    """Unit tests for _try_fetch_markdown()."""

    def _call(self, url="https://example.com"):
        from tools.browser_tool import _try_fetch_markdown
        return _try_fetch_markdown(url)

    # ── markdown_header strategy ──────────────────────────────────────

    def test_header_returns_markdown_when_content_type_matches(self):
        """Accept header probe succeeds when server returns text/markdown."""
        md_resp = _make_response(content_type="text/markdown", text="# Hello")
        cfg = {"browser": {"markdown_header": True, "markdown_provider": ""}}
        with patch("tools.browser_tool.httpx.get", return_value=md_resp) as mock_get, \
             patch("hermes_cli.config.load_config", return_value=cfg):
            result = self._call()
        assert result == "# Hello"
        # Verify Accept header was sent
        mock_get.assert_called_once()
        call_headers = mock_get.call_args[1]["headers"]
        assert call_headers["Accept"] == "text/markdown"

    def test_header_returns_none_when_content_type_is_html(self):
        """Accept header probe returns None when server ignores the header."""
        html_resp = _make_response(content_type="text/html", text="<html></html>")
        cfg = {"browser": {"markdown_header": True, "markdown_provider": ""}}
        with patch("tools.browser_tool.httpx.get", return_value=html_resp), \
             patch("hermes_cli.config.load_config", return_value=cfg):
            result = self._call()
        assert result is None

    def test_header_skipped_when_disabled(self):
        """When markdown_header=False, Accept header probe is not attempted."""
        cfg = {"browser": {"markdown_header": False, "markdown_provider": ""}}
        with patch("tools.browser_tool.httpx.get") as mock_get, \
             patch("hermes_cli.config.load_config", return_value=cfg):
            result = self._call()
        mock_get.assert_not_called()
        assert result is None

    def test_header_http_error_falls_through(self):
        """HTTP error on header probe falls through gracefully."""
        cfg = {"browser": {"markdown_header": True, "markdown_provider": ""}}
        with patch("tools.browser_tool.httpx.get", side_effect=httpx.ConnectError("fail")), \
             patch("hermes_cli.config.load_config", return_value=cfg):
            result = self._call()
        assert result is None

    # ── URL scheme validation ───────────���────────────────────────────

    def test_rejects_file_scheme(self):
        """file:// URLs are rejected to prevent SSRF."""
        cfg = {"browser": {"markdown_header": True, "markdown_provider": ""}}
        with patch("tools.browser_tool.httpx.get") as mock_get, \
             patch("hermes_cli.config.load_config", return_value=cfg):
            result = self._call("file:///etc/passwd")
        mock_get.assert_not_called()
        assert result is None

    def test_rejects_ftp_scheme(self):
        """ftp:// URLs are rejected."""
        cfg = {"browser": {"markdown_header": True, "markdown_provider": ""}}
        with patch("tools.browser_tool.httpx.get") as mock_get, \
             patch("hermes_cli.config.load_config", return_value=cfg):
            result = self._call("ftp://internal.host/data")
        mock_get.assert_not_called()
        assert result is None

    # ── markdown_provider strategy ────────────────────────────────────

    def test_provider_returns_markdown(self):
        """Provider proxy returns content → use it."""
        html_resp = _make_response(content_type="text/html")
        md_resp = _make_response(content_type="text/plain", text="# From provider")
        cfg = {"browser": {"markdown_header": True, "markdown_provider": "https://md.example.com/"}}
        with patch("tools.browser_tool.httpx.get", side_effect=[html_resp, md_resp]) as mock_get, \
             patch("hermes_cli.config.load_config", return_value=cfg):
            result = self._call("https://target.com")
        assert result == "# From provider"
        # Second call should be to provider URL
        assert mock_get.call_args_list[1][0][0] == "https://md.example.com/https://target.com"

    def test_provider_trailing_slash_normalized(self):
        """Provider URL without trailing slash gets one added."""
        html_resp = _make_response(content_type="text/html")
        md_resp = _make_response(content_type="text/plain", text="# Normalized")
        cfg = {"browser": {"markdown_header": True, "markdown_provider": "https://md.example.com"}}
        with patch("tools.browser_tool.httpx.get", side_effect=[html_resp, md_resp]) as mock_get, \
             patch("hermes_cli.config.load_config", return_value=cfg):
            result = self._call("https://target.com")
        assert result == "# Normalized"
        assert mock_get.call_args_list[1][0][0] == "https://md.example.com/https://target.com"

    def test_provider_empty_response_returns_none(self):
        """Provider returns empty/whitespace body → treat as failure."""
        html_resp = _make_response(content_type="text/html")
        empty_resp = _make_response(content_type="text/plain", text="   ")
        cfg = {"browser": {"markdown_header": True, "markdown_provider": "https://md.example.com/"}}
        with patch("tools.browser_tool.httpx.get", side_effect=[html_resp, empty_resp]), \
             patch("hermes_cli.config.load_config", return_value=cfg):
            result = self._call()
        assert result is None

    def test_provider_skipped_when_empty_string(self):
        """No provider configured → only header probe runs."""
        html_resp = _make_response(content_type="text/html")
        cfg = {"browser": {"markdown_header": True, "markdown_provider": ""}}
        with patch("tools.browser_tool.httpx.get", return_value=html_resp) as mock_get, \
             patch("hermes_cli.config.load_config", return_value=cfg):
            result = self._call()
        # Only one call (header probe), no provider call
        assert mock_get.call_count == 1
        assert result is None

    def test_provider_http_error_returns_none(self):
        """Provider HTTP error → returns None."""
        html_resp = _make_response(content_type="text/html")
        cfg = {"browser": {"markdown_header": True, "markdown_provider": "https://md.example.com/"}}
        with patch("tools.browser_tool.httpx.get", side_effect=[html_resp, httpx.ConnectError("fail")]), \
             patch("hermes_cli.config.load_config", return_value=cfg):
            result = self._call()
        assert result is None

    # ── Both strategies combined ──────────────────────────────────────

    def test_header_success_skips_provider(self):
        """When header probe succeeds, provider is never called."""
        md_resp = _make_response(content_type="text/markdown", text="# Direct")
        cfg = {"browser": {"markdown_header": True, "markdown_provider": "https://md.example.com/"}}
        with patch("tools.browser_tool.httpx.get", return_value=md_resp) as mock_get, \
             patch("hermes_cli.config.load_config", return_value=cfg):
            result = self._call()
        assert result == "# Direct"
        assert mock_get.call_count == 1  # Only header probe

    # ── Config load failure ───────────────────────────────────────────

    def test_config_load_failure_uses_defaults(self):
        """If load_config raises, defaults are used (header=True, provider='')."""
        md_resp = _make_response(content_type="text/markdown", text="# Fallback")
        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("no config")), \
             patch("tools.browser_tool.httpx.get", return_value=md_resp):
            result = self._call()
        assert result == "# Fallback"


# ---------------------------------------------------------------------------
# browser_navigate — fast-path integration
# ---------------------------------------------------------------------------

class TestBrowserNavigateFastPath:
    """Tests for the fast-path and interactive bypass in browser_navigate()."""

    def test_fastpath_sets_read_only_true(self):
        """Markdown fast-path response includes read_only=True."""
        from tools.browser_tool import browser_navigate
        with patch("tools.browser_tool._active_sessions", {}), \
             patch("tools.browser_tool._try_fetch_markdown", return_value="# Page"):
            result = json.loads(browser_navigate("https://example.com"))
        assert result["success"] is True
        assert result["read_only"] is True
        assert result["markdown"] == "# Page"

    def test_fastpath_skipped_when_interactive(self):
        """interactive=True bypasses the markdown fast-path entirely."""
        from tools.browser_tool import browser_navigate
        with patch("tools.browser_tool._active_sessions", {}), \
             patch("tools.browser_tool._try_fetch_markdown") as mock_md, \
             patch("tools.browser_tool._get_session_info", return_value={"session_name": "s1", "_first_nav": True}), \
             patch("tools.browser_tool._maybe_start_recording"), \
             patch("tools.browser_tool._run_browser_command", return_value={
                 "success": True, "data": {"title": "Example", "url": "https://example.com"}
             }):
            result = json.loads(browser_navigate("https://example.com", interactive=True))
        mock_md.assert_not_called()
        assert result["success"] is True
        assert "read_only" not in result

    def test_fastpath_none_falls_through_to_browser(self):
        """When _try_fetch_markdown returns None, real browser is used."""
        from tools.browser_tool import browser_navigate
        with patch("tools.browser_tool._active_sessions", {}), \
             patch("tools.browser_tool._try_fetch_markdown", return_value=None), \
             patch("tools.browser_tool._get_session_info", return_value={"session_name": "s1", "_first_nav": True}), \
             patch("tools.browser_tool._maybe_start_recording"), \
             patch("tools.browser_tool._run_browser_command", return_value={
                 "success": True, "data": {"title": "HN", "url": "https://news.ycombinator.com"}
             }):
            result = json.loads(browser_navigate("https://news.ycombinator.com"))
        assert result["success"] is True
        assert "read_only" not in result

    def test_fastpath_skipped_when_session_exists(self):
        """Fast-path is skipped when a live session already exists for the task."""
        from tools.browser_tool import browser_navigate
        # Simulate an existing session for the "default" task
        with patch("tools.browser_tool._active_sessions", {"default": {"session_name": "s1"}}), \
             patch("tools.browser_tool._try_fetch_markdown") as mock_md, \
             patch("tools.browser_tool._get_session_info", return_value={"session_name": "s1", "_first_nav": False}), \
             patch("tools.browser_tool._maybe_start_recording"), \
             patch("tools.browser_tool._run_browser_command", return_value={
                 "success": True, "data": {"title": "Example", "url": "https://example.com"}
             }):
            result = json.loads(browser_navigate("https://example.com"))
        mock_md.assert_not_called()
        assert result["success"] is True
        assert "read_only" not in result
