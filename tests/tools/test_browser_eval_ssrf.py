"""Tests that browser_console(expression=...) cannot bypass the SSRF guard.

browser_snapshot / browser_vision re-check the page URL before returning
content, but ``_browser_eval`` returns arbitrary JS results directly. Two
sub-paths could read private content without ever touching snapshot/vision:

  1. Direct fetch:  ``fetch('http://127.0.0.1/secret').then(r => r.text())``
     — the page URL stays public, so the post-eval recheck can't see it.
     Closed by a pre-scan of the expression for private-host URL literals.
  2. Navigate-then-read:  ``location.href = 'http://127.0.0.1/'`` then a later
     eval reads ``document.body.innerText`` — closed by re-checking the page
     URL after the eval runs.

This is the sibling fix for the eval return-value path of issue #44731.
"""

import json

import pytest

from tools import browser_tool


PRIVATE_URL = "http://127.0.0.1:8080/secret"
PUBLIC_URL = "https://example.com/page"
METADATA_URL = "http://169.254.169.254/latest/meta-data/"


@pytest.fixture(autouse=True)
def _no_camofox(monkeypatch):
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    # No supervisor — force the subprocess fallback path by default.
    monkeypatch.setattr(browser_tool, "_last_session_key", lambda key: key)


def _eval(expression, task_id="test"):
    return json.loads(browser_tool._browser_eval(expression, task_id=task_id))


# ---------------------------------------------------------------------------
# Sub-path 1: direct private-host fetch literal in the expression (pre-scan)
# ---------------------------------------------------------------------------


class TestExpressionPreScan:
    def _guard_on(self, monkeypatch):
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)

    def test_blocks_private_fetch_literal(self, monkeypatch):
        self._guard_on(monkeypatch)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False)
        monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda url: False)

        called = {"n": 0}

        def _run(task_id, command, args=None, **kwargs):
            called["n"] += 1
            return {"success": True, "data": {"result": "leaked-content"}}

        monkeypatch.setattr(browser_tool, "_run_browser_command", _run)

        result = _eval(f"fetch('{PRIVATE_URL}').then(r => r.text())")
        assert result["success"] is False
        assert "private or internal address" in result["error"]
        assert PRIVATE_URL in result["error"]
        # Expression never executed — blocked before any browser command.
        assert called["n"] == 0

    def test_blocks_metadata_fetch_literal(self, monkeypatch):
        self._guard_on(monkeypatch)
        # Public-safe to is_safe_url, but the always-blocked floor catches IMDS.
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
        monkeypatch.setattr(
            browser_tool, "_is_always_blocked_url",
            lambda url: "169.254.169.254" in url,
        )
        monkeypatch.setattr(
            browser_tool, "_run_browser_command",
            lambda *a, **k: {"success": True, "data": {"result": "creds"}},
        )

        result = _eval(f"fetch('{METADATA_URL}')")
        assert result["success"] is False
        assert "private or internal address" in result["error"]

    def test_allows_public_fetch_literal(self, monkeypatch):
        self._guard_on(monkeypatch)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
        monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda url: False)
        # After the (public) eval, the page-URL recheck must also see a public URL.
        monkeypatch.setattr(
            browser_tool, "_run_browser_command",
            lambda task_id, command, args=None, **k: (
                {"success": True, "data": {"result": PUBLIC_URL}}
                if args == ["window.location.href"]
                else {"success": True, "data": {"result": "ok"}}
            ),
        )

        result = _eval(f"fetch('{PUBLIC_URL}').then(r => r.text())")
        assert result["success"] is True
        assert result["result"] == "ok"

    def test_skips_prescan_for_local_backend(self, monkeypatch):
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
        monkeypatch.setattr(
            browser_tool, "_run_browser_command",
            lambda *a, **k: {"success": True, "data": {"result": "local-ok"}},
        )
        result = _eval(f"fetch('{PRIVATE_URL}')")
        assert result["success"] is True
        assert result["result"] == "local-ok"

    def test_skips_prescan_for_local_sidecar(self, monkeypatch):
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: True)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(
            browser_tool, "_run_browser_command",
            lambda *a, **k: {"success": True, "data": {"result": "sidecar-ok"}},
        )
        result = _eval(f"fetch('{PRIVATE_URL}')")
        assert result["success"] is True

    def test_skips_prescan_when_allow_private(self, monkeypatch):
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: True)
        monkeypatch.setattr(
            browser_tool, "_run_browser_command",
            lambda *a, **k: {"success": True, "data": {"result": "allowed"}},
        )
        result = _eval(f"fetch('{PRIVATE_URL}')")
        assert result["success"] is True


# ---------------------------------------------------------------------------
# Sub-path 2: navigate-then-read (post-eval page-URL recheck)
# ---------------------------------------------------------------------------


class TestPostEvalPageRecheck:
    def _guard_on(self, monkeypatch):
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)

    def test_blocks_when_page_navigated_private(self, monkeypatch):
        self._guard_on(monkeypatch)
        # Expression itself has no URL literal (reads the DOM), so the pre-scan
        # passes; the danger is that the page was navigated to a private URL by
        # an earlier eval. The recheck must catch it.
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False)
        monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda url: False)
        monkeypatch.setattr(
            browser_tool, "_run_browser_command",
            lambda task_id, command, args=None, **k: (
                {"success": True, "data": {"result": PRIVATE_URL}}
                if args == ["window.location.href"]
                else {"success": True, "data": {"result": "secret DOM text"}}
            ),
        )

        result = _eval("document.body.innerText")
        assert result["success"] is False
        assert "private or internal address" in result["error"]
        assert PRIVATE_URL in result["error"]

    def test_allows_when_page_public(self, monkeypatch):
        self._guard_on(monkeypatch)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
        monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda url: False)
        monkeypatch.setattr(
            browser_tool, "_run_browser_command",
            lambda task_id, command, args=None, **k: (
                {"success": True, "data": {"result": PUBLIC_URL}}
                if args == ["window.location.href"]
                else {"success": True, "data": {"result": "public DOM text"}}
            ),
        )

        result = _eval("document.body.innerText")
        assert result["success"] is True
        assert result["result"] == "public DOM text"

    def test_fail_open_when_url_probe_fails(self, monkeypatch):
        """If the window.location.href probe errors, don't block (fail-open)."""
        self._guard_on(monkeypatch)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False)
        monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda url: False)

        def _run(task_id, command, args=None, **k):
            if args == ["window.location.href"]:
                return {"success": False, "error": "CDP probe failed"}
            return {"success": True, "data": {"result": "dom text"}}

        monkeypatch.setattr(browser_tool, "_run_browser_command", _run)

        result = _eval("document.body.innerText")
        assert result["success"] is True
        assert result["result"] == "dom text"


# ---------------------------------------------------------------------------
# Helper-level unit tests
# ---------------------------------------------------------------------------


class TestExpressionScanHelper:
    def test_returns_first_private_literal(self, monkeypatch):
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: "127.0.0.1" not in url)
        monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda url: False)
        out = browser_tool._expression_targets_private_url(
            "fetch('https://example.com'); fetch('http://127.0.0.1/x')"
        )
        assert out == "http://127.0.0.1/x"

    def test_none_when_no_url(self, monkeypatch):
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
        monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda url: False)
        assert browser_tool._expression_targets_private_url("document.title") is None

    def test_strips_trailing_punctuation(self, monkeypatch):
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False)
        monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda url: False)
        out = browser_tool._expression_targets_private_url("location.href='http://10.0.0.1/';")
        assert out == "http://10.0.0.1/"
