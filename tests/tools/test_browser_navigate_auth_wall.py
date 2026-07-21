"""Integration tests for the browser_navigate auth-wall response (#24288).

Drives ``tools.browser_tool.browser_navigate`` end-to-end with the
underlying agent-browser CLI mocked.  Asserts that the JSON response
gains an ``auth_wall_warning`` field exactly when the heuristic says
"this landed on a login page AND no CDP override is attached".

Complements ``tests/tools/test_browser_auth_wall_hint.py`` (which
pins the pure helpers) by catching wiring regressions between the
helper and the navigation result.
"""
from __future__ import annotations

import json

import pytest

from tools import browser_tool


def _success_result(url="https://example.com", title="Example"):
    """Mock agent-browser command success payload."""
    return {"success": True, "data": {"title": title, "url": url}}


@pytest.fixture
def _patched_navigate(monkeypatch):
    """Stub every external dependency of ``browser_navigate`` so the
    handler runs deterministically.

    The default state matches "local-backend, no CDP, no SSRF, no
    bot detection".  Individual tests override the agent-browser
    result via ``_run_browser_command`` to simulate landing on
    specific pages.
    """
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "check_website_access", lambda url: None)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
    monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: True)
    monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
    monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda url: False)
    monkeypatch.setattr(
        browser_tool,
        "_get_session_info",
        lambda task_id: {
            "session_name": f"s_{task_id}",
            "bb_session_id": None,
            "cdp_url": None,
            "features": {"local": True},
            "_first_nav": False,
        },
    )
    # Default: no CDP override -- the auth-wall hint is gated on this.
    monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: "")
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    yield monkeypatch


# ---------------------------------------------------------------------------
# Positive cases: hint must be emitted
# ---------------------------------------------------------------------------


class TestAuthWallWarningEmitted:
    """Pages that redirect to a login page MUST trip the warning when
    no CDP is attached."""

    def test_github_login_redirect_emits_warning(self, _patched_navigate):
        """The exact symptom #24288 was filed about: navigating to
        an authenticated GitHub page redirects to /login."""
        _patched_navigate.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _success_result(
                url="https://github.com/login?return_to=%2FNousResearch%2Fhermes-agent%2Fissues%2Fnew",
                title="Sign in to GitHub - GitHub",
            ),
        )
        result = json.loads(
            browser_tool.browser_navigate(
                "https://github.com/NousResearch/hermes-agent/issues/new",
            )
        )
        assert result["success"] is True
        assert "auth_wall_warning" in result, (
            "GitHub login redirect must surface the remediation hint"
        )
        warning = result["auth_wall_warning"]
        assert "/browser connect" in warning
        assert "--remote-debugging-port=9222" in warning

    def test_google_signin_redirect_emits_warning(self, _patched_navigate):
        _patched_navigate.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _success_result(
                url="https://accounts.google.com/signin/v2/identifier?continue=...",
                title="Sign in - Google Accounts",
            ),
        )
        result = json.loads(
            browser_tool.browser_navigate("https://mail.google.com/mail/u/0/")
        )
        assert "auth_wall_warning" in result

    def test_microsoft_oauth_redirect_emits_warning(self, _patched_navigate):
        _patched_navigate.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _success_result(
                url="https://login.microsoftonline.com/common/oauth2/v2.0/authorize?...",
                title="Sign in to your account",
            ),
        )
        result = json.loads(
            browser_tool.browser_navigate("https://office.com/launch")
        )
        assert "auth_wall_warning" in result

    def test_title_only_signal_emits_warning(self, _patched_navigate):
        """Some auth walls keep the original URL but swap the page to a
        login form -- title heuristic must catch those."""
        _patched_navigate.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _success_result(
                url="https://app.example.com/dashboard",  # URL unchanged
                title="Sign in to Acme",
            ),
        )
        result = json.loads(
            browser_tool.browser_navigate("https://app.example.com/dashboard")
        )
        assert "auth_wall_warning" in result


# ---------------------------------------------------------------------------
# Negative cases: hint must NOT be emitted
# ---------------------------------------------------------------------------


class TestAuthWallWarningSuppressed:
    """The hint must stay silent when:
      - The page isn't a login page.
      - The user IS already attached via CDP (then they have cookies,
        the hint would be misleading).
    """

    def test_normal_page_does_not_emit_warning(self, _patched_navigate):
        _patched_navigate.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _success_result(
                url="https://github.com/NousResearch/hermes-agent",
                title="NousResearch/hermes-agent: AI agent framework",
            ),
        )
        result = json.loads(
            browser_tool.browser_navigate("https://github.com/NousResearch/hermes-agent")
        )
        assert "auth_wall_warning" not in result

    def test_documentation_page_does_not_emit_warning(self, _patched_navigate):
        _patched_navigate.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _success_result(
                url="https://docs.python.org/3/library/asyncio.html",
                title="asyncio -- Asynchronous I/O -- Python documentation",
            ),
        )
        result = json.loads(
            browser_tool.browser_navigate("https://docs.python.org/3/library/asyncio.html")
        )
        assert "auth_wall_warning" not in result

    def test_cdp_attached_suppresses_warning_even_on_login_page(
        self, _patched_navigate,
    ):
        """When the user IS already attached via /browser connect, the
        cookies of their real Chrome are in play -- the login page
        either won't appear or is the user's destination.  Either way
        the hint would be misleading noise."""
        _patched_navigate.setattr(
            browser_tool,
            "_get_cdp_override",
            lambda: "ws://127.0.0.1:9222/devtools/browser/abc",
        )
        _patched_navigate.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _success_result(
                url="https://github.com/login",
                title="Sign in to GitHub",
            ),
        )
        result = json.loads(browser_tool.browser_navigate("https://github.com/login"))
        assert "auth_wall_warning" not in result, (
            "CDP-attached sessions must NOT see the hint -- the user "
            "is already in their real browser and the cookies are in "
            "play.  Surfacing the hint here is confusing."
        )

    def test_failed_navigation_does_not_emit_warning(self, _patched_navigate):
        """Hint is gated on a successful nav -- a failed one bypasses
        the whole post-success block."""
        _patched_navigate.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: {"success": False, "error": "DNS failed"},
        )
        result = json.loads(
            browser_tool.browser_navigate("https://github.com/login")
        )
        assert result["success"] is False
        assert "auth_wall_warning" not in result


# ---------------------------------------------------------------------------
# Schema docstring guarantee
# ---------------------------------------------------------------------------


class TestSchemaMentionsAuthWallField:
    """The model can't act on a JSON field it doesn't know to look at.
    Pin that the schema description mentions the new key so future
    schema refactors don't silently drop the hint's discoverability."""

    def test_browser_navigate_schema_describes_auth_wall_field(self):
        schemas = {
            s["name"]: s for s in browser_tool.BROWSER_TOOL_SCHEMAS
        }
        nav_schema = schemas["browser_navigate"]
        desc = nav_schema["description"]
        assert "auth_wall_warning" in desc, (
            "browser_navigate schema must mention the auth_wall_warning "
            "field so the model parses it as actionable.  Got: "
            f"{desc[:200]!r}..."
        )


# ---------------------------------------------------------------------------
# Bug-shape regression anchor
# ---------------------------------------------------------------------------


class TestBug24288Repro:
    """#24288 anchor.

    Pre-fix behaviour: navigating to a GitHub page that requires login
    silently redirected to /login, the model got back a login form
    snapshot, and nothing in the response told it that
    ``/browser connect`` exists.  The model + user were both stuck.

    Post-fix contract: the same nav now ships a structured
    ``auth_wall_warning`` field with three concrete remediations.
    This anchor asserts the end-to-end contract so a future refactor
    that drops the detector or the schema mention fires immediately.
    """

    def test_github_issue_new_login_redirect_surfaces_remediation(
        self, _patched_navigate,
    ):
        # Verbatim repro from the issue body: navigate to "issue new",
        # redirect to /login.
        _patched_navigate.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _success_result(
                url="https://github.com/login?return_to=%2FNousResearch%2Fhermes-agent%2Fissues%2Fnew",
                title="Sign in to GitHub - GitHub",
            ),
        )
        result = json.loads(
            browser_tool.browser_navigate(
                "https://github.com/NousResearch/hermes-agent/issues/new",
            )
        )

        # Anchor 1: success path + warning present.
        assert result["success"] is True, (
            "#24288 regression: failed nav response -- the auth-wall "
            "hint sits on the success branch"
        )
        assert "auth_wall_warning" in result, (
            "#24288 regression: GitHub login redirect must surface "
            "auth_wall_warning -- without it the model retries and "
            "the user is stuck"
        )

        # Anchor 2: every named remediation from the issue + the
        # suggested fix is present.  Removing any of these reduces
        # the hint's value.
        warning = result["auth_wall_warning"]
        assert "/browser connect" in warning, (
            "#24288 regression: lost the /browser connect remediation"
        )
        assert "--remote-debugging-port=9222" in warning, (
            "#24288 regression: lost the manual Chrome launch flag"
        )
        assert "auto_attach_local_chrome" in warning, (
            "#24288 regression: lost the auto-attach config flag"
        )
        assert "gh" in warning.lower(), (
            "#24288 regression: lost the GitHub-specific gh CLI "
            "workaround that the issue reporter explicitly named"
        )
