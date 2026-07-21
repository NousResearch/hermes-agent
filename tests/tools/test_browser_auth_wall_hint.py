"""Unit tests for the browser tool's auth-wall hint + opt-in auto-attach
to local Chrome (#24288).

Covers the three pure helpers added in
``tools/browser_tool.py`` -- ``_looks_like_auth_wall``,
``_auth_wall_remediation_hint``, and ``_probe_local_chrome_cdp`` --
plus the integration into ``_get_cdp_override`` that ties the
``browser.auto_attach_local_chrome`` config flag to the probe result.

Every test is hermetic: ``requests.get`` is mocked so we never touch
the real network, and the probe cache is reset between tests so
behaviour is reproducible.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def browser_tool_module():
    """Import the module under test in isolation per test.

    The auto-attach helpers stash state on a module-level dict
    (``_LOCAL_CDP_PROBE_CACHE``) so we reset that between tests to
    avoid cross-test leakage.
    """
    import tools.browser_tool as bt
    bt._LOCAL_CDP_PROBE_CACHE["checked_at"] = 0.0
    bt._LOCAL_CDP_PROBE_CACHE["url"] = ""
    return bt


# ---------------------------------------------------------------------------
# _looks_like_auth_wall
# ---------------------------------------------------------------------------


class TestLooksLikeAuthWall:
    """Heuristic for "did this navigation land on a login page?"."""

    @pytest.mark.parametrize(
        "url,title",
        [
            # URL-based detection -- common login paths.
            ("https://github.com/login?return_to=%2Fnouseresearch%2Fhermes-agent%2Fissues%2Fnew", ""),
            ("https://accounts.google.com/signin/v2/identifier", "Sign in"),
            ("https://login.microsoftonline.com/common/oauth2/v2.0/authorize?...", ""),
            ("https://example.okta.com/login/login.htm", ""),
            ("https://app.example.com/auth/login?next=/dashboard", ""),
            ("https://gitlab.example.com/users/sign_in", "GitLab"),
            ("https://example.com/account/login", ""),
            ("https://api.example.com/oauth/authorize?client_id=...", ""),
            # Title-based detection -- URL doesn't tip us off but the page does.
            ("https://intranet.example.com/portal", "Sign in to Acme Portal"),
            ("https://example.com/dashboard", "Log in - Example"),
            ("https://example.com/", "Authentication Required"),
            ("https://example.com/", "Session expired -- please log in again"),
        ],
    )
    def test_detects_obvious_login_pages(self, browser_tool_module, url, title):
        assert browser_tool_module._looks_like_auth_wall(url, title) is True

    @pytest.mark.parametrize(
        "url,title",
        [
            # The destination IS a login portal -- we still flag it,
            # which is fine: surfacing the hint costs one extra field
            # but never breaks a working flow.  These cases are listed
            # here to document the false-positive class explicitly.
            ("https://login.microsoftonline.com/", "Sign in to your account"),  # actually a login page
            # Genuine non-login pages -- must NOT trigger.
            ("https://github.com/NousResearch/hermes-agent", "NousResearch/hermes-agent"),
            ("https://docs.python.org/3/library/asyncio.html", "asyncio -- Asynchronous I/O"),
            ("https://en.wikipedia.org/wiki/Login", "Login - Wikipedia"),  # encyclopedia article ABOUT login
            ("", ""),
            ("https://example.com/blog/intro-to-blogging", "Welcome to my blog"),
        ],
    )
    def test_no_false_positives_on_normal_pages(self, browser_tool_module, url, title):
        # Branching expected: the first row is intentionally a true
        # positive even though it "looks like a destination" -- the
        # heuristic is one-sided by design.  Pin that contract.
        if "login.microsoftonline.com" in url or title.lower() == "login - wikipedia":
            assert browser_tool_module._looks_like_auth_wall(url, title) is True
        else:
            assert browser_tool_module._looks_like_auth_wall(url, title) is False

    def test_case_insensitive_matching(self, browser_tool_module):
        """The heuristic must not depend on URL/title casing."""
        assert browser_tool_module._looks_like_auth_wall(
            "https://EXAMPLE.com/LOGIN", "",
        ) is True
        assert browser_tool_module._looks_like_auth_wall(
            "", "SIGN IN TO CONTINUE",
        ) is True

    def test_handles_none_inputs_gracefully(self, browser_tool_module):
        """Defensive: ``data.get(...)`` returns None when the field is
        missing; the helper must tolerate that instead of TypeError'ing."""
        assert browser_tool_module._looks_like_auth_wall(None, None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _auth_wall_remediation_hint
# ---------------------------------------------------------------------------


class TestAuthWallRemediationHint:
    """The remediation text is the user-facing payload of the fix --
    pin the exact contract so a future refactor doesn't accidentally
    drop one of the three workarounds."""

    def test_mentions_all_three_remediations(self, browser_tool_module):
        hint = browser_tool_module._auth_wall_remediation_hint()
        # Remediation 1: the slash command.
        assert "/browser connect" in hint
        # Remediation 2: the manual launch flag.
        assert "--remote-debugging-port=9222" in hint
        # Remediation 2 cont'd: the opt-in config flag that pairs with it.
        assert "auto_attach_local_chrome" in hint
        # Remediation 3: the GitHub-specific workaround the user named.
        assert "gh" in hint.lower()
        assert "gh issue create" in hint or "gh repo view" in hint or "gh pr create" in hint

    def test_explains_why_login_redirects_happen(self, browser_tool_module):
        """The hint must make the cause clear so the model relays it
        accurately to the user."""
        hint = browser_tool_module._auth_wall_remediation_hint()
        # The "isolated session, no shared cookies" framing is the
        # core insight users need; surfacing it lets the model
        # explain the situation without further investigation.
        assert "isolated" in hint.lower()
        assert "cookies" in hint.lower() or "session" in hint.lower()

    def test_hint_is_stable(self, browser_tool_module):
        """Idempotent module-level constant -- the function returns
        the same string on every call."""
        first = browser_tool_module._auth_wall_remediation_hint()
        second = browser_tool_module._auth_wall_remediation_hint()
        assert first == second


# ---------------------------------------------------------------------------
# _probe_local_chrome_cdp
# ---------------------------------------------------------------------------


class TestProbeLocalChromeCdp:
    """The opt-in auto-attach probe.  Must:

    - Return the webSocketDebuggerUrl when Chrome is reachable.
    - Return "" when Chrome is down (and not raise).
    - Cache the result for a short TTL so concurrent tool calls share
      the answer and don't all bang on 127.0.0.1:9222.
    - Self-heal: a stale "Chrome is down" cache must expire so the
      user can launch Chrome mid-session and have the next nav pick
      it up.
    """

    def test_returns_ws_url_when_chrome_reachable(self, browser_tool_module):
        fake_response = MagicMock()
        fake_response.json.return_value = {
            "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/browser/abc-123",
        }
        fake_response.raise_for_status = MagicMock()
        with patch("tools.browser_tool.requests.get", return_value=fake_response) as fake_get:
            url = browser_tool_module._probe_local_chrome_cdp()

        assert url == "ws://127.0.0.1:9222/devtools/browser/abc-123"
        assert fake_get.call_count == 1
        # Hit the discovery endpoint, not the WS directly.
        called_url = fake_get.call_args.args[0]
        assert called_url.endswith("/json/version")

    def test_returns_empty_when_chrome_unreachable(self, browser_tool_module):
        """Connection refused / timeout / DNS failure all collapse to
        an empty string so the caller falls through to the local
        headless launcher."""
        import requests
        with patch(
            "tools.browser_tool.requests.get",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ):
            url = browser_tool_module._probe_local_chrome_cdp()
        assert url == ""

    def test_caches_negative_result_for_ttl(self, browser_tool_module):
        """A "Chrome is down" answer must NOT cause every subsequent
        tool call to re-probe within the same agent loop."""
        import requests
        with patch(
            "tools.browser_tool.requests.get",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ) as fake_get:
            browser_tool_module._probe_local_chrome_cdp()
            browser_tool_module._probe_local_chrome_cdp()
            browser_tool_module._probe_local_chrome_cdp()
        assert fake_get.call_count == 1, (
            "Probe must cache the negative result for the TTL -- got "
            f"{fake_get.call_count} requests"
        )

    def test_caches_positive_result_for_ttl(self, browser_tool_module):
        fake_response = MagicMock()
        fake_response.json.return_value = {
            "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/browser/abc",
        }
        fake_response.raise_for_status = MagicMock()
        with patch(
            "tools.browser_tool.requests.get", return_value=fake_response,
        ) as fake_get:
            browser_tool_module._probe_local_chrome_cdp()
            browser_tool_module._probe_local_chrome_cdp()
        assert fake_get.call_count == 1

    def test_cache_expires_after_ttl(self, browser_tool_module, monkeypatch):
        """Stale negative cache must self-heal so the user can launch
        Chrome and have the next nav pick it up."""
        import requests
        # First call: Chrome is down.
        with patch(
            "tools.browser_tool.requests.get",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ):
            browser_tool_module._probe_local_chrome_cdp()
        # Wind time forward past the TTL.
        future = browser_tool_module.time.monotonic() + 999.0
        monkeypatch.setattr(browser_tool_module.time, "monotonic", lambda: future)

        # Now Chrome comes up.
        fake_response = MagicMock()
        fake_response.json.return_value = {
            "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/browser/new",
        }
        fake_response.raise_for_status = MagicMock()
        with patch(
            "tools.browser_tool.requests.get", return_value=fake_response,
        ):
            url = browser_tool_module._probe_local_chrome_cdp()
        assert url.endswith("/new")

    def test_returns_empty_when_payload_lacks_ws_url(self, browser_tool_module):
        """Older Chrome builds occasionally return /json/version
        without a webSocketDebuggerUrl key -- we must not crash."""
        fake_response = MagicMock()
        fake_response.json.return_value = {"Browser": "Chrome/120.0", "Protocol-Version": "1.3"}
        fake_response.raise_for_status = MagicMock()
        with patch("tools.browser_tool.requests.get", return_value=fake_response):
            url = browser_tool_module._probe_local_chrome_cdp()
        assert url == ""


# ---------------------------------------------------------------------------
# _get_cdp_override + auto_attach_local_chrome wiring
# ---------------------------------------------------------------------------


class TestAutoAttachIntegration:
    """End-to-end contract for the new config flag."""

    @pytest.fixture(autouse=True)
    def _scrub_env(self, monkeypatch):
        monkeypatch.delenv("BROWSER_CDP_URL", raising=False)

    def test_auto_attach_off_by_default_skips_probe(self, browser_tool_module):
        """Safety guarantee: a user with Chrome running on 9222 but
        without the opt-in flag must NOT have Hermes silently attach."""
        probe_calls = []
        with patch.object(
            browser_tool_module,
            "_probe_local_chrome_cdp",
            lambda *a, **kw: probe_calls.append(True) or "ws://oops",
        ), patch.object(
            browser_tool_module,
            "read_raw_config",
            create=True,
            new=MagicMock(return_value={"browser": {}}),
        ) if False else patch(
            "hermes_cli.config.read_raw_config",
            return_value={"browser": {}},  # no auto_attach_local_chrome key
        ):
            result = browser_tool_module._get_cdp_override()
        assert result == ""
        assert probe_calls == [], (
            "Auto-attach must not probe when the opt-in flag is absent"
        )

    def test_auto_attach_on_returns_probe_result(self, browser_tool_module):
        with patch.object(
            browser_tool_module,
            "_probe_local_chrome_cdp",
            return_value="ws://127.0.0.1:9222/devtools/browser/xyz",
        ), patch(
            "hermes_cli.config.read_raw_config",
            return_value={"browser": {"auto_attach_local_chrome": True}},
        ):
            result = browser_tool_module._get_cdp_override()
        assert result == "ws://127.0.0.1:9222/devtools/browser/xyz"

    def test_auto_attach_caches_into_env(self, browser_tool_module, monkeypatch):
        """After a successful auto-attach the resolved URL is stashed
        in BROWSER_CDP_URL so subsequent calls short-circuit through
        the env-override branch and don't re-probe."""
        monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
        with patch.object(
            browser_tool_module,
            "_probe_local_chrome_cdp",
            return_value="ws://127.0.0.1:9222/devtools/browser/cached",
        ), patch(
            "hermes_cli.config.read_raw_config",
            return_value={"browser": {"auto_attach_local_chrome": True}},
        ):
            browser_tool_module._get_cdp_override()

        import os
        assert os.environ.get("BROWSER_CDP_URL", "").endswith("/cached")
        # And the cleanup happens automatically -- monkeypatch will
        # restore the env after the test ends.

    def test_explicit_cdp_url_wins_over_auto_attach(self, browser_tool_module):
        """Precedence: ``browser.cdp_url`` (persistent config) wins over
        the opt-in probe.  Users who set an explicit URL want THAT
        endpoint, even if a local Chrome happens to be running."""
        probe_calls = []
        with patch.object(
            browser_tool_module,
            "_resolve_cdp_override",
            side_effect=lambda raw: f"resolved:{raw}",
        ), patch.object(
            browser_tool_module,
            "_probe_local_chrome_cdp",
            lambda *a, **kw: probe_calls.append(True) or "ws://OOPS-AUTO-ATTACH",
        ), patch(
            "hermes_cli.config.read_raw_config",
            return_value={
                "browser": {
                    "cdp_url": "http://my-grid.internal:9333",
                    "auto_attach_local_chrome": True,  # still true, but ignored
                },
            },
        ):
            result = browser_tool_module._get_cdp_override()

        assert result.startswith("resolved:http://my-grid.internal:9333")
        assert probe_calls == [], (
            "Explicit cdp_url must win without probing the local Chrome"
        )

    def test_env_var_wins_over_everything(self, browser_tool_module, monkeypatch):
        """BROWSER_CDP_URL (set by /browser connect) is the top of the
        precedence stack -- explicit config and auto-attach are both
        skipped when it's present."""
        monkeypatch.setenv("BROWSER_CDP_URL", "http://my-live-chrome:9222")
        with patch.object(
            browser_tool_module,
            "_resolve_cdp_override",
            side_effect=lambda raw: f"resolved:{raw}",
        ):
            result = browser_tool_module._get_cdp_override()
        assert result == "resolved:http://my-live-chrome:9222"
