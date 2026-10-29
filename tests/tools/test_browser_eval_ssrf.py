"""Page-driven JS must not bypass the browser SSRF guard via URL literals.

``fetch('http://127.0.0.1/secret')`` never updates ``location.href``, so no page-URL recheck can see it.
``browser_exec`` pre-scans every URL literal in the code it runs (``_code_url_verdict``) and
``browser_cdp`` pre-scans ``Runtime.evaluate`` expressions (``expression_targets_private_url``): a
private-host literal is refused before anything reaches the browser.

Sibling fix for the eval return-value path of issue #44731.
"""

import json

import pytest

from tools import browser_session_cloud, browser_url_policy, browser_use_cli

PRIVATE_URL = "http://127.0.0.1:8080/secret"
PUBLIC_URL = "https://example.com/page"
METADATA_URL = "http://169.254.169.254/latest/meta-data/"


@pytest.fixture(autouse=True)
def _no_policy_or_hybrid(monkeypatch):
    monkeypatch.setattr(browser_url_policy, "check_website_access", lambda url: None)
    monkeypatch.setattr(browser_url_policy, "hybrid_routes_locally", lambda url: False)


class TestExpressionPreScan:
    def _guard_on(self, monkeypatch):
        monkeypatch.setattr(browser_session_cloud, "is_local_backend", lambda: False)
        monkeypatch.setattr(browser_session_cloud, "allow_private_urls", lambda: False)

    def test_blocks_private_fetch_literal(self, monkeypatch):
        self._guard_on(monkeypatch)
        monkeypatch.setattr(browser_url_policy, "is_safe_url", lambda url: False)
        monkeypatch.setattr(browser_url_policy, "is_always_blocked_url", lambda url: False)

        def _never_spawn():
            raise AssertionError("browser-harness must not run for a private URL literal")

        monkeypatch.setattr(browser_use_cli, "_find_cli", _never_spawn)

        result = json.loads(browser_use_cli.browser_exec(f"js(\"fetch('{PRIVATE_URL}').then(r => r.text())\")"))
        assert "private or internal address" in result["error"]

    def test_blocks_metadata_fetch_literal(self, monkeypatch):
        self._guard_on(monkeypatch)
        # Public-safe to is_safe_url, but the always-blocked floor catches IMDS.
        monkeypatch.setattr(browser_url_policy, "is_safe_url", lambda url: True)
        monkeypatch.setattr(browser_url_policy, "is_always_blocked_url", lambda url: "169.254.169.254" in url)

        verdict = browser_use_cli._code_url_verdict(f"js(\"fetch('{METADATA_URL}')\")")
        assert verdict.blocked is not None
        assert "cloud metadata endpoint" in verdict.blocked

    def test_allows_public_fetch_literal(self, monkeypatch):
        self._guard_on(monkeypatch)
        monkeypatch.setattr(browser_url_policy, "is_safe_url", lambda url: True)
        monkeypatch.setattr(browser_url_policy, "is_always_blocked_url", lambda url: False)

        verdict = browser_use_cli._code_url_verdict(f"js(\"fetch('{PUBLIC_URL}').then(r => r.text())\")")
        assert verdict.blocked is None

    def test_skips_prescan_when_allow_private(self, monkeypatch):
        monkeypatch.setattr(browser_session_cloud, "is_local_backend", lambda: False)
        monkeypatch.setattr(browser_session_cloud, "allow_private_urls", lambda: True)
        monkeypatch.setattr(browser_url_policy, "is_safe_url", lambda url: False)
        monkeypatch.setattr(browser_url_policy, "is_always_blocked_url", lambda url: False)

        verdict = browser_use_cli._code_url_verdict(f"js(\"fetch('{PRIVATE_URL}')\")")
        assert verdict.blocked is None


class TestExpressionScanHelper:
    def test_returns_first_private_literal(self, monkeypatch):
        monkeypatch.setattr(browser_url_policy, "is_safe_url", lambda url: "127.0.0.1" not in url)
        monkeypatch.setattr(browser_url_policy, "is_always_blocked_url", lambda url: False)
        out = browser_url_policy.expression_targets_private_url(
            "fetch('https://example.com'); fetch('http://127.0.0.1/x')"
        )
        assert out == "http://127.0.0.1/x"

    def test_strips_trailing_punctuation(self, monkeypatch):
        monkeypatch.setattr(browser_url_policy, "is_safe_url", lambda url: False)
        monkeypatch.setattr(browser_url_policy, "is_always_blocked_url", lambda url: False)
        out = browser_url_policy.expression_targets_private_url("location.href='http://10.0.0.1/';")
        assert out == "http://10.0.0.1/"


@pytest.mark.parametrize("local_backend,allow_private", [(True, False), (False, True)])
def test_raw_cdp_keeps_the_metadata_floor_when_the_private_guard_is_off(monkeypatch, local_backend, allow_private):
    from tools import browser_cdp_tool, browser_session_cloud
    monkeypatch.setattr(browser_session_cloud, "is_local_backend", lambda: local_backend)
    monkeypatch.setattr(browser_session_cloud, "allow_private_urls", lambda: allow_private)
    guard = browser_cdp_tool._browser_cdp_private_guard
    for method in ("Page.navigate", "Target.createTarget", "Network.loadNetworkResource", "Fetch.continueRequest"):
        assert "cloud metadata" in guard(method=method, params={"url": "http://169.254.169.254/latest/meta-data/"})
        assert guard(method=method, params={"url": "http://192.168.1.5/"}) is None
    for template in ("location.href='{}'", "fetch('{}').then(r => r.text())"):
        metadata = {"expression": template.format("http://169.254.169.254/latest/meta-data/")}
        assert "cloud metadata" in guard(method="Runtime.evaluate", params=metadata)
        lan = {"expression": template.format("http://192.168.1.5/")}
        assert guard(method="Runtime.evaluate", params=lan) is None


@pytest.mark.parametrize("local_backend", [False, True])
def test_raw_cdp_url_params_are_screened_even_when_the_open_page_is_public(monkeypatch, local_backend):
    """A public page does not vouch for a call's own URLs: unlisted resource-loading methods keep the metadata
    floor with the private guard on or off, and the private-address check whenever the guard is on."""
    from tools import browser_cdp_tool, browser_session_cloud
    monkeypatch.setattr(browser_session_cloud, "is_local_backend", lambda: local_backend)
    monkeypatch.setattr(browser_session_cloud, "allow_private_urls", lambda: False)
    monkeypatch.setattr(browser_cdp_tool, "_current_page_private_url", lambda endpoint, target_id: None)
    guard = browser_cdp_tool._browser_cdp_private_guard
    metadata = "http://metadata.google.internal/computeMetadata/v1/"
    calls = (("Network.loadNetworkResource", {"frameId": "F", "url": metadata, "options": {}}),
             ("Runtime.callFunctionOn", {"functionDeclaration": f"() => fetch('{metadata}')"}),
             ("Page.addScriptToEvaluateOnNewDocument", {"source": f"fetch(`{metadata}`)"}))
    for method, params in calls:
        assert "cloud metadata" in guard(method=method, params=params, endpoint="ws://x", target_id="T")
    loopback = {"frameId": "F", "url": "http://127.0.0.1:8080/", "options": {}}
    verdict = guard(method="Network.loadNetworkResource", params=loopback, endpoint="ws://x", target_id="T")
    assert (verdict is None) == local_backend
    public = {"frameId": "F", "url": "https://93.184.216.34/", "options": {}}
    assert guard(method="Network.loadNetworkResource", params=public, endpoint="ws://x", target_id="T") is None
    # Typing a URL fetches nothing: text input stays allowed, metadata and LAN addresses alike.
    for typed in (metadata, "http://192.168.1.10/"):
        assert guard(method="Input.insertText", params={"text": typed}, endpoint="ws://x", target_id="T") is None
