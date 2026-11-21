"""Page-driven JS must not bypass the browser SSRF guard via URL literals.

``fetch('http://127.0.0.1/secret')`` never updates ``location.href``, so no page-URL recheck can see it.
``browser_exec`` pre-scans every URL literal in the code it runs (``_code_url_verdict``) and
``browser_cdp`` pre-scans ``Runtime.evaluate`` expressions (``expression_targets_private_url``): a
private-host literal is refused before anything reaches the browser.

Sibling fix for the eval return-value path of issue #44731.
"""

import functools
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


@pytest.fixture
def cdp_guard(monkeypatch, local_backend):
    """The raw CDP guard with a public page open: private guard on for a remote backend, off for a local one."""
    from tools import browser_cdp_tool
    monkeypatch.setattr(browser_session_cloud, "is_local_backend", lambda: local_backend)
    monkeypatch.setattr(browser_session_cloud, "allow_private_urls", lambda: False)
    monkeypatch.setattr(browser_cdp_tool, "_current_page_private_url", lambda endpoint, target_id: None)
    return functools.partial(browser_cdp_tool._browser_cdp_private_guard, endpoint="ws://x", target_id="T")


@pytest.mark.parametrize("local_backend", [False, True])
def test_raw_cdp_url_params_are_screened_even_when_the_open_page_is_public(cdp_guard, local_backend):
    """A public page does not vouch for a call's own URLs: unlisted resource-loading methods keep the metadata
    floor with the private guard on or off, and the private-address check whenever the guard is on."""
    metadata = "http://metadata.google.internal/computeMetadata/v1/"
    calls = (("Network.loadNetworkResource", {"frameId": "F", "url": metadata, "options": {}}),
             ("Runtime.callFunctionOn", {"functionDeclaration": f"() => fetch('{metadata}')"}),
             ("Page.addScriptToEvaluateOnNewDocument", {"source": f"fetch(`{metadata}`)"}))
    for method, params in calls:
        assert "cloud metadata" in cdp_guard(method=method, params=params)
    loopback = {"frameId": "F", "url": "http://127.0.0.1:8080/", "options": {}}
    assert (cdp_guard(method="Network.loadNetworkResource", params=loopback) is None) == local_backend
    public = {"frameId": "F", "url": "https://93.184.216.34/", "options": {}}
    assert cdp_guard(method="Network.loadNetworkResource", params=public) is None
    # Typing a URL fetches nothing: text input stays allowed, metadata and LAN addresses alike.
    for typed in (metadata, "http://192.168.1.10/"):
        assert cdp_guard(method="Input.insertText", params={"text": typed}) is None


@pytest.mark.parametrize("local_backend", [False, True])
def test_raw_cdp_params_the_call_never_fetches_are_not_screened(cdp_guard):
    """Blocklist and interception patterns, dialog text and cookie scope URLs name hosts without fetching
    them: wildcard and loopback patterns pass, and so does blocking the metadata address itself."""
    calls = (("Network.setBlockedURLs", {"urls": ["https://*.doubleclick.net/*", "http://127.0.0.1/*",
                                                   "http://169.254.169.254/*"]}),
             ("Fetch.enable", {"patterns": [{"urlPattern": "https://*.example.invalid/*"}]}),
             ("Page.handleJavaScriptDialog", {"accept": True, "promptText": "http://127.0.0.1:8080/"}),
             ("Network.setCookie", {"name": "a", "value": "b", "url": "http://localhost:3000/"}))
    for method, params in calls:
        assert cdp_guard(method=method, params=params) is None, method
    # An attribute the page then loads is still a fetch.
    src = {"nodeId": 1, "name": "src", "value": "http://169.254.169.254/latest/meta-data/"}
    assert "cloud metadata" in cdp_guard(method="DOM.setAttributeValue", params=src)


@pytest.mark.parametrize("local_backend", [False, True])
def test_raw_cdp_reads_bracketed_ipv6_hosts_whole(cdp_guard):
    """``[...]`` hosts are one literal: a public IPv6 URL passes, an IPv4-mapped metadata address does not."""
    public = {"frameId": "F", "url": "https://[2606:4700:4700::1111]/dns", "options": {}}
    assert cdp_guard(method="Network.loadNetworkResource", params=public) is None
    mapped = {"functionDeclaration": "() => fetch('http://[::ffff:a9fe:a9fe]/latest/meta-data/')"}
    assert "cloud metadata" in cdp_guard(method="Runtime.callFunctionOn", params=mapped)


@pytest.mark.parametrize("local_backend", [False, True])
def test_raw_cdp_metadata_floor_reads_scheme_relative_urls(cdp_guard):
    """``//host``, ``///host`` and ``\\\\host`` load with the page's scheme; a ``//`` comment, a path, or a
    ``//`` inside another URL's query is not a second target."""
    for target in ("//169.254.169.254/latest/meta-data/", "///169.254.169.254/latest/", "\\\\169.254.169.254\\latest",
                   "//metadata.google.internal/computeMetadata/v1/", "//2852039166/", "//0xA9FEA9FE/latest/",
                   "//0251.0376.0251.0376/", "//169.254.43518/", "//169.254.169.254\x01"):
        verdict = cdp_guard(method="Runtime.evaluate", params={"expression": f"location.href='{target}'"})
        assert "cloud metadata" in verdict, target
    html = {"expression": "document.body.innerHTML = '<img src=//169.254.169.254/latest/>'"}
    assert "cloud metadata" in cdp_guard(method="Runtime.evaluate", params=html)
    src = {"nodeId": 1, "name": "src", "value": "//0xa9fea9fe/latest/"}
    assert "cloud metadata" in cdp_guard(method="DOM.setAttributeValue", params=src)
    harmless = {"expression": "// note: see docs\nconst p = 'a//b'; fetch('https://93.184.216.34//x')"}
    assert cdp_guard(method="Runtime.evaluate", params=harmless) is None
    # A //TODO comment or a scheme-relative public script is not an http(s) literal for the private check.
    script = {"functionDeclaration": "function(){ //TODO\n import('//cdn.jsdelivr.net/npm/a.js') }"}
    assert cdp_guard(method="Runtime.callFunctionOn", params=script) is None
    for url in ("https://1.1.1.1/?next=//169.254.169.254/", "https://[2606:4700:4700::1111]//169.254.169.254/x"):
        assert cdp_guard(method="Network.loadNetworkResource", params={"frameId": "F", "url": url}) is None, url


@pytest.mark.parametrize("local_backend", [False, True])
def test_raw_cdp_reads_the_host_after_userinfo(cdp_guard, local_backend):
    """``)``, ``]``, ``<`` and ``>`` are legal in userinfo, so ``http://x)@169.254.169.254/`` goes to the
    metadata host: a literal cut at one of them inside its authority is also read past the last ``@``."""
    for userinfo in ("1.1.1.1)", "1.1.1.1]", "1.1.1.1<", "1.1.1.1>", "1.1.1.1)@1.1.1.1]"):
        for prefix in ("http://", "//"):
            target = f"{prefix}{userinfo}@169.254.169.254/latest/meta-data/"
            call = {"functionDeclaration": f"() => fetch('{target}')"}
            assert "cloud metadata" in cdp_guard(method="Runtime.callFunctionOn", params=call), target
    html = {"expression": "document.body.innerHTML = '<img src=\"http://1.1.1.1)@169.254.169.254/latest/\">'"}
    assert "(http://169.254.169.254/latest/)" in cdp_guard(method="Runtime.evaluate", params=html)
    raw = {"frameId": "F", "url": "http://1.1.1.1]@169.254.169.254/latest/", "options": {}}
    assert "cloud metadata" in cdp_guard(method="Network.loadNetworkResource", params=raw)
    loopback = {"functionDeclaration": "() => fetch('http://1.1.1.1)@127.0.0.1:8080/')"}
    assert (cdp_guard(method="Runtime.callFunctionOn", params=loopback) is None) == local_backend
    # The literal as cut is still read too: an address after an unquoted href does not hide the href.
    href = {"functionDeclaration": "() => el.innerHTML = '<a href=http://127.0.0.1:8080>me@93.184.216.34</a>'"}
    assert (cdp_guard(method="Runtime.callFunctionOn", params=href) is None) == local_backend
    # An @ after the authority ended (path, query) leaves the host alone.
    path = {"frameId": "F", "url": "https://93.184.216.34/a)@169.254.169.254/", "options": {}}
    assert cdp_guard(method="Network.loadNetworkResource", params=path) is None


@pytest.mark.parametrize("local_backend", [False, True])
def test_raw_cdp_scheme_relative_literals_cost_no_dns_lookup(cdp_guard, monkeypatch):
    """``//word`` comments are judged by their literal host: no resolver call per comment."""
    from tools import url_safety
    lookups = []
    monkeypatch.setattr(url_safety, "_getaddrinfo", lambda host, port=None: lookups.append(host) or [])
    script = "\n".join(f"//step{i} fill the form" for i in range(20)) + "\n//eslint-disable-next-line\nrun()"
    assert cdp_guard(method="Runtime.evaluate", params={"expression": script}) is None
    assert lookups == []
