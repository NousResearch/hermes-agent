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
def test_raw_cdp_reads_the_host_after_any_userinfo(cdp_guard, local_backend):
    """Userinfo may also hold a quote, backtick, space or ``[``: the metadata host after the last ``@`` is
    still read, and a raw ``url`` the parser rejects fails closed instead of passing as not metadata."""
    for userinfo in ('1.1.1.1"', "1.1.1.1`", "1.1.1.1 x", "1.1.1.1[", "1.1.1.1'"):
        quote = '"' if "'" in userinfo else "'"
        call = {"functionDeclaration": f"() => fetch({quote}http://{userinfo}@169.254.169.254/latest/{quote})"}
        assert "cloud metadata" in cdp_guard(method="Runtime.callFunctionOn", params=call), userinfo
    raw = {"frameId": "F", "url": "http://a[@169.254.169.254/latest/", "options": {}}
    assert "cloud metadata" in cdp_guard(method="Network.loadNetworkResource", params=raw)
    broken = {"frameId": "F", "url": "http://[93.184.216.34/", "options": {}}
    assert "cannot be parsed" in cdp_guard(method="Network.loadNetworkResource", params=broken)
    # Past a quote, an @ is a mail address in the next string, not this URL's host.
    mail = {"functionDeclaration": "() => send('https://93.184.216.34', 'me@127.0.0.1')"}
    assert cdp_guard(method="Runtime.callFunctionOn", params=mail) is None
    # Nor is text with a control character after an ``@`` past a quote a metadata host.
    text = {"functionDeclaration": "() => { fetch('https://93.184.216.34'); s = 'a@b\x01c' }"}
    assert cdp_guard(method="Runtime.callFunctionOn", params=text) is None
    # But a control character right after the metadata host ends it: a browser strips it and loads the host.
    for literal in ('http://"@169.254.169.254\x01', 'http://"@169.254.169.254\x00', '//"@169.254.169.254\x1f'):
        call = {"functionDeclaration": f"() => fetch('{literal}')"}
        assert "cloud metadata" in cdp_guard(method="Runtime.callFunctionOn", params=call), repr(literal)


def test_metadata_read_past_quotes_returns_one_literal_per_url_head():
    """Many metadata ``@host`` after one long URL head, or many URL heads each with one, still refuse the
    call, and the literals returned stay about as long as the text: one per head, not one per ``@`` that
    repeats the head, and each cut at the next head."""
    for text in ("'http://" + "/" * 20_000 + "@169.254.169.254 " * 2_000, "http://@169.254.169.254/" * 2_000):
        literals = browser_url_policy.url_literals(text, protocol_relative=True)
        assert any(browser_url_policy.literal_targets_metadata(lit) for lit in literals)
        assert sum(map(len, literals)) < 4 * len(text), text[:30]


class _CountingPattern:
    """A literal regex that counts the characters it is asked to match, to bound ``url_literals``' work."""

    def __init__(self, pattern):
        self.pattern, self.chars = pattern, 0

    def finditer(self, text):
        return self.pattern.finditer(text)

    def match(self, text):
        self.chars += len(text)
        return self.pattern.match(text)


def test_browser_exec_url_literals_work_stays_linear():
    """With ``browser_exec``'s literal regex, the work stays about as large as the text: a URL head of
    backslashes (which that regex never matches) followed by many metadata ``@host`` is read once, not once
    per ``@``; and a run of ``http://u@host\\n`` literals reads each host only up to the next URL head."""
    for text in ("'http://" + "\\" * 20_000 + "@169.254.169.254 " * 2_000,
                 'print("' + "http://u@93.184.216.34\\n" * 2_000):
        counting = _CountingPattern(browser_use_cli._URL_RE)
        literals = browser_url_policy.url_literals(text, pattern=counting)
        assert counting.chars < 4 * len(text), (text[:30], counting.chars, len(text))
        assert sum(map(len, literals)) < 4 * len(text), text[:30]


@pytest.mark.parametrize("local_backend", [False, True])
def test_metadata_floor_reads_the_last_at_from_every_url_head(cdp_guard, local_backend):
    """An earlier ``@`` (``http://u@1.1.1.1"@host/``) or a quote or space right after the scheme
    (``http://"@host/``) does not hide the metadata host a browser loads, in raw CDP or ``browser_exec``."""
    for target in ('http://u@1.1.1.1"@169.254.169.254/', "http://u@1.1.1.1 x@169.254.169.254/",
                   'http://"@169.254.169.254/', "http:// @169.254.169.254/", '//"@169.254.169.254/'):
        call = {"functionDeclaration": f"() => location.href = '{target}'"}
        assert "cloud metadata" in cdp_guard(method="Runtime.callFunctionOn", params=call), target
        if target.startswith("http"):
            verdict = browser_use_cli._code_url_verdict(f"new_tab('{target}')").blocked
            # Guard on, the earlier ``u@1.1.1.1`` literal is refused first, as any userinfo URL is.
            assert verdict and ("cloud metadata" in verdict or not local_backend), target
    # Nor does a later mail address once the URL's string has closed, with no ``/`` after the host.
    for target in ('http://"@169.254.169.254', "http:// @169.254.169.254", '//"@169.254.169.254'):
        code = f"location.href = '{target}'; const email = 'x@93.184.216.34'"
        assert "cloud metadata" in cdp_guard(method="Runtime.callFunctionOn", params={"functionDeclaration": code})
        if target.startswith("http"):
            assert "cloud metadata" in (browser_use_cli._code_url_verdict(code).blocked or ""), target
    # Within one authority the last ``@`` wins; a public URL carrying another URL as query data is its host only.
    for url in ("http://a@169.254.169.254@93.184.216.34/", "https://93.184.216.34/?next=http://u@93.184.216.34/"):
        verdict = cdp_guard(method="Network.loadNetworkResource", params={"frameId": "F", "url": url, "options": {}})
        assert verdict is None if local_backend else "cloud metadata" not in (verdict or ""), url
    # Mail addresses after a bare scheme or an earlier URL are still not hosts.
    mail = {"functionDeclaration": "() => send('https://93.184.216.34', 'see http:// docs', 'me@127.0.0.1')"}
    assert cdp_guard(method="Runtime.callFunctionOn", params=mail) is None
    assert browser_use_cli._code_url_verdict(mail["functionDeclaration"]).blocked is None


@pytest.mark.parametrize("local_backend", [False, True])
def test_browser_exec_reads_the_host_after_userinfo(monkeypatch, local_backend):
    """``browser_exec`` reads the host after userinfo like raw CDP does; ``\\`` still ends a literal, as in
    Python source it starts an escape."""
    monkeypatch.setattr(browser_session_cloud, "is_local_backend", lambda: local_backend)
    monkeypatch.setattr(browser_session_cloud, "allow_private_urls", lambda: False)
    for code in ('new_tab("http://1.1.1.1)@169.254.169.254/latest/")', "new_tab('http://1.1.1.1\"@169.254.169.254/')"):
        assert "cloud metadata" in (browser_use_cli._code_url_verdict(code).blocked or ""), code
    assert browser_use_cli._code_url_verdict('print("opened http://93.184.216.34\\n")').blocked is None
    # The host after userinfo is cut where the literal is: a \n escape is not part of it. With the guard on, any
    # userinfo URL is refused as ambiguous (unchanged); off, nothing is refused.
    verdict = browser_use_cli._code_url_verdict('print("https://u@93.184.216.34\\n")').blocked
    assert verdict is None if local_backend else "private or internal" in verdict


@pytest.mark.parametrize("local_backend", [False, True])
def test_raw_cdp_scheme_relative_literals_cost_no_dns_lookup(cdp_guard, monkeypatch):
    """``//word`` comments are judged by their literal host: no resolver call per comment."""
    from tools import url_safety
    lookups = []
    monkeypatch.setattr(url_safety, "_getaddrinfo", lambda host, port=None: lookups.append(host) or [])
    script = "\n".join(f"//step{i} fill the form" for i in range(20)) + "\n//eslint-disable-next-line\nrun()"
    assert cdp_guard(method="Runtime.evaluate", params={"expression": script}) is None
    assert lookups == []
