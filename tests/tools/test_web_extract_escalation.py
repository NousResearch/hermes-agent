import asyncio
import itertools
import json
import sys
import threading
import types

import pytest


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


class FakeWebSocket:
    def __init__(self, responder):
        self.sent = []
        self._inbound = asyncio.Queue()
        self._responder = responder

    async def send(self, raw):
        message = json.loads(raw)
        self.sent.append(message)
        self._responder(self, message)

    async def recv(self):
        return await self._inbound.get()

    def feed(self, payload):
        self._inbound.put_nowait(json.dumps(payload))


class FakeConnection:
    def __init__(self, websocket):
        self.websocket = websocket

    async def __aenter__(self):
        return self.websocket

    async def __aexit__(self, exc_type, exc, traceback):
        return False


def install_fake_websockets(monkeypatch, websocket):
    module = types.SimpleNamespace(
        connect=lambda *args, **kwargs: FakeConnection(websocket)
    )
    monkeypatch.setitem(sys.modules, "websockets", module)


def standard_cdp_response(websocket, message, *, reply_to_close=True):
    method = message["method"]
    result = {}

    if method == "Target.createTarget":
        result = {"targetId": "target-1"}
    elif method == "Target.attachToTarget":
        result = {"sessionId": "session-1"}
    elif method == "Runtime.evaluate":
        expression = message.get("params", {}).get("expression", "")
        if expression == "location.href":
            result = {
                "result": {
                    "value": "https://public.example/final",
                }
            }
        else:
            result = {
                "result": {
                    "value": json.dumps(
                        {
                            "title": "Public page",
                            "content": "public content",
                        }
                    )
                }
            }
    elif method == "Page.navigate":
        websocket.feed({"id": message["id"], "result": {}})
        websocket.feed(
            {
                "sessionId": "session-1",
                "method": "Page.loadEventFired",
                "params": {},
            }
        )
        return
    elif method == "Target.closeTarget" and not reply_to_close:
        return

    websocket.feed({"id": message["id"], "result": result})


class FakeCloudProvider:
    def __init__(
        self,
        name="browser-use",
        available=True,
        close_result=True,
        create_started=None,
        create_release=None,
    ):
        self.name = name
        self.available = available
        self.close_result = close_result
        self.create_started = create_started
        self.create_release = create_release
        self.created = []
        self.closed = []
        self.emergency_cleanups = []

    def is_available(self):
        return self.available

    def create_session(self, task_id):
        self.created.append(task_id)
        if self.create_started is not None:
            self.create_started.set()
        if self.create_release is not None:
            self.create_release.wait(timeout=2)
        return {
            "bb_session_id": "cloud-session-1",
            "cdp_url": "wss://cloud.example/devtools/browser/1",
        }

    def close_session(self, session_id):
        self.closed.append(session_id)
        return self.close_result

    def emergency_cleanup(self, session_id):
        self.emergency_cleanups.append(session_id)


def test_block_signal_detects_jina_captcha_notice():
    from tools import web_routing

    result = {
        "url": "https://example.com",
        "content": "Warning: This page maybe requiring CAPTCHA",
        "raw_content": "Warning: This page maybe requiring CAPTCHA",
        "metadata": {"status_code": 200},
    }

    assert web_routing._extract_block_signal(result)


def test_block_signal_does_not_treat_404_as_escalation():
    from tools import web_routing

    result = {
        "url": "https://example.com/missing",
        "error": "Jina Reader returned 404",
        "metadata": {"status_code": 404},
    }

    assert web_routing._extract_block_signal(result) is None
    assert web_routing._extract_is_technical_failure(result) is False


def test_block_signal_detects_jina_wrapped_403_even_on_full_page():
    """Jina wraps upstream 403 in a 200 response — must still escalate."""
    from tools import web_routing

    jina_403 = (
        "Title: \n\n"
        "URL Source: https://www.reddit.com/r/peptides/\n\n"
        "Warning: Target URL returned error 403: Forbidden\n\n"
        "Markdown Content:\nYou've been blocked by network security.\n\n"
        "To continue, log in to your Reddit account or use your developer token\n"
    )
    assert len(jina_403.strip()) > web_routing._NEAR_EMPTY_BLOCK_CHARS

    result = {
        "url": "https://www.reddit.com/r/peptides/",
        "content": jina_403,
        "raw_content": jina_403,
        "metadata": {"status_code": 200},
    }

    signal = web_routing._extract_block_signal(result)
    assert signal
    assert "403" in signal


def test_block_signal_detects_g2_jina_captcha_notice_under_200_chars():
    """Regression: Jina's G2 CAPTCHA warning is 193 chars and must escalate."""
    from tools import web_routing

    jina_notice = (
        "Title: g2.com\n\n"
        "URL Source: https://www.g2.com/products/firecrawl/reviews\n\n"
        "Warning: This page maybe requiring CAPTCHA, please make sure you are authorized to access this page.\n\n"
        "Markdown Content:\n\n"
    )
    assert len(jina_notice.strip()) < 200

    result = {
        "url": "https://www.g2.com/products/firecrawl/reviews",
        "content": jina_notice,
        "raw_content": jina_notice,
        "metadata": {"status_code": 200},
    }

    signal = web_routing._extract_block_signal(result)
    assert signal
    assert "captcha" in signal.lower()


def test_hard_interstitial_detects_datadome_from_browser_lane():
    """A full-size DataDome interstitial from Browser Use is classified as blocked."""
    from tools import web_routing

    datadome_page = (
        '- Iframe "DataDome Device Check" [ref=e1]\n'
        "  - generic\n"
        "    - paragraph\n"
        '      - StaticText "Access is temporarily restricted"\n'
    ) * 3
    assert len(datadome_page.strip()) > web_routing._HARD_INTERSTITIAL_MIN_CHARS

    result = {
        "url": "https://blocked.example",
        "content": datadome_page,
        "raw_content": datadome_page,
        "metadata": {"lane": "browser_use_cloud"},
    }

    assert web_routing._is_hard_interstitial(result)
    # Should NOT be caught by normal block signal (page is too large)
    assert web_routing._extract_block_signal(result) is None


def test_block_signal_does_not_escalate_on_marker_in_full_page():
    """A page with real content that merely mentions 'captcha' is NOT a block."""
    from tools import web_routing

    real_page = "This is a normal article discussing captcha detection systems. " * 20
    result = {
        "url": "https://example.com",
        "content": real_page,
        "raw_content": real_page,
        "metadata": {"status_code": 200},
    }

    assert web_routing._extract_block_signal(result) is None


def test_block_signal_detects_403_alone():
    from tools import web_routing

    result = {
        "url": "https://blocked.example",
        "content": "some content",
        "raw_content": "some content",
        "metadata": {"status_code": 403},
    }

    signal = web_routing._extract_block_signal(result)
    assert signal
    assert "403" in signal


def test_block_signal_detects_429_alone():
    from tools import web_routing

    result = {
        "url": "https://rate-limited.example",
        "error": "Rate limit",
        "metadata": {"status_code": 429},
    }

    signal = web_routing._extract_block_signal(result)
    assert signal
    assert "429" in signal


@pytest.mark.asyncio
async def test_jina_block_skips_unsafe_automatic_home_chrome_and_uses_cloud(
    monkeypatch,
):
    from tools import web_routing

    class JinaProvider:
        name = "jina"

        async def extract(self, urls, **kwargs):
            return [{
                "url": urls[0],
                "content": "Warning: This page maybe requiring CAPTCHA",
                "raw_content": "Warning: This page maybe requiring CAPTCHA",
                "metadata": {"status_code": 200},
            }]

    async def cloud_success(url, lanes):
        return {
            "url": url,
            "title": "OK",
            "content": "isolated cloud content",
            "raw_content": "isolated cloud content",
        }

    monkeypatch.setattr(
        web_routing, "_extract_via_browser_use_cloud", cloud_success
    )

    results = await web_routing._extract_with_jina_escalation(JinaProvider(), ["https://blocked.example"], format=None)

    assert results[0]["title"] == "OK"
    assert results[0]["content"] == "isolated cloud content"
    assert results[0]["routing"]["lanes"] == [
        "jina",
        "home_chrome_cdp",
        "browser_use_cloud",
    ]


@pytest.mark.asyncio
async def test_large_initial_jina_interstitial_escalates_to_home_chrome(monkeypatch):
    from tools import web_routing

    interstitial = (
        "DataDome Device Check\n"
        "Access is temporarily restricted\n"
        + ("Complete the browser verification challenge. " * 20)
    )
    initial = {
        "content": interstitial,
        "raw_content": interstitial,
    }
    assert web_routing._extract_block_signal(initial) is None
    assert web_routing._is_hard_interstitial(initial)

    class JinaProvider:
        name = "jina"

        async def extract(self, urls, **kwargs):
            return [{
                "url": urls[0],
                "content": interstitial,
                "raw_content": interstitial,
                "metadata": {"status_code": 200},
            }]

    calls = []

    async def chrome_refusal(url):
        calls.append(("chrome", url))
        return {"url": url, "error": "automatic Home Chrome is disabled"}

    async def cloud_success(url, lanes):
        calls.append(("cloud", url))
        return {
            "url": url,
            "title": "Recovered",
            "content": "isolated cloud content",
            "raw_content": "isolated cloud content",
        }

    monkeypatch.setattr(
        web_routing, "_extract_via_home_chrome_cdp", chrome_refusal
    )
    monkeypatch.setattr(
        web_routing, "_extract_via_browser_use_cloud", cloud_success
    )

    results = await web_routing._extract_with_jina_escalation(
        JinaProvider(), ["https://blocked.example"], format=None
    )

    assert calls == [
        ("chrome", "https://blocked.example"),
        ("cloud", "https://blocked.example"),
    ]
    assert results[0]["content"] == "isolated cloud content"
    assert results[0]["routing"]["lanes"] == [
        "jina",
        "home_chrome_cdp",
        "browser_use_cloud",
    ]
    assert "interstitial" in results[0]["routing"]["escalation_reason"].lower()


@pytest.mark.asyncio
async def test_automatic_home_chrome_lane_fails_closed():
    from tools import web_routing

    result = await web_routing._extract_via_home_chrome_cdp(
        "https://public.example"
    )

    assert result.get("error")
    assert "disabled" in result["error"].lower()
    assert "explicit browser" in result["error"].lower()


@pytest.mark.asyncio
async def test_browser_use_escalation_logs_once(monkeypatch, tmp_path):
    from tools import web_routing

    class JinaProvider:
        name = "jina"

        async def extract(self, urls, **kwargs):
            return [{
                "url": urls[0],
                "content": "DataDome captcha",
                "raw_content": "DataDome captcha",
                "metadata": {"status_code": 403},
            }]

    async def blocked_chrome(url):
        return {"url": url, "content": "Cloudflare captcha", "raw_content": "Cloudflare captcha"}

    async def browser_use_success(url, lanes):
        web_routing._append_capped_lane_log(url, lanes, "success")
        return {"url": url, "title": "OK", "content": "browser use content", "raw_content": "browser use content"}

    monkeypatch.setattr(web_routing, "_extract_via_home_chrome_cdp", blocked_chrome)
    monkeypatch.setattr(web_routing, "_extract_via_browser_use_cloud", browser_use_success)
    monkeypatch.setattr(web_routing, "get_hermes_home", lambda: tmp_path)

    results = await web_routing._extract_with_jina_escalation(JinaProvider(), ["https://blocked.example"], format=None)

    assert results[0]["content"] == "browser use content"
    log = tmp_path.joinpath("logs/lane_escalation.log").read_text()
    assert "https://blocked.example" in log
    assert "lanes=jina,home_chrome_cdp,browser_use_cloud" in log


def test_lane_log_removes_userinfo_query_and_fragment(monkeypatch, tmp_path):
    from tools import web_routing

    monkeypatch.setattr(web_routing, "get_hermes_home", lambda: tmp_path)

    web_routing._append_capped_lane_log(
        "https://user:password@example.com/path?access_token=secret#private",
        ["jina", "browser_use_cloud"],
        "failed",
    )

    log = tmp_path.joinpath("logs/lane_escalation.log").read_text()
    assert "https://example.com/path" in log
    assert "user" not in log
    assert "password" not in log
    assert "access_token" not in log
    assert "secret" not in log
    assert "private" not in log


@pytest.mark.asyncio
async def test_cdp_call_times_out_when_no_response_arrives():
    from tools import web_routing

    class SilentWebSocket:
        async def send(self, raw):
            return None

        async def recv(self):
            await asyncio.Future()

    with pytest.raises(TimeoutError, match="Runtime.evaluate"):
        await asyncio.wait_for(
            web_routing._cdp_call(
                SilentWebSocket(),
                itertools.count(1),
                "Runtime.evaluate",
                timeout_s=0.02,
            ),
            timeout=0.25,
        )


@pytest.mark.asyncio
async def test_cdp_events_do_not_reset_command_deadline():
    from tools import web_routing

    class EventFloodWebSocket:
        async def send(self, raw):
            return None

        async def recv(self):
            await asyncio.sleep(0.005)
            return json.dumps(
                {
                    "method": "Page.frameNavigated",
                    "params": {},
                }
            )

    with pytest.raises(TimeoutError, match="Page.navigate"):
        await asyncio.wait_for(
            web_routing._cdp_call(
                EventFloodWebSocket(),
                itertools.count(1),
                "Page.navigate",
                timeout_s=0.03,
            ),
            timeout=0.25,
        )


@pytest.mark.asyncio
async def test_cdp_rejects_unsafe_initial_url_before_connect(monkeypatch):
    from tools import web_routing

    async def unsafe(_url):
        return False

    connected = []

    def unexpected_connect(*args, **kwargs):
        connected.append(True)
        raise AssertionError("unsafe URL must be rejected before CDP connection")

    monkeypatch.setattr(web_routing, "async_is_safe_url", unsafe)
    monkeypatch.setattr(web_routing, "check_website_access", lambda url: None)
    monkeypatch.setitem(
        sys.modules,
        "websockets",
        types.SimpleNamespace(connect=unexpected_connect),
    )

    result = await web_routing._extract_via_cdp(
        "http://127.0.0.1/private",
        "ws://chrome.example/devtools/browser/1",
        "test_lane",
    )

    assert connected == []
    assert result.get("error")
    assert any(
        token in result["error"].lower()
        for token in ("private", "internal", "unsafe")
    )


@pytest.mark.asyncio
async def test_cdp_fetch_blocks_private_redirect_before_page_extraction(monkeypatch):
    from tools import web_routing

    private_url = "http://127.0.0.1/admin"

    async def safe_url(candidate):
        return candidate != private_url

    def responder(websocket, message):
        if message["method"] == "Page.navigate":
            websocket.feed({"id": message["id"], "result": {}})
            websocket.feed(
                {
                    "sessionId": "session-1",
                    "method": "Fetch.requestPaused",
                    "params": {
                        "requestId": "redirect-1",
                        "request": {"url": private_url},
                        "resourceType": "Document",
                    },
                }
            )
            websocket.feed(
                {
                    "sessionId": "session-1",
                    "method": "Page.loadEventFired",
                    "params": {},
                }
            )
            return
        standard_cdp_response(websocket, message)

    websocket = FakeWebSocket(responder)
    install_fake_websockets(monkeypatch, websocket)
    monkeypatch.setattr(web_routing, "async_is_safe_url", safe_url)
    monkeypatch.setattr(web_routing, "check_website_access", lambda url: None)

    result = await asyncio.wait_for(
        web_routing._extract_via_cdp(
            "https://public.example/start",
            "ws://chrome.example/devtools/browser/1",
            "test_lane",
        ),
        timeout=0.25,
    )

    methods = [message["method"] for message in websocket.sent]
    assert methods.index("Fetch.enable") < methods.index("Page.navigate")
    assert "Runtime.evaluate" not in methods

    enable = next(
        message for message in websocket.sent
        if message["method"] == "Fetch.enable"
    )
    patterns = enable["params"]["patterns"]
    assert {
        (pattern.get("urlPattern"), pattern.get("requestStage"))
        for pattern in patterns
    } == {
        ("http://*", "Request"),
        ("https://*", "Request"),
    }

    failed = [
        message for message in websocket.sent
        if message["method"] == "Fetch.failRequest"
    ]
    assert [message["params"]["requestId"] for message in failed] == [
        "redirect-1"
    ]
    assert not any(
        message["method"] == "Fetch.continueRequest"
        and message["params"].get("requestId") == "redirect-1"
        for message in websocket.sent
    )
    assert result.get("error")
    assert any(
        token in result["error"].lower()
        for token in ("private", "internal")
    )


@pytest.mark.asyncio
async def test_cdp_cleanup_timeout_does_not_hide_success(monkeypatch):
    from tools import web_routing

    def responder(websocket, message):
        standard_cdp_response(
            websocket,
            message,
            reply_to_close=False,
        )

    websocket = FakeWebSocket(responder)
    install_fake_websockets(monkeypatch, websocket)

    async def allow_url(url, lane, **kwargs):
        return None

    monkeypatch.setattr(web_routing, "_navigation_block_reason", allow_url)
    monkeypatch.setattr(web_routing, "_CDP_CLEANUP_TIMEOUT_S", 0.01)

    result = await asyncio.wait_for(
        web_routing._extract_via_cdp(
            "https://public.example/start",
            "ws://chrome.example/devtools/browser/1",
            "test_lane",
        ),
        timeout=0.25,
    )

    assert result["content"] == "public content"
    assert any(
        message["method"] == "Target.closeTarget"
        for message in websocket.sent
    )


@pytest.mark.asyncio
async def test_cdp_continues_safe_document_and_signed_subframe(monkeypatch):
    from tools import web_routing

    initial_url = "https://public.example/start"
    signed_subframe = (
        "https://cdn.example/embed?"
        "X-Amz-Signature=page-generated-signature"
    )
    navigate_id = None

    def responder(websocket, message):
        nonlocal navigate_id
        method = message["method"]
        if method == "Page.navigate":
            navigate_id = message["id"]
            websocket.feed(
                {
                    "sessionId": "session-1",
                    "method": "Fetch.requestPaused",
                    "params": {
                        "requestId": "document-1",
                        "request": {"url": initial_url},
                        "resourceType": "Document",
                        "frameId": "",
                    },
                }
            )
            return
        if (
            method == "Fetch.continueRequest"
            and message["params"]["requestId"] == "document-1"
        ):
            websocket.feed({"id": message["id"], "result": {}})
            websocket.feed({"id": navigate_id, "result": {}})
            websocket.feed(
                {
                    "sessionId": "session-1",
                    "method": "Fetch.requestPaused",
                    "params": {
                        "requestId": "subframe-1",
                        "request": {"url": signed_subframe},
                        "resourceType": "Document",
                        "frameId": "child-frame",
                    },
                }
            )
            return
        if (
            method == "Fetch.continueRequest"
            and message["params"]["requestId"] == "subframe-1"
        ):
            websocket.feed({"id": message["id"], "result": {}})
            websocket.feed(
                {
                    "sessionId": "session-1",
                    "method": "Page.loadEventFired",
                    "params": {},
                }
            )
            return
        standard_cdp_response(websocket, message)

    websocket = FakeWebSocket(responder)
    install_fake_websockets(monkeypatch, websocket)

    async def safe_url(_url):
        return True

    monkeypatch.setattr(web_routing, "async_is_safe_url", safe_url)
    monkeypatch.setattr(web_routing, "check_website_access", lambda url: None)

    result = await web_routing._extract_via_cdp(
        initial_url,
        "wss://cloud.example/devtools/browser/1",
        "browser_use_cloud",
    )

    assert result["content"] == "public content"
    continued = {
        message["params"]["requestId"]
        for message in websocket.sent
        if message["method"] == "Fetch.continueRequest"
    }
    assert continued == {"document-1", "subframe-1"}
    assert not any(
        message["method"] == "Fetch.failRequest"
        for message in websocket.sent
    )


@pytest.mark.asyncio
async def test_browser_use_lane_honors_local_profile_config(monkeypatch, tmp_path):
    from tools import browser_tool, web_routing
    from plugins.browser.browser_use import provider as browser_use_provider

    tmp_path.joinpath("config.yaml").write_text(
        "browser:\n  cloud_provider: local\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(browser_tool, "_cached_cloud_provider", None)
    monkeypatch.setattr(browser_tool, "_cloud_provider_resolved", False)

    constructed = []

    class UnexpectedBrowserUse:
        def __init__(self):  # pragma: no cover - must not run
            constructed.append(True)
            raise AssertionError("local profile must not construct Browser Use")

    async def allow_url(url, lane, **kwargs):
        return None

    monkeypatch.setattr(
        browser_use_provider,
        "BrowserUseBrowserProvider",
        UnexpectedBrowserUse,
    )
    monkeypatch.setattr(web_routing, "_navigation_block_reason", allow_url)

    result = await web_routing._extract_via_browser_use_cloud(
        "https://public.example",
        ["jina", "home_chrome_cdp", "browser_use_cloud"],
    )

    assert constructed == []
    assert result.get("error")
    assert "browser use" in result["error"].lower()


@pytest.mark.asyncio
async def test_browser_use_lane_refuses_different_selected_provider(monkeypatch):
    from tools import browser_tool_cloud, web_routing

    provider = FakeCloudProvider(name="browserbase")

    async def allow_url(url, lane, **kwargs):
        return None

    monkeypatch.setattr(
        browser_tool_cloud, "_get_cloud_provider", lambda: provider
    )
    monkeypatch.setattr(web_routing, "_navigation_block_reason", allow_url)

    result = await web_routing._extract_via_browser_use_cloud(
        "https://public.example",
        ["jina", "home_chrome_cdp", "browser_use_cloud"],
    )

    assert provider.created == []
    assert result.get("error")


@pytest.mark.asyncio
async def test_browser_use_lane_refuses_unavailable_selected_provider(monkeypatch):
    from tools import browser_tool_cloud, web_routing

    provider = FakeCloudProvider(available=False)

    async def allow_url(url, lane, **kwargs):
        return None

    monkeypatch.setattr(
        browser_tool_cloud, "_get_cloud_provider", lambda: provider
    )
    monkeypatch.setattr(web_routing, "_navigation_block_reason", allow_url)

    result = await web_routing._extract_via_browser_use_cloud(
        "https://public.example",
        ["jina", "home_chrome_cdp", "browser_use_cloud"],
    )

    assert provider.created == []
    assert result.get("error")


@pytest.mark.asyncio
async def test_browser_use_lane_rejects_credential_query_before_session(monkeypatch):
    from tools import browser_tool_cloud, web_routing

    provider = FakeCloudProvider()
    monkeypatch.setattr(
        browser_tool_cloud, "_get_cloud_provider", lambda: provider
    )

    result = await web_routing._extract_via_browser_use_cloud(
        "https://public.example/?access_token=secret",
        ["jina", "home_chrome_cdp", "browser_use_cloud"],
    )

    assert provider.created == []
    assert "credential" in result["error"].lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "error", "exception"])
async def test_browser_use_session_closes_for_every_extract_outcome(
    monkeypatch, outcome
):
    from tools import browser_tool_cloud, web_routing

    provider = FakeCloudProvider()

    async def allow_url(url, lane, **kwargs):
        return None

    async def extract(url, cdp_url, lane):
        if outcome == "exception":
            raise RuntimeError("CDP failed")
        if outcome == "error":
            return {"url": url, "error": "CDP failed"}
        return {
            "url": url,
            "content": "cloud content",
            "raw_content": "cloud content",
        }

    monkeypatch.setattr(
        browser_tool_cloud, "_get_cloud_provider", lambda: provider
    )
    monkeypatch.setattr(web_routing, "_navigation_block_reason", allow_url)
    monkeypatch.setattr(web_routing, "_extract_via_cdp", extract)

    result = await web_routing._extract_via_browser_use_cloud(
        "https://public.example",
        ["jina", "home_chrome_cdp", "browser_use_cloud"],
    )

    assert len(provider.created) == 1
    assert provider.closed == ["cloud-session-1"]
    if outcome == "success":
        assert result["content"] == "cloud content"
    else:
        assert result.get("error")


@pytest.mark.asyncio
async def test_browser_use_false_close_is_reported_and_emergency_cleanup_runs(
    monkeypatch,
):
    from tools import browser_tool_cloud, web_routing

    provider = FakeCloudProvider(close_result=False)

    async def allow_url(url, lane, **kwargs):
        return None

    async def extract(url, cdp_url, lane):
        return {
            "url": url,
            "content": "cloud content",
            "raw_content": "cloud content",
        }

    monkeypatch.setattr(
        browser_tool_cloud, "_get_cloud_provider", lambda: provider
    )
    monkeypatch.setattr(web_routing, "_navigation_block_reason", allow_url)
    monkeypatch.setattr(web_routing, "_extract_via_cdp", extract)

    result = await web_routing._extract_via_browser_use_cloud(
        "https://public.example",
        ["jina", "home_chrome_cdp", "browser_use_cloud"],
    )

    assert provider.closed == ["cloud-session-1", "cloud-session-1"]
    assert provider.emergency_cleanups == ["cloud-session-1"]
    assert "cleanup failed" in result["error"].lower()


@pytest.mark.asyncio
async def test_browser_use_creation_cancellation_reconciles_and_closes_session(
    monkeypatch,
):
    from tools import browser_tool_cloud, web_routing

    create_started = threading.Event()
    create_release = threading.Event()
    provider = FakeCloudProvider(
        create_started=create_started,
        create_release=create_release,
    )

    async def allow_url(url, lane, **kwargs):
        return None

    async def unexpected_extract(url, cdp_url, lane):  # pragma: no cover
        raise AssertionError("cancelled creation must not begin extraction")

    monkeypatch.setattr(
        browser_tool_cloud, "_get_cloud_provider", lambda: provider
    )
    monkeypatch.setattr(web_routing, "_navigation_block_reason", allow_url)
    monkeypatch.setattr(web_routing, "_extract_via_cdp", unexpected_extract)

    task = asyncio.create_task(
        web_routing._extract_via_browser_use_cloud(
            "https://public.example",
            ["jina", "home_chrome_cdp", "browser_use_cloud"],
        )
    )
    started = await asyncio.wait_for(
        asyncio.to_thread(create_started.wait, 2),
        timeout=3,
    )
    assert started is True

    task.cancel()
    create_release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert provider.closed == ["cloud-session-1"]
