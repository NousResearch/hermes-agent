"""The browser opens only web URLs, on every backend.

Local backends (Camofox, local Chromium) skip the SSRF/scheme checks on the premise that the agent's
terminal can read and reach everything anyway. With kernel-enforced secret isolation that premise no
longer holds: ``file:///opt/data/.env`` (or a local HTML file that iframes it) would put Hermes secrets
straight into the page snapshot handed to the model. Only http(s) and about:blank may be opened.
"""

import json

import pytest

from tools import browser_tool
from tools import browser_tool_cloud as bt_cloud
from tools import browser_tool_session as bt_session

REFUSED = [
    "file:///opt/data/.env", "FILE:///etc/passwd", "file://localhost/opt/data/auth.json",
    "view-source:file:///opt/data/.env", "view-source:https://example.com", "chrome://settings",
    "about:config", "resource:///modules/", "jar:file:///tmp/x.zip!/a.html", "javascript:alert(1)",
    "data:text/html,<iframe src=file:///opt/data/.env>", "ftp://example.com/x", "blob:https://e.com/u",
]


@pytest.fixture
def local_browser(monkeypatch):
    opened = []
    monkeypatch.setattr(browser_tool, "check_website_access", lambda url: None)
    monkeypatch.setattr(bt_cloud, "_is_local_backend", lambda: True)
    monkeypatch.setattr(bt_session, "_get_session_info", lambda task_id: {
        "session_name": f"s_{task_id}", "bb_session_id": None, "cdp_url": None,
        "features": {"local": True}, "_first_nav": False})
    monkeypatch.setattr(bt_session, "_run_browser_command",
                        lambda key, cmd, args, **kw: opened.append(args[0]) or
                        {"success": True, "data": {"title": "OK", "url": args[0]}})
    monkeypatch.setattr(browser_tool, "_attach_auto_snapshot", lambda *a, **k: None)
    return opened


@pytest.mark.parametrize("camofox", [False, True])
@pytest.mark.parametrize("url", REFUSED)
def test_non_web_schemes_are_refused_on_local_backends(monkeypatch, local_browser, url, camofox):
    sent = []
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: camofox)
    monkeypatch.setattr(browser_tool, "_camofox", lambda *a, **k: sent.append(a) or json.dumps({"success": True}))
    result = json.loads(browser_tool.browser_navigate(url, task_id="scheme"))
    assert result["success"] is False
    assert not local_browser and not sent


@pytest.mark.parametrize("url", ["https://example.com/", "http://127.0.0.1:8000/index.html", "about:blank"])
def test_web_urls_still_open_on_local_backends(monkeypatch, local_browser, url):
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    result = json.loads(browser_tool.browser_navigate(url, task_id="scheme"))
    assert result["success"] is True, result
    assert local_browser == [url]
