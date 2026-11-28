"""A persistent-profile Chromium is quit over CDP (``Browser.close``) before any signal.

Headless Chromium exits on SIGTERM without committing its pending cookie writes (the cookie store flushes on
a ~30s timer and on an orderly quit), so with ``browser.user_data_dir`` set a sign-in made shortly before the
browser was stopped was lost and the next launch was signed out again.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from unittest.mock import patch

import pytest

from tools import browser_launcher as launcher


def _browser(**over):
    base = dict(key="k", pid=4242, start_time=1, cdp_url="ws://127.0.0.1:9/devtools/browser/x",
                http_url="http://127.0.0.1:9", user_data_dir="/tmp/p", owned=True, ephemeral=False, home_key="h")
    base.update(over)
    return launcher.LaunchedBrowser(**base)


@pytest.mark.parametrize("over", [{"ephemeral": True}, {"owned": False}, {"foreign": True}, {"pid": None}])
def test_graceful_close_never_touches_throwaway_attached_or_foreign_browsers(over):
    with patch("websockets.sync.client.connect") as connect:
        assert launcher._graceful_close(_browser(**over), None) is False
    connect.assert_not_called()


def test_graceful_close_sends_browser_close_and_waits_for_exit():
    sent = []

    class _WS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def send(self, msg):
            sent.append(json.loads(msg))

        def recv(self, timeout=None):
            raise TimeoutError

    alive = iter([True, False])
    with patch("websockets.sync.client.connect", return_value=_WS()), \
         patch.object(launcher, "_pid_alive", side_effect=lambda pid: next(alive)):
        assert launcher._graceful_close(_browser(), None) is True
    assert sent == [{"id": 1, "method": "Browser.close"}]


def test_stop_skips_the_signal_when_the_browser_quit_cleanly(monkeypatch):
    browser = _browser(ephemeral=True)  # the unrecorded path: owned, stopped by this process
    monkeypatch.setitem(launcher._browsers, "k", browser)
    calls = []
    with patch.object(launcher, "_graceful_close", side_effect=lambda b, p: calls.append("close") or True), \
         patch.object(launcher, "_terminate", side_effect=lambda *a: calls.append("term")):
        launcher.stop_browser("k")
    assert calls == ["close"]
    assert "k" not in launcher._browsers


def test_stop_falls_back_to_the_signal(monkeypatch):
    browser = _browser(ephemeral=True)
    monkeypatch.setitem(launcher._browsers, "k", browser)
    calls = []
    with patch.object(launcher, "_graceful_close", side_effect=lambda b, p: calls.append("close") or False), \
         patch.object(launcher, "_terminate", side_effect=lambda *a: calls.append("term")):
        launcher.stop_browser("k")
    assert calls == ["close", "term"]


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("HERMES_E2E_BROWSER", "").strip() != "1",
                    reason="real-browser E2E: set HERMES_E2E_BROWSER=1 to opt in")
def test_real_chromium_keeps_a_fresh_cookie_across_a_stop(tmp_path, monkeypatch):
    import shutil

    import websockets

    exe = next((shutil.which(n) for n in ("google-chrome", "chromium", "chromium-browser") if shutil.which(n)), None)
    if not exe:
        pytest.skip("Chrome/Chromium not installed")
    profile = str(tmp_path / "profile")
    monkeypatch.setattr("tools.browser_session.build_browser_env", lambda *a, **k: dict(os.environ))  # no runtime manifest read

    async def cdp(url, method, params):
        async with websockets.connect(url, max_size=2 ** 24, proxy=None) as ws:
            await ws.send(json.dumps({"id": 1, "method": method, "params": params}))
            while True:
                reply = json.loads(await ws.recv())
                if reply.get("id") == 1:
                    return reply

    browser, err = launcher.launch_chromium("persist-e2e", executable=exe, user_data_dir=profile, ephemeral=False)
    assert browser is not None, err
    try:
        asyncio.run(cdp(browser.cdp_url, "Storage.setCookies", {"cookies": [{
            "name": "probe", "value": "kept", "domain": "example.com", "path": "/", "secure": True,
            "expires": time.time() + 3600}]}))
    finally:
        launcher.stop_browser("persist-e2e")  # immediately: well inside the cookie store's flush timer
    browser, err = launcher.launch_chromium("persist-e2e", executable=exe, user_data_dir=profile, ephemeral=False)
    assert browser is not None, err
    try:
        cookies = asyncio.run(cdp(browser.cdp_url, "Storage.getCookies", {}))["result"]["cookies"]
    finally:
        launcher.stop_browser("persist-e2e")
    assert [c["value"] for c in cookies if c["name"] == "probe"] == ["kept"]
