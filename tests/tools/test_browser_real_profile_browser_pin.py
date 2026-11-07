"""Tests for ``browser.real_profile_browser`` (which family real-profile browsing may drive).

Native behavior follows the OS default browser. Detection is Chromium-only, so a host
whose default is Safari or Firefox reports "not a supported Chromium browser" and the
user has no way to say "use Chrome's profile anyway".

Invariants under test:
- pin set + installed   -> the pinned family launches, the OS default is ignored
- pin unset             -> detection decides, exactly the pre-patch behavior
- pin is read per call  -> a config change lands on the next browser use, never cached
"""
from tools import browser_session_real_profile as rp


class _FreeLock:
    def acquire(self, timeout=None):
        return True

    def release(self):
        pass


def _prepare(monkeypatch, *, pinned="", detected=None):
    """Real-profile prerequisites: consent on, no Lightpanda, nothing running, lock free.

    Returns the list of browser names the launch was asked for.
    """
    from hermes_cli import browser_connect
    from tools import browser_launcher, browser_lightpanda, browser_session_cloud

    settings = {"real_profile_browser": pinned}
    monkeypatch.setattr(browser_session_cloud, "use_real_profile", lambda: True)
    monkeypatch.setattr(browser_session_cloud, "browser_cfg", lambda key, default=None: settings.get(key, default))
    monkeypatch.setattr(browser_lightpanda, "using_lightpanda_engine", lambda: False)
    monkeypatch.setattr(browser_launcher, "get_browser", lambda _key: None)
    monkeypatch.setattr(browser_connect, "detect_default_chromium", lambda: detected)
    monkeypatch.setattr(rp, "_home_lock", _FreeLock)

    launched = []

    def fake_launch(browser_name):
        launched.append(browser_name)
        return "ws://127.0.0.1:9222/devtools/browser/test", None

    monkeypatch.setattr(rp, "_launch", fake_launch)
    return launched


class TestPinnedBrowser:
    def test_pin_launches_the_named_family_even_when_the_default_is_not_chromium(self, monkeypatch):
        launched = _prepare(monkeypatch, pinned="chrome", detected=None)

        cdp_url, error = rp.real_profile_cdp()

        assert error is None
        assert cdp_url is not None
        assert launched == ["chrome"]

    def test_without_the_pin_detection_still_decides(self, monkeypatch):
        launched = _prepare(monkeypatch, pinned="", detected=None)

        cdp_url, error = rp.real_profile_cdp()

        assert cdp_url is None
        assert error is not None and "not a supported Chromium browser" in error
        assert launched == []

    def test_without_the_pin_a_chromium_default_still_works(self, monkeypatch):
        launched = _prepare(monkeypatch, pinned="", detected="brave")

        cdp_url, error = rp.real_profile_cdp()

        assert error is None
        assert cdp_url is not None
        assert launched == ["brave"]

    def test_pin_is_normalized_and_read_per_call(self, monkeypatch):
        from tools import browser_session_cloud

        settings = {"real_profile_browser": " Chrome "}
        monkeypatch.setattr(browser_session_cloud, "browser_cfg", lambda key, default=None: settings.get(key, default))
        assert rp._pinned_browser() == "chrome"

        settings["real_profile_browser"] = ""
        assert rp._pinned_browser() is None

    def test_a_pin_outside_the_supported_families_fails_closed_with_its_own_reason(self, monkeypatch):
        launched = _prepare(monkeypatch, pinned="firefox", detected="chrome")

        cdp_url, error = rp.real_profile_cdp()

        assert cdp_url is None
        assert error is not None and "not a supported browser family" in error
        assert launched == []
