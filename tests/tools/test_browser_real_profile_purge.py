"""Tests for the real-profile cookie-purge detection (#96993).

Chrome >= 151 on Windows binds cookie encryption to the original profile, so
the copy-browser's first launch actively purges the cookies the snapshot just
copied in (556 -> 6, 3507 -> ~0 in the issue's measurements). The fix detects
the drop and surfaces a notice on the first navigation instead of letting the
agent discover site-by-site login failures.

The detection is deliberately platform-independent (a before/after cookie
count around the launch), so these tests run everywhere: they build real
SQLite cookie DBs in tmp_path and exercise the counting helper, the purge
predicate, and the notice -> session -> navigation wiring.
"""

import os
import sqlite3

import pytest


def _write_cookie_db(path: str, count: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [("host.example", "c", b"v10x")] * count
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS cookies (host_key TEXT, name TEXT, encrypted_value BLOB)"
        )
        conn.execute("DELETE FROM cookies")
        conn.executemany(
            "INSERT INTO cookies VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def _clean_real_profile_state():
    import tools.browser_tool as bt

    bt._real_profile_cdp_cache.pop("cdp", None)
    bt._real_profile_purge_notice.pop("msg", None)
    yield
    bt._real_profile_cdp_cache.pop("cdp", None)
    bt._real_profile_purge_notice.pop("msg", None)


class TestCountRealProfileCookies:
    def test_counts_network_location(self, tmp_path):
        from tools.browser_tool_real_profile import _count_real_profile_cookies

        copy_dir = str(tmp_path)
        _write_cookie_db(os.path.join(copy_dir, "Default", "Network", "Cookies"), 7)
        assert _count_real_profile_cookies(copy_dir) == 7

    def test_falls_back_to_legacy_root_location(self, tmp_path):
        from tools.browser_tool_real_profile import _count_real_profile_cookies

        copy_dir = str(tmp_path)
        _write_cookie_db(os.path.join(copy_dir, "Default", "Cookies"), 3)
        assert _count_real_profile_cookies(copy_dir) == 3

    def test_prefers_network_location_when_both_exist(self, tmp_path):
        from tools.browser_tool_real_profile import _count_real_profile_cookies

        copy_dir = str(tmp_path)
        _write_cookie_db(os.path.join(copy_dir, "Default", "Cookies"), 100)
        _write_cookie_db(os.path.join(copy_dir, "Default", "Network", "Cookies"), 9)
        assert _count_real_profile_cookies(copy_dir) == 9

    def test_missing_db_returns_none(self, tmp_path):
        from tools.browser_tool_real_profile import _count_real_profile_cookies

        assert _count_real_profile_cookies(str(tmp_path)) is None

    def test_non_sqlite_file_returns_none(self, tmp_path):
        from tools.browser_tool_real_profile import _count_real_profile_cookies

        db = tmp_path / "Default" / "Network" / "Cookies"
        db.parent.mkdir(parents=True)
        db.write_text("this is not a database")
        assert _count_real_profile_cookies(str(tmp_path)) is None

    def test_empty_jar_is_zero_not_none(self, tmp_path):
        from tools.browser_tool_real_profile import _count_real_profile_cookies

        copy_dir = str(tmp_path)
        _write_cookie_db(os.path.join(copy_dir, "Default", "Network", "Cookies"), 0)
        assert _count_real_profile_cookies(copy_dir) == 0


class TestCookiesPurgedAfterLaunch:
    @pytest.mark.parametrize(
        "before,after,expected",
        [
            # The reported purge shapes: 556 -> 6 and 3507 -> ~0.
            (556, 6, True),
            (3507, 0, True),
            # Normal startup churn (expired-cookie sweep, visitor cookies).
            (556, 550, False),
            (120, 121, False),
            # Baseline too small to be worth a warning.
            (4, 0, False),
            # Unknown counts (locked / missing DB) never claim a purge.
            (None, 0, False),
            (556, None, False),
            (None, None, False),
            # Exactly halving is not a purge; anything past it is.
            (100, 50, False),
            (100, 49, True),
        ],
    )
    def test_predicate(self, before, after, expected):
        from tools.browser_tool_real_profile import _cookies_purged_after_launch

        assert _cookies_purged_after_launch(before, after) is expected


class TestPurgeNoticeWiring:
    def test_create_local_session_carries_notice_once(self):
        """A pending purge notice lands on the session features and is consumed."""
        import tools.browser_tool as bt
        from tools.browser_tool_session import _create_local_session

        bt._real_profile_purge_notice["msg"] = "purge notice under test"
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "tools.browser_tool_real_profile._real_profile_cdp",
                lambda: ("http://127.0.0.1:9222", None),
            )
            mp.setattr(
                "tools.browser_tool_cdp._resolve_cdp_override",
                lambda u: u,
            )
            session = _create_local_session("t-purge")
        assert session["real_profile_purge_warning"] == "purge notice under test"
        assert session["features"]["real_profile_cookies_purged"] is True
        # One-shot: consumed, so a second session (e.g. after a re-snapshot
        # that restores the cookies) does not re-warn.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "tools.browser_tool_real_profile._real_profile_cdp",
                lambda: ("http://127.0.0.1:9222", None),
            )
            mp.setattr(
                "tools.browser_tool_cdp._resolve_cdp_override",
                lambda u: u,
            )
            session2 = _create_local_session("t-purge-2")
        assert "real_profile_purge_warning" not in session2
        assert "real_profile_cookies_purged" not in session2["features"]

    def test_navigation_surfaces_the_warning(self):
        """The session-carried notice is surfaced on the first navigation's response."""
        from tools.browser_tool import _add_navigate_warnings

        response: dict = {}
        _add_navigate_warnings(
            response,
            "Example Domain",
            {
                "features": {"local": True, "real_profile": True},
                "real_profile_purge_warning": "purge notice under test",
            },
        )
        assert response["real_profile_purge_warning"] == "purge notice under test"

    def test_navigation_without_notice_stays_clean(self):
        from tools.browser_tool import _add_navigate_warnings

        response: dict = {}
        _add_navigate_warnings(response, "Example Domain", None)
        assert "real_profile_purge_warning" not in response


def _arm_real_profile_launch(monkeypatch, copy_dir, on_launch):
    """Stub everything around the real-profile launch; keep the counting real."""
    import tools.browser_tool_real_profile as rp
    import hermes_cli.browser_connect as bc

    monkeypatch.setattr("tools.browser_tool_cloud._use_real_profile", lambda: True)
    monkeypatch.setattr(
        "tools.browser_tool_lightpanda_fallback._using_lightpanda_engine", lambda: False
    )
    monkeypatch.setattr(rp, "_real_profile_unsupported_reason", lambda browser: None)
    monkeypatch.setattr(rp, "_cdp_http_ready", lambda c: False)
    monkeypatch.setattr(rp, "_agent_browser_get_cdp", lambda s: None)
    monkeypatch.setattr(rp, "_surviving_chrome_cdp", lambda d: None)

    monkeypatch.setattr(bc, "detect_default_chromium", lambda system=None: "chrome")
    monkeypatch.setattr(bc, "real_profile_copy_dir", lambda browser: copy_dir)
    monkeypatch.setattr(
        bc, "snapshot_real_profile", lambda browser, src=None: (copy_dir, None)
    )
    monkeypatch.setattr(bc, "chromium_executable", lambda browser: "/fake/chrome")

    def fake_launch(real_binary, dir_):
        on_launch()
        return 9222, None

    monkeypatch.setattr(rp, "_launch_real_profile_chrome", fake_launch)
    monkeypatch.setattr(
        rp,
        "_attach_agent_browser_to_real_profile",
        lambda port, dir_: ("http://127.0.0.1:9222", None),
    )


class TestPurgeDetectionAroundLaunch:
    def test_detected_drop_sets_notice_and_keeps_launching(self, tmp_path, monkeypatch):
        """End-to-end through _real_profile_cdp: snapshot counts 8, launch leaves 1.

        The heavy launch path is stubbed; what is under test is the real
        counting against real DB files and the decision that follows it —
        the purge must produce a notice, not a launch failure.
        """
        import tools.browser_tool as bt
        import tools.browser_tool_real_profile as rp

        copy_dir = str(tmp_path / "browser-profile" / "chrome")
        cookies = os.path.join(copy_dir, "Default", "Network", "Cookies")
        _write_cookie_db(cookies, 8)

        def purge_on_launch():
            # The purge: Chrome rewrites the jar down to one survivor.
            _write_cookie_db(cookies, 1)

        _arm_real_profile_launch(monkeypatch, copy_dir, purge_on_launch)

        cdp, err = rp._real_profile_cdp()
        assert cdp == "http://127.0.0.1:9222"
        assert err is None
        assert "#96993" in bt._real_profile_purge_notice["msg"]

    def test_surviving_jar_sets_no_notice(self, tmp_path, monkeypatch):
        """No meaningful drop -> no notice; the launch result is untouched."""
        import tools.browser_tool as bt
        import tools.browser_tool_real_profile as rp

        copy_dir = str(tmp_path / "browser-profile" / "chrome")
        cookies = os.path.join(copy_dir, "Default", "Network", "Cookies")
        _write_cookie_db(cookies, 8)

        def churn_on_launch():
            # Normal startup churn only: one cookie expired.
            _write_cookie_db(cookies, 7)

        _arm_real_profile_launch(monkeypatch, copy_dir, churn_on_launch)

        cdp, err = rp._real_profile_cdp()
        assert cdp == "http://127.0.0.1:9222"
        assert err is None
        assert "msg" not in bt._real_profile_purge_notice

    def test_missing_post_launch_db_never_claims_purge(self, tmp_path, monkeypatch):
        """A post-launch DB that cannot be read must not fabricate a purge notice."""
        import tools.browser_tool as bt
        import tools.browser_tool_real_profile as rp

        copy_dir = str(tmp_path / "browser-profile" / "chrome")
        cookies = os.path.join(copy_dir, "Default", "Network", "Cookies")
        _write_cookie_db(cookies, 8)

        def remove_db_on_launch():
            os.unlink(cookies)

        _arm_real_profile_launch(monkeypatch, copy_dir, remove_db_on_launch)

        cdp, err = rp._real_profile_cdp()
        assert cdp == "http://127.0.0.1:9222"
        assert err is None
        assert "msg" not in bt._real_profile_purge_notice
