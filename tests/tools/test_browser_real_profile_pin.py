"""Tests for the LOCAL real_profile_pin patch (browser.real_profile_pin).

Native behavior: snapshot copies whichever Chromium profile was last used
(Local State -> profile.last_used). The pin lets a machine with a work
profile and a personal profile lock each Hermes install to one identity so
last-used roulette can never give the agent the wrong principal.

Invariants under test:
- pin set + exists       -> pinned profile is copied, last_used ignored
- pin set + missing      -> FAIL CLOSED (error), never silently last_used
- pin unset              -> native last_used behavior, byte-for-byte
"""
import json
import os

import pytest


class TestRealProfilePin:
    def _make_profile(self, root, last_used="Profile 2"):
        """Synthetic Chromium user-data-dir with two profiles + last_used."""
        for prof in ("Default", "Profile 2", "Profile 4"):
            (root / prof / "Network").mkdir(parents=True)
            (root / prof / "Cookies").write_text(f"cookies-{prof}")
            (root / prof / "Login Data").write_text(f"logins-{prof}")
            (root / prof / "Preferences").write_text("{}")
        (root / "Crashpad").mkdir()
        (root / "Local State").write_text(
            json.dumps({"os_crypt": {}, "profile": {"last_used": last_used}})
        )
        return root

    def test_pin_wins_over_last_used(self, tmp_path, monkeypatch):
        import hermes_cli.browser_connect as bc

        src = self._make_profile(tmp_path / "real", last_used="Profile 4")
        home = tmp_path / "hermes-home"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
        monkeypatch.setattr(bc, "_real_profile_pin", lambda: "Profile 2")

        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert err is None and dst
        got = (home / "browser-profile" / "chrome" / "Default" / "Cookies").read_text()
        assert got == "cookies-Profile 2", "pin must override last_used"

    def test_bad_pin_fails_closed(self, tmp_path, monkeypatch):
        import hermes_cli.browser_connect as bc

        src = self._make_profile(tmp_path / "real")
        monkeypatch.setattr(bc, "get_hermes_home", lambda: tmp_path / "hh")
        monkeypatch.setattr(bc, "_real_profile_pin", lambda: "Profile 99")

        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert dst is None
        assert err and "real_profile_pin" in err and "Profile 99" in err
        # Nothing may have been copied when the pin failed closed
        assert not (tmp_path / "hh" / "browser-profile" / "chrome" / "Default").exists()

    def test_no_pin_keeps_native_last_used(self, tmp_path, monkeypatch):
        import hermes_cli.browser_connect as bc

        src = self._make_profile(tmp_path / "real", last_used="Profile 4")
        home = tmp_path / "hermes-home"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
        monkeypatch.setattr(bc, "_real_profile_pin", lambda: None)

        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert err is None and dst
        got = (home / "browser-profile" / "chrome" / "Default" / "Cookies").read_text()
        assert got == "cookies-Profile 4", "no pin = native last_used"

    def test_re_sync_respects_pin_when_last_used_flips(self, tmp_path, monkeypatch):
        """The wrong-principal regression: session 2 with different last_used
        must NOT overlay a different profile's auth onto the pinned copy."""
        import hermes_cli.browser_connect as bc

        src = self._make_profile(tmp_path / "real", last_used="Profile 2")
        home = tmp_path / "hermes-home"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
        monkeypatch.setattr(bc, "_real_profile_pin", lambda: "Profile 2")

        dst1, err1 = bc.snapshot_real_profile("chrome", src=str(src))
        assert err1 is None

        # User browses HM (Profile 4) in between; last_used flips.
        (src / "Local State").write_text(
            json.dumps({"os_crypt": {}, "profile": {"last_used": "Profile 4"}})
        )
        (src / "Profile 2" / "Cookies").write_text("cookies-Profile 2-v2")

        dst2, err2 = bc.snapshot_real_profile("chrome", src=str(src))
        assert err2 is None and dst2 == dst1
        got = (home / "browser-profile" / "chrome" / "Default" / "Cookies").read_text()
        assert got == "cookies-Profile 2-v2", "auth re-sync must stay on the pin"
