"""Real-profile browser + profile SELECTION (browser.real_profile_browser and the
discovery the desktop picker reads).

The invariant under every case is the wrong-principal rule (#95549): a selection may
resolve to exactly the identity the user chose, or fail closed with a fixable message.
It may never silently fall through to a different browser or a different profile.
"""
import json

import pytest


def _make_user_data_dir(root, profiles=("Default", "Profile 2"), last_used="Default",
                        names=None):
    """Synthetic Chromium user-data-dir with profile dirs + a Local State info_cache."""
    names = names or {}
    for prof in profiles:
        (root / prof / "Network").mkdir(parents=True)
        (root / prof / "Cookies").write_text(f"cookies-{prof}")
        (root / prof / "Login Data").write_text(f"logins-{prof}")
        (root / prof / "Preferences").write_text("{}")
    (root / "Local State").write_text(json.dumps({
        "os_crypt": {},
        "profile": {
            "last_used": last_used,
            "info_cache": {p: {"name": names.get(p, p)} for p in profiles}}}))
    return root


@pytest.fixture
def bc():
    import hermes_cli.browser_connect as module
    return module


class TestResolveRealProfileBrowser:
    """browser.real_profile_browser: pin wins, bad pin fails closed, unset = OS default."""

    def test_unset_follows_os_default(self, bc, monkeypatch):
        monkeypatch.setattr(bc, "_real_profile_browser_override", lambda: None)
        monkeypatch.setattr(bc, "detect_default_chromium", lambda system=None: "brave")

        assert bc.resolve_real_profile_browser() == ("brave", None)

    def test_pin_overrides_os_default(self, bc, tmp_path, monkeypatch):
        _make_user_data_dir(tmp_path / "chrome-data")
        monkeypatch.setattr(bc, "_real_profile_browser_override", lambda: "chrome")
        monkeypatch.setattr(bc, "detect_default_chromium", lambda system=None: "brave")
        monkeypatch.setattr(bc, "real_profile_data_dir",
                            lambda browser, system=None: str(tmp_path / "chrome-data"))
        monkeypatch.setattr(bc, "chromium_executable", lambda browser, system=None: "/usr/bin/x")

        browser, err = bc.resolve_real_profile_browser()
        assert (browser, err) == ("chrome", None), "an explicit pin must beat the OS default"

    def test_unknown_browser_fails_closed(self, bc, monkeypatch):
        monkeypatch.setattr(bc, "_real_profile_browser_override", lambda: "firefox")
        monkeypatch.setattr(bc, "detect_default_chromium", lambda system=None: "chrome")

        browser, err = bc.resolve_real_profile_browser()
        assert browser is None, "must never fall back to the default browser"
        assert err and "firefox" in err and "real_profile_browser" in err

    def test_pinned_browser_without_profile_dir_fails_closed(self, bc, tmp_path, monkeypatch):
        monkeypatch.setattr(bc, "_real_profile_browser_override", lambda: "edge")
        monkeypatch.setattr(bc, "detect_default_chromium", lambda system=None: "chrome")
        monkeypatch.setattr(bc, "real_profile_data_dir",
                            lambda browser, system=None: str(tmp_path / "missing"))

        browser, err = bc.resolve_real_profile_browser()
        assert browser is None
        assert err and "edge" in err.lower()

    def test_pinned_browser_without_binary_fails_closed(self, bc, tmp_path, monkeypatch):
        """A leftover ~/.config dir from an uninstalled browser must fail HERE, with a
        fixable message, not later at launch with 'binary could not be found'."""
        _make_user_data_dir(tmp_path / "chrome-data")
        monkeypatch.setattr(bc, "_real_profile_browser_override", lambda: "chrome")
        monkeypatch.setattr(bc, "real_profile_data_dir",
                            lambda browser, system=None: str(tmp_path / "chrome-data"))
        monkeypatch.setattr(bc, "chromium_executable", lambda browser, system=None: None)

        browser, err = bc.resolve_real_profile_browser()
        assert browser is None
        assert err and "not installed" in err

    def test_case_and_whitespace_tolerant(self, bc, tmp_path, monkeypatch):
        """A hand-edited config.yaml ('  Brave ') resolves like the canonical key."""
        _make_user_data_dir(tmp_path / "brave-data")
        monkeypatch.setattr(bc, "_browser_setting",
                            lambda key: "  Brave " if key == "real_profile_browser" else None)
        monkeypatch.setattr(bc, "real_profile_data_dir",
                            lambda browser, system=None: str(tmp_path / "brave-data"))
        monkeypatch.setattr(bc, "chromium_executable", lambda browser, system=None: "/usr/bin/x")

        assert bc.resolve_real_profile_browser() == ("brave", None)


class TestListProfilesInDataDir:
    def test_lists_directories_with_display_names(self, bc, tmp_path):
        src = _make_user_data_dir(
            tmp_path / "data", profiles=("Default", "Profile 1", "Profile 2"),
            last_used="Profile 1", names={"Profile 1": "Work", "Profile 2": "Personal"})

        rows = bc.list_profiles_in_data_dir(str(src))
        assert [r["directory"] for r in rows] == ["Default", "Profile 1", "Profile 2"]
        assert {r["directory"]: r["name"] for r in rows}["Profile 2"] == "Personal"
        assert [r["directory"] for r in rows if r["last_used"]] == ["Profile 1"]

    def test_pin_target_is_the_directory_key(self, bc, tmp_path, monkeypatch):
        """Contract with browser.real_profile_pin: every `directory` the picker offers must
        be a pin _resolve_source_profile accepts — otherwise the UI can write a config the
        launch path rejects."""
        src = _make_user_data_dir(tmp_path / "data", profiles=("Default", "Profile 2"),
                                  names={"Profile 2": "Personal"})

        for row in bc.list_profiles_in_data_dir(str(src)):
            monkeypatch.setattr(bc, "_real_profile_pin", lambda d=row["directory"]: d)
            assert bc._resolve_source_profile(str(src)) == (row["directory"], None)

    def test_missing_local_state_degrades_to_directory_names(self, bc, tmp_path):
        src = _make_user_data_dir(tmp_path / "data")
        (src / "Local State").unlink()

        rows = bc.list_profiles_in_data_dir(str(src))
        assert [r["name"] for r in rows] == ["Default", "Profile 2"]

    def test_non_profile_directories_are_ignored(self, bc, tmp_path):
        src = _make_user_data_dir(tmp_path / "data")
        (src / "Crashpad").mkdir()
        (src / "ShaderCache").mkdir()

        assert [r["directory"] for r in bc.list_profiles_in_data_dir(str(src))] == [
            "Default", "Profile 2"]

    def test_absent_dir_is_empty_not_error(self, bc, tmp_path):
        assert bc.list_profiles_in_data_dir(str(tmp_path / "nope")) == []
        assert bc.list_profiles_in_data_dir(None) == []


class TestListRealProfileCandidates:
    def test_resolved_pair_matches_what_a_launch_would_use(self, bc, tmp_path, monkeypatch):
        """The picker's `resolved_browser`/`resolved_profile` is the SAME answer the
        launch path gets — one resolver, or the UI lies about the identity."""
        src = _make_user_data_dir(tmp_path / "brave-data",
                                  profiles=("Default", "Profile 1", "Profile 2"),
                                  last_used="Profile 1")
        monkeypatch.setattr(bc, "real_profile_data_dir",
                            lambda browser, system=None: str(src) if browser == "brave" else None)
        monkeypatch.setattr(bc, "chromium_executable",
                            lambda browser, system=None: "/usr/bin/x" if browser == "brave" else None)
        monkeypatch.setattr(bc, "_real_profile_browser_override", lambda: "brave")
        monkeypatch.setattr(bc, "_real_profile_pin", lambda: "Profile 2")

        info = bc.list_real_profile_candidates(system="Linux")
        assert info["resolved_browser"] == "brave"
        assert info["resolved_profile"] == "Profile 2"
        assert info["error"] is None
        # And it agrees with the resolvers the launch path calls.
        assert bc.resolve_real_profile_browser("Linux")[0] == info["resolved_browser"]
        assert bc._resolve_source_profile(str(src))[0] == info["resolved_profile"]

    def test_every_supported_browser_has_a_row_with_a_label(self, bc, monkeypatch):
        monkeypatch.setattr(bc, "detect_default_chromium", lambda system=None: None)
        monkeypatch.setattr(bc, "_real_profile_browser_override", lambda: None)

        info = bc.list_real_profile_candidates(system="Linux")
        keys = [row["key"] for row in info["browsers"]]
        assert keys == list(bc.real_profile_browser_keys())
        assert all(row["label"] and row["label"] != row["key"] for row in info["browsers"]), \
            "each row needs a human label for the picker"

    def test_non_chromium_default_surfaces_a_fixable_error(self, bc, monkeypatch):
        monkeypatch.setattr(bc, "detect_default_chromium", lambda system=None: None)
        monkeypatch.setattr(bc, "_real_profile_browser_override", lambda: None)

        info = bc.list_real_profile_candidates(system="Linux")
        assert info["resolved_browser"] is None
        assert info["error"] and "Chromium" in info["error"]

    def test_unsupported_channel_default_is_flagged_not_normalized(self, bc, monkeypatch):
        monkeypatch.setattr(bc, "detect_default_chromium",
                            lambda system=None: bc.UNSUPPORTED_CHANNEL)
        monkeypatch.setattr(bc, "_real_profile_browser_override", lambda: None)

        info = bc.list_real_profile_candidates(system="Linux")
        assert info["detected_unsupported_channel"] is True
        assert info["resolved_browser"] is None, "a Beta/Dev default must not resolve to stable"
        assert info["detected_default"] is None

    def test_bad_pin_error_reaches_the_picker(self, bc, tmp_path, monkeypatch):
        src = _make_user_data_dir(tmp_path / "data")
        monkeypatch.setattr(bc, "real_profile_data_dir",
                            lambda browser, system=None: str(src) if browser == "chrome" else None)
        monkeypatch.setattr(bc, "chromium_executable",
                            lambda browser, system=None: "/usr/bin/x" if browser == "chrome" else None)
        monkeypatch.setattr(bc, "_real_profile_browser_override", lambda: "chrome")
        monkeypatch.setattr(bc, "_real_profile_pin", lambda: "Profile 99")

        info = bc.list_real_profile_candidates(system="Linux")
        assert info["error"] and "Profile 99" in info["error"]

    @pytest.mark.parametrize("system,supported", [
        ("Linux", True), ("Darwin", True), ("Windows", True), ("FreeBSD", False)])
    def test_platform_support_is_reported(self, bc, monkeypatch, system, supported):
        """The desktop shows/hides the picker from this flag, so it must be honest per OS."""
        monkeypatch.setattr(bc, "detect_default_chromium", lambda system=None: None)
        monkeypatch.setattr(bc, "_real_profile_browser_override", lambda: None)

        assert bc.list_real_profile_candidates(system=system)["supported"] is supported


class TestSnapshotIdentitySwap:
    """Re-pinning must REBUILD the snapshot, not overlay auth onto the old identity."""

    def test_changing_the_pin_rebuilds_the_tree(self, bc, tmp_path, monkeypatch):
        src = _make_user_data_dir(tmp_path / "real", profiles=("Default", "Profile 2"),
                                  last_used="Default")
        # A file that only exists in Default: if the tree is reused, it survives the swap.
        (src / "Default" / "Bookmarks").write_text("default-bookmarks")
        home = tmp_path / "hermes-home"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)

        monkeypatch.setattr(bc, "_real_profile_pin", lambda: "Default")
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert err is None
        copy = home / "browser-profile" / "chrome" / "Default"
        assert (copy / "Bookmarks").exists()

        # User re-pins to Profile 2 (which has no Bookmarks).
        monkeypatch.setattr(bc, "_real_profile_pin", lambda: "Profile 2")
        dst2, err2 = bc.snapshot_real_profile("chrome", src=str(src))
        assert err2 is None and dst2 == dst
        assert (copy / "Cookies").read_text() == "cookies-Profile 2"
        assert not (copy / "Bookmarks").exists(), \
            "re-pinning must rebuild, not overlay onto the previous identity's tree"

    def test_same_pin_reuses_the_tree(self, bc, tmp_path, monkeypatch):
        """The rebuild must trigger on a CHANGE only — a stable pin keeps the fast path."""
        src = _make_user_data_dir(tmp_path / "real", last_used="Default")
        home = tmp_path / "hermes-home"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
        monkeypatch.setattr(bc, "_real_profile_pin", lambda: "Default")

        bc.snapshot_real_profile("chrome", src=str(src))
        copy = home / "browser-profile" / "chrome" / "Default"
        (copy / "marker-of-reuse").write_text("kept")

        bc.snapshot_real_profile("chrome", src=str(src))
        assert (copy / "marker-of-reuse").exists(), "an unchanged pin must not rebuild"


class TestSnapshotCleanup:
    def test_keep_removes_only_other_browsers(self, bc, tmp_path, monkeypatch):
        """Switching browsers must not leave the old browser's cookie copy on disk."""
        home = tmp_path / "hermes-home"
        for browser in ("chrome", "brave"):
            (home / "browser-profile" / browser / "Default").mkdir(parents=True)
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)

        bc.cleanup_real_profile_snapshots(keep="brave")
        assert (home / "browser-profile" / "brave").is_dir()
        assert not (home / "browser-profile" / "chrome").exists()

    def test_no_keep_removes_everything(self, bc, tmp_path, monkeypatch):
        home = tmp_path / "hermes-home"
        (home / "browser-profile" / "chrome" / "Default").mkdir(parents=True)
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)

        bc.cleanup_real_profile_snapshots()
        assert not (home / "browser-profile").exists()

    def test_absent_store_is_a_no_op(self, bc, tmp_path, monkeypatch):
        monkeypatch.setattr(bc, "get_hermes_home", lambda: tmp_path / "empty-home")
        bc.cleanup_real_profile_snapshots()
        bc.cleanup_real_profile_snapshots(keep="chrome")


class TestProfileScopedSelection:
    """Two Hermes profiles on one machine must be able to browse as different identities.

    This is the whole point of the feature: the settings live in each profile's own
    config.yaml, so the resolver must read the CURRENT HERMES_HOME, never a cached one.
    """

    def test_two_homes_resolve_to_different_browsers(self, bc, tmp_path, monkeypatch):
        """E2E through the REAL config loader: write two profiles' config.yaml, point
        HERMES_HOME at each in turn, and confirm the resolver reads that home's file."""
        for browser_dir in ("chrome-data", "brave-data"):
            _make_user_data_dir(tmp_path / browser_dir, profiles=("Default", "Profile 2"))
        monkeypatch.setattr(bc, "real_profile_data_dir", lambda browser, system=None: {
            "chrome": str(tmp_path / "chrome-data"),
            "brave": str(tmp_path / "brave-data")}.get(browser))
        monkeypatch.setattr(bc, "chromium_executable", lambda browser, system=None: "/usr/bin/x")

        homes = {}
        for name, browser, pin in (("coder", "chrome", "Profile 2"), ("omar", "brave", "Default")):
            home = tmp_path / name
            home.mkdir()
            (home / "config.yaml").write_text(
                "browser:\n"
                "  use_real_profile: true\n"
                f"  real_profile_browser: {browser}\n"
                f"  real_profile_pin: '{pin}'\n")
            homes[name] = home

        seen = {}
        for name, home in homes.items():
            monkeypatch.setenv("HERMES_HOME", str(home))
            browser, err = bc.resolve_real_profile_browser("Linux")
            assert err is None, err
            src = bc.real_profile_data_dir(browser)
            seen[name] = (browser, bc._resolve_source_profile(src)[0])

        assert seen["coder"] == ("chrome", "Profile 2")
        assert seen["omar"] == ("brave", "Default")
        assert seen["coder"] != seen["omar"], \
            "per-profile config must give per-profile browsing identities"

    def test_snapshot_stores_are_per_home(self, bc, tmp_path, monkeypatch):
        """Two profiles browsing as different identities must not share one snapshot dir,
        or the second launch would overwrite the first's cookies."""
        seen = set()
        for name in ("coder", "omar"):
            monkeypatch.setenv("HERMES_HOME", str(tmp_path / name))
            seen.add(bc.real_profile_copy_dir("chrome"))
        assert len(seen) == 2, "snapshot dir must be scoped to HERMES_HOME"
