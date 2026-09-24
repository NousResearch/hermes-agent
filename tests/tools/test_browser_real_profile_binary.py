"""Tests for the LOCAL real_profile_binary patch (browser.real_profile_binary).

Native behavior: real-profile browsing launches the DETECTED real browser binary.
The override lets a machine point that launch at another binary — e.g. the packaged
Chrome-for-Testing — so a headless automation instance is never the REAL Google
Chrome.app holding the macOS Launch Services registration ("dock trap": dock
clicks activate a windowless Chrome).

Invariants under test:
- override set + file exists    -> that path is returned
- override is a glob            -> NEWEST match wins (numeric revision sort)
- override set + missing        -> None (fall back to the real browser, never block)
- override unset / blank / odd  -> None, native behavior
"""
import os

import pytest


class TestRealProfileBinaryOverride:
    @pytest.fixture(autouse=True)
    def _patch_setting(self, monkeypatch):
        import hermes_cli.browser_connect as bc

        self._bc = bc
        self._setting = {}
        monkeypatch.setattr(bc, "_browser_setting", lambda key: self._setting.get(key))
        yield

    def test_unset_returns_none(self):
        from hermes_cli.browser_connect import real_profile_binary_override
        assert real_profile_binary_override() is None

    def test_blank_and_non_string_return_none(self):
        from hermes_cli.browser_connect import real_profile_binary_override
        self._setting["real_profile_binary"] = "   "
        assert real_profile_binary_override() is None
        self._setting["real_profile_binary"] = None
        assert real_profile_binary_override() is None
        self._setting["real_profile_binary"] = 42
        assert real_profile_binary_override() is None

    def test_existing_file_wins(self, tmp_path):
        from hermes_cli.browser_connect import real_profile_binary_override
        binary = tmp_path / "chrome-for-testing"
        binary.write_text("#!/bin/sh\n")
        self._setting["real_profile_binary"] = str(binary)
        assert real_profile_binary_override() == str(binary)

    def test_home_expansion(self, monkeypatch, tmp_path):
        from hermes_cli.browser_connect import real_profile_binary_override
        binary = tmp_path / "bin" / "chrome"
        binary.parent.mkdir()
        binary.write_text("#!/bin/sh\n")
        monkeypatch.setenv("HOME", str(tmp_path))
        self._setting["real_profile_binary"] = "~/bin/chrome"
        assert real_profile_binary_override() == str(binary)

    def test_glob_picks_newest_revision(self, tmp_path):
        """Playwright caches multiple chromium-<rev> dirs; the highest revision must win,
        including the 999 -> 1000 digit-boundary case a plain string max() gets wrong."""
        from hermes_cli.browser_connect import real_profile_binary_override
        for rev in ("1234", "999", "1000"):
            d = tmp_path / f"chromium-{rev}" / "chrome-mac"
            d.mkdir(parents=True)
            (d / "Google Chrome for Testing").write_text("#!/bin/sh\n")
        self._setting["real_profile_binary"] = str(tmp_path / "chromium-*" / "chrome-mac" / "Google Chrome for Testing")
        got = real_profile_binary_override()
        assert got == str(tmp_path / "chromium-1234" / "chrome-mac" / "Google Chrome for Testing")

    def test_glob_no_match_returns_none(self, tmp_path):
        from hermes_cli.browser_connect import real_profile_binary_override
        self._setting["real_profile_binary"] = str(tmp_path / "chromium-*" / "nope")
        assert real_profile_binary_override() is None

    def test_missing_file_returns_none(self, tmp_path):
        from hermes_cli.browser_connect import real_profile_binary_override
        self._setting["real_profile_binary"] = str(tmp_path / "does-not-exist")
        assert real_profile_binary_override() is None

    def test_call_site_wiring_present(self):
        """Drift guard: the real-profile launch path must consult the override (and only fall
        back to chromium_executable). Catches a refactor silently reverting to the real binary."""
        import inspect
        import tools.browser_tool_real_profile as tool

        src = inspect.getsource(tool._real_profile_cdp)
        assert "real_profile_binary_override()" in src, "override call missing from _real_profile_cdp"
        assert "real_profile_binary_override() or chromium_executable(" in src, (
            "override must take precedence over the detected real binary")
