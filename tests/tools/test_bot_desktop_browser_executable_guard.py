"""The persistent-profile version floor on the Bot Desktop browser selection (#135932).

Chromium SIGTRAPs opening a user-data-dir written by a newer build, so a launch choice older
than the profile's ``Last Version`` is not a slowdown but a hard crash ('CDP response channel
closed'). The selection must floor on the profile: an explicit pin older than it falls back to
a bundled/system build that can still open it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.bot_desktop import browser


@pytest.fixture(autouse=True)
def _no_env_pin(monkeypatch):
    monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)


@pytest.fixture(autouse=True)
def _fresh_version_cache():
    # Hold the real lru-wrapped probe: tests monkeypatch the module attribute, and the cache
    # must be cleared through the original object, not whatever replaced the name mid-test.
    original = browser._chromium_major
    original.cache_clear()
    yield
    original.cache_clear()


def _fake_chromium(path: Path) -> str:
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def _written_profile(tmp_path, last_version="149.0.7827.55") -> Path:
    profile = tmp_path / "browser-profile"
    profile.mkdir()
    (profile / "Last Version").write_text(last_version, encoding="utf-8")
    return profile


def test_explicit_pin_older_than_the_profile_falls_back_to_a_newer_bundle(tmp_path, monkeypatch):
    pinned, bundled = _fake_chromium(tmp_path / "pinned-chromium"), _fake_chromium(tmp_path / "bundled-chromium")
    profile = _written_profile(tmp_path)
    monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", pinned)
    monkeypatch.setattr(browser, "_chromium_major", lambda exe: 145 if exe == pinned else 149)
    monkeypatch.setattr(browser, "_managed_executable", lambda: bundled)
    monkeypatch.setattr(browser, "_system_executable", lambda: None)

    assert browser.executable(profile=profile) == bundled


def test_explicit_pin_that_opens_the_profile_is_kept(tmp_path, monkeypatch):
    pinned = _fake_chromium(tmp_path / "pinned-chromium")
    profile = _written_profile(tmp_path)
    monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", pinned)
    monkeypatch.setattr(browser, "_chromium_major", lambda _: 149)

    assert browser.executable(profile=profile) == pinned


def test_explicit_pin_stands_when_no_newer_browser_exists(tmp_path, monkeypatch):
    pinned = _fake_chromium(tmp_path / "pinned-chromium")
    profile = _written_profile(tmp_path)
    monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", pinned)
    monkeypatch.setattr(browser, "_chromium_major", lambda _: 145)
    monkeypatch.setattr(browser, "_managed_executable", lambda: None)
    monkeypatch.setattr(browser, "_system_executable", lambda: None)

    assert browser.executable(profile=profile) == pinned, \
        "every candidate would crash the same way, and the user chose this pin"


def test_without_a_profile_the_explicit_pin_wins_without_any_version_probe(tmp_path, monkeypatch):
    pinned = _fake_chromium(tmp_path / "pinned-chromium")
    monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", pinned)

    def _no_probe(exe):
        raise AssertionError("no version probe is allowed without a profile")

    monkeypatch.setattr(browser, "_chromium_major", _no_probe)

    assert browser.executable() == pinned


def test_a_first_run_profile_never_blocks_the_pin(tmp_path, monkeypatch):
    pinned = _fake_chromium(tmp_path / "pinned-chromium")
    profile = tmp_path / "browser-profile"
    profile.mkdir()  # no Last Version yet: nothing written the jar
    monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", pinned)

    assert browser.executable(profile=profile) == pinned


def test_dock_launch_and_env_for_agent_floor_on_the_persistent_profile(monkeypatch):
    seen = {}

    def fake_executable(*, profile=None):
        seen["profile"] = profile
        return "/usr/bin/chromium"

    monkeypatch.setattr(browser, "executable", fake_executable)
    monkeypatch.setattr(browser, "profile_dir", lambda: Path("/state/bd-profile"))

    assert browser.dock_launch() == ("/usr/bin/chromium", "/state/bd-profile")
    env = browser.env_for_agent({})

    assert env["AGENT_BROWSER_EXECUTABLE_PATH"] == "/usr/bin/chromium"
    assert env["AGENT_BROWSER_PROFILE"] == "/state/bd-profile"
    assert seen["profile"] == Path("/state/bd-profile")


def test_profile_chromium_major_parses_chromiums_last_version_file(tmp_path):
    profile = tmp_path / "p"
    profile.mkdir()

    assert browser._profile_chromium_major(None) is None
    assert browser._profile_chromium_major(profile) is None
    (profile / "Last Version").write_text("145.0.7632.6", encoding="utf-8")
    assert browser._profile_chromium_major(profile) == 145
    (profile / "Last Version").write_text("garbage", encoding="utf-8")
    assert browser._profile_chromium_major(profile) is None


def test_chromium_major_reads_the_binary_version_line(tmp_path):
    exe = tmp_path / "chrome"
    exe.write_text('#!/bin/sh\necho "Chromium 145.0.7632.6"\n', encoding="utf-8")
    exe.chmod(0o755)

    assert browser._chromium_major(str(exe)) == 145
    assert browser._chromium_major(str(tmp_path / "missing")) is None
