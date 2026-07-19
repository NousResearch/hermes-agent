"""Tests for the ``browser.browsers_path`` config key.

Playwright puts the Chromium build, headless shell, and ffmpeg (~500MB) in a
fixed per-platform cache on the system drive unless ``PLAYWRIGHT_BROWSERS_PATH``
says otherwise (issue #66795). ``browser.browsers_path`` is the persistent,
config-file way to redirect that, bridged to the env var for subprocesses.

Precedence guard: an already-set ``PLAYWRIGHT_BROWSERS_PATH`` must always win,
so the new key cannot relocate the Docker image's baked-in browsers
(``Dockerfile`` sets ``ENV PLAYWRIGHT_BROWSERS_PATH=/opt/hermes/.playwright``)
or override an operator's shell export.
"""

import os
from unittest.mock import patch

import pytest

from tools import browser_tool as bt


def _cfg(path):
    return {"browser": {"browsers_path": path}}


@pytest.fixture(autouse=True)
def _reset_chromium_cache():
    bt._cached_chromium_installed = None
    yield
    bt._cached_chromium_installed = None


class TestConfiguredBrowsersPath:
    def test_reads_config_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        with patch("hermes_cli.config.read_raw_config", return_value=_cfg(str(tmp_path))):
            assert bt._configured_browsers_path() == str(tmp_path)

    def test_expands_user_home(self, monkeypatch):
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        with patch("hermes_cli.config.read_raw_config",
                   return_value=_cfg("~/browsers")):
            expected = os.path.expanduser("~/browsers")
            assert bt._configured_browsers_path() == expected

    def test_windows_drive_path_passed_through_verbatim(self, monkeypatch):
        """No os.path.join / separator rewriting — "D:/x" must survive as-is.

        The reporter's case is a Windows user relocating to a second drive;
        normalizing separators here would break on the non-native platform.
        """
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        with patch("hermes_cli.config.read_raw_config",
                   return_value=_cfg("D:/ms-playwright")):
            assert bt._configured_browsers_path() == "D:/ms-playwright"

    def test_empty_when_unset(self, monkeypatch):
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        with patch("hermes_cli.config.read_raw_config", return_value={}):
            assert bt._configured_browsers_path() == ""

    def test_empty_when_config_unreadable(self, monkeypatch):
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        with patch("hermes_cli.config.read_raw_config",
                   side_effect=Exception("no config")):
            assert bt._configured_browsers_path() == ""

    def test_empty_when_browser_section_is_not_a_dict(self, monkeypatch):
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        with patch("hermes_cli.config.read_raw_config",
                   return_value={"browser": "nonsense"}):
            assert bt._configured_browsers_path() == ""


class TestBrowsersPathEnvOverrides:
    def test_injects_configured_path(self, monkeypatch, tmp_path):
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        with patch("hermes_cli.config.read_raw_config", return_value=_cfg(str(tmp_path))):
            assert bt._browsers_path_env_overrides() == {
                "PLAYWRIGHT_BROWSERS_PATH": str(tmp_path)
            }

    def test_env_var_wins_over_config(self, monkeypatch, tmp_path):
        """Docker (Dockerfile ENV) and shell exports must not be overridden."""
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", "/opt/hermes/.playwright")
        with patch("hermes_cli.config.read_raw_config", return_value=_cfg(str(tmp_path))):
            assert bt._browsers_path_env_overrides() == {}

    def test_no_override_when_nothing_configured(self, monkeypatch):
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        with patch("hermes_cli.config.read_raw_config", return_value={}):
            assert bt._browsers_path_env_overrides() == {}


class TestBuildBrowserEnv:
    def test_subprocess_env_carries_configured_path(self, monkeypatch, tmp_path):
        """The agent-browser / `agent-browser install` subprocess is what
        actually downloads and launches Chromium, so the value has to reach
        its environment, not just Hermes' own process.
        """
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        with patch("hermes_cli.config.read_raw_config", return_value=_cfg(str(tmp_path))):
            env = bt._build_browser_env()
        assert env["PLAYWRIGHT_BROWSERS_PATH"] == str(tmp_path)

    def test_subprocess_env_untouched_when_key_unset(self, monkeypatch):
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        with patch("hermes_cli.config.read_raw_config", return_value={}):
            env = bt._build_browser_env()
        assert "PLAYWRIGHT_BROWSERS_PATH" not in env


class TestChromiumSearchRootsConfig:
    def test_configured_path_is_searched_first(self, monkeypatch, tmp_path):
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        with patch("hermes_cli.config.read_raw_config", return_value=_cfg(str(tmp_path))):
            roots = bt._chromium_search_roots()
        assert roots[0] == str(tmp_path)

    def test_default_caches_still_searched(self, monkeypatch, tmp_path):
        """Relocating the cache must not stop Hermes finding an existing
        install in Playwright's default location.
        """
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        with patch("hermes_cli.config.read_raw_config", return_value=_cfg(str(tmp_path))):
            roots = bt._chromium_search_roots()
        home = os.path.expanduser("~")
        assert os.path.join(home, ".cache", "ms-playwright") in roots

    def test_env_var_still_wins(self, monkeypatch, tmp_path):
        env_dir = tmp_path / "from-env"
        cfg_dir = tmp_path / "from-config"
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(env_dir))
        with patch("hermes_cli.config.read_raw_config", return_value=_cfg(str(cfg_dir))):
            roots = bt._chromium_search_roots()
        assert roots[0] == str(env_dir)
        assert str(cfg_dir) not in roots

    def test_env_var_zero_suppresses_config_path(self, monkeypatch, tmp_path):
        """"0" is Playwright's "keep browsers next to the package" sentinel.

        It is an explicit env-var choice, so the config fallback must not
        quietly re-add a directory the user opted out of.
        """
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", "0")
        with patch("hermes_cli.config.read_raw_config", return_value=_cfg(str(tmp_path))):
            roots = bt._chromium_search_roots()
        assert "0" not in roots
        assert str(tmp_path) not in roots

    def test_chromium_found_under_configured_path(self, monkeypatch, tmp_path):
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(bt.shutil, "which", lambda _name: None)
        (tmp_path / "chromium-1228").mkdir()
        with patch("hermes_cli.config.read_raw_config", return_value=_cfg(str(tmp_path))):
            assert bt._chromium_installed() is True


class TestDefaultConfig:
    def test_key_present_with_empty_default(self):
        """Empty default = Playwright's per-platform cache, i.e. no behavior
        change for anyone who does not set it.
        """
        from hermes_cli.config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["browser"]["browsers_path"] == ""
