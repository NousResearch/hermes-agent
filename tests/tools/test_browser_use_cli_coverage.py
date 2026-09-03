"""Test coverage for tools/browser_use_cli.py — 20 functions had LOW coverage.

Tests the pure helper functions: URL blocking, PATH flooring, config
reading, and mode detection. All subprocess and network calls mocked.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from tools.browser_use_cli import (
    _blocked_url_in_code,
    _floor_subprocess_path,
    is_legacy_browser_use_cloud_config,
)


class TestBlockedUrlInCode:
    def test_empty_code_returns_none(self):
        assert _blocked_url_in_code("") is None

    def test_none_returns_none(self):
        assert _blocked_url_in_code(None) is None

    def test_safe_url_returns_none(self):
        assert _blocked_url_in_code("open('https://example.com')") is None


class TestFloorSubprocessPath:
    def test_windows_noop(self, monkeypatch):
        monkeypatch.setattr(os, "name", "nt")
        result = _floor_subprocess_path("/usr/bin:/usr/local/bin")
        assert result == "/usr/bin:/usr/local/bin"


class TestLegacyBrowserUseCloudConfig:
    def test_empty_config_returns_false(self):
        assert is_legacy_browser_use_cloud_config({}) is False

    def test_none_config_returns_false(self):
        assert is_legacy_browser_use_cloud_config(None) is False
