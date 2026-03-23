from pathlib import Path
from unittest.mock import Mock, patch


def test_playwright_registry_dir_uses_default_linux_cache(monkeypatch):
    from tools.browser_tool import _playwright_registry_dir

    monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/pw-cache")

    with patch("tools.browser_tool.sys.platform", "linux"):
        assert _playwright_registry_dir() == Path("/tmp/pw-cache/ms-playwright")


def test_playwright_registry_dir_resolves_relative_env_path(monkeypatch):
    from tools.browser_tool import _playwright_registry_dir

    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", "relative-browsers")
    monkeypatch.setenv("INIT_CWD", "/tmp/project-root")

    assert _playwright_registry_dir() == Path("/tmp/project-root/relative-browsers")


def test_check_browser_requirements_requires_local_playwright_runtime():
    from tools.browser_tool import check_browser_requirements

    with (
        patch("tools.browser_tool._find_agent_browser"),
        patch("tools.browser_tool._get_cloud_provider", return_value=None),
        patch("tools.browser_tool._has_local_playwright_browser", return_value=False),
    ):
        assert check_browser_requirements() is False

    with (
        patch("tools.browser_tool._find_agent_browser"),
        patch("tools.browser_tool._get_cloud_provider", return_value=None),
        patch("tools.browser_tool._has_local_playwright_browser", return_value=True),
    ):
        assert check_browser_requirements() is True


def test_check_browser_requirements_uses_cloud_provider_config_when_selected():
    from tools.browser_tool import check_browser_requirements

    provider = Mock()
    provider.is_configured.side_effect = [False, True]

    with (
        patch("tools.browser_tool._find_agent_browser"),
        patch("tools.browser_tool._get_cloud_provider", return_value=provider),
        patch("tools.browser_tool._has_local_playwright_browser") as mock_local_check,
    ):
        assert check_browser_requirements() is False
        assert check_browser_requirements() is True

    mock_local_check.assert_not_called()