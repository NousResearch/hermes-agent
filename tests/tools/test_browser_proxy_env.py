"""Tests for browser subprocess proxy env stripping (#14372 / #15372)."""


class TestBrowserProxyEnvStripped:
    """Proxy env vars must be stripped by the production env builder."""

    def test_build_browser_env_strips_proxy_vars(self, monkeypatch):
        """_build_browser_env removes proxy vars returned by subprocess env builder."""
        from tools import browser_tool
        from tools.environments import local

        proxy_vars = {
            "HTTP_PROXY": "http://proxy:8080",
            "HTTPS_PROXY": "http://proxy:8080",
            "NO_PROXY": "localhost",
            "ALL_PROXY": "socks5://proxy:1080",
            "http_proxy": "http://proxy:8080",
            "https_proxy": "http://proxy:8080",
            "no_proxy": "localhost",
            "all_proxy": "socks5://proxy:1080",
        }

        def fake_subprocess_env(inherit_credentials=False):
            assert inherit_credentials is False
            return {**proxy_vars, "SAFE_ENV": "kept"}

        monkeypatch.setattr(local, "hermes_subprocess_env", fake_subprocess_env)

        env = browser_tool._build_browser_env()

        assert env["SAFE_ENV"] == "kept"
        for var in proxy_vars:
            assert var not in env, f"{var} should have been stripped"

    def test_build_browser_env_preserves_browser_backend_keys(self, monkeypatch):
        """Proxy stripping must not remove explicitly allowed browser backend keys."""
        from tools import browser_tool
        from tools.environments import local

        monkeypatch.setattr(local, "hermes_subprocess_env", lambda inherit_credentials=False: {})
        monkeypatch.setenv("BROWSERBASE_API_KEY", "browserbase-key")
        monkeypatch.setenv("HTTP_PROXY", "http://proxy:8080")

        env = browser_tool._build_browser_env()

        assert env["BROWSERBASE_API_KEY"] == "browserbase-key"
        assert "HTTP_PROXY" not in env
