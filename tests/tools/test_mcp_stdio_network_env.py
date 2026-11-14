"""Stdio MCP children inherit the parent's proxy and CA-bundle settings, never its secrets."""

from unittest.mock import patch


def test_proxy_env_vars_forwarded_from_parent():
    """Parent proxy keys (both cases) and CA-bundle keys must reach stdio MCP children."""
    from tools.mcp_tool_config import _build_safe_env

    fake_env = {
        "PATH": "/usr/bin",
        "SSL_CERT_FILE": "/etc/corp/ca.pem",
        "NODE_EXTRA_CA_CERTS": "/etc/corp/ca.pem",
        "http_proxy": "http://proxy.example:8080",
        "https_proxy": "http://proxy.example:8080",
        "no_proxy": "localhost,127.0.0.1",
        "HTTP_PROXY": "http://proxy.example:8080",
        "HTTPS_PROXY": "http://proxy.example:8443",
        "NO_PROXY": "localhost",
        "SECRET_KEY": "should_not_appear",
    }
    with patch.dict("os.environ", fake_env, clear=True):
        result = _build_safe_env(None)
    assert result["http_proxy"] == "http://proxy.example:8080"
    assert result["https_proxy"] == "http://proxy.example:8080"
    assert result["no_proxy"] == "localhost,127.0.0.1"
    assert result["HTTP_PROXY"] == "http://proxy.example:8080"
    assert result["HTTPS_PROXY"] == "http://proxy.example:8443"
    assert result["NO_PROXY"] == "localhost"
    assert result["SSL_CERT_FILE"] == result["NODE_EXTRA_CA_CERTS"] == "/etc/corp/ca.pem"
    assert "SECRET_KEY" not in result


def test_user_env_proxy_overrides_parent():
    """Server config env: wins; unset parent proxy keys are not invented."""
    from tools.mcp_tool_config import _build_safe_env

    fake_env = {
        "PATH": "/usr/bin",
        "http_proxy": "http://parent-proxy:8080",
        "HTTPS_PROXY": "http://parent-proxy:8443",
    }
    user_env = {
        "http_proxy": "http://server-config-proxy:9000",
    }
    with patch.dict("os.environ", fake_env, clear=True):
        result = _build_safe_env(user_env)
    assert result["http_proxy"] == "http://server-config-proxy:9000"
    assert result["HTTPS_PROXY"] == "http://parent-proxy:8443"
    assert "https_proxy" not in result
    assert "no_proxy" not in result
    assert "NO_PROXY" not in result
    assert "HTTP_PROXY" not in result
