"""Managed-scope guards that hold whatever the source of the network policy.

A managed ``security.fake_ip_ranges`` entry obeys the same reserved-class rule as the profile's own, and a managed
``terminal.env_passthrough`` list cannot tunnel a provider credential into sandboxed children either.
"""
import socket
from unittest.mock import patch

import pytest

from hermes_cli import managed_scope
from tools import env_passthrough, url_safety


def _reset() -> None:
    managed_scope.invalidate_managed_cache()
    url_safety._reset_allow_private_cache()
    env_passthrough._config_passthrough.clear()


@pytest.fixture
def managed(tmp_path, monkeypatch):
    """Write the managed config.yaml (HERMES_MANAGED_DIR) and drop every cache that holds these keys."""
    md = tmp_path / "managed"
    md.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(md))
    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)

    def write(text: str) -> None:
        (md / "config.yaml").write_text(text, encoding="utf-8")
        _reset()

    yield write
    _reset()


def _resolves_to(ip: str):
    return patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))])


def test_a_managed_fake_ip_entry_cannot_excuse_a_reserved_class(managed):
    managed("security:\n  fake_ip_ranges: [10.0.0.0/8, 198.18.0.0/15]\n")
    with _resolves_to("10.0.0.5"):
        assert url_safety.is_safe_url("https://example.com/") is False
    with _resolves_to("198.18.0.2"):
        assert url_safety.is_safe_url("https://example.com/") is True


def test_a_managed_env_passthrough_still_refuses_provider_credentials(managed):
    from tools.environments.local_env_policy import _HERMES_PROVIDER_ENV_BLOCKLIST

    credential = min(_HERMES_PROVIDER_ENV_BLOCKLIST)
    managed(f"terminal:\n  env_passthrough: [{credential}, SSL_CERT_FILE]\n")
    assert not env_passthrough.is_env_passthrough(credential)
    assert env_passthrough.is_env_passthrough("SSL_CERT_FILE")
