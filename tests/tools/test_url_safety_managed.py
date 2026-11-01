"""Managed scope pins the SSRF policy the way load_config() pins every other key.

An operator behind a fake-ip TUN proxy declares ``security.fake_ip_ranges`` in
``/etc/hermes/config.yaml``; the profile's own config.yaml never names it. url_safety must read
the managed value, and a managed value must win over the profile's.
"""

import socket
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import managed_scope
from tools.url_safety import _global_allow_private_urls, _reset_allow_private_cache, is_safe_url


@pytest.fixture
def managed(tmp_path, monkeypatch):
    md = tmp_path / "managed"
    md.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(md))
    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
    managed_scope.invalidate_managed_cache()
    _reset_allow_private_cache()
    yield md
    managed_scope.invalidate_managed_cache()
    _reset_allow_private_cache()


def _write_managed(md: Path, body: str) -> None:
    (md / "config.yaml").write_text(body, encoding="utf-8")
    managed_scope.invalidate_managed_cache()
    _reset_allow_private_cache()


def _resolves_to(ip):
    return patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))])


def test_managed_fake_ip_ranges_unblock_the_proxy_sentinel(managed):
    with patch("hermes_cli.config.read_raw_config", return_value={}):
        with _resolves_to("198.18.5.86"):
            assert is_safe_url("https://seranking.com/") is False
        _write_managed(managed, "security:\n  fake_ip_ranges:\n    - 198.18.0.0/15\n")
        with _resolves_to("198.18.5.86"):
            assert is_safe_url("https://seranking.com/") is True
        with _resolves_to("10.0.0.8"):
            assert is_safe_url("https://internal.example/") is False


def test_managed_fake_ip_ranges_replace_the_profile_list(managed):
    profile = {"security": {"fake_ip_ranges": ["198.18.0.0/15"]}}
    _write_managed(managed, "security:\n  fake_ip_ranges: []\n")
    with patch("hermes_cli.config.read_raw_config", return_value=profile), _resolves_to("198.18.5.86"):
        assert is_safe_url("https://seranking.com/") is False


def test_managed_allow_private_urls_applies_and_wins(managed):
    with patch("hermes_cli.config.read_raw_config", return_value={}):
        _write_managed(managed, "security:\n  allow_private_urls: true\n")
        assert _global_allow_private_urls() is True
    profile = {"security": {"allow_private_urls": True}}
    with patch("hermes_cli.config.read_raw_config", return_value=profile):
        _write_managed(managed, "security:\n  allow_private_urls: false\n")
        assert _global_allow_private_urls() is False


def test_without_managed_scope_the_profile_value_decides(managed):
    profile = {"security": {"fake_ip_ranges": ["198.18.0.0/15"], "allow_private_urls": False}}
    with patch("hermes_cli.config.read_raw_config", return_value=profile):
        assert _global_allow_private_urls() is False
        with _resolves_to("198.18.5.86"):
            assert is_safe_url("https://seranking.com/") is True
