"""A managed ``allow_private_urls`` pin decides the private-URL toggle before anything the user controls.

The managed-scope guide: a pinned key wins over the profile's ``config.yaml``, its ``.env`` and the shell
environment. For this toggle the user-controlled inputs are ``HERMES_ALLOW_PRIVATE_URLS`` (shell or profile
``.env``) and the profile's two spellings of the key (``security.allow_private_urls`` and the legacy
``browser.allow_private_urls``). Without a pin the order stays environment variable, then config.
"""
import socket
from unittest.mock import patch

import pytest

from hermes_cli import managed_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import url_safety


def _reset() -> None:
    managed_scope.invalidate_managed_cache()
    url_safety._reset_allow_private_cache()


@pytest.fixture
def managed(tmp_path, monkeypatch):
    """Write the managed config.yaml (HERMES_MANAGED_DIR) and drop the caches that hold the toggle."""
    md = tmp_path / "managed"
    md.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(md))
    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)

    def write(text: str) -> None:
        (md / "config.yaml").write_text(text, encoding="utf-8")
        _reset()

    yield write
    _reset()


def _profile(config: dict):
    return patch("hermes_cli.config.read_raw_config", return_value=config)


def _resolves_to(ip: str):
    return patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))])


@pytest.mark.parametrize("env_value", ["true", "1", "yes"])
def test_a_managed_false_pin_beats_the_environment_variable(managed, monkeypatch, env_value):
    managed("security:\n  allow_private_urls: false\n")
    monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", env_value)
    with _profile({}):
        assert url_safety._global_allow_private_urls() is False
        with _resolves_to("10.0.0.5"):
            assert url_safety.is_safe_url("https://intranet.example/") is False


@pytest.mark.parametrize("env_value", ["false", "0", "no"])
def test_a_managed_true_pin_beats_the_environment_variable(managed, monkeypatch, env_value):
    managed("security:\n  allow_private_urls: true\n")
    monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", env_value)
    with _profile({}):
        assert url_safety._global_allow_private_urls() is True
        with _resolves_to("10.0.0.5"):
            assert url_safety.is_safe_url("https://intranet.example/") is True


def test_a_managed_false_pin_beats_the_profiles_legacy_key(managed):
    managed("security:\n  allow_private_urls: false\n")
    with _profile({"browser": {"allow_private_urls": True}}):
        assert url_safety._global_allow_private_urls() is False


def test_a_managed_false_pin_beats_a_routed_profiles_own_file(managed, tmp_path):
    """The profile-scoped read (a multiplex gateway turn) resolves the pin the same way, from real files."""
    managed("security:\n  allow_private_urls: false\n")
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "security:\n  allow_private_urls: true\nbrowser:\n  allow_private_urls: true\n", encoding="utf-8")
    token = set_hermes_home_override(home)
    try:
        assert url_safety._global_allow_private_urls() is False
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize("pinned, env_value, profile", [
    (False, "true", {"security": {"allow_private_urls": True}}),
    (True, "false", {"security": {"allow_private_urls": False}}),
])
def test_a_managed_pin_on_the_legacy_key_decides_too(managed, monkeypatch, pinned, env_value, profile):
    managed(f"browser:\n  allow_private_urls: {str(pinned).lower()}\n")
    monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", env_value)
    with _profile(profile):
        assert url_safety._global_allow_private_urls() is pinned


def test_the_preferred_key_decides_when_the_managed_scope_pins_both(managed):
    managed("security:\n  allow_private_urls: false\nbrowser:\n  allow_private_urls: true\n")
    with _profile({}):
        assert url_safety._global_allow_private_urls() is False


@pytest.mark.parametrize("managed_text", [
    "security:\n  fake_ip_ranges: [198.18.0.0/15]\nterminal:\n  env_passthrough: [SSL_CERT_FILE]\n",  # a sandbox launcher's
    "security:\n  allow_private_urls: null\n",  # a null value pins nothing
])
@pytest.mark.parametrize("env_value, profile, expected", [
    ("true", {"security": {"allow_private_urls": False}}, True),
    ("false", {"security": {"allow_private_urls": True}, "browser": {"allow_private_urls": True}}, False),
])
def test_without_a_pin_the_environment_variable_still_decides(managed, monkeypatch, managed_text, env_value,
                                                             profile, expected):
    managed(managed_text)
    monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", env_value)
    with _profile(profile):
        assert url_safety._global_allow_private_urls() is expected


def test_without_a_pin_either_profile_key_still_opts_in(managed):
    managed("security:\n  fake_ip_ranges: [198.18.0.0/15]\n")
    with _profile({"security": {"allow_private_urls": False}, "browser": {"allow_private_urls": True}}):
        assert url_safety._global_allow_private_urls() is True


_NULLS = [
    "security:\n  allow_private_urls: null\n",
    "browser:\n  allow_private_urls: null\n",
    "security:\n  allow_private_urls: null\nbrowser:\n  allow_private_urls: null\n",
]


@pytest.mark.parametrize("managed_text", _NULLS)
@pytest.mark.parametrize("profile, expected", [
    ({"security": {"allow_private_urls": True}}, True),
    ({"browser": {"allow_private_urls": True}}, True),
    ({"security": {"allow_private_urls": False}}, False),
    ({}, False),
])
def test_a_managed_null_leaves_the_profiles_own_keys_deciding(managed, managed_text, profile, expected):
    """A null on either spelling pins nothing. With the environment variable unset, the profile's
    own keys decide as they do without a managed file: the null does not erase them."""
    managed(managed_text)
    with _profile(profile):
        assert url_safety._global_allow_private_urls() is expected


@pytest.mark.parametrize("managed_text", _NULLS)
@pytest.mark.parametrize("spelling", ["security", "browser"])
def test_a_managed_null_leaves_a_routed_profiles_own_file_deciding(managed, tmp_path, managed_text, spelling):
    """The same through the real loaders and a profile-scoped read (a multiplex gateway turn)."""
    managed(managed_text)
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(f"{spelling}:\n  allow_private_urls: true\n", encoding="utf-8")
    token = set_hermes_home_override(home)
    try:
        assert url_safety._global_allow_private_urls() is True
        with _resolves_to("10.0.0.5"):
            assert url_safety.is_safe_url("https://intranet.example/") is True
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize("managed_text, pinned", [
    ("security:\n  allow_private_urls: null\nbrowser:\n  allow_private_urls: false\n", False),
    ("security:\n  allow_private_urls: null\nbrowser:\n  allow_private_urls: true\n", True),
])
def test_a_null_on_one_spelling_leaves_a_pin_on_the_other_in_force(managed, managed_text, pinned):
    managed(managed_text)
    with _profile({"security": {"allow_private_urls": not pinned}, "browser": {"allow_private_urls": not pinned}}):
        assert url_safety._global_allow_private_urls() is pinned
