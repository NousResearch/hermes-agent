"""The private-URL toggle and the fake-ip ranges are read from config once per process (outside a
profile-scoped turn) and re-read only after the test reset."""

from unittest.mock import patch

import pytest

from tools import url_safety


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
    url_safety._reset_allow_private_cache()
    yield
    url_safety._reset_allow_private_cache()


def _config(value: dict):
    reads = []

    def read_raw_config():
        reads.append(1)
        return value

    return reads, patch("hermes_cli.config.read_raw_config", read_raw_config)


def test_private_url_toggle_is_read_once_until_reset():
    reads, cfg = _config({"security": {"allow_private_urls": True}})
    with cfg:
        assert url_safety._global_allow_private_urls() is True
        assert url_safety._global_allow_private_urls() is True
    assert len(reads) == 1
    reads, cfg = _config({})
    with cfg:
        assert url_safety._global_allow_private_urls() is True
        url_safety._reset_allow_private_cache()
        assert url_safety._global_allow_private_urls() is False
        assert url_safety._global_allow_private_urls() is False
    assert len(reads) == 1


def test_an_empty_fake_ip_declaration_is_remembered_too():
    reads, cfg = _config({})
    with cfg:
        assert url_safety._global_fake_ip_ranges() == ()
        assert url_safety._global_fake_ip_ranges() == ()
    assert len(reads) == 1
    reads, cfg = _config({"security": {"fake_ip_ranges": ["198.18.0.0/15"]}})
    with cfg:
        assert url_safety._global_fake_ip_ranges() == ()
        url_safety._reset_allow_private_cache()
        assert [str(n) for n in url_safety._global_fake_ip_ranges()] == ["198.18.0.0/15"]
    assert len(reads) == 1
