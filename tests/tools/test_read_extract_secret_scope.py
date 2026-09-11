"""Hosted OCR resolves its credential through the profile secret scope.

The desktop cron ticker installs a profile secret scope WITHOUT the multiplex flag
(``cron/scheduler.py`` builds the scope per job; ``set_multiplex_active`` is only set by a
multiplexing gateway and the external worker), so a raw ``os.environ`` read resolves the
launch profile's key for a routed profile's job.
"""

import pytest

from agent.secret_scope import (UnscopedSecretError, build_profile_secret_scope,
                                reset_secret_scope, set_multiplex_active, set_secret_scope)
from hermes_cli.env_loader import hydrate_profile_secret_sources


def _routed_home(tmp_path, monkeypatch, key="company-ZZZZ"):
    """A routed profile home whose .env key differs from the launch profile's os.environ one."""
    home = tmp_path / "routed"
    home.mkdir()
    (home / ".env").write_text(f"FIRECRAWL_API_KEY={key}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("FIRECRAWL_API_KEY", "personal-AAAA")
    return home


def test_hosted_ocr_uses_routed_scope_without_the_multiplex_flag(tmp_path, monkeypatch):
    """The desktop-ticker shape: scope installed, multiplex flag off."""
    home = _routed_home(tmp_path, monkeypatch)
    hydrate_profile_secret_sources(home)
    token = set_secret_scope(build_profile_secret_scope(home))
    try:
        from tools import read_extract
        assert read_extract._hosted_ocr_config() == (True, "company-ZZZZ", None)
    finally:
        reset_secret_scope(token)


def test_hosted_ocr_uses_routed_scope_under_multiplex(tmp_path, monkeypatch):
    """The multiplexing gateway shape: scope installed, flag on."""
    home = _routed_home(tmp_path, monkeypatch)
    hydrate_profile_secret_sources(home)
    set_multiplex_active(True)
    token = set_secret_scope(build_profile_secret_scope(home))
    try:
        from tools import read_extract
        assert read_extract._hosted_ocr_config() == (True, "company-ZZZZ", None)
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)


def test_hosted_ocr_keeps_single_profile_environment_fallback(monkeypatch):
    """No scope, no multiplex: the process env stays the source (systemd / ``op run`` installs)."""
    monkeypatch.setenv("FIRECRAWL_API_KEY", "personal-AAAA")
    token = set_secret_scope(None)
    try:
        from tools import read_extract
        assert read_extract._hosted_ocr_config() == (True, "personal-AAAA", None)
    finally:
        reset_secret_scope(token)


def test_hosted_ocr_probe_survives_the_multiplex_fail_closed_path(monkeypatch):
    """Boot-time check_fns run before any scope exists; the probe reports unavailable, never raises."""
    monkeypatch.setenv("FIRECRAWL_API_KEY", "personal-AAAA")
    set_multiplex_active(True)
    token = set_secret_scope(None)
    try:
        from agent.secret_scope import get_secret
        with pytest.raises(UnscopedSecretError):
            get_secret("FIRECRAWL_API_KEY", "")
        from tools import read_extract
        assert read_extract._hosted_ocr_config() == (False, None, None)
        assert read_extract.hosted_ocr_available() is False
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)
