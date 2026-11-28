"""Sibling launch-home check: _is_routed_home compares through the hermes_home_key cache.

tools.environments.local._is_routed_home is named alongside secret_scope._is_process_home by
hermes_constants.pin_process_hermes_home as a launch-home identity check; both must share the
one identity seam (hermes_constants.hermes_home_key), not a raw Path.resolve() round-trip per
call (#145022)."""

import pathlib

import pytest

from hermes_constants import reset_hermes_home_key_cache
from tools.environments.local import _is_routed_home


@pytest.fixture
def resolve_calls(monkeypatch):
    calls = []
    real = pathlib.Path.resolve

    def counting(self, *args, **kwargs):
        calls.append(str(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "resolve", counting)
    return calls


def _launch_home(tmp_path, monkeypatch):
    home = tmp_path / "launch"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    reset_hermes_home_key_cache()
    return home


def test_warm_routed_home_comparison_pays_no_syscalls(tmp_path, monkeypatch, resolve_calls):
    home = _launch_home(tmp_path, monkeypatch)
    assert _is_routed_home(home) is False  # cold: resolves populate the resolved-key cache
    resolve_calls.clear()
    for _ in range(3):
        assert _is_routed_home(home) is False
    assert resolve_calls == []


def test_symlinked_launch_home_is_not_routed(tmp_path, monkeypatch):
    """Symlink identity survives the cache: a home reached through a symlink is still the
    process's own, so it is not routed."""
    real_home = tmp_path / "real"
    real_home.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real_home, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(link))
    reset_hermes_home_key_cache()
    assert _is_routed_home(real_home) is False
    assert _is_routed_home(link) is False


def test_satellite_and_missing_homes_are_routed(tmp_path, monkeypatch):
    _launch_home(tmp_path, monkeypatch)
    satellite = tmp_path / "satellite"
    satellite.mkdir()
    assert _is_routed_home(satellite) is True
    assert _is_routed_home(tmp_path / "ghost") is True
