"""Eager multi-profile activation: when a host arms the fail-closed secret guard, and when it must not.

``profiles_to_serve`` takes the multiplex flag as an ARGUMENT and never reads config, so the eager
gate has to consult the operator's setting itself; and ``named_profile_has_identity`` accepts an
empty ``.env``, which is all a crashed ``hermes profile create`` leaves behind. Both made a host
arm a one-way, process-wide guard it was never meant to arm.
"""
import pytest

from agent import secret_scope
from tui_gateway import launch_profile_policy


@pytest.fixture
def two_profile_host(tmp_path, monkeypatch):
    """Fake HOME so ``profiles/`` never resolves to the live install (see hermes-agent-dev)."""
    home = tmp_path / "fakehome" / ".hermes"
    (home / "profiles" / "b").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    from profiles.paths import _get_profiles_root
    assert str(_get_profiles_root()).startswith(str(tmp_path))
    return home / "profiles" / "b"


def test_activates_for_a_real_second_profile(two_profile_host):
    (two_profile_host / "config.yaml").write_text("{}\n", encoding="utf-8")
    assert launch_profile_policy.activate_multi_profile_hosting_eagerly() is True
    assert secret_scope.is_multiplex_active()


def test_the_retired_opt_out_no_longer_disarms_the_credential_guard(two_profile_host, monkeypatch):
    """``gateway.multiplex_profiles: false`` is retired as a topology opt-out, so it must not
    disarm this guard: a multi-home host that skipped activation because of a stale ``false``
    would serve the second profile with the LAUNCH profile's credentials -- the exact fail-open
    the guard exists to prevent. The host is multi-profile; that is the whole question."""
    (two_profile_host / "config.yaml").write_text("{}\n", encoding="utf-8")
    from hermes_cli import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "load_config", lambda *a, **k: {"gateway": {"multiplex_profiles": False}})

    assert launch_profile_policy.activate_multi_profile_hosting_eagerly() is True
    assert secret_scope.is_multiplex_active()


def test_a_crashed_profile_create_shell_is_not_a_second_tenant(two_profile_host):
    """An EMPTY ``.env`` is enough for ``named_profile_has_identity`` — not for flipping the host."""
    (two_profile_host / ".env").write_text("", encoding="utf-8")
    assert launch_profile_policy.activate_multi_profile_hosting_eagerly() is False
    assert not secret_scope.is_multiplex_active()


def test_unreadable_profiles_dir_fails_closed_and_says_so(two_profile_host, monkeypatch):
    """Silently returning False left the guard off for the process lifetime with zero log lines."""
    from gateway import profile_serving as profiles_mod

    def _boom(multiplex):
        raise PermissionError("profiles/ is unreadable")

    monkeypatch.setattr(profiles_mod, "profiles_to_serve", _boom)

    assert launch_profile_policy.activate_multi_profile_hosting_eagerly() is True
    assert secret_scope.is_multiplex_active()
