"""Tests: vault.* JSON-RPC handlers (tui_gateway/methods_vault.py).

The Desktop's Settings → Credential Vault panel. Contracts:
- vault.list returns metadata only — secret values must never appear in
  any response envelope;
- vault.add validates via VaultStore.add_item and surfaces clean,
  secret-free error messages;
- vault.remove reports {removed: bool} idempotently.
"""

from __future__ import annotations

import json

import pytest

import tui_gateway.server as srv


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _result(envelope):
    assert "error" not in envelope, envelope
    return envelope["result"]


def _error(envelope):
    assert "error" in envelope, envelope
    return envelope["error"]


_LOGIN_PARAMS = {
    "kind": "login",
    "label": "Example login",
    "origin": "https://example.com",
    "secret": {
        "identifier_type": "email",
        "identifier": "user@example.com",
        "password": "s3cret-pw-9000",
    },
}


def test_add_then_list_is_password_free(home):
    out = _result(srv._methods["vault.add"](1, dict(_LOGIN_PARAMS)))
    assert out["id"].startswith("vault_")
    # add's own envelope must not echo the secret back
    assert "s3cret-pw-9000" not in json.dumps(out)

    listed = _result(srv._methods["vault.list"](2, {}))
    assert len(listed["items"]) == 1
    item = listed["items"][0]
    assert item["id"] == out["id"]
    assert item["kind"] == "login"
    assert item["label"] == "Example login"
    assert item["origin"] == "https://example.com"
    assert item["created_at"]
    # Identifier is agent-visible metadata (design: only the password is secret).
    assert item["identifier"] == "user@example.com"
    assert item["identifier_type"] == "email"
    dumped = json.dumps(listed)
    assert "s3cret-pw-9000" not in dumped
    assert "password" not in dumped


def test_add_validation_errors_are_clean(home):
    err = _error(
        srv._methods["vault.add"](
            1,
            {
                "kind": "login",
                "label": "no origin",
                "secret": {
                    "identifier_type": "email",
                    "identifier": "user@example.com",
                    "password": "s3cret-pw-9000",
                },
            },
        )
    )
    assert err["code"] == 5095
    assert "s3cret-pw-9000" not in json.dumps(err)

    err = _error(srv._methods["vault.add"](2, {"kind": "login", "label": "x"}))
    assert err["code"] == 5095

    err = _error(
        srv._methods["vault.add"](
            3, {"kind": "wat", "label": "x", "secret": {"password": "s3cret-pw-9000"}}
        )
    )
    assert err["code"] == 5095
    assert "s3cret-pw-9000" not in json.dumps(err)


def _sources_rows(home):
    out = _result(srv._methods["vault.sources"](90, {}))
    return {row["name"]: row for row in out["sources"]}


def test_enabling_a_detected_manager_reports_it_enabled(home, monkeypatch):
    """The Settings toggle and `hermes vault sources --enable` both clear the opt-out override;
    the shipped default must then agree with `is_enabled()`'s zero-config contract — an installed
    manager becomes a login source instead of silently staying off (#109546)."""
    monkeypatch.setattr(
        "agent.vault_backends.base.is_installed", lambda name: name == "bitwarden"
    )
    _result(
        srv._methods["vault.source.set"](80, {"name": "bitwarden", "enabled": True})
    )
    rows = _sources_rows(home)
    assert rows["bitwarden"]["installed"] is True
    assert rows["bitwarden"]["enabled"] is True


def test_disabling_a_manager_persists_the_opt_out(home, monkeypatch):
    monkeypatch.setattr(
        "agent.vault_backends.base.is_installed", lambda name: name == "bitwarden"
    )
    _result(
        srv._methods["vault.source.set"](81, {"name": "bitwarden", "enabled": False})
    )
    rows = _sources_rows(home)
    assert rows["bitwarden"]["installed"] is True
    assert rows["bitwarden"]["enabled"] is False


def test_undetected_manager_stays_off(home, monkeypatch):
    monkeypatch.setattr("agent.vault_backends.base.is_installed", lambda name: False)
    rows = _sources_rows(home)
    assert rows["bitwarden"]["installed"] is False


def test_source_set_tolerates_scalar_vault_section(home, monkeypatch):
    """A hand-edited ``vault: true`` in config.yaml must not crash the Desktop
    Credential Vault source toggle: the malformed section is coerced to a dict
    before the opt-out is written (same YAML-shape hazard class _voice_cfg_dict
    documents for voice.*, #19835)."""
    monkeypatch.setattr(
        "agent.vault_backends.base.is_installed", lambda name: name == "bitwarden"
    )
    (home / "config.yaml").write_text("vault: true\n")
    _result(
        srv._methods["vault.source.set"](82, {"name": "bitwarden", "enabled": False})
    )
    rows = _sources_rows(home)
    assert rows["bitwarden"]["enabled"] is False


def test_remove_is_idempotent(home):
    item_id = _result(srv._methods["vault.add"](1, dict(_LOGIN_PARAMS)))["id"]
    assert _result(srv._methods["vault.remove"](2, {"id": item_id}))["removed"] is True
    assert _result(srv._methods["vault.remove"](3, {"id": item_id}))["removed"] is False
    assert _result(srv._methods["vault.list"](4, {}))["items"] == []


def test_remove_requires_id(home):
    err = _error(srv._methods["vault.remove"](1, {}))
    assert err["code"] == 5095


def test_launch_profile_vault_rpcs_stay_scoped_once_the_process_multiplexes(home, monkeypatch):
    """Once a second profile has been served, ``get_secret`` fails closed for unscoped reads. The
    launch profile's vault.* calls (Desktop sends no ``profile`` for it) must still bind the launch
    secret scope — otherwise every enabled manager's token read raises UnscopedSecretError and the
    Passwords & Logins panel shows "Could not load vault items" until the gateway restarts."""
    from agent.secret_scope import set_multiplex_active

    monkeypatch.setattr("agent.vault_backends.base.is_installed", lambda name: name == "onepassword")
    set_multiplex_active(True)  # conftest resets the latch per test
    assert _sources_rows(home)["onepassword"]["enabled"] is True


_FAKE_BW_STATUS = """#!/usr/bin/env python3
import json, os, sys
accepted = os.environ.get("BW_SESSION") == "dummy-toggle-external-session"
print(json.dumps({"status": "unlocked" if accepted else "locked"}))
"""


@pytest.fixture
def external_bitwarden(home, tmp_path, monkeypatch):
    """Bitwarden with an opt-in session file and a fake ``bw`` that only answers ``status``."""
    from agent.vault_backends import unlock

    monkeypatch.setattr("agent.vault_backends.base.is_installed", lambda name: True)
    bw = tmp_path / "bw"
    bw.write_text(_FAKE_BW_STATUS, encoding="utf-8")
    bw.chmod(0o700)
    session = tmp_path / "session"
    session.write_text("dummy-toggle-external-session", encoding="utf-8")
    session.chmod(0o600)
    (home / "config.yaml").write_text(
        f"vault:\n  bitwarden:\n    binary_path: {bw}\n    session_file: {session}\n", encoding="utf-8")
    yield unlock
    unlock.lock()


def _bitwarden_usable(home) -> bool:
    return _sources_rows(home)["bitwarden"]["unlocked"]


def _toggle(enabled: bool) -> None:
    _result(srv._methods["vault.source.set"](83, {"name": "bitwarden", "enabled": enabled}))


@pytest.mark.platforms("posix")
def test_reenabling_a_disabled_source_restores_its_external_session(home, external_bitwarden):
    assert _bitwarden_usable(home)
    _toggle(False)
    assert external_bitwarden.is_fenced("bitwarden")
    _toggle(True)
    assert not external_bitwarden.is_fenced("bitwarden")
    assert _bitwarden_usable(home)


@pytest.mark.platforms("posix")
def test_enabling_an_already_enabled_source_keeps_a_manual_lock(home, external_bitwarden):
    _result(srv._methods["vault.lock"](84, {"name": "bitwarden"}))
    _toggle(True)
    assert external_bitwarden.is_fenced("bitwarden")
    assert not _bitwarden_usable(home)
    assert _sources_rows(home)["bitwarden"]["enabled"] is True


@pytest.mark.platforms("posix")
def test_disable_then_enable_after_a_manual_lock_restores_the_external_session(home, external_bitwarden):
    _result(srv._methods["vault.lock"](85, {"name": "bitwarden"}))
    _toggle(False)
    _toggle(True)
    assert not external_bitwarden.is_fenced("bitwarden")
    assert _bitwarden_usable(home)


@pytest.mark.platforms("posix")
def test_failed_enable_save_keeps_the_source_fenced(home, external_bitwarden, monkeypatch):
    _toggle(False)

    def refuse(*args, **kwargs):
        raise OSError("config.yaml is read-only")

    with monkeypatch.context() as failing:
        failing.setattr("hermes_cli.config.save_config", refuse)
        with pytest.raises(OSError):
            srv._methods["vault.source.set"](86, {"name": "bitwarden", "enabled": True})
    assert external_bitwarden.is_fenced("bitwarden")
    assert _sources_rows(home)["bitwarden"]["enabled"] is False


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("lock_params", [{}, {"name": "bitwarden"}])
def test_a_lock_between_enable_save_and_clear_keeps_the_source_fenced(
        home, external_bitwarden, monkeypatch, lock_params):
    _toggle(False)
    from hermes_cli import config
    real_save = config.save_config
    stale = []

    def save_then_lock(cfg, **kwargs):
        real_save(cfg, **kwargs)
        stale.append(external_bitwarden.begin_unlock("bitwarden"))
        _result(srv._methods["vault.lock"](88, lock_params))

    monkeypatch.setattr(config, "save_config", save_then_lock)
    _toggle(True)
    assert external_bitwarden.is_fenced("bitwarden")
    assert not _bitwarden_usable(home)
    assert not external_bitwarden.store_session_token("bitwarden", "dummy-stale", stale[0])


@pytest.mark.platforms("posix")
def test_enable_that_does_not_persist_keeps_the_fence_and_reports_disabled(home, external_bitwarden, monkeypatch):
    _toggle(False)
    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: True)
    out = _result(srv._methods["vault.source.set"](89, {"name": "bitwarden", "enabled": True}))
    assert out["enabled"] is False
    assert external_bitwarden.is_fenced("bitwarden")


@pytest.mark.platforms("posix")
def test_reenabling_bitwarden_leaves_other_backends_and_profiles_fenced(home, external_bitwarden, tmp_path, monkeypatch):
    other_profile = tmp_path / "profile-b"
    with monkeypatch.context() as other:
        other.setenv("HERMES_HOME", str(other_profile))
        external_bitwarden.lock()
    _result(srv._methods["vault.lock"](87, {"name": "onepassword"}))
    _toggle(False)
    _toggle(True)
    assert not external_bitwarden.is_fenced("bitwarden")
    assert external_bitwarden.is_fenced("onepassword")
    with monkeypatch.context() as other:
        other.setenv("HERMES_HOME", str(other_profile))
        assert external_bitwarden.is_fenced("bitwarden")
        assert external_bitwarden.is_fenced("onepassword")
