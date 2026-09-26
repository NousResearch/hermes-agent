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


def test_set_password_replaces_in_place_and_keeps_otp_and_origin(home):
    """In-place replacement (#123915): same handle, same origin, TOTP seed untouched, envelopes password-free."""
    item_id = _result(srv._methods["vault.add"](1, {
        "kind": "login",
        "label": "Canva",
        "origin": "https://canva.com",
        "secret": {
            "identifier_type": "email",
            "identifier": "user@example.com",
            "password": "old-pw-123",
            "otp_secret": "JBSWY3DPEHPK3PXP",
        },
    }))["id"]
    out = _result(srv._methods["vault.set_password"](2, {"id": item_id, "password": "new-pw-456"}))
    assert out["id"] == item_id
    assert "new-pw-456" not in json.dumps(out)
    listed = _result(srv._methods["vault.list"](3, {}))
    assert len(listed["items"]) == 1
    item = listed["items"][0]
    assert item["id"] == item_id
    assert item["origin"] == "https://canva.com"
    assert item["identifier"] == "user@example.com"
    assert item.get("has_otp") is True
    assert "new-pw-456" not in json.dumps(listed)


def test_set_password_errors_are_clean_and_leave_the_entry_untouched(home):
    item_id = _result(srv._methods["vault.add"](1, dict(_LOGIN_PARAMS)))["id"]
    err = _error(srv._methods["vault.set_password"](2, {"id": item_id, "password": "  "}))
    assert err["code"] == 5095
    err = _error(srv._methods["vault.set_password"](3, {"id": "vault_nope", "password": "new-pw"}))
    assert err["code"] == 5095
    err = _error(srv._methods["vault.set_password"](4, {"id": item_id}))
    assert err["code"] == 5095
    listed = _result(srv._methods["vault.list"](5, {}))
    assert len(listed["items"]) == 1
    assert "s3cret-pw-9000" not in json.dumps(_error(srv._methods["vault.set_password"](6, {"id": item_id, "password": ""})))
    # the stored password still fills: nothing was half-written
    assert listed["items"][0]["id"] == item_id


def test_launch_profile_vault_rpcs_stay_scoped_once_the_process_multiplexes(home, monkeypatch):
    """Once a second profile has been served, ``get_secret`` fails closed for unscoped reads. The
    launch profile's vault.* calls (Desktop sends no ``profile`` for it) must still bind the launch
    secret scope — otherwise every enabled manager's token read raises UnscopedSecretError and the
    Passwords & Logins panel shows "Could not load vault items" until the gateway restarts."""
    from agent.secret_scope import set_multiplex_active

    monkeypatch.setattr("agent.vault_backends.base.is_installed", lambda name: name == "onepassword")
    set_multiplex_active(True)  # conftest resets the latch per test
    assert _sources_rows(home)["onepassword"]["enabled"] is True
