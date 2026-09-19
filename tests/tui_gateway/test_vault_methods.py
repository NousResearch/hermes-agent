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
    assert "origin is required" in err["message"]
    assert "s3cret-pw-9000" not in json.dumps(err)

    err = _error(srv._methods["vault.add"](2, {"kind": "login", "label": "x"}))
    assert err["code"] == 5095
    assert "secret payload is required" in err["message"]

    err = _error(
        srv._methods["vault.add"](
            3, {"kind": "wat", "label": "x", "secret": {"password": "s3cret-pw-9000"}}
        )
    )
    assert err["code"] == 5095
    assert "unknown vault kind" in err["message"]
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


def test_source_set_tolerates_scalar_named_section(home, monkeypatch):
    """``vault: {bitwarden: true}`` must not crash either: the named section is
    coerced before ``enabled`` is written."""
    monkeypatch.setattr(
        "agent.vault_backends.base.is_installed", lambda name: name == "bitwarden"
    )
    (home / "config.yaml").write_text("vault:\n  bitwarden: true\n")
    _result(
        srv._methods["vault.source.set"](83, {"name": "bitwarden", "enabled": False})
    )
    rows = _sources_rows(home)
    assert rows["bitwarden"]["enabled"] is False


def test_source_set_enable_tolerates_scalar_vault_section(home, monkeypatch):
    """The enable arm (``section.pop``) must also survive a scalar ``vault:``
    section: coercion happens before either write shape."""
    monkeypatch.setattr(
        "agent.vault_backends.base.is_installed", lambda name: name == "bitwarden"
    )
    (home / "config.yaml").write_text("vault: true\n")
    _result(
        srv._methods["vault.source.set"](84, {"name": "bitwarden", "enabled": True})
    )
    rows = _sources_rows(home)
    assert rows["bitwarden"]["enabled"] is True


def test_vault_list_reports_corrupt_store_cleanly(home):
    """A decryptable-but-malformed vault file degrades to the 5095 envelope
    carrying the VaultError text, never a traceback."""
    from agent.vault_store import VaultStore

    store = VaultStore(home / "vault")
    store.add_item(
        "login",
        "site",
        {"identifier_type": "email", "identifier": "u", "password": "p"},
        origin="https://x.example",
    )
    store._vault_path.write_bytes(store._fernet().encrypt(b"{not json"))

    err = _error(srv._methods["vault.list"](95, {}))
    assert err["code"] == 5095
    assert "corrupted" in err["message"]


def test_remove_is_idempotent(home):
    item_id = _result(srv._methods["vault.add"](1, dict(_LOGIN_PARAMS)))["id"]
    assert _result(srv._methods["vault.remove"](2, {"id": item_id}))["removed"] is True
    assert _result(srv._methods["vault.remove"](3, {"id": item_id}))["removed"] is False
    assert _result(srv._methods["vault.list"](4, {}))["items"] == []


def test_remove_requires_id(home):
    err = _error(srv._methods["vault.remove"](1, {}))
    assert err["code"] == 5095
