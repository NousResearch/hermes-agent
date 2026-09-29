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


def test_native_app_unlock_accepts_empty_password(home, monkeypatch):
    class OnePassword:
        name = "onepassword"
        needs_unlock = True
        supports_app_unlock = True

        def auth_capabilities(self):
            return {"mode": "interactive", "methods": ["app", "password"],
                    "native_app_eligible": True, "reason": None}

        def __init__(self):
            self.received = None

        def unlock(self, password):
            self.received = password

    backend = OnePassword()
    monkeypatch.setattr("agent.vault_backends.enabled_backends", lambda: [backend])

    result = _result(srv._methods["vault.unlock"](83, {"name": "onepassword", "password": ""}))

    assert backend.received == ""
    assert result == {"name": "onepassword", "unlocked": True}


def test_sources_include_secret_free_backend_auth_capabilities(home, monkeypatch):
    capability = {"mode": "service_account", "methods": [],
                  "native_app_eligible": False, "reason": None}

    class OnePassword:
        name = "onepassword"
        needs_unlock = True
        supports_app_unlock = True  # Legacy flag must not override host capability metadata.

        def auth_capabilities(self):
            return capability

        def is_unlocked(self):
            return False

    monkeypatch.setattr("agent.vault_backends.enabled_backends", lambda: [OnePassword()])
    row = _sources_rows(home)["onepassword"]

    assert row["auth_capabilities"] == capability
    assert "token" not in json.dumps(row).lower()
    assert "secret" not in json.dumps(row).lower()


@pytest.mark.parametrize(
    ("capability", "requested_method", "password", "allowed"),
    [
        ({"mode": "interactive", "methods": ["app", "password"], "native_app_eligible": True, "reason": None}, "app", "", True),
        ({"mode": "interactive", "methods": ["app", "password"], "native_app_eligible": True, "reason": None}, "password", "pw", True),
        ({"mode": "service_account", "methods": [], "native_app_eligible": False, "reason": None}, "app", "", False),
        ({"mode": "connect", "methods": [], "native_app_eligible": False, "reason": None}, "password", "pw", False),
    ],
)
def test_unlock_validates_explicit_method_against_backend_capabilities(
    home, monkeypatch, capability, requested_method, password, allowed
):
    class Backend:
        name = "onepassword"
        needs_unlock = True

        def auth_capabilities(self):
            return capability

        def unlock(self, password, *, method=None):
            self.received = (password, method)

    backend = Backend()
    monkeypatch.setattr("agent.vault_backends.enabled_backends", lambda: [backend])
    result = srv._methods["vault.unlock"](
        84, {"name": backend.name, "password": password, "method": requested_method}
    )

    if allowed:
        assert _result(result) == {"name": backend.name, "unlocked": True}
        assert backend.received == (password, requested_method)
    else:
        error = _error(result)
        assert error["code"] == 5095
        assert "not available" in error["message"].lower()
        assert not hasattr(backend, "received")


def test_legacy_backend_without_capability_metadata_keeps_old_unlock_request(home, monkeypatch):
    class LegacyBackend:
        name = "bitwarden"
        needs_unlock = True
        supports_app_unlock = False

        def unlock(self, password):
            self.received = password

    backend = LegacyBackend()
    monkeypatch.setattr("agent.vault_backends.enabled_backends", lambda: [backend])
    result = _result(srv._methods["vault.unlock"](
        86, {"name": "bitwarden", "password": "old-client"}
    ))
    assert result == {"name": "bitwarden", "unlocked": True}
    assert backend.received == "old-client"


def test_native_unlock_error_with_empty_password_is_not_corrupted_by_redaction(home, monkeypatch):
    class OnePassword:
        name = "onepassword"
        needs_unlock = True
        supports_app_unlock = True

        def unlock(self, password):
            assert password == ""
            raise RuntimeError("desktop authorization was declined")

    monkeypatch.setattr("agent.vault_backends.enabled_backends", lambda: [OnePassword()])

    error = _error(srv._methods["vault.unlock"](85, {"name": "onepassword", "password": ""}))

    assert error["message"] == "desktop authorization was declined"


def test_empty_password_remains_rejected_for_managers_without_native_unlock(home, monkeypatch):
    class Bitwarden:
        name = "bitwarden"
        needs_unlock = True
        supports_app_unlock = False

        def unlock(self, password):
            raise AssertionError("empty password must not reach Bitwarden")

    monkeypatch.setattr("agent.vault_backends.enabled_backends", lambda: [Bitwarden()])

    error = _error(srv._methods["vault.unlock"](84, {"name": "bitwarden", "password": ""}))

    assert error["code"] == 5095
    assert "required" in error["message"]


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
