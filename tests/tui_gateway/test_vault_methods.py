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


@pytest.fixture
def external_vault_rpc(home, monkeypatch):
    import subprocess
    from agent.vault_backends import base, bitwarden, unlock

    class Peer:
        def write(self, _obj):
            return True

        def close(self):
            pass

    backend = bitwarden.BitwardenLoginBackend({"binary_path": "fake-bw"})
    other = home / "other-profile"
    other.mkdir()
    monkeypatch.setattr(srv, "_profile_home", lambda profile: other if profile == "other" else home)
    monkeypatch.setattr(base, "enabled_backends", lambda: [backend])
    monkeypatch.setattr("agent.vault_backends.enabled_backends", lambda: [backend])
    monkeypatch.setattr(base, "is_installed", lambda _name: True)
    monkeypatch.setattr(bitwarden, "run_with_secret_env",
                        lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, "fake-settings-token", ""))
    monkeypatch.setattr(bitwarden, "run_cli", lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, json.dumps([
        {"id": "fake-item", "type": 1, "name": "Fake login", "login": {"uris": [{"uri": "https://example.test"}]}}
    ]), ""))
    peers = [Peer(), Peer()]
    for peer in peers:
        srv.register_live_transport(peer)

    def rpc(peer, method, **params):
        return srv.dispatch({"jsonrpc": "2.0", "id": 42, "method": method, "params": params}, peer)

    yield peers, backend, rpc, unlock
    for peer in peers:
        srv.unregister_live_transport(peer)
    unlock.lock_all_profiles()
    unlock.set_current_session_id(None)


def test_settings_unlock_is_private_to_its_live_transport_and_profile(external_vault_rpc):
    peers, backend, rpc, unlock = external_vault_rpc
    a, b = peers
    assert _result(rpc(a, "vault.unlock", name="bitwarden", password="fake-password"))["unlocked"]
    status = lambda peer, **params: next(row for row in _result(rpc(peer, "vault.sources", **params))["sources"]
                                        if row["name"] == "bitwarden")["unlocked"]
    assert status(a)
    assert len(_result(rpc(a, "vault.list"))["items"]) == 1
    # Unknown ownership fields must fail validation, not grant access or be silently ignored.
    rejected = rpc(b, "vault.sources", session_id="forged-owner", owner="forged-owner")
    assert _error(rejected)["code"] == 4000
    assert not status(b)
    assert status(a), "Rejected ownership spoofing must not revoke the legitimate transport"
    assert _result(rpc(b, "vault.list"))["items"] == []
    assert not status(a, profile="other")
    assert _result(rpc(a, "vault.lock", profile="other"))["locked"]
    assert status(a)
    unlock.set_current_session_id("unrelated-conversation")
    assert not backend.is_unlocked(), "Settings must not implicitly authorize a conversation"
    assert _result(rpc(b, "vault.lock", name="bitwarden"))["locked"]
    assert not status(a), "Explicit Lock remains profile-wide"


def test_settings_disconnect_revokes_inflight_and_queued_unlocks(external_vault_rpc, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from agent.vault_backends import bitwarden

    peers, _backend, rpc, _unlock = external_vault_rpc
    a, b = peers
    started, finish = threading.Event(), threading.Event()
    original = bitwarden.run_with_secret_env

    def delayed(*args, **kwargs):
        started.set()
        assert finish.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(bitwarden, "run_with_secret_env", delayed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(rpc, a, "vault.unlock", name="bitwarden", password="fake-password")
        try:
            assert started.wait(10)
            srv.unregister_live_transport(a)
        finally:
            finish.set()
        assert _error(pending.result(timeout=10))["code"] == 5095
    # A queued request carrying the disconnected transport cannot mint a replacement owner.
    assert _error(rpc(a, "vault.unlock", name="bitwarden", password="fake-password"))["code"] == 5095
    assert _result(rpc(b, "vault.unlock", name="bitwarden", password="fake-password"))["unlocked"]
    srv.unregister_live_transport(b)
    assert _result(rpc(b, "vault.list"))["items"] == [], "Disconnect also revokes committed tokens"


def test_launch_profile_vault_rpcs_stay_scoped_once_the_process_multiplexes(home, monkeypatch):
    """Once a second profile has been served, ``get_secret`` fails closed for unscoped reads. The
    launch profile's vault.* calls (Desktop sends no ``profile`` for it) must still bind the launch
    secret scope — otherwise every enabled manager's token read raises UnscopedSecretError and the
    Passwords & Logins panel shows "Could not load vault items" until the gateway restarts."""
    from agent.secret_scope import set_multiplex_active

    monkeypatch.setattr("agent.vault_backends.base.is_installed", lambda name: name == "onepassword")
    set_multiplex_active(True)  # conftest resets the latch per test
    assert _sources_rows(home)["onepassword"]["enabled"] is True
