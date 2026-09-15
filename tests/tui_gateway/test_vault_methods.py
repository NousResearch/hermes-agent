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


def test_remove_is_idempotent(home):
    item_id = _result(srv._methods["vault.add"](1, dict(_LOGIN_PARAMS)))["id"]
    assert _result(srv._methods["vault.remove"](2, {"id": item_id}))["removed"] is True
    assert _result(srv._methods["vault.remove"](3, {"id": item_id}))["removed"] is False
    assert _result(srv._methods["vault.list"](4, {}))["items"] == []


def test_remove_requires_id(home):
    err = _error(srv._methods["vault.remove"](1, {}))
    assert err["code"] == 5095


def test_list_scrubs_unicode_tag_characters_from_manager_supplied_text(home):
    # A shared/imported vault can carry invisible TAG characters in its titles
    # and usernames; they must not reach the model through vault.list.
    params = dict(_LOGIN_PARAMS)
    params["label"] = "Example\U000E0000 login\U000E0069\U000E0067\U000E006E\U000E006F\U000E0072\U000E0065"
    params["secret"] = dict(_LOGIN_PARAMS["secret"],
                            identifier="user@example.com\U000E0000")
    out = _result(srv._methods["vault.add"](1, params))

    listed = _result(srv._methods["vault.list"](2, {}))
    item = listed["items"][0]
    assert item["id"] == out["id"]  # handle untouched: fills route by it
    assert item["label"] == "Example login"
    assert item["identifier"] == "user@example.com"
    dumped = json.dumps(listed)
    assert "\U000E0000" not in dumped
    assert "\U000E0069" not in dumped
    assert "\U000E0072" not in dumped
    assert "\U000E007F" not in dumped

    # Valid emoji tag sequences (TR51 flags) are the one legitimate use of
    # plane-14 characters and must survive the scrub.
    flag = "\U0001F3F4\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F"  # Scotland
    params2 = dict(_LOGIN_PARAMS, label=f"{flag} Bank")
    srv._methods["vault.add"](3, params2)
    labels = {i["label"] for i in _result(srv._methods["vault.list"](4, {}))["items"]}
    assert f"{flag} Bank" in labels
