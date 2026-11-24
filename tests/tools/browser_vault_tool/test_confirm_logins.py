"""``vault.confirm_logins``: a login on a listed site (subdomains included, ``*`` = every login) is filled only
after the user approves it; a declined or unanswerable prompt writes nothing and never resolves the secret."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.vault_store import VaultStore
from tools import browser_vault_tool

_LOGIN = {"identifier_type": "username", "identifier": "me", "password": "pw-canary-confirm"}


@pytest.fixture
def store(tmp_path):
    return VaultStore(base_dir=tmp_path / "vault")


def _config(listed):
    return patch("hermes_cli.config.load_config_readonly", return_value={"vault": {"confirm_logins": listed}})


@pytest.mark.parametrize(
    "listed, want",
    [
        (["bank.example"], True),
        (["https://bank.example"], True),
        (["https://bank.example/login?x=1"], True),
        ("*", True),
        (["mail.example", "*"], True),
        (["other.example"], False),
        (["ank.example"], False),  # a suffix that is not a whole label never matches
        (["www.bank.example"], False),  # a subdomain entry does not cover the parent
        ([], False),
        (None, False),
        (["not a site", "bank.example"], True),  # an unparseable entry is skipped, not fatal
    ],
)
def test_which_sites_need_approval(listed, want):
    meta = SimpleNamespace(origin="https://bank.example", allowed_origins=())
    with _config(listed):
        assert browser_vault_tool._login_needs_confirm(meta, "https://bank.example") is want


def test_listed_parent_covers_subdomains_and_manager_origins():
    meta = SimpleNamespace(origin="https://www.bank.example", allowed_origins=("https://app.bank.example",))
    with _config(["bank.example"]):
        assert browser_vault_tool._login_needs_confirm(meta, "https://www.bank.example") is True
    other = SimpleNamespace(origin="https://shop.example", allowed_origins=("https://pay.bank.example",))
    with _config(["bank.example"]):
        assert browser_vault_tool._login_needs_confirm(other, "https://shop.example") is True


def test_missing_or_malformed_vault_section_never_asks():
    meta = SimpleNamespace(origin="https://bank.example", allowed_origins=())
    for cfg in ({}, {"vault": True}, {"vault": {"confirm_logins": ""}}):
        with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
            assert browser_vault_tool._login_needs_confirm(meta, "https://bank.example") is False


@pytest.mark.parametrize("answer", ["decline", "cancel", None])
def test_declined_fill_writes_nothing_and_leaks_nothing(store, answer):
    meta = store.add_item(kind="login", label="Bank", origin="https://bank.example", secret=_LOGIN)
    secret_exprs = []
    with patch("agent.vault_store.get_vault_store", return_value=store), \
         patch.object(browser_vault_tool, "_eval_js",
                      return_value={"success": True, "result": "https://bank.example/login"}), \
         patch.object(browser_vault_tool, "_eval_js_secret", side_effect=lambda t, e: secret_exprs.append(e)), \
         _config(["bank.example"]), \
         patch("tools.approval_prompt.request_elicitation_consent", return_value=answer) as ask:
        raw = browser_vault_tool.browser_vault_fill(meta.id)
    out = json.loads(raw)
    assert out["success"] is False and out["error_type"] == "login_declined"
    assert ask.call_args.kwargs["surface"] == "vault-login"
    assert "Bank" in ask.call_args.args[0] and "https://bank.example" in ask.call_args.args[0]
    assert secret_exprs == [] and _LOGIN["password"] not in raw


def test_unlisted_site_fills_without_asking(store):
    meta = store.add_item(kind="login", label="Bank", origin="https://bank.example", secret=_LOGIN)
    with patch("agent.vault_store.get_vault_store", return_value=store), \
         patch.object(browser_vault_tool, "_eval_js",
                      return_value={"success": True, "result": "https://bank.example/login"}), \
         patch.object(browser_vault_tool, "_eval_js_secret", return_value={"success": False, "error": "stop"}), \
         _config(["other.example"]), \
         patch("tools.approval_prompt.request_elicitation_consent") as ask:
        out = json.loads(browser_vault_tool.browser_vault_fill(meta.id))
    ask.assert_not_called()
    assert out.get("error_type") != "login_declined"


def test_cards_keep_their_own_prompt_even_with_star(store):
    """``*`` is about logins: a card still goes through the payment prompt, never the login one."""
    card = {"card_number": "4111111111111111", "exp_month": "7", "exp_year": "2029", "cvc": "123",
            "cardholder_name": "A User", "billing_postal_code": "94110"}
    meta = store.add_item(kind="payment", label="Visa", origin="https://bank.example", secret=card)
    with patch("agent.vault_store.get_vault_store", return_value=store), \
         patch.object(browser_vault_tool, "_eval_js",
                      return_value={"success": True, "result": "https://bank.example/pay"}), \
         _config("*"), \
         patch("tools.approval_prompt.request_elicitation_consent", return_value="decline") as ask:
        out = json.loads(browser_vault_tool.browser_vault_fill(meta.id))
    assert out["error_type"] == "payment_declined" and ask.call_args.kwargs["surface"] == "vault-payment"
