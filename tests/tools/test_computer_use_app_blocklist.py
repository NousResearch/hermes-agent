"""Computer-use app blocklist (Cowork-inspired): deterministic denial at the tool boundary.

Claude Cowork ships an app blocklist for computer use — "Prevent Claude from accessing certain
apps by adding them to a blocklist. Any requests from Claude to use blocked applications will be
automatically denied" — with trading/crypto apps blocked by default. These tests pin the Hermes
port's contract: the built-in sensitive list denies BOTH the explicit ``app=`` path and the
sticky-target path pre-approval, token matching never over-blocks lookalike names, and the
config knobs (blocked_apps / unblocked_apps / block_sensitive_apps) all take effect.
"""
import json
from types import SimpleNamespace

import pytest

from tools.computer_use import app_blocklist
from tools.computer_use.tool import _dispatch, _reject_unsafe


@pytest.fixture
def cu_config(monkeypatch):
    """Point the blocklist at a controlled computer_use config block."""
    def set_cfg(**cfg):
        monkeypatch.setattr(app_blocklist, "_cfg", lambda: cfg)
        return cfg
    return set_cfg


def test_sensitive_default_blocks_both_paths_without_overblocking(cu_config):
    cu_config()  # empty config: built-in sensitive list active by default
    # Explicit app= target denied before approval, structured code, and capture is covered too.
    for action, args in (("capture", {"app": "Ledger Live"}), ("type", {"text": "x", "app": "1Password"})):
        out = json.loads(_reject_unsafe(action, args))
        assert out["code"] == "app_blocked"
    # Sticky-target path: input to a blocklisted current target is denied even without app=.
    backend = SimpleNamespace(_last_app="Robinhood")
    out = json.loads(_dispatch(backend, "type", {"text": "sell everything"}))
    assert out["code"] == "app_blocked" and "Robinhood" in out["error"]
    # Token matching never over-blocks lookalikes or unrelated apps; unknown target fails open.
    for name in ("GitKraken", "kate", "Ledger of Accounts.xlsx - LibreOffice", ""):
        assert app_blocklist.blocked_app_match(name) is None, name
    assert _reject_unsafe("click", {"element": 1}) is None  # no app= → no explicit-path denial


def test_config_knobs_extend_exempt_and_disable(cu_config):
    # blocked_apps extends; matching is case/punctuation-insensitive token sequence.
    cu_config(blocked_apps=["My Bank"])
    assert app_blocklist.blocked_app_match("my-bank Desktop") == "My Bank"
    assert app_blocklist.blocked_app_match("MyBankHelper") is None  # one token ≠ two-token pattern
    # unblocked_apps exempts a built-in pattern (false-positive valve) but not the others.
    cu_config(unblocked_apps=["kraken"])
    assert app_blocklist.blocked_app_match("Kraken Pro") is None
    assert app_blocklist.blocked_app_match("Coinbase") == "coinbase"
    # block_sensitive_apps=False disables the built-in list; explicit blocked_apps still applies.
    cu_config(block_sensitive_apps=False, blocked_apps=["coinbase"])
    assert app_blocklist.blocked_app_match("Robinhood") is None
    assert app_blocklist.blocked_app_match("Coinbase") == "coinbase"
