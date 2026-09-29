"""Additive vault auth metadata remains valid across mixed client/backend versions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tui_gateway.contracts.profiles_vault_complete_foreign_subagents import (
    VaultSource,
    VaultUnlockParams,
)


def test_legacy_source_rows_without_auth_metadata_remain_valid():
    source = VaultSource(
        name="onepassword",
        display_name="1Password",
        enabled=True,
        needs_unlock=True,
        unlocked=False,
        installed=True,
    )
    assert source.auth_capabilities is None


@pytest.mark.parametrize("mode", ["interactive", "service_account", "connect", "unavailable"])
def test_capabilities_accept_each_auth_mode(mode):
    source = VaultSource.model_validate({
        "name": "onepassword",
        "display_name": "1Password",
        "enabled": True,
        "needs_unlock": True,
        "unlocked": False,
        "installed": True,
        "auth_capabilities": {
            "mode": mode,
            "methods": ["password"] if mode == "interactive" else [],
            "native_app_eligible": False,
            "reason": None,
        },
    })
    assert source.auth_capabilities is not None
    assert source.auth_capabilities.mode == mode


def test_unlock_method_is_optional_for_older_clients_and_restricted_for_new_clients():
    assert VaultUnlockParams(name="onepassword", password="masked").method is None
    assert VaultUnlockParams(name="onepassword", method="app").method == "app"
    assert VaultUnlockParams(name="onepassword", method="password").method == "password"
    with pytest.raises(ValidationError):
        VaultUnlockParams(name="onepassword", method="connect")
