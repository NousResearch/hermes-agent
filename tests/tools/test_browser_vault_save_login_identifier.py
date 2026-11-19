"""browser_vault_save_login classifies the identifier the user typed: a phone number however it is
grouped, an email, else a username. The fill reads a saved phone back as ``tel``."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agent.vault_store import VaultStore


@pytest.fixture()
def store(tmp_path):
    return VaultStore(base_dir=tmp_path / "vault")


@pytest.mark.parametrize("identifier, id_type", [
    ("+48600100200", "phone"),
    ("+48 600 100 200", "phone"),
    ("(600) 100-200", "phone"),
    ("600.100.200", "phone"),
    ("tek@acme.test", "email"),
    ("tek-2024", "username"),
    ("12 34", "username"),  # too few digits for a phone number
    ("600100200x", "username"),  # a digit run with a letter is a handle
])
def test_a_phone_number_is_stored_as_phone_however_it_was_typed(store, monkeypatch, identifier, id_type):
    """The fill reads a saved phone back as ``tel``, which tolerates a mask's grouping; stored as a
    username, a number typed with spaces ends in ``fill_reverted`` on a site that groups it
    differently. The identifier itself is kept as the user typed it."""
    from agent.vault_backends import unlock as unlock_mod
    from tools import browser_vault_tool

    unlock_mod.set_save_login_prompt_callback(lambda origin, site: {"identifier": identifier, "password": "pw-x"})
    monkeypatch.setattr(browser_vault_tool, "_current_page_origin", lambda task_id: "https://acme.test")
    monkeypatch.setattr(browser_vault_tool, "browser_vault_fill",
                        lambda handle, task_id=None: json.dumps({"success": True, "filled_fields": 2}))
    try:
        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch("agent.vault_backends.unlock.can_prompt_here", return_value=True):
            out = json.loads(browser_vault_tool.browser_vault_save_login(task_id="t1"))
    finally:
        unlock_mod.set_save_login_prompt_callback(None)
    [meta] = store.list_items()
    assert (out["identifier_type"], meta.identifier_type, meta.identifier) == (id_type, id_type, identifier)
