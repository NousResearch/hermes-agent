"""Multiplex invariant: tool-side profile credentials / targets never leak from the default profile.

Under ``gateway.multiplex_profiles`` ``os.environ`` holds the DEFAULT profile's ``.env``; a secondary
profile's turn runs with a secret scope that may not define a var at all. Every reader below must
then see "unset", never the default profile's value (`agent/secret_scope.py::get_secret` contract).
"""
from __future__ import annotations

import pytest

from agent import secret_scope


@pytest.fixture
def secondary_scope(monkeypatch):
    """Multiplex ON with a secondary profile's (empty) secret scope installed."""
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({})
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)


def test_scoped_tool_credential_gates_ignore_default_profile_environ(monkeypatch, secondary_scope, tmp_path):
    from tools import browser_use_cli, read_extract, tool_backend_helpers

    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-default")
    monkeypatch.setenv("MODAL_TOKEN_ID", "id-default")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "secret-default")
    monkeypatch.setenv("BROWSER_USE_API_KEY", "bu-default")
    monkeypatch.setattr(tool_backend_helpers.Path, "home", lambda: tmp_path)  # no ~/.modal.toml

    enabled, api_key, _ = read_extract._hosted_ocr_config()
    assert (enabled, api_key) == (False, None)
    assert tool_backend_helpers.has_direct_modal_credentials() is False
    assert browser_use_cli.is_legacy_browser_use_cloud_config({"cloud_provider": "browser-use"}) is False


def test_weixin_home_channel_resolves_from_profile_scope_not_environ(monkeypatch, secondary_scope):
    from tools import send_message_tool

    class _NoHome:
        def get_home_channel(self, platform):
            return None

    monkeypatch.setenv("WEIXIN_HOME_CHANNEL", "wx-default-chat")
    chat_id, err = send_message_tool._home_chat_id(_NoHome(), None, "weixin")
    assert chat_id is None and err
    # The secondary's own .env value is honoured.
    token = secret_scope.set_secret_scope({"WEIXIN_HOME_CHANNEL": "wx-secondary-chat"})
    try:
        assert send_message_tool._home_chat_id(_NoHome(), None, "weixin") == ("wx-secondary-chat", None)
    finally:
        secret_scope.reset_secret_scope(token)
