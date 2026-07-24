"""Tests for the discord tool's react / unreact actions."""
import json

import pytest

import tools.discord_tool as dt


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    monkeypatch.setattr(dt, "_get_bot_token", lambda: "test-token")
    # Default: no server_actions allowlist configured (all actions available).
    monkeypatch.setattr(dt, "_load_allowed_actions_config", lambda: None)


def _spy(monkeypatch):
    calls = []

    def _fake_request(method, path, token, params=None, body=None, timeout=15):
        calls.append({"method": method, "path": path})
        return None  # Discord returns 204 for add/remove reaction

    monkeypatch.setattr(dt, "_discord_request", _fake_request)
    return calls


def test_react_issues_put_with_encoded_unicode_emoji(monkeypatch):
    calls = _spy(monkeypatch)
    out = json.loads(dt.discord_core(
        "react", channel_id="C1", message_id="M1", emoji="✅",
    ))
    assert out["success"] is True
    assert calls == [{
        "method": "PUT",
        "path": "/channels/C1/messages/M1/reactions/%E2%9C%85/@me",
    }]


def test_react_encodes_custom_emoji_name_id(monkeypatch):
    calls = _spy(monkeypatch)
    dt.discord_core("react", channel_id="C1", message_id="M1", emoji="party:123")
    assert calls[0]["path"] == "/channels/C1/messages/M1/reactions/party%3A123/@me"


def test_react_encodes_reserved_delimiter_hash(monkeypatch):
    """INV-5: a '#'-bearing emoji must be percent-encoded, not corrupt the path."""
    calls = _spy(monkeypatch)
    dt.discord_core("react", channel_id="C1", message_id="M1", emoji="#\u20e3")
    assert calls[0]["path"] == "/channels/C1/messages/M1/reactions/%23%E2%83%A3/@me"


def test_unreact_issues_delete(monkeypatch):
    calls = _spy(monkeypatch)
    out = json.loads(dt.discord_core(
        "unreact", channel_id="C1", message_id="M1", emoji="✅",
    ))
    assert out["success"] is True
    assert calls[0]["method"] == "DELETE"
    assert calls[0]["path"] == "/channels/C1/messages/M1/reactions/%E2%9C%85/@me"


def test_react_missing_emoji_returns_error(monkeypatch):
    _spy(monkeypatch)
    out = json.loads(dt.discord_core("react", channel_id="C1", message_id="M1", emoji=""))
    assert "error" in out
    assert "emoji" in out["error"].lower()
    assert out.get("success") is not True


def test_react_non_403_failure_does_not_fabricate_success(monkeypatch):
    """B1 fake-green guard: a 404 (message gone) must surface an error, never success:True."""
    def _raise_404(method, path, token, params=None, body=None, timeout=15):
        raise dt.DiscordAPIError(404, "Unknown Message")

    monkeypatch.setattr(dt, "_discord_request", _raise_404)
    out = json.loads(dt.discord_core("react", channel_id="C1", message_id="M1", emoji="✅"))
    assert out.get("success") is not True
    assert "error" in out


def test_react_403_gives_actionable_hint(monkeypatch):
    def _raise_403(method, path, token, params=None, body=None, timeout=15):
        raise dt.DiscordAPIError(403, "Missing Permissions")

    monkeypatch.setattr(dt, "_discord_request", _raise_403)
    out = json.loads(dt.discord_core("react", channel_id="C1", message_id="M1", emoji="✅"))
    assert "error" in out
    assert "ADD_REACTIONS" in out["error"]


def test_react_blocked_by_server_actions_allowlist(monkeypatch):
    _spy(monkeypatch)
    monkeypatch.setattr(dt, "_load_allowed_actions_config", lambda: ["fetch_messages"])
    out = json.loads(dt.discord_core("react", channel_id="C1", message_id="M1", emoji="✅"))
    assert "error" in out
    assert "disabled by config" in out["error"]


def test_react_available_when_server_actions_unset(monkeypatch):
    """RC-C / D-1: with discord.server_actions unset, react runs (mirrors pin_message)."""
    calls = _spy(monkeypatch)
    monkeypatch.setattr(dt, "_load_allowed_actions_config", lambda: None)
    out = json.loads(dt.discord_core("react", channel_id="C1", message_id="M1", emoji="✅"))
    assert out["success"] is True
    assert len(calls) == 1


def test_react_unreact_in_core_schema():
    assert dt._STATIC_CORE_SCHEMA is not None
    enum = dt._STATIC_CORE_SCHEMA["parameters"]["properties"]["action"]["enum"]
    assert "react" in enum and "unreact" in enum
    assert "emoji" in dt._STATIC_CORE_SCHEMA["parameters"]["properties"]
    # RC1: unreact description states the own-reaction scope.
    unreact_line = next(
        (d for n, _s, d in dt._ACTION_MANIFEST if n == "unreact"), "",
    )
    assert "OWN" in unreact_line or "own" in unreact_line
