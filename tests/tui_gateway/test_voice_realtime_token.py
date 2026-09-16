"""Tests for the voice.realtime_token RPC (xAI ephemeral token mint).

Browser surfaces call this to get a short-lived credential plus the
server-built session.update payload. No network: requests.post and the
credential resolver are monkeypatched.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

import tui_gateway.server as srv

# Bound before the autouse fixture below replaces the module attribute, so the profile-scope test
# can exercise the real effective-config read against on-disk homes.
_REAL_LOAD_CFG = srv._load_cfg


def _call(params: dict) -> dict:
    return srv._methods["voice.realtime_token"](1, params)


class _FakeResponse(SimpleNamespace):
    def json(self):
        return self._json


def _response(status_code=200, json_body=None, text=""):
    return _FakeResponse(status_code=status_code, _json=json_body or {}, text=text)


@pytest.fixture
def _creds(monkeypatch):
    import tools.xai_http as xai_http

    resolver = MagicMock(return_value={"api_key": "xai-live-key"})
    monkeypatch.setattr(xai_http, "resolve_xai_http_credentials", resolver)
    return resolver


@pytest.fixture
def _post(monkeypatch):
    import requests

    post = MagicMock(return_value=_response(
        200, {"value": "eph-token-1", "expires_at": 1234567890}
    ))
    monkeypatch.setattr(requests, "post", post)
    return post


@pytest.fixture(autouse=True)
def _realtime_enabled(monkeypatch):
    monkeypatch.setattr(
        srv,
        "_load_cfg",
        lambda: {"voice": {"realtime": {"enabled": True, "brain": "supervisor"}}},
    )


class TestTokenMint:
    def test_disabled_realtime_is_rejected(self, _creds, _post, monkeypatch):
        monkeypatch.setattr(srv, "_load_cfg", lambda: {"voice": {"realtime": {"enabled": False}}})
        env = _call({})
        assert env["error"]["code"] == 4030
        assert "not enabled" in env["error"]["message"]
        _post.assert_not_called()

    def test_dispatch_validates_against_the_wire_contract(self, _creds, _post):
        """Through the real dispatcher (not the bare handler): the grant must satisfy the declared
        result model (strict mode raises ContractViolation on a mismatch) and an unknown param key
        is a client/backend version skew answered with 4000, never silently ignored."""
        ok = srv.handle_request({
            "jsonrpc": "2.0", "id": 7, "method": "voice.realtime_token",
            "params": {"keyterms": ["hermes-agent"], "expires_seconds": 120},
        })
        assert ok["result"]["token"] == "eph-token-1"
        assert ok["result"]["model"]
        skew = srv.handle_request({
            "jsonrpc": "2.0", "id": 8, "method": "voice.realtime_token",
            "params": {"expires": 120},
        })
        assert skew["error"]["code"] == 4000
        assert "expires" in skew["error"]["message"]

    def test_happy_path_returns_token_and_session_config(self, _creds, _post):
        env = _call({})
        result = env["result"]
        assert result["token"] == "eph-token-1"
        assert result["expires_at"] == 1234567890
        assert result["url"].startswith("wss://api.x.ai/v1/realtime?model=")
        # Session payload is the supervisor config from the single Python
        # source of truth — browser surfaces send it verbatim.
        session = result["session_update"]["session"]
        tool_names = [t["name"] for t in session["tools"]]
        assert "consult_hermes" in tool_names
        assert "steer_hermes" in tool_names
        assert session["audio"]["output"]["format"]["rate"] == 24000
        # Mint call shape.
        args, kwargs = _post.call_args
        assert args[0] == "https://api.x.ai/v1/realtime/client_secrets"
        assert kwargs["json"] == {"expires_after": {"seconds": 300}}
        assert kwargs["headers"]["Authorization"] == "Bearer xai-live-key"

    def test_mint_url_follows_configured_realtime_url(self, _creds, _post, monkeypatch):
        monkeypatch.setattr(
            srv, "_load_cfg",
            lambda: {"voice": {"realtime": {
                "enabled": True,
                "url": "wss://proxy.example.com/v1/realtime",
            }}},
        )
        result = _call({})["result"]
        assert _post.call_args[0][0] == (
            "https://proxy.example.com/v1/realtime/client_secrets"
        )
        assert result["url"].startswith("wss://proxy.example.com/v1/realtime?model=")

    def test_supervisor_forced_even_when_config_says_ears(self, _creds, _post, monkeypatch):
        monkeypatch.setattr(
            srv, "_load_cfg",
            lambda: {"voice": {"realtime": {"enabled": True, "brain": "ears"}}},
        )
        session = _call({})["result"]["session_update"]["session"]
        assert "tools" in session  # ears payload has no tools

    def test_context_keyterms_merge_with_voice_vocabulary(self, _creds, _post):
        session = _call({
            "keyterms": ["hermes-agent", "Repository", "", "hermes-agent"],
        })["result"]["session_update"]["session"]
        terms = session["audio"]["input"]["transcription"]["keyterms"]
        assert "Hermes" in terms
        assert "repository" in terms
        assert "hermes-agent" in terms
        assert sum(term.casefold() == "repository" for term in terms) == 1

    def test_expiry_is_clamped(self, _creds, _post):
        _call({"expires_seconds": 999999})
        assert _post.call_args[1]["json"] == {"expires_after": {"seconds": 3600}}
        _call({"expires_seconds": 1})
        assert _post.call_args[1]["json"] == {"expires_after": {"seconds": 60}}
        _call({"expires_seconds": "soon"})
        assert _post.call_args[1]["json"] == {"expires_after": {"seconds": 300}}

    def test_nested_client_secret_shape_is_supported(self, _creds, _post):
        _post.return_value = _response(
            200, {"client_secret": {"value": "eph-2", "expires_at": 42}}
        )
        result = _call({})["result"]
        assert result["token"] == "eph-2"
        assert result["expires_at"] == 42

    def test_401_refreshes_credentials_and_retries_once(self, _creds, _post):
        _creds.side_effect = [
            {"api_key": "xai-stale"},
            {"api_key": "xai-fresh"},
        ]
        _post.side_effect = [
            _response(401, {}, "unauthorized"),
            _response(200, {"value": "eph-3"}),
        ]
        result = _call({})["result"]
        assert result["token"] == "eph-3"
        assert _creds.call_args_list[1][1] == {
            "force_refresh": True, "api_key_hint": "xai-stale",
        }
        assert _post.call_args_list[1][1]["headers"]["Authorization"] == "Bearer xai-fresh"

    def test_missing_credentials_is_a_typed_error(self, _creds, _post):
        _creds.return_value = {"api_key": ""}
        _creds.side_effect = None
        env = _call({})
        assert env["error"]["code"] == 4030
        _post.assert_not_called()

    def test_upstream_http_error_is_surfaced(self, _creds, _post):
        _post.return_value = _response(500, {}, "boom")
        _post.side_effect = None
        env = _call({})
        assert env["error"]["code"] == 5030
        assert "500" in env["error"]["message"]

    def test_empty_token_is_an_error(self, _creds, _post):
        _post.return_value = _response(200, {"unexpected": True})
        env = _call({})
        assert env["error"]["code"] == 5030


def _write_voice_cfg(home: Path, *, realtime_enabled: bool, url: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"voice": {"realtime": {"enabled": realtime_enabled, "url": url}}}),
        encoding="utf-8",
    )


def test_params_profile_selects_that_profiles_realtime_config(tmp_path, monkeypatch, _creds, _post):
    """The dashboard serves several profiles from one backend and runs the voice card's consults in
    ``params.profile``'s session; the token must be minted from THAT profile's ``voice.realtime``
    block, not the launch profile's (which here has realtime off and a different endpoint)."""
    launch = tmp_path / "launch"
    worker = tmp_path / "profiles" / "code"
    _write_voice_cfg(launch, realtime_enabled=False, url="wss://launch.example/v1/realtime")
    _write_voice_cfg(worker, realtime_enabled=True, url="wss://worker.example/v1/realtime")
    monkeypatch.setattr(srv, "_load_cfg", _REAL_LOAD_CFG)
    monkeypatch.setattr(srv, "_hermes_home", launch)
    monkeypatch.setattr(srv, "_profile_home", lambda name: worker if (name or "").strip() == "code" else None)
    srv._cfg_cache = srv._cfg_sig = srv._cfg_path = None

    assert _call({})["error"]["code"] == 4030  # launch profile: realtime off
    _post.assert_not_called()

    result = _call({"profile": "code"})["result"]
    assert result["token"] == "eph-token-1"
    assert _post.call_args[0][0] == "https://worker.example/v1/realtime/client_secrets"
    assert result["url"].startswith("wss://worker.example/v1/realtime?model=")
