"""Grok-Live config/status surface: mode resolver widening, credential order, status contract.

These pin INVARIANTS from docs/grok-live-voice/SPEC.md (§1, §3, §8, §11), not snapshots:
  * ``voice_chat_mode()`` is the single source of truth and now returns ``grok-live``;
  * ``resolve_grok_live_status()`` has exactly the gpt-live status shape and never leaks a
    credential;
  * auth order is OAuth-first in ``auto`` — an env XAI_API_KEY must NEVER shadow a working
    SuperGrok OAuth token (the highest-value behavioral guarantee in the spec).
"""

import base64
import datetime
import json

import pytest

import tools.voice_live as voice_live
import tools.voice_live_grok as grok
from hermes_cli.config_defaults import DEFAULT_CONFIG


# ── voice_chat_mode() widening (SPEC §1) ─────────────────────────────────────

@pytest.mark.parametrize("raw", ["grok-live", "grok_live", "Grok-Live", "groklive", " grok-live "])
def test_voice_chat_mode_accepts_grok_live_spellings(raw):
    assert voice_live.voice_chat_mode({"voice_chat_mode": raw}) == "grok-live"


@pytest.mark.parametrize("raw,expected", [
    ("gpt-live", "gpt-live"),
    ("gpt_live", "gpt-live"),
    ("chained", "chained"),
    ("", "chained"),
    (None, "chained"),
    ("realtime", "chained"),  # unknown values still fall back, never crash
])
def test_voice_chat_mode_existing_engines_unchanged(raw, expected):
    assert voice_live.voice_chat_mode({"voice_chat_mode": raw}) == expected


def test_grok_module_does_not_reimplement_the_mode_resolver():
    # If the grok module exposes the resolver at all, it must BE the shared one (re-exported),
    # never a local reimplementation (SPEC §1: single source of truth).
    assert grok.voice_chat_mode is voice_live.voice_chat_mode
    assert "voice_chat_mode" not in grok.__dict__ or grok.__dict__["voice_chat_mode"] is voice_live.voice_chat_mode
    assert grok.GROK_LIVE_MODE == "grok-live"


# ── config defaults (SPEC §3) ────────────────────────────────────────────────

def test_default_config_carries_the_grok_live_block():
    block = DEFAULT_CONFIG["voice"]["grok_live"]
    assert block["model"] == "grok-voice-latest"
    assert block["voice"] == "eve"
    assert block["auth"] == "auto"
    assert block["instructions"] == ""
    assert block["vad_threshold"] == 0.75
    assert block["silence_ms"] == 700
    assert block["prefix_ms"] == 333
    assert block["speed"] == 1.0


# ── auth resolution order (SPEC §8 — AUTHORITATIVE) ─────────────────────────

def _write_oauth_auth_json(home, *, expired=False, access_token="oauth-token-abc"):
    now = datetime.datetime.now(datetime.timezone.utc)
    last_refresh = (now - datetime.timedelta(hours=2) if expired else now)
    id_payload = base64.urlsafe_b64encode(json.dumps({"aud": "test-client"}).encode()).decode().rstrip("=")
    doc = {
        "providers": {
            "xai-oauth": {
                "last_refresh": last_refresh.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                "discovery": {"token_endpoint": "https://auth.example.invalid/token"},
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": "refresh-xyz",
                    "id_token": f"header.{id_payload}.sig",
                    "expires_in": 3600,
                },
            }
        }
    }
    (home / "auth.json").write_text(json.dumps(doc))
    return doc


def test_auto_mode_prefers_oauth_over_a_present_api_key(tmp_path, monkeypatch):
    """THE pinning test (SPEC §8): oauth present → oauth used even when XAI_API_KEY is also set."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("XAI_API_KEY", "zero-credit-key")
    _write_oauth_auth_json(tmp_path)
    assert grok._resolve_grok_credentials({"auth": "auto"}) == "oauth-token-abc"


def test_auto_mode_falls_back_to_api_key_when_no_oauth(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("XAI_API_KEY", "fallback-key")
    assert grok._resolve_grok_credentials({"auth": "auto"}) == "fallback-key"


def test_oauth_mode_never_uses_the_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("XAI_API_KEY", "zero-credit-key")
    assert grok._resolve_grok_credentials({"auth": "oauth"}) == ""


def test_apikey_mode_ignores_the_oauth_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("XAI_API_KEY", "forced-key")
    _write_oauth_auth_json(tmp_path)
    assert grok._resolve_grok_credentials({"auth": "apikey"}) == "forced-key"


def test_api_key_reads_from_dot_env_when_env_var_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    (tmp_path / ".env").write_text('# comment\nXAI_API_KEY="dotenv-key"\n')
    assert grok._xai_api_key() == "dotenv-key"


def test_expired_oauth_token_refreshes_and_writes_back_atomically(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    _write_oauth_auth_json(tmp_path, expired=True, access_token="stale-token")

    class _Resp:
        def read(self):
            return json.dumps({"access_token": "fresh-token", "expires_in": 3600}).encode()

    monkeypatch.setattr(grok.urllib.request, "urlopen", lambda req, timeout=0: _Resp())
    assert grok._oauth_token() == "fresh-token"
    # The shared store is updated for the CLI's own xai-oauth lane, with no temp file left behind.
    doc = json.loads((tmp_path / "auth.json").read_text())
    assert doc["providers"]["xai-oauth"]["tokens"]["access_token"] == "fresh-token"
    assert not (tmp_path / "auth.json.tmp").exists()


def test_refresh_failure_yields_no_credential_and_keeps_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    _write_oauth_auth_json(tmp_path, expired=True, access_token="stale-token")

    def _boom(req, timeout=0):
        raise OSError("network down")

    monkeypatch.setattr(grok.urllib.request, "urlopen", _boom)
    assert grok._oauth_token() == ""
    assert json.loads((tmp_path / "auth.json").read_text())["providers"]["xai-oauth"][
        "tokens"]["access_token"] == "stale-token"


# ── resolve_grok_live_status() (SPEC §11) ────────────────────────────────────

def test_status_shape_and_no_credential_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    status = grok.resolve_grok_live_status()
    assert set(status) == {"mode", "available", "reason", "model", "voice"}
    assert status == {
        "mode": "chained",
        "available": False,
        "reason": "no xAI credential (SuperGrok login via `hermes login` or set XAI_API_KEY)",
        "model": "grok-voice-latest",
        "voice": "eve",
    }


def test_status_available_with_oauth_token_and_never_leaks_it(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    _write_oauth_auth_json(tmp_path, access_token="secret-oauth-token")
    status = grok.resolve_grok_live_status()
    assert status["available"] is True
    assert status["reason"] is None
    assert "secret-oauth-token" not in json.dumps(status)


def test_status_reflects_selected_mode_and_config_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("XAI_API_KEY", "some-key")
    (tmp_path / "config.yaml").write_text(
        "voice:\n"
        "  voice_chat_mode: grok-live\n"
        "  grok_live:\n"
        "    model: grok-voice-experimental\n"
        "    voice: ara\n"
    )
    status = grok.resolve_grok_live_status()
    assert status["mode"] == "grok-live"
    assert status["available"] is True
    assert status["model"] == "grok-voice-experimental"
    assert status["voice"] == "ara"


# ── persona (SPEC §5) ────────────────────────────────────────────────────────

def test_live_instructions_default_is_the_persona_and_appends_extras():
    assert grok.live_instructions({}) == grok.GROK_LIVE_PERSONA
    merged = grok.live_instructions({"instructions": "Speak Lao first."})
    assert merged.startswith(grok.GROK_LIVE_PERSONA)
    assert merged.endswith("Speak Lao first.")
