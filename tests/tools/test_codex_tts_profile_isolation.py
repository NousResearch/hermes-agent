"""A live-voice credential keeps its issuing profile through OAuth refresh."""

import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from hermes_constants import (
    get_hermes_home,
    get_hermes_home_override,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tools import tts_streaming, tts_tool_codex


def _token(account):
    claims = {"exp": 4_102_444_800, "https://api.openai.com/auth": {"chatgpt_account_id": account}}
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"e30.{payload}.signature"


def _profile(tmp_path, name):
    home = tmp_path / ".hermes" / "profiles" / name
    home.mkdir(parents=True)
    (home / "auth.json").write_text(json.dumps({
        "version": 1,
        "active_provider": "openai-codex",
        "providers": {"openai-codex": {"tokens": {
            "access_token": _token(name), "refresh_token": f"refresh-{name}",
        }}},
        "credential_pool": {"openai-codex": [{
            "id": name, "source": "device_code", "auth_type": "oauth", "priority": 0,
            "access_token": _token(name), "refresh_token": f"refresh-{name}",
        }]},
    }))
    return home


def _streamer(home):
    scope = set_hermes_home_override(home)
    try:
        return tts_streaming.OpenAICodexStreamer({"provider": "openai-codex"}, {})
    finally:
        reset_hermes_home_override(scope)


@pytest.mark.parametrize("ambient_scope", [False, True])
@pytest.mark.parametrize("refresh_fails", [False, True])
def test_live_streamer_refresh_stays_with_issuing_profile(
    monkeypatch, tmp_path, ambient_scope, refresh_fails
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    owner = _profile(tmp_path, "owner")
    ambient = _profile(tmp_path, "ambient")
    monkeypatch.setenv("HERMES_HOME", str(owner))
    streamer = _streamer(owner)
    ambient_before = (ambient / "auth.json").read_bytes()
    calls, refresh_scopes = [], []

    def refresh(access_token, refresh_token):
        refresh_scopes.append((get_hermes_home(), access_token, refresh_token))
        if refresh_fails:
            raise RuntimeError("refresh unavailable")
        return {"access_token": _token("owner-fresh"), "refresh_token": "refresh-owner-fresh"}

    def synthesize(_text, token, **kwargs):
        calls.append((token, kwargs.get("account_id")))
        if token != _token("owner-fresh"):
            raise RuntimeError("ChatGPT synthesis failed (HTTP 401)")
        # Stop before decoding: this regression exercises real profile/pool I/O,
        # and cancellation after network completion must still retain its refresh.
        streamer.cancel()
        return SimpleNamespace(audio=b"ID3speech")

    monkeypatch.setattr("hermes_cli.auth.refresh_codex_oauth_pure", refresh)
    monkeypatch.setattr(tts_tool_codex, "synthesize_codex_speech", synthesize)
    monkeypatch.setenv("HERMES_HOME", str(ambient))
    scope = set_hermes_home_override(ambient if ambient_scope else None)
    try:
        if refresh_fails:
            with pytest.raises(RuntimeError, match="HTTP 401"):
                list(streamer.stream("Stay with my profile."))
        else:
            assert list(streamer.stream("Stay with my profile.")) == []
        assert get_hermes_home() == ambient
        assert get_hermes_home_override() == (str(ambient) if ambient_scope else None)
    finally:
        reset_hermes_home_override(scope)

    assert os.environ["HERMES_HOME"] == str(ambient)
    assert (ambient / "auth.json").read_bytes() == ambient_before
    assert refresh_scopes == [(owner, _token("owner"), "refresh-owner")]
    if not refresh_fails:
        store = json.loads((owner / "auth.json").read_text())
        assert store["providers"]["openai-codex"]["tokens"] == {
            "access_token": _token("owner-fresh"), "refresh_token": "refresh-owner-fresh",
        }
        assert store["credential_pool"]["openai-codex"][0]["access_token"] == _token("owner-fresh")
        assert streamer.credentials["account_id"] == "owner-fresh"
        assert calls == [(_token("owner"), "owner"), (_token("owner-fresh"), "owner-fresh")]


def test_simultaneous_voice_sessions_refresh_without_global_profile_swap(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    homes = {name: _profile(tmp_path, name) for name in ("alpha", "beta")}
    ambient = _profile(tmp_path, "ambient")
    monkeypatch.setenv("HERMES_HOME", str(ambient))
    streamers = {name: _streamer(home) for name, home in homes.items()}
    ambient_before = (ambient / "auth.json").read_bytes()
    refresh_overlap = Barrier(2, timeout=10)

    def refresh(access_token, refresh_token):
        name = refresh_token.removeprefix("refresh-")
        assert access_token == _token(name)
        assert get_hermes_home() == homes[name]
        assert os.environ["HERMES_HOME"] == str(ambient)
        # Both network calls must enter while the other is active. A global
        # environment swap/lock would serialize them or expose the wrong home.
        refresh_overlap.wait()
        return {"access_token": _token(f"{name}-fresh"), "refresh_token": f"refresh-{name}-fresh"}

    def synthesize(name, token, **kwargs):
        if token == _token(name):
            raise RuntimeError("ChatGPT synthesis failed (HTTP 401)")
        assert token == _token(f"{name}-fresh")
        assert kwargs["account_id"] == f"{name}-fresh"
        streamers[name].cancel()
        return SimpleNamespace(audio=b"ID3speech")

    def run(name):
        assert list(streamers[name].stream(name)) == []
        assert get_hermes_home_override() is None
        assert get_hermes_home() == ambient

    monkeypatch.setattr("hermes_cli.auth.refresh_codex_oauth_pure", refresh)
    monkeypatch.setattr(tts_tool_codex, "synthesize_codex_speech", synthesize)
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(run, name) for name in homes]
        for future in futures:
            future.result(timeout=15)

    assert (ambient / "auth.json").read_bytes() == ambient_before
    for name, home in homes.items():
        store = json.loads((home / "auth.json").read_text())
        assert store["providers"]["openai-codex"]["tokens"]["refresh_token"] == f"refresh-{name}-fresh"
        assert store["credential_pool"]["openai-codex"][0]["access_token"] == _token(f"{name}-fresh")
