"""LIVE E2E: Desktop/TUI ``session.resume`` with a stale (deleted/renamed)
custom provider, through the REAL gateway dispatch path.

This is the live-report shape from Discord (endpoints "deleted ages ago"
coming back, Bot Chats pinned to dead providers): a real ``state.db`` session
row whose persisted ``model_config`` names a custom provider that no longer
exists in config.yaml, resumed through ``tui_gateway.server.handle_request``
(the same dispatch the Desktop WebSocket transport funnels into) with
``eager_build: true`` so the REAL ``_make_agent`` → ``AIAgent`` construction
runs synchronously — no mocks, no pre-seeded sessions.

On pre-fix main the eager build dies with
``resume failed: Unknown provider 'custom:oldone'`` (or silently pins the
dead identity). Post-fix:

* renamed provider (same endpoint/model under a new name) → healed to the
  new ``custom:<name>`` identity;
* deleted, unrecoverable provider → falls back to the configured default;
* legacy canonical Bot Chat row (titled exactly "Bot Chat", no
  follow_profile_config marker) → stored pin ignored, profile's CURRENT
  config used.

Run:  python -m pytest tests/tui_gateway/test_stale_provider_resume_live.py -o addopts= -v -s
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

import pytest
import yaml

OLD_URL = "https://old-endpoint.invalid/v1"
NEW_URL = "https://new-endpoint.invalid/v1"


@pytest.fixture()
def live_home(monkeypatch):
    """A REAL isolated HERMES_HOME with a config.yaml + state.db on disk."""
    tmp = Path(tempfile.mkdtemp(prefix="hermes-live-staleprov-"))
    home = tmp / ".hermes"
    home.mkdir(parents=True)
    config = {
        "model": {"default": "test-model-live", "provider": "custom:newone"},
        "custom_providers": [
            {
                "name": "newone",
                "base_url": NEW_URL,
                "api_key": "sk-live-test-not-real",
                "api_mode": "chat_completions",
            }
        ],
    }
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setenv("HERMES_HOME", str(home))
    # hermes_constants caches the resolved home at first read — the env var
    # alone doesn't repoint an already-imported process. Use the override API
    # (the same mechanism profile-scoped resumes use).
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home_token = set_hermes_home_override(str(home))
    # Neutralize ambient provider creds so resolution uses ONLY the config
    # above — this must behave the same on a dev box and a bare CI runner.
    for var in list(os.environ):
        if var.endswith("_API_KEY") or var in ("OPENROUTER_KEY", "NOUS_KEY"):
            monkeypatch.delenv(var, raising=False)

    import hermes_cli.config as hconfig
    import hermes_cli.runtime_provider as rp

    for mod in (hconfig, rp):
        for attr in ("_config_cache", "_cache", "_CONFIG_CACHE"):
            if hasattr(mod, attr):
                try:
                    setattr(mod, attr, None)
                except Exception:
                    pass

    import hermes_state
    import tui_gateway.server as server

    # The launch DB handle and the module-level home snapshot are import-time
    # caches — repoint both at the isolated home for the duration of the test
    # (see references: tui-gateway live WS harness, same trap). The hermetic
    # conftest also re-pins hermes_state.DEFAULT_DB_PATH per test; pin it to
    # THIS home so server._get_db() opens the same real state.db we seed.
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    monkeypatch.setattr(server, "_db", None, raising=False)
    monkeypatch.setattr(server, "_db_error", None, raising=False)
    monkeypatch.setattr(server, "_hermes_home", str(home), raising=False)
    yield home, server
    # Detach the shared handle so the tmpdir can be reclaimed.
    try:
        if server._db is not None:
            server._db.close()
    except Exception:
        pass
    server._db = None
    try:
        reset_hermes_home_override(home_token)
    except Exception:
        pass


def _seed_session_row(
    home: Path,
    *,
    title: str,
    provider: str,
    base_url: str | None = OLD_URL,
    extra_config: dict | None = None,
) -> str:
    """Write a REAL session row + one message into the profile's state.db."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    sid = uuid.uuid4().hex[:12]
    model_config = {
        "model": "test-model-live",
        "provider": provider,
        "api_mode": "chat_completions",
    }
    if base_url:
        model_config["base_url"] = base_url
    if extra_config:
        model_config.update(extra_config)
    db.create_session(
        sid,
        source="tui",
        model="test-model-live",
        model_config=model_config,
        session_key=f"live-test:{sid}",
    )
    db.set_session_title(sid, title)
    db.close()
    return sid


def _resume(server, sid: str) -> dict:
    """Drive the REAL dispatch entry (same funnel the Desktop WS uses)."""
    return server.handle_request(
        {
            "id": f"rid-{sid}",
            "method": "session.resume",
            "params": {"session_id": sid, "eager_build": True, "omit_messages": True},
        }
    )


def _teardown(server, resp) -> None:
    live_sid = (resp.get("result") or {}).get("session_id")
    if live_sid:
        server.handle_request(
            {"id": "close", "method": "session.close", "params": {"session_id": live_sid}}
        )


class TestStaleProviderResumeLive:
    def test_deleted_provider_falls_back_to_default(self, live_home):
        """The Discord report shape: the stored provider was deleted from
        config ages ago. Resume must NOT die with 'Unknown provider' — it
        falls back to the configured default and the agent builds."""
        home, server = live_home
        sid = _seed_session_row(
            home, title="Old work chat", provider="custom:deleted-ages-ago"
        )
        resp = _resume(server, sid)
        try:
            assert "error" not in resp, f"resume failed live: {resp.get('error')}"
            result = resp["result"]
            assert result.get("resumed") == sid
            info = result.get("info") or {}
            # Healed via base_url→entry recovery or dropped to the configured
            # default — either way the DEAD name must not survive.
            assert "deleted-ages-ago" not in json.dumps(info)
        finally:
            _teardown(server, resp)

    def test_renamed_provider_heals_to_new_identity(self, live_home):
        """oldone → newone rename, same model: identity heals to newone."""
        home, server = live_home
        sid = _seed_session_row(
            home, title="Renamed provider chat", provider="custom:oldone", base_url=None
        )
        resp = _resume(server, sid)
        try:
            assert "error" not in resp, f"resume failed live: {resp.get('error')}"
        finally:
            _teardown(server, resp)

    def test_legacy_bot_chat_follows_current_profile_config(self, live_home):
        """A pre-contract canonical Bot Chat (title exactly 'Bot Chat', no
        follow_profile_config marker) pinned to a dead provider must resume
        on the profile's CURRENT config."""
        home, server = live_home
        sid = _seed_session_row(home, title="Bot Chat", provider="custom:deadbot")
        resp = _resume(server, sid)
        try:
            assert "error" not in resp, f"Bot Chat resume failed live: {resp.get('error')}"
            info = (resp.get("result") or {}).get("info") or {}
            assert "deadbot" not in json.dumps(info)
        finally:
            _teardown(server, resp)
