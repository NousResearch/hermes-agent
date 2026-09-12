"""LIVE E2E (#103498): ``session.resume`` must rebuild with the STORED runtime.

Report shape: a chat stored as ``gemini-flash`` + thinking off is resumed
while the profile defaults are ``vision-exp`` + high. Pre-fix, the
recreated agent silently adopted the profile defaults:

* cold/deferred resume of a row WITHOUT a provider pin (default-routing
  shape) kept the model but rebuilt thinking from the profile (off→high);
* lazy resume dropped the whole stored runtime (model AND thinking).

Both go through the REAL ``session.resume`` dispatch with a REAL
``state.db`` row; the cold path additionally runs the REAL ``_make_agent``
→ ``AIAgent`` construction (offline-safe: the default route is a fake
local custom provider, same trick as test_stale_provider_resume_live.py).

Run:  scripts/run_tests.sh tests/tui_gateway/test_resume_restores_stored_runtime.py
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pytest
import yaml

STORED_MODEL = "gemini-flash"
THINKING_OFF = {"enabled": False}


@pytest.fixture()
def live_home(monkeypatch):
    """Isolated HERMES_HOME whose profile defaults differ from the stored row."""
    tmp = Path(tempfile.mkdtemp(prefix="hermes-live-103498-"))
    home = tmp / ".hermes"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({
            "model": {"default": "vision-exp", "provider": "custom:def"},
            "agent": {"reasoning_effort": "high"},
            "custom_providers": [
                {
                    "name": "def",
                    "base_url": "https://default.invalid/v1",
                    "api_key": "fake-key-for-construction-only",
                    "api_mode": "chat_completions",
                }
            ],
        })
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home_token = set_hermes_home_override(str(home))
    import os

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

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    monkeypatch.setattr(server, "_db", None, raising=False)
    monkeypatch.setattr(server, "_db_error", None, raising=False)
    monkeypatch.setattr(server, "_hermes_home", str(home), raising=False)
    yield home, server
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


def _seed_row(home: Path, *, model_config: dict) -> str:
    from hermes_state import SessionDB

    sid = "s-" + uuid.uuid4().hex[:10]
    db = SessionDB(db_path=home / "state.db")
    db.create_session(
        sid, source="tui", model=STORED_MODEL, model_config=dict(model_config)
    )
    db.set_session_title(sid, f"stored-runtime {sid}")
    db.close()
    return sid


def _close(server, resp) -> None:
    live_sid = (resp.get("result") or {}).get("session_id")
    if live_sid:
        server.handle_request({
            "id": "close",
            "method": "session.close",
            "params": {"session_id": live_sid},
        })


class TestResumeRestoresStoredRuntime:
    def test_cold_resume_rebuilds_thinking_off_without_provider_pin(
        self,
        live_home,
    ):
        """Default-routing row (no provider pin): the rebuilt agent keeps
        the stored model AND the stored thinking-off, not the profile high."""
        home, server = live_home
        sid = _seed_row(
            home,
            model_config={
                "model": STORED_MODEL,
                "reasoning_config": dict(THINKING_OFF),
            },
        )
        resp = server.handle_request({
            "id": "rid",
            "method": "session.resume",
            "params": {"session_id": sid, "omit_messages": True},
        })
        try:
            assert "error" not in resp, f"resume failed live: {resp.get('error')}"
            record = server._sessions[resp["result"]["session_id"]]
            agent = server._make_agent(
                resp["result"]["session_id"],
                sid,
                **server._deferred_build_agent_kwargs(record, None),
            )
            assert agent.model == STORED_MODEL
            assert agent.reasoning_config == THINKING_OFF
        finally:
            _close(server, resp)

    def test_lazy_resume_carries_stored_runtime(self, live_home):
        """A lazy (watch) resume pins the stored runtime on the record so
        the deferred first-prompt build restores model + thinking."""
        home, server = live_home
        sid = _seed_row(
            home,
            model_config={
                "model": STORED_MODEL,
                "reasoning_config": dict(THINKING_OFF),
            },
        )
        resp = server.handle_request({
            "id": "rid",
            "method": "session.resume",
            "params": {"session_id": sid, "lazy": True, "omit_messages": True},
        })
        try:
            assert "error" not in resp, f"resume failed live: {resp.get('error')}"
            record = server._sessions[resp["result"]["session_id"]]
            overrides = record.get("resume_runtime_overrides") or {}
            assert (record.get("model_override") or {}).get("model") == STORED_MODEL
            assert overrides.get("reasoning_config_override") == THINKING_OFF
            kw = server._deferred_build_agent_kwargs(record, None)
            assert (kw.get("model_override") or {}).get("model") == STORED_MODEL
            assert kw.get("reasoning_config_override") == THINKING_OFF
        finally:
            _close(server, resp)
