"""``session.resume`` resolves a stored session's provider overrides under the SESSION profile's scope.

Under a multiplexed gateway launched as profile A, resuming profile B's stored session used to
evaluate B's persisted ``custom:<name>`` provider against A's config: the deferred (Desktop
``defer_history``) and cold paths called ``_stored_session_runtime_overrides`` outside
``_profile_build_scope(ctx.profile_home)``, so a provider defined only in B was "unroutable",
got healed to A's entry for the same endpoint, and the build in B died with
``Unknown provider 'custom:local-code'`` (#115607). Two real homes, A→B: B's ``providers.*``
entry must survive resume; a launch-profile row keeps resolving against the launch config.
"""

from __future__ import annotations

import pytest

from hermes_state import SessionDB
from tui_gateway import server

_BASE_URL = "http://127.0.0.1:59999/v1"
_STORED = "20260919-000000-abcd"


class _NoopHydration:
    def __call__(self, *_a, **_k):
        return None


def _write_home(home, config_text: str) -> None:
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(config_text)
    (home / ".env").write_text("")


def _store(home, provider: str) -> None:
    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session(_STORED, "desktop", model="local-code",
                          model_config={"model": "local-code", "provider": provider, "base_url": _BASE_URL})
        db.append_message(_STORED, "user", "hi")
        db.append_message(_STORED, "assistant", "hello")
    finally:
        db.close()


@pytest.fixture()
def homes(monkeypatch, tmp_path):
    """Launch home A owns ``custom_providers: Local-Code``; secondary B owns ``providers.local-vllm`` — the
    same endpoint under a different name, so a cross-profile heal is observable."""
    launch, secondary = tmp_path / "a", tmp_path / "b"
    _write_home(launch, "model:\n  default: gpt-4o\n  provider: openai\n"
                        f"custom_providers:\n  - name: Local-Code\n    base_url: {_BASE_URL}\n    api_key: k-a\n")
    _write_home(secondary, "model:\n  default: local-code\n  provider: custom:local-vllm\n"
                           f"providers:\n  local-vllm:\n    api: {_BASE_URL}\n    api_key: k-b\n")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_hermes_home", str(launch))
    monkeypatch.setattr(server, "_profile_home", lambda p: secondary if p == "b" else None)
    monkeypatch.setattr(server, "_get_db", lambda: SessionDB(db_path=launch / "state.db"))
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_resume_hydration", _NoopHydration())
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda *a, **k: None)
    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *a, **k: None)
    monkeypatch.setattr(server, "_default_session_cwd", lambda *a, **k: str(tmp_path))
    known = set(server._sessions)
    yield {"a": launch, "b": secondary}
    with server._sessions_lock:
        for sid in [s for s in server._sessions if s not in known]:
            server._sessions.pop(sid, None)


def _resume(**params):
    resp = server.handle_request({"id": "1", "method": "session.resume",
                                  "params": {"session_id": _STORED, "source": "desktop", **params}})
    assert "error" not in resp, resp
    return server._sessions[resp["result"]["session_id"]]["resume_runtime_overrides"]


@pytest.mark.parametrize("path", [{"defer_history": True, "omit_messages": True}, {}], ids=["deferred", "cold"])
def test_secondary_profile_overrides_resolve_against_its_own_config(homes, path):
    _store(homes["b"], "custom:local-vllm")

    overrides = _resume(profile="b", **path)

    assert overrides["provider_override"] == "custom:local-vllm"
    assert overrides["model_override"]["provider"] == "custom:local-vllm"


def test_launch_profile_overrides_still_resolve_against_launch_config(homes):
    _store(homes["a"], "custom:local-code")

    overrides = _resume(defer_history=True, omit_messages=True)

    assert overrides["provider_override"] == "custom:local-code"
