"""``session.resume`` must rebuild the agent with the STORED session runtime (#103498).

During a disconnect -> reconnect window a stored session (e.g. ``gemini`` + thinking
off) could resume with the profile DEFAULT model/thinking (e.g. ``vision-exp`` + high)
while the persisted row stayed correct. The resumed runtime — not the row — was wrong.

Pinned here, end to end through the real ``session.resume`` RPC + the real deferred
agent build (``_start_agent_build``), with a fake ``_make_agent`` that honors the same
override/default rule as the real one (an explicit ``reasoning_config_override`` wins,
else the ambient default). Two holes, one contract:

* a lazy resume upgraded by the first prompt rebuilt a pure profile-default agent
  (the lazy record carried no overrides at all);
* a stored provider that is no longer routable dropped the WHOLE overrides dict, so
  the stored model/reasoning/tier silently reverted (only the dead provider pin may
  fall back).

The ``follow_profile_config`` case keeps following the profile (the #101514 direction
is explicitly NOT changed by this fix).
"""

from __future__ import annotations

import types

import pytest

from tui_gateway import server

STORED_MODEL = "gemini"
STORED_REASONING_OFF = {"enabled": False}
PROFILE_DEFAULT_MODEL = "vision-exp"

_HISTORY = [{"role": "user", "content": "hello"}]


class _FakeDB:
    def __init__(self, row):
        self._row = row

    def get_session(self, _target):
        return dict(self._row)

    def get_session_by_title(self, _title):
        return None

    def reopen_session(self, _target):
        pass

    def get_messages_as_conversation(self, _target, **_kwargs):
        return [dict(m) for m in _HISTORY]

    def get_resume_conversations(self, _target):
        return self.get_messages_as_conversation(_target), self.get_messages_as_conversation(_target)

    def get_ancestor_display_prefix(self, _sid):
        return []


def _stored_row(**model_config):
    return {
        "id": "stored-1",
        "title": "test chat",
        "model": STORED_MODEL,
        "billing_provider": "gemini",
        "model_config": {"provider": "gemini", "reasoning_config": dict(STORED_REASONING_OFF), **model_config},
        "message_count": 1,
    }


def _install(monkeypatch, row, *, routable):
    """Fake the agent seam + ambient profile defaults."""
    db = _FakeDB(row)
    events = []

    def fake_make_agent(_sid, _key, **kwargs):
        override = kwargs.get("model_override") or {}
        # Same rule as the real _make_agent: an explicit reasoning override wins,
        # otherwise the agent falls back to the ambient (profile default) config.
        return types.SimpleNamespace(
            model=override.get("model") or PROFILE_DEFAULT_MODEL,
            provider=(override.get("provider") or "vision-provider"),
            reasoning_config=kwargs.get("reasoning_config_override", {"enabled": True, "effort": "high"}),
        )

    monkeypatch.setattr(server, "_profile_session_db", lambda _home: (db, False))
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_set_session_context", lambda *a: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda tokens: None)
    monkeypatch.setattr(server, "_make_agent", fake_make_agent)
    monkeypatch.setattr(server, "_wire_session_agent", lambda *a: True)
    monkeypatch.setattr(server, "_announce_built_agent", lambda *a: None)
    monkeypatch.setattr(server, "_emit", lambda *a: events.append(a))
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_resume_hydration", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *a: None)
    monkeypatch.setattr(server, "_is_routable_provider", lambda _p: routable)
    # Ambient profile defaults DIFFER from the stored runtime: any leak reads as the default.
    monkeypatch.setattr(server, "_config_model_target", lambda: (PROFILE_DEFAULT_MODEL, None))


def _resume_and_build(monkeypatch, params, row, *, routable=True):
    _install(monkeypatch, row, routable=routable)
    before = set(server._sessions)
    resp = server.handle_request({"id": "1", "method": "session.resume", "params": params})
    assert resp is not None and "result" in resp, resp
    sid = resp["result"]["session_id"]
    session = server._sessions[sid]
    try:
        if session.get("resume_history_ready") is not None:
            session["resume_history_ready"].set()  # transcript hydration is stubbed out
        server._start_agent_build(sid, session)
        session["_agent_build_thread"].join(timeout=15)
        assert session.get("agent") is not None, session.get("agent_error")
        return session["agent"]
    finally:
        for live_sid in set(server._sessions) - before:
            server._sessions.pop(live_sid, None)


@pytest.mark.parametrize(
    ("params", "routable", "row_kwargs"),
    [
        ({"session_id": "stored-1"}, True, {}),
        ({"session_id": "stored-1", "defer_history": True}, True, {}),
        ({"session_id": "stored-1", "lazy": True}, True, {}),
        ({"session_id": "stored-1"}, False, {"provider": "removed-provider", "billing_provider": "removed-provider"}),
    ],
    ids=["cold", "deferred", "lazy-upgrade", "unroutable-provider"],
)
def test_resume_restores_stored_model_and_thinking(monkeypatch, params, routable, row_kwargs):
    agent = _resume_and_build(monkeypatch, params, _stored_row(**row_kwargs), routable=routable)
    assert agent.model == STORED_MODEL
    assert agent.reasoning_config == STORED_REASONING_OFF


def test_follow_profile_config_session_still_follows_profile(monkeypatch):
    agent = _resume_and_build(
        monkeypatch, {"session_id": "stored-1"}, _stored_row(follow_profile_config=True), routable=True
    )
    assert agent.model == PROFILE_DEFAULT_MODEL
