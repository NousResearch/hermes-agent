"""A chat whose agent is still building reports ITS OWN model, not the profile default.

A cold ``session.resume`` restores the stored row's model into the live record
(``model_override``) and builds the agent off the response path. While that build
is in flight (``status: starting``) every other attach of the same chat answers
through ``_live_session_payload`` → ``_fallback_session_info``: ``session.activate``
(switching back to the chat) and a second ``session.resume`` that reuses the live
record (another client opening it). Those answers reported the profile default, so
the Desktop model picker showed the default for the length of the build (seconds)
on a chat that runs another model, until the build's ``session.info`` corrected it.

Contract: during the build, both answers carry the stored chat's model and
provider; a chat with no pick of its own still gets the profile default.
"""

from __future__ import annotations

import threading

import pytest

from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_messages, seed_session
from tui_gateway import server

PROFILE_DEFAULT = ("profile-default-model", "")


@pytest.fixture
def building(tmp_path, monkeypatch):
    """Cold-resume a stored chat with the real handlers and hold its agent build in flight."""
    db = open_persistence(profile_scope(home=tmp_path), eager=True)
    entered, release = threading.Event(), threading.Event()
    sids: list[str] = []

    def blocked_make_agent(*_args, **_kwargs):
        entered.set()
        release.wait(10)
        raise RuntimeError("test released the build")

    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_session_default_route", lambda _session: PROFILE_DEFAULT)
    monkeypatch.setattr(server, "_make_agent", blocked_make_agent)
    monkeypatch.setattr(server, "_default_session_cwd", lambda: str(tmp_path))
    monkeypatch.setattr(server, "_emit", lambda *_args: None)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *_args: None)

    def resume_while_building(model: str | None = None, provider: str | None = None) -> str:
        """Seed the stored chat (no model = it follows the profile), cold-resume it, return the live sid."""
        pick = {"model": model, "model_config": {"model": model, "provider": provider}} if model else {}
        seed_session(db, "stored-chat", "desktop", title="Picker test", **pick)
        seed_messages(db, "stored-chat", {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"})
        response = server.handle_request({"id": "cold", "method": "session.resume", "params": {"session_id": "stored-chat"}})
        assert "error" not in response, response
        sids.append(sid := response["result"]["session_id"])
        assert entered.wait(5), "the agent build never started"
        return sid

    yield resume_while_building
    release.set()
    for sid in sids:
        if (record := server._sessions.get(sid)) is not None:
            assert record["agent_ready"].wait(5)
        server._sessions.pop(sid, None)
    db.close()


def _activate(sid: str) -> dict:
    """Switching back to the chat."""
    return server.handle_request({"id": "activate", "method": "session.activate",
                                  "params": {"session_id": sid, "omit_messages": True}})["result"]


def _resume_again(_sid: str) -> dict:
    """Another client opening the chat: the stored id resolves to the live record."""
    return server.handle_request({"id": "reuse", "method": "session.resume",
                                  "params": {"session_id": "stored-chat", "omit_messages": True}})["result"]


@pytest.mark.parametrize("attach", [_activate, _resume_again], ids=["session.activate", "reuse-live session.resume"])
def test_attach_during_build_reports_the_chats_own_model(building, attach):
    sid = building("claude-haiku-5-5", "anthropic")

    result = attach(sid)

    assert result["session_id"] == sid
    assert result["status"] == "starting"
    assert (result["info"]["model"], result["info"].get("provider")) == ("claude-haiku-5-5", "anthropic")


def test_attach_during_build_of_a_profile_following_chat_reports_the_profile_default(building):
    sid = building()

    result = _activate(sid)

    assert result["status"] == "starting"
    assert result["info"]["model"] == PROFILE_DEFAULT[0]
    assert "provider" not in result["info"]
