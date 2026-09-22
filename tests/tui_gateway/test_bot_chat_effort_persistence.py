"""A Bot Chat effort pick must land in the bot's own profile config.

The desktop effort picker sends ``config.set {key: "reasoning"}`` with no
scope, so ``_set_reasoning`` takes its session-scoped branch and only pins the
live session. A Bot Chat rebuilds model and effort from its member profile's
``config.yaml`` on every resume, so the pick silently snapped back to the
configured value on the next backend rebuild (app restart, idle recycle).

Fix: when the session is a Bot Chat, the pick is ALSO written as
``agent.reasoning_effort`` into that bot's profile config, using the same
``_session_profile_runtime_scope`` write ``_set_model`` already performs for
model picks. Plain chats keep their existing session-scoped behaviour.

Detection reuses the gateway's own identity signal: the persisted
``follow_profile_config`` marker, with the bare ``"Bot Chat"`` title compare as
the legacy fallback ``_row_follows_profile`` also keeps for rows written
before the marker existed.
"""

import contextlib
from unittest import mock

import tui_gateway.server as server  # noqa: F401 — binds the gateway's globals onto the method modules


def _params(value="high", scope=""):
    return {"key": "reasoning", "value": value, "scope": scope, "session_id": ""}


def _pick(session, value="high", scope=""):
    with mock.patch.object(server, "_session_profile_runtime_scope",
                           side_effect=lambda s: contextlib.nullcontext()) as scope_cm, \
         mock.patch.object(server, "_write_config_key") as write:
        result = server._set_reasoning("r1", _params(value, scope), "reasoning", value, session)
    return result, write, scope_cm


def test_bot_chat_marker_persists_pick_to_profile_config():
    """The persisted marker is the identity — a marked row writes the profile config."""
    session = {"follow_profile_config": True, "agent": None}
    _result, write, scope_cm = _pick(session)
    write.assert_called_once_with("agent.reasoning_effort", "high")
    scope_cm.assert_called_once_with(session)
    assert session["create_reasoning_override"] is not None


def test_bot_chat_legacy_title_persists_pick_to_profile_config():
    """Rows written before the marker existed are still recognised by title."""
    session = {"title": "Bot Chat", "agent": None}
    _result, write, scope_cm = _pick(session)
    write.assert_called_once_with("agent.reasoning_effort", "high")
    scope_cm.assert_called_once_with(session)


def test_plain_chat_stays_session_scoped():
    """A normal chat must never rewrite a profile config from a menu pick."""
    session = {"agent": None}
    _result, write, scope_cm = _pick(session)
    write.assert_not_called()
    scope_cm.assert_not_called()
    assert session["create_reasoning_override"] is not None


def test_global_scope_writes_the_global_key_but_not_a_profile():
    """``scope: global`` is the existing global path and stays untouched."""
    session = {"follow_profile_config": True, "agent": None}
    _result, write, scope_cm = _pick(session, scope="global")
    write.assert_called_once_with("agent.reasoning_effort", "high")
    scope_cm.assert_not_called()


def test_unknown_value_is_rejected_without_writing_anything():
    session = {"follow_profile_config": True, "agent": None}
    _result, write, scope_cm = _pick(session, value="bogus")
    write.assert_not_called()
    scope_cm.assert_not_called()
    assert "create_reasoning_override" not in session
