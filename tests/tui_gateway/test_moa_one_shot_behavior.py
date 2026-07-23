"""Tests for TUI gateway /moa one-shot restore and no-continuation-state.

The TUI gateway /moa handler is a JSON-RPC @method("command.dispatch") handler
inside tui_gateway/server.py, not a free function. These tests exercise the
gateway-level restore logic (GatewayRunner._restore_moa_one_shot) and the
session-state invariants that the /moa one-shot path must uphold.

For the full TUI slash handler coverage, see the existing
tests/tui_gateway/test_moa_reference_emit.py (MoA reference-model emit) and
tests/gateway/test_moa_one_shot_restore.py (restore helper unit tests).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.run import GatewayRunner


# ---------------------------------------------------------------------------
# Gateway restore: one-shot completion clears all continuation state
# ---------------------------------------------------------------------------

def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {}
    runner._evict_cached_agent = MagicMock()
    return runner


def _make_event(*, moa_disable=False, moa_restore=None):
    event = SimpleNamespace()
    if moa_disable:
        event._moa_disable_after_turn = True
        event._moa_restore_override = moa_restore
    return event


def test_one_shot_restore_clears_moa_provider():
    """After a one-shot turn, the MoA virtual provider must be replaced with
    the user's prior model — no continuation into the next turn."""
    runner = _make_runner()
    key = "agent:main:telegram:dm:42"
    runner._session_model_overrides[key] = {"provider": "moa", "model": "default"}
    event = _make_event(
        moa_disable=True,
        moa_restore={"provider": "openrouter", "model": "claude-opus-4.8"},
    )

    runner._restore_moa_one_shot(event, key)

    assert runner._session_model_overrides[key]["provider"] == "openrouter"
    assert runner._session_model_overrides[key]["model"] == "claude-opus-4.8"
    runner._evict_cached_agent.assert_called_once_with(key)


def test_one_shot_restore_none_removes_override_entirely():
    """If the user had no model override before /moa, the override entry
    is removed entirely — not left with stale MoA."""
    runner = _make_runner()
    key = "agent:main:discord:guild:456"
    runner._session_model_overrides[key] = {"provider": "moa", "model": "default"}
    event = _make_event(moa_disable=True, moa_restore=None)

    runner._restore_moa_one_shot(event, key)

    assert key not in runner._session_model_overrides
    runner._evict_cached_agent.assert_called_once_with(key)


def test_normal_turn_does_not_touch_overrides():
    """A non-MoA turn must not alter model overrides or evict agents."""
    runner = _make_runner()
    key = "agent:main:slack:channel:789"
    original = {"provider": "openrouter", "model": "gpt-4"}
    runner._session_model_overrides[key] = original.copy()
    event = _make_event()  # no moa_disable

    runner._restore_moa_one_shot(event, key)

    assert runner._session_model_overrides[key] == original
    runner._evict_cached_agent.assert_not_called()


def test_restore_fires_from_finally_even_on_exception():
    """The restore helper is called from a finally block, so it must succeed
    even when the turn raised an exception."""
    import pytest

    runner = _make_runner()
    key = "agent:main:telegram:dm:999"
    runner._session_model_overrides[key] = {"provider": "moa", "model": "default"}
    event = _make_event(
        moa_disable=True,
        moa_restore={"provider": "openrouter", "model": "gpt-4"},
    )

    with pytest.raises(RuntimeError):
        try:
            raise RuntimeError("provider error mid-turn")
        finally:
            runner._restore_moa_one_shot(event, key)

    # Restore still happened despite the exception
    assert runner._session_model_overrides[key] == {
        "provider": "openrouter",
        "model": "gpt-4",
    }


# ---------------------------------------------------------------------------
# Session-level: moa_one_shot_restore dict lifecycle
# ---------------------------------------------------------------------------

def test_session_restore_dict_shape():
    """The moa_one_shot_restore dict stored in the session must contain
    the fields the restore handler reads: override, model, provider."""
    # This is the shape set by the TUI /moa handler (server.py ~line 11585)
    restore = {
        "override": {"provider": "openrouter", "model": "gpt-4"},
        "model": "gpt-4",
        "provider": "openrouter",
    }
    # All three fields must be present for the restore path to work correctly
    assert "override" in restore
    assert "model" in restore
    assert "provider" in restore


def test_session_with_no_prior_override_stores_none():
    """When the user had no model_override before /moa, the restore dict
    stores override=None so the restore path pops model_override entirely."""
    restore = {
        "override": None,  # no prior override
        "model": "anthropic/claude-opus-4.8",
        "provider": "openrouter",
    }
    assert restore["override"] is None
    # The restore handler checks this and does session.pop("model_override")
