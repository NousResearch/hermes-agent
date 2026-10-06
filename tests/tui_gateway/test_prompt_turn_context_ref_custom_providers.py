"""Regression test: the ``@`` context-reference expansion path in the TUI gateway's turn
preparation must thread the agent's ``custom_providers`` into context-length resolution.

Mirrors ``agent.context_compressor.ContextCompressor._resolve_context_length``'s fix
(71516214c3, "thread custom_providers into context-length resolution", see
``tests/agent/test_context_compressor_custom_providers.py``) — a per-model
``context_length`` override in ``custom_providers`` was invisible to ``@file``/``@url``
injection-limit sizing because ``_prepare_turn_input`` called ``get_model_context_length()``
without it, so a custom-provider model's real (often larger) window was silently replaced
by the hardcoded catalog default, needlessly truncating or refusing large ``@`` references.
"""

import threading
from types import SimpleNamespace
from unittest.mock import patch

from tui_gateway import prompt_turn
from tui_gateway.method_ctx import rebind


def _noop(*_args, **_kwargs):
    return None


def test_context_reference_expansion_threads_agent_custom_providers():
    custom_providers = [{"base_url": "https://custom.example.com/v1", "models": {"big-model": {"context_length": 999999}}}]
    agent = SimpleNamespace(
        model="big-model", base_url="https://custom.example.com/v1", api_key="", provider="custom",
        _config_context_length=None, _custom_providers=custom_providers,
    )
    session = {
        "session_key": "s1", "agent": agent, "profile_home": None,
        "history_lock": threading.RLock(), "history": [], "history_version": 0, "cols": 80,
    }
    st = prompt_turn._TurnRun(agent=agent, one_turn_restore=True, terminal_callback=None, receipt_committed=False)

    captured = {}

    def fake_get_model_context_length(*_args, **kwargs):
        captured.update(kwargs)
        return 999999

    prepare = rebind(prompt_turn._prepare_turn_input, {
        "_set_session_context": lambda *a, **k: "token",
        "_wire_callbacks": _noop,
        "_sync_bot_capabilities": _noop,
        "_session_cwd": lambda session: "/tmp",
        "_register_session_cwd": _noop,
        "make_stream_renderer": lambda cols: "streamer",
        "_start_turn_voice": lambda: (None, False),
        "_prepend_note": lambda msg, note: msg,
        "_pending_reaction_notes": lambda session: "",
        "_hud_surface_note": lambda session: "",
    })

    with (
        patch("agent.model_metadata.get_model_context_length", side_effect=fake_get_model_context_length),
        patch(
            "agent.context_references.preprocess_context_references",
            return_value=SimpleNamespace(blocked=False, message="expanded prompt", warnings=[]),
        ),
    ):
        result = prepare("sid1", session, st, "@notes.txt please summarize", [])

    assert result is not None
    assert captured.get("custom_providers") == custom_providers


def test_context_reference_expansion_defaults_custom_providers_to_none():
    """An agent that never got ``_custom_providers`` set (e.g. a stub in another test) must
    not raise — the call defaults to ``None``, same as every other caller of this resolver."""
    agent = SimpleNamespace(
        model="m", base_url="", api_key="", provider="", _config_context_length=None,
    )
    session = {
        "session_key": "s1", "agent": agent, "profile_home": None,
        "history_lock": threading.RLock(), "history": [], "history_version": 0, "cols": 80,
    }
    st = prompt_turn._TurnRun(agent=agent, one_turn_restore=True, terminal_callback=None, receipt_committed=False)

    captured = {}

    def fake_get_model_context_length(*_args, **kwargs):
        captured.update(kwargs)
        return 128000

    prepare = rebind(prompt_turn._prepare_turn_input, {
        "_set_session_context": lambda *a, **k: "token",
        "_wire_callbacks": _noop,
        "_sync_bot_capabilities": _noop,
        "_session_cwd": lambda session: "/tmp",
        "_register_session_cwd": _noop,
        "make_stream_renderer": lambda cols: "streamer",
        "_start_turn_voice": lambda: (None, False),
        "_prepend_note": lambda msg, note: msg,
        "_pending_reaction_notes": lambda session: "",
        "_hud_surface_note": lambda session: "",
    })

    with (
        patch("agent.model_metadata.get_model_context_length", side_effect=fake_get_model_context_length),
        patch(
            "agent.context_references.preprocess_context_references",
            return_value=SimpleNamespace(blocked=False, message="expanded prompt", warnings=[]),
        ),
    ):
        result = prepare("sid1", session, st, "@notes.txt please summarize", [])

    assert result is not None
    assert captured.get("custom_providers") is None
