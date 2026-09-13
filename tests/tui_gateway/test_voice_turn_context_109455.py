"""Structured voice_context for TUI/Desktop gateway turns (#109455).

Desktop's GPT-Live full-duplex mode already tags its turns with ``surface: "voice-live"``
(see tools/voice_live.py, tests/tui_gateway/test_voice_live_delegation.py); these tests cover
turning that existing signal into the same structured ``{input_modality, voice_session_active,
client_surface}`` shape ``pre_llm_call`` hooks/plugins get for CLI voice input, and threading it
into the actual turn via ``agent.run_conversation(voice_context=...)``.

``voice_context`` is threaded as an explicit per-request argument end-to-end (``prompt.submit``
-> ``_handle_busy_submit``/``_enqueue_prompt`` -> the queued-prompt dict -> ``_run_prompt_submit``
-> ``_invoke_agent``), exactly like the existing ``turn_author`` — NOT stashed on the shared
``session`` dict. An earlier version of this change stored it as ``session["voice_turn_context"]``
and read it back in ``_invoke_agent``; two independent reviewers caught that a session dict is
mutated on every submit, so a value stored there would misattribute to whichever turn next reads
it — a queued prompt still waiting behind a busy turn, or an internal follow-up (auto-continue,
goal continuation, a heartbeat/background notification) that calls ``_run_prompt_submit`` directly
and never goes through ``prompt.submit`` at all. The tests below pin the explicit-threading fix
and the state-bleed scenario it closes.
"""

from __future__ import annotations

import threading
import types

import pytest

from tui_gateway import server

# ``_invoke_agent``/``_TurnRun``/etc. reference server.py globals (``_emit``, ``sys``, ...) as bare
# names; they only resolve once ``method_ctx.bind_module`` rebinds them onto ``server`` at import
# time (see prompt_turn.py's module docstring), so tests must call the rebound ``server.<name>``,
# not the raw ``prompt_turn``/``session_auto_continue`` module attribute.
_TurnRun = server._TurnRun
_invoke_agent = server._invoke_agent
_enqueue_prompt = server._enqueue_prompt
_sanitize_queued_entry_vs_inflight_user = server._sanitize_queued_entry_vs_inflight_user


def _session(**extra):
    return {
        "agent": types.SimpleNamespace(valid_tool_names=set()),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": True,
        "transport": None,
        "attached_images": [],
        **extra,
    }


_VOICE_CONTEXT = {"input_modality": "voice", "voice_session_active": True, "client_surface": "voice-live"}


class TestPromptSubmitComputesVoiceContextWithoutMutatingSession:
    """``prompt.submit`` no longer stashes the signal on the session — it's a local, threaded
    straight into the turn invocation for THIS request only."""

    @pytest.fixture
    def busy_session(self):
        session = _session()
        server._sessions["sid"] = session
        yield session
        server._sessions.pop("sid", None)

    def test_live_surface_submit_leaves_no_trace_on_the_session(self, busy_session):
        server._methods["prompt.submit"](
            "r1", {"session_id": "sid", "text": "what's the weather", "queued": True, "surface": "voice-live"})

        assert "voice_turn_context" not in busy_session

    def test_hud_surface_submit_also_leaves_no_trace(self, busy_session):
        server._methods["prompt.submit"](
            "r1", {"session_id": "sid", "text": "x", "queued": True, "surface": "hud"})

        assert "voice_turn_context" not in busy_session


class _FakeAgentWithVoiceContext:
    """A ``run_conversation`` whose signature actually declares ``voice_context`` (unlike a bare
    ``MagicMock``, on which ``inspect.signature`` can't feature-detect the parameter) — mirrors
    the real ``agent.turn_facade.TurnFacadeMixin.run_conversation`` contract closely enough for
    ``_invoke_agent``'s ``"voice_context" in run_params`` guard to behave as it does in production.
    """

    def __init__(self):
        self.valid_tool_names = set()
        self.captured_kwargs: dict = {}

    def run_conversation(self, run_message, *, conversation_history=None, stream_callback=None,
                         persist_user_message=None, task_id=None, turn_author=None,
                         voice_context=None, **_ignored):
        self.captured_kwargs = dict(voice_context=voice_context, turn_author=turn_author)
        return {"final_response": "ok", "messages": [], "completed": True}


def _turn_run(agent):
    return _TurnRun(agent=agent, one_turn_restore=None, terminal_callback=None, receipt_committed=False)


class TestInvokeAgentThreadsVoiceContext:
    def test_voice_context_reaches_run_conversation(self):
        agent = _FakeAgentWithVoiceContext()
        session = _session()
        st = _turn_run(agent)

        _invoke_agent("sid", session, st, "what's the weather", "what's the weather", None, [], None, None,
                      None, _VOICE_CONTEXT)

        assert agent.captured_kwargs["voice_context"] == _VOICE_CONTEXT

    def test_typed_turn_omits_voice_context_kwarg(self):
        agent = _FakeAgentWithVoiceContext()
        session = _session()
        st = _turn_run(agent)

        _invoke_agent("sid", session, st, "hello", "hello", None, [], None, None)

        assert agent.captured_kwargs["voice_context"] is None

    def test_no_state_bleed_across_invocations_on_the_same_session(self):
        """Regression: a prior design stored the signal on ``session["voice_turn_context"])``,
        so a voice turn's flag survived on the shared dict and leaked into the NEXT turn on the
        same session — including an internal follow-up that never passed voice_context at all.
        Explicit-argument threading makes that structurally impossible: two calls sharing one
        session dict must not influence each other."""
        agent = _FakeAgentWithVoiceContext()
        session = _session()
        st = _turn_run(agent)

        _invoke_agent("sid", session, st, "what's the weather", "what's the weather", None, [], None, None,
                      None, _VOICE_CONTEXT)
        assert agent.captured_kwargs["voice_context"] == _VOICE_CONTEXT

        # A same-session follow-up (auto-continue, goal continuation, a heartbeat) that never
        # passes voice_context must not inherit the previous turn's flag.
        _invoke_agent("sid", session, st, "continuing...", "continuing...", None, [], None, None)
        assert agent.captured_kwargs["voice_context"] is None


class TestEnqueuePromptPreservesVoiceContext:
    """The queued-prompt envelope (mirrors ``turn_author``'s existing handling)."""

    def test_voice_tagged_prompt_is_queued_with_its_context(self):
        session = _session()

        _enqueue_prompt(session, "what's the weather", None, voice_context=_VOICE_CONTEXT)

        assert session["queued_prompt"]["voice_context"] == _VOICE_CONTEXT

    def test_voice_tagged_prompt_does_not_merge_with_a_plain_queued_slot(self):
        session = _session()
        _enqueue_prompt(session, "first", None)
        assert "queued_prompts" not in session  # plain text-only queue: still just one slot

        _enqueue_prompt(session, "second (voice)", None, voice_context=_VOICE_CONTEXT)

        # A voice-tagged entry must stay a separate envelope, not get glued onto the plain one —
        # merging would make the combined text's voice_context ambiguous.
        assert session["queued_prompt"]["text"] == "first"
        assert session["queued_prompts"][0]["text"] == "second (voice)"
        assert session["queued_prompts"][0]["voice_context"] == _VOICE_CONTEXT

    def test_plain_prompt_after_a_voice_tagged_one_does_not_merge_into_it(self):
        session = _session()
        _enqueue_prompt(session, "first (voice)", None, voice_context=_VOICE_CONTEXT)

        _enqueue_prompt(session, "second", None)

        assert session["queued_prompt"]["text"] == "first (voice)"
        assert session["queued_prompts"][0]["text"] == "second"
        assert "voice_context" not in session["queued_prompts"][0]


class TestSanitizeQueuedEntryKeepsVoiceTaggedEnvelopes:
    def test_voice_tagged_self_duplicate_text_is_not_dropped(self):
        entry = {"text": "what's the weather", "voice_context": _VOICE_CONTEXT}

        result = _sanitize_queued_entry_vs_inflight_user(entry, "what's the weather")

        assert result == entry

    def test_untagged_self_duplicate_text_is_still_droppable(self):
        # Baseline: without a voice_context (or turn_author), the existing self-duplicate
        # detection is untouched by this change.
        entry = {"text": "what's the weather"}

        result = _sanitize_queued_entry_vs_inflight_user(entry, "what's the weather")

        assert result is None
