"""Contract tests for the optional per-turn composer-mode frame (``note``/``mode``).

The frame rides ``prompt.submit`` and the steer/redirect RPCs as an OPTIONAL
pair: the note reaches the model through the per-turn ``api_content`` sidecar
(the user's own ``content`` is never polluted) and the opaque ``mode`` label
rides ``display_metadata`` (display-only, popped from every outbound copy).
Everything below is pure: no gateway, no agent construction.
"""
from agent.interrupt_control import InterruptControlMixin
from agent.prompt_builder import steer_user_row
from tui_gateway.methods_prompt import parse_turn_note
from tui_gateway.session_auto_continue import _enqueue_prompt, _sanitize_queued_entry_vs_inflight_user


class _CorrectionAgent(InterruptControlMixin):
    """__init__-less stub: the mixin's lock helpers fall back to unlocked slots."""

    def __init__(self):
        self._pending_steer = None
        self._pending_redirect = None


def test_parse_turn_note_sanitizes_and_caps():
    assert parse_turn_note({"note": "  keep it short  ", "mode": " plan "}) == ("keep it short", "plan")
    assert parse_turn_note({"note": 123, "mode": None}) == ("", "")
    assert parse_turn_note({"note": "x" * 9000})[0] == "x" * 8192
    assert parse_turn_note({"mode": "m" * 64})[1] == "m" * 32


def test_steer_row_keeps_the_users_words_clean():
    row = steer_user_row("hello", note="MODE: plan", mode="plan")
    # The note never leaks into the transcript bytes...
    assert "MODE: plan" not in row["content"]
    assert row["content"].strip().endswith("[/OUT-OF-BAND USER MESSAGE]")
    # ...it rides the API-side bytes, and the label stays display-only.
    assert row["api_content"].startswith("MODE: plan")
    assert row["display_metadata"] == {"mode": "plan"}
    # A frame-less steer keeps the exact pre-existing row shape.
    plain = steer_user_row("hello")
    assert "api_content" not in plain and "display_metadata" not in plain


def test_correction_frame_is_one_shot_per_slot():
    agent = _CorrectionAgent()
    assert agent.steer("fix the bug", note="MODE: plan", mode="plan") is True
    assert agent._pending_steer.endswith("fix the bug")
    assert agent._take_correction_note("_pending_steer") == ("MODE: plan", "plan")
    # Consumed: a later delivery must never inherit a stale frame.
    assert agent._take_correction_note("_pending_steer") == ("", "")
    assert agent.steer("plain") is True
    assert agent._take_correction_note("_pending_steer") == ("", "")


def test_frames_concatenate_and_the_mode_label_is_last_wins():
    agent = _CorrectionAgent()
    agent._queue_correction_note("_pending_redirect", "first", "plan")
    agent._queue_correction_note("_pending_redirect", "second", "debug")
    assert agent._take_correction_note("_pending_redirect") == ("first\n\nsecond", "debug")


def test_a_framed_queued_prompt_never_merges_into_a_plain_slot():
    session: dict = {}
    _enqueue_prompt(session, "first", None)
    _enqueue_prompt(session, "second", None, turn_note="MODE: plan", turn_mode="plan")
    # The framed arrival did NOT merge into the plain slot (its frame would die there).
    assert session["queued_prompt"]["text"] == "first"
    assert session["queued_prompts"][-1]["note"] == "MODE: plan"
    assert session["queued_prompts"][-1]["mode"] == "plan"


def test_a_plain_arrival_never_merges_into_a_framed_slot():
    session: dict = {}
    _enqueue_prompt(session, "framed", None, turn_note="MODE: plan", turn_mode="plan")
    _enqueue_prompt(session, "plain", None)
    assert session["queued_prompt"]["text"] == "framed"
    assert session["queued_prompt"]["note"] == "MODE: plan"
    assert session["queued_prompts"][-1]["text"] == "plain"


def test_sanitize_keeps_a_framed_entry():
    entry = {"text": "x", "note": "MODE: plan", "mode": "plan"}
    # Same text as the live prompt, but framed: it survives instead of being dropped.
    assert _sanitize_queued_entry_vs_inflight_user(entry, "x") == entry
