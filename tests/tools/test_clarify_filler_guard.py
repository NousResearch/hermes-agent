"""Filler guard in tools/clarify_tool.py: a call where ANY question is terse free-text (the
post-compaction echo shape, alone or inside a batch), fired within the burst window of the same
session's previous clarify call, bounces wholly with a corrective error before any card shows."""

import contextlib
import json
from types import SimpleNamespace

import pytest

import tools.clarify_tool as clarify_mod
from gateway.session_context import scoped_current_session_id
from tools import approval_context
from tools.clarify_tool import FILLER_GUARD_WINDOW_SECONDS, FILLER_MAX_WORDS, clarify_tool
from tools.registry import registry

FILLER_TEXTS = ["state", "list", "x", "noop", "what next now"]


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class RecordingCallback:
    def __init__(self):
        self.calls = []

    def __call__(self, questions):
        self.calls.append(questions)
        return {"answers": {q["qid"]: "ok" for q in questions}, "outcome": "submitted"}


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    # Mutate (never rebind) the module's state so no other test sees a stale dict.
    clarify_mod._last_clarify_at.clear()
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    fake = FakeClock()
    monkeypatch.setattr(clarify_mod, "time", SimpleNamespace(monotonic=fake))
    # Start every test with no resolvable session on either seam, whatever an earlier test bound.
    with session(""), scoped_current_session_id(""):
        yield fake
    clarify_mod._last_clarify_at.clear()


@contextlib.contextmanager
def session(session_id):
    tokens = approval_context.set_current_observability_context(session_id=session_id)
    try:
        yield
    finally:
        approval_context.reset_current_observability_context(tokens)


def _ask(questions, callback):
    return json.loads(clarify_tool(questions, callback=callback))


def _one(text, **extra):
    return [{"question": text, **extra}]


def _rejected(result, callback):
    return "error" in result and "responses" not in result and not callback.calls


def _shown(result, callback):
    return "responses" in result and len(callback.calls) == 1


def _prime(clock, session_id="A"):
    """A real, substantive clarify call that opens the window."""
    with session(session_id):
        assert "responses" in _ask(_one("Which database should the migration target?"), RecordingCallback())


class TestRejects:
    @pytest.mark.parametrize("text", FILLER_TEXTS)
    def test_terse_free_text_within_window_is_rejected_without_callback(self, clock, text):
        assert len(text.split()) <= FILLER_MAX_WORDS
        _prime(clock)
        clock.advance(5)
        cb = RecordingCallback()
        with session("A"):
            result = _ask(_one(text), cb)
        assert _rejected(result, cb)

    def test_rejected_just_below_the_window(self, clock):
        _prime(clock)
        clock.advance(FILLER_GUARD_WINDOW_SECONDS - 0.001)
        cb = RecordingCallback()
        with session("A"):
            assert _rejected(_ask(_one("state"), cb), cb)

    def test_accepted_at_exactly_the_window(self, clock):
        _prime(clock)
        clock.advance(FILLER_GUARD_WINDOW_SECONDS)
        cb = RecordingCallback()
        with session("A"):
            assert _shown(_ask(_one("state"), cb), cb)

    def test_blank_choices_list_counts_as_no_choices(self, clock):
        _prime(clock)
        clock.advance(1)
        cb = RecordingCallback()
        with session("A"):
            assert _rejected(_ask(_one("list", choices=[]), cb), cb)

    def test_rejected_filler_arms_the_window(self, clock):
        _prime(clock)
        clock.advance(20)
        with session("A"):
            first, second = RecordingCallback(), RecordingCallback()
            assert _rejected(_ask(_one("state"), first), first)
            clock.advance(20)  # 40s after the prime, 20s after the rejected filler
            assert _rejected(_ask(_one("list"), second), second)

    def test_all_filler_batch_is_rejected(self, clock):
        _prime(clock)
        clock.advance(1)
        cb = RecordingCallback()
        with session("A"):
            assert _rejected(_ask([{"question": "state"}, {"question": "list"}], cb), cb)

    def test_mixed_batch_with_a_substantive_question_is_rejected(self, clock):
        # Option B (owner call 2026-10-28): one filler question bounces the WHOLE call; the
        # corrective error tells the model to resend the real questions substantively.
        _prime(clock)
        clock.advance(1)
        cb = RecordingCallback()
        with session("A"):
            assert _rejected(_ask([{"question": "Which database should the migration target?"},
                                   {"question": "state"}], cb), cb)

    def test_real_dispatch_path_rejects(self, clock):
        _prime(clock)
        clock.advance(3)
        cb = RecordingCallback()
        with session("A"):
            raw = registry.dispatch("clarify", {"questions": _one("noop")}, callback=cb)
        result = json.loads(raw) if isinstance(raw, str) else raw
        assert _rejected(result, cb)

    def test_unresolvable_session_still_guarded(self, clock):
        first, second = RecordingCallback(), RecordingCallback()
        assert _shown(_ask(_one("x"), first), first)
        clock.advance(2)
        assert _rejected(_ask(_one("x"), second), second)


class TestAllows:
    def test_first_call_in_session_is_shown(self, clock):
        cb = RecordingCallback()
        with session("A"):
            assert _shown(_ask(_one("state"), cb), cb)

    def test_gap_beyond_window_is_shown(self, clock):
        _prime(clock)
        clock.advance(FILLER_GUARD_WINDOW_SECONDS + 0.001)
        cb = RecordingCallback()
        with session("A"):
            assert _shown(_ask(_one("state"), cb), cb)

    def test_more_than_max_words_is_shown(self, clock):
        _prime(clock)
        clock.advance(1)
        cb = RecordingCallback()
        text = " ".join(["word"] * (FILLER_MAX_WORDS + 1))
        with session("A"):
            assert _shown(_ask(_one(text), cb), cb)

    def test_choices_present_is_shown(self, clock):
        _prime(clock)
        clock.advance(1)
        cb = RecordingCallback()
        with session("A"):
            assert _shown(_ask(_one("Pick one", choices=["a", "b"]), cb), cb)

    def test_other_session_is_not_gated(self, clock):
        _prime(clock, "A")
        clock.advance(1)
        cb = RecordingCallback()
        with session("B"):
            assert _shown(_ask(_one("state"), cb), cb)

    def test_gateway_session_id_isolates_when_approval_id_unset(self, clock):
        with scoped_current_session_id("gw-A"):
            _ask(_one("Which database should the migration target?"), RecordingCallback())
        clock.advance(1)
        cb, gated = RecordingCallback(), RecordingCallback()
        with scoped_current_session_id("gw-B"):
            assert _shown(_ask(_one("state"), cb), cb)
        with scoped_current_session_id("gw-A"):
            assert _rejected(_ask(_one("state"), gated), gated)

    @pytest.mark.parametrize("malformed", [
        [],
        [{"question": "   "}],
        [{"question": "state", "choices": ["  ", ""]}],
        [{"question": "state", "choices": "a,b"}],
        ["state"],
    ])
    def test_malformed_call_does_not_arm_the_window(self, clock, malformed):
        bad = RecordingCallback()
        with session("A"):
            assert _rejected(_ask(malformed, bad), bad)
            clock.advance(1)
            cb = RecordingCallback()
            assert _shown(_ask(_one("state"), cb), cb)


class TestStateBounded:
    def test_stale_session_keys_are_pruned_on_write(self, clock):
        for index in range(500):
            clock.advance(1)
            with session(f"s{index}"):
                _ask(_one("Which database should the migration target?"), RecordingCallback())
        horizon = clarify_mod._FILLER_STATE_TTL_SECONDS
        now = clock()
        assert clarify_mod._last_clarify_at
        assert all(now - at <= horizon for at in clarify_mod._last_clarify_at.values())
        assert len(clarify_mod._last_clarify_at) <= horizon + 1

        clock.advance(horizon * 10)
        with session("late"):
            _ask(_one("Which database should the migration target?"), RecordingCallback())
        assert list(clarify_mod._last_clarify_at) == ["late"]


class TestClockDiscipline:
    def test_record_never_moves_backward_under_a_stale_sample(self, clock):
        # A caller that sampled the clock early must not rewind a later call's record when it
        # finally acquires the lock: the stored timestamp is monotone per session.
        _prime(clock)                      # records at 1000.0
        clock.advance(20)                  # a second substantive call records at 1020.0
        with session("A"):
            assert "responses" in _ask(_one("Which index should we rebuild first?"), RecordingCallback())
        clock.now = 1001.0                 # stale sample from a caller that paused before the lock
        with session("A"):
            _ask(_one("Please confirm whether we should deploy the migration now."), RecordingCallback())
        assert clarify_mod._last_clarify_at["A"] == 1020.0


class TestIdentityFailure:
    def test_session_lookup_import_failure_falls_back_to_global_key(self, clock, monkeypatch):
        # A guard must never break a legitimate clarify: if the identity seam cannot even be
        # imported, the call proceeds on the shared global key instead of raising.
        import sys
        monkeypatch.setitem(sys.modules, "gateway.session_context", None)
        first = RecordingCallback()
        assert _shown(_ask(_one("Which database should the migration target?"), first), first)
        clock.advance(1)
        filler = RecordingCallback()
        assert _rejected(_ask(_one("state"), filler), filler)  # still guarded, just coarsely
