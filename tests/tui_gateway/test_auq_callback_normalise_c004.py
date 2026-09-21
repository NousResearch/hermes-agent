"""C004-R1 RED tests — TUI gateway AUQ callback result normalisation.

Small behavioural tests for the module helper
``_normalise_auq_callback_result(questions, result)`` which:

  * returns ``result`` unchanged when it is a dict (the happy path — the
    TUI renderer responded with a ``{index: label}`` dict);
  * otherwise (empty string, None, or any non-dict timeout/interruption
    fallback) returns a per-question skipped dict
    ``{i: "__skipped__" for i in range(len(questions))}``.

The helper is wired around the ``_block`` call in
``ask_user_questions_callback`` so a timeout or interruption that returns
``""`` / ``None`` is translated into skipped answers rather than being
passed to the tool as a non-dict (which would error).
"""
import pytest

from tui_gateway.server import _normalise_auq_callback_result


_QUESTIONS = [
    {"question": "Q1", "options": [
        {"label": "A", "recommended": True},
        {"label": "B"},
    ]},
    {"question": "Q2", "options": [
        {"label": "C", "recommended": True},
        {"label": "D"},
    ]},
]


class TestNormaliseAuqCallbackResultDictPassthrough:
    """When the TUI returns a dict, it must pass through unchanged."""

    def test_dict_passthrough(self):
        result = {0: "A", 1: "D"}
        assert _normalise_auq_callback_result(_QUESTIONS, result) is result

    def test_partial_dict_passthrough(self):
        """Even a partial dict is passed through — the tool layer, not the
        normaliser, is responsible for coercing missing keys to (skipped)."""
        result = {0: "A"}
        assert _normalise_auq_callback_result(_QUESTIONS, result) is result


class TestNormaliseAuqCallbackResultFallback:
    """When ``_block`` returns a non-dict (timeout / interruption fallback),
    the helper must return a per-question skipped dict."""

    @pytest.mark.parametrize("fallback", ["", None, "timeout"])
    def test_non_dict_returns_skipped_dict(self, fallback):
        out = _normalise_auq_callback_result(_QUESTIONS, fallback)
        assert out == {0: "__skipped__", 1: "__skipped__"}, (
            f"expected skipped dict for fallback {fallback!r}, got {out}"
        )

    def test_empty_questions_returns_empty_dict(self):
        out = _normalise_auq_callback_result([], "")
        assert out == {}
