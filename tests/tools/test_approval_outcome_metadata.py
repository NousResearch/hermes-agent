"""Regression coverage for preserving every user-facing approval outcome."""

from __future__ import annotations

import json

import pytest

from tools import approval
from tools import code_execution_tool
from tools import terminal_tool


_EXPECTED_USER_SUMMARY_OUTCOMES = (
    "denied",
    "timeout",
    "notify_failed",
    "cancelled",
    "blocked",
)


def test_user_summary_outcomes_are_complete():
    assert approval._USER_SUMMARY_OUTCOMES == set(_EXPECTED_USER_SUMMARY_OUTCOMES)


@pytest.mark.parametrize("outcome", _EXPECTED_USER_SUMMARY_OUTCOMES)
def test_terminal_preserves_all_user_summary_outcomes(outcome):
    assert terminal_tool._approval_outcome_fields({"outcome": outcome}) == {
        "approval_outcome": outcome
    }


@pytest.mark.parametrize("outcome", _EXPECTED_USER_SUMMARY_OUTCOMES)
def test_execute_code_preserves_all_user_summary_outcomes(outcome):
    result = json.loads(code_execution_tool._error_result(
        "approval rejected",
        user_summary="A human-readable outcome.",
        approval_outcome=outcome,
    ))

    assert result["approval_outcome"] == outcome
