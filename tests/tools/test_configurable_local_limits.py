"""Focused contracts for disabling local work-count limits.

These tests intentionally cover only configurable turn/tool/iteration ceilings.  Timeouts,
cancellation, watchdogs, provider budgets, rate limits, and approval boundaries are independent.
"""

import sys

import pytest

from hermes_cli.config import TURN_LIMIT_UNLIMITED, resolve_local_limit
from tools.code_execution_tool import DEFAULT_MAX_TOOL_CALLS, _resolve_max_tool_calls
from tools.delegate_tool import DEFAULT_MAX_ITERATIONS, _resolve_max_iterations


@pytest.mark.parametrize("raw", [None, "unlimited", "none", "null", "inf", 0, -1])
def test_generic_local_limit_accepts_explicit_unlimited_values(raw):
    assert resolve_local_limit(raw, default=17) == TURN_LIMIT_UNLIMITED


@pytest.mark.parametrize("raw, expected", [(1, 1), (250, 250), ("42", 42)])
def test_generic_local_limit_preserves_positive_finite_caps(raw, expected):
    assert resolve_local_limit(raw, default=17) == expected


def test_generic_local_limit_uses_default_for_invalid_values():
    assert resolve_local_limit("not-a-limit", default=17) == 17
    assert resolve_local_limit(True, default=17) == 17


@pytest.mark.parametrize("raw", [None, "unlimited", "none", 0, -1])
def test_code_execution_tool_calls_can_be_unlimited(raw):
    assert _resolve_max_tool_calls({"max_tool_calls": raw}) == sys.maxsize


def test_code_execution_missing_limit_keeps_compatibility_default():
    assert _resolve_max_tool_calls({}) == DEFAULT_MAX_TOOL_CALLS


@pytest.mark.parametrize("raw", [None, "unlimited", "none", 0, -1])
def test_delegation_iterations_can_be_unlimited(raw):
    assert _resolve_max_iterations({"max_iterations": raw}) == sys.maxsize


def test_delegation_missing_limit_keeps_compatibility_default():
    assert _resolve_max_iterations({}) == DEFAULT_MAX_ITERATIONS
