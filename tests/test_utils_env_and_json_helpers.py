"""Tests for shared env-var helpers and safe_json_loads."""

import json

import pytest

from utils import env_bool, env_float, env_int, safe_json_loads


# ─── env_int / env_float / env_bool ────────────────────────────────────────


def test_env_int_parses_and_falls_back(monkeypatch):
    monkeypatch.setenv("HERMES_TEST_INT", "42")
    assert env_int("HERMES_TEST_INT") == 42

    monkeypatch.setenv("HERMES_TEST_INT", " 7 ")
    assert env_int("HERMES_TEST_INT") == 7

    monkeypatch.setenv("HERMES_TEST_INT", "not-a-number")
    assert env_int("HERMES_TEST_INT") == 0
    assert env_int("HERMES_TEST_INT", default=5) == 5

    monkeypatch.delenv("HERMES_TEST_INT", raising=False)
    assert env_int("HERMES_TEST_INT") == 0
    assert env_int("HERMES_TEST_INT", default=-1) == -1


def test_env_float_parses_and_falls_back(monkeypatch):
    monkeypatch.setenv("HERMES_TEST_FLOAT", "3.5")
    assert env_float("HERMES_TEST_FLOAT") == 3.5

    monkeypatch.setenv("HERMES_TEST_FLOAT", "not-a-float")
    assert env_float("HERMES_TEST_FLOAT") == 0.0
    assert env_float("HERMES_TEST_FLOAT", default=1.5) == 1.5

    monkeypatch.delenv("HERMES_TEST_FLOAT", raising=False)
    assert env_float("HERMES_TEST_FLOAT") == 0.0


def test_env_bool_uses_shared_truthy_rules(monkeypatch):
    monkeypatch.setenv("HERMES_TEST_BOOL", "yes")
    assert env_bool("HERMES_TEST_BOOL") is True

    monkeypatch.setenv("HERMES_TEST_BOOL", "0")
    assert env_bool("HERMES_TEST_BOOL") is False

    monkeypatch.delenv("HERMES_TEST_BOOL", raising=False)
    assert env_bool("HERMES_TEST_BOOL") is False
    assert env_bool("HERMES_TEST_BOOL", default=True) is True


# ─── safe_json_loads ───────────────────────────────────────────────────────


def test_safe_json_loads_parses_valid_json():
    assert safe_json_loads('{"a": 1}') == {"a": 1}
    assert safe_json_loads("[1, 2, 3]") == [1, 2, 3]


def test_safe_json_loads_returns_default_on_malformed_json():
    assert safe_json_loads("{not json", default="fallback") == "fallback"
    assert safe_json_loads("", default="fallback") == "fallback"


def test_safe_json_loads_returns_default_on_non_string_input():
    assert safe_json_loads(None, default="fallback") == "fallback"
    assert safe_json_loads(12345, default="fallback") == "fallback"


def test_safe_json_loads_returns_default_on_deeply_nested_json():
    """Deeply nested JSON raises RecursionError inside json.loads, which is
    not a JSONDecodeError — it must still fall back to default instead of
    crashing the caller (tool_guardrails parses untrusted tool output)."""
    deep = "[" * 100_000 + "]" * 100_000
    assert safe_json_loads(deep, default="fallback") == "fallback"
