"""Tests for the env_int / env_bool environment-variable helpers."""

from utils import env_bool, env_int


def test_env_int_parses_valid_integer(monkeypatch):
    monkeypatch.setenv("HERMES_TEST_INT", "42")
    assert env_int("HERMES_TEST_INT") == 42


def test_env_int_strips_whitespace(monkeypatch):
    monkeypatch.setenv("HERMES_TEST_INT", "  123  ")
    assert env_int("HERMES_TEST_INT") == 123


def test_env_int_falls_back_on_invalid(monkeypatch):
    monkeypatch.setenv("HERMES_TEST_INT", "not_a_number")
    assert env_int("HERMES_TEST_INT", 7) == 7


def test_env_int_falls_back_when_unset(monkeypatch):
    monkeypatch.delenv("HERMES_TEST_INT", raising=False)
    assert env_int("HERMES_TEST_INT", 5) == 5


def test_env_bool_reads_truthy_strings(monkeypatch):
    monkeypatch.setenv("HERMES_TEST_BOOL", "YeS")
    assert env_bool("HERMES_TEST_BOOL") is True

    monkeypatch.setenv("HERMES_TEST_BOOL", "off")
    assert env_bool("HERMES_TEST_BOOL") is False


def test_env_bool_honors_default_when_unset(monkeypatch):
    monkeypatch.delenv("HERMES_TEST_BOOL", raising=False)
    assert env_bool("HERMES_TEST_BOOL", default=True) is True
    assert env_bool("HERMES_TEST_BOOL", default=False) is False


def test_env_bool_honors_default_when_empty(monkeypatch):
    monkeypatch.setenv("HERMES_TEST_BOOL", "   ")
    assert env_bool("HERMES_TEST_BOOL", default=True) is True


def test_env_bool_set_value_overrides_default(monkeypatch):
    monkeypatch.setenv("HERMES_TEST_BOOL", "false")
    assert env_bool("HERMES_TEST_BOOL", default=True) is False
