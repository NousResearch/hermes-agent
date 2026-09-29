"""Tests for shared utility coercion/parsing helpers."""

from utils import (
    env_bool,
    env_int,
    env_var_enabled,
    is_truthy_value,
    safe_json_loads,
)


def test_is_truthy_value_accepts_common_truthy_strings():
    assert is_truthy_value("true") is True
    assert is_truthy_value(" YES ") is True
    assert is_truthy_value("on") is True
    assert is_truthy_value("1") is True


def test_is_truthy_value_respects_default_for_none():
    assert is_truthy_value(None, default=True) is True
    assert is_truthy_value(None, default=False) is False


def test_is_truthy_value_rejects_falsey_strings():
    assert is_truthy_value("false") is False
    assert is_truthy_value("0") is False
    assert is_truthy_value("off") is False


def test_env_var_enabled_uses_shared_truthy_rules(monkeypatch):
    monkeypatch.setenv("HERMES_TEST_BOOL", "YeS")
    assert env_var_enabled("HERMES_TEST_BOOL") is True

    monkeypatch.setenv("HERMES_TEST_BOOL", "no")
    assert env_var_enabled("HERMES_TEST_BOOL") is False


def test_safe_json_loads_returns_default_on_invalid_json():
    assert safe_json_loads("{bad", default={"fallback": True}) == {"fallback": True}
    assert safe_json_loads(None, default=[]) == []


def test_safe_json_loads_parses_valid_json():
    result = safe_json_loads('{"ok": 1}')
    assert result == {"ok": 1}


def test_env_int_parses_integer_and_falls_back(monkeypatch):
    monkeypatch.setenv("HERMES_TEST_INT", "42")
    assert env_int("HERMES_TEST_INT", default=7) == 42

    monkeypatch.setenv("HERMES_TEST_INT", "not-a-number")
    assert env_int("HERMES_TEST_INT", default=7) == 7

    monkeypatch.delenv("HERMES_TEST_INT", raising=False)
    assert env_int("HERMES_TEST_INT", default=7) == 7


def test_env_bool_uses_shared_truthy_value(monkeypatch):
    monkeypatch.setenv("HERMES_TEST_BOOL2", "on")
    assert env_bool("HERMES_TEST_BOOL2", default=False) is True

    monkeypatch.setenv("HERMES_TEST_BOOL2", "off")
    assert env_bool("HERMES_TEST_BOOL2", default=True) is False

    monkeypatch.delenv("HERMES_TEST_BOOL2", raising=False)
    assert env_bool("HERMES_TEST_BOOL2", default=True) is False
