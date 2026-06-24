"""Tests for shared environment-value helpers."""

from utils import env_positive_int, env_var_enabled, is_truthy_value


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


def test_env_positive_int_accepts_positive_values_and_falls_back(monkeypatch):
    monkeypatch.setenv("HERMES_TEST_INT", "4096")
    assert env_positive_int("HERMES_TEST_INT", 16) == 4096

    for raw in ("", "0", "-1", "nan", "12.5", "large"):
        monkeypatch.setenv("HERMES_TEST_INT", raw)
        assert env_positive_int("HERMES_TEST_INT", 16) == 16
