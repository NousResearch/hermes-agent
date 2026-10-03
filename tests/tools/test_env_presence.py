"""Tests for tools.env_presence — presence-only (never value) checks."""

import os

from tools.env_presence import env_presence


class TestLocalTarget:
    def test_present_variable(self, monkeypatch):
        monkeypatch.setenv("ROB_TEST_PRESENT_VAR", "irrelevant-value")
        result = env_presence(["ROB_TEST_PRESENT_VAR"], target="local")
        assert result.presence["ROB_TEST_PRESENT_VAR"] == "PRESENT"

    def test_absent_variable(self, monkeypatch):
        monkeypatch.delenv("ROB_TEST_DEFINITELY_ABSENT_VAR", raising=False)
        result = env_presence(["ROB_TEST_DEFINITELY_ABSENT_VAR"], target="local")
        assert result.presence["ROB_TEST_DEFINITELY_ABSENT_VAR"] == "ABSENT"

    def test_never_returns_value(self, monkeypatch):
        monkeypatch.setenv("ROB_TEST_SECRET_LOOKING_VAR", "hunter2-should-never-appear")
        result = env_presence(["ROB_TEST_SECRET_LOOKING_VAR"])
        assert "hunter2" not in str(result.presence)
        assert result.presence["ROB_TEST_SECRET_LOOKING_VAR"] == "PRESENT"

    def test_default_target_is_local(self, monkeypatch):
        monkeypatch.setenv("ROB_TEST_DEFAULT_TARGET", "x")
        result = env_presence(["ROB_TEST_DEFAULT_TARGET"])
        assert result.target == "local"
        assert result.presence["ROB_TEST_DEFAULT_TARGET"] == "PRESENT"


class TestTargetValidation:
    def test_unknown_target_scheme_errors_without_crashing(self):
        result = env_presence(["X"], target="ftp:something")
        assert result.error is not None
        assert result.presence == {}

    def test_unsafe_container_name_rejected(self):
        result = env_presence(["X"], target="docker:; rm -rf /")
        assert result.error is not None

    def test_unsafe_unit_name_rejected(self):
        result = env_presence(["X"], target="systemd:$(touch /tmp/x)")
        assert result.error is not None
