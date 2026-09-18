"""Tests for tools/secure_input.py."""

import json
import os
import sys
from pathlib import Path

import pytest

from tools.secure_input import (
    SECURE_INPUT_SCHEMA,
    _append_to_env_file,
    _is_valid_env_name,
    check_secure_input_requirements,
    secure_input_tool,
)


# ---------------------------------------------------------------------------
# secure_input_tool
# ---------------------------------------------------------------------------


class TestSecureInputTool:
    def test_missing_key(self):
        result = json.loads(secure_input_tool(key="", callback=lambda k, p: "val"))
        assert "error" in result

    def test_whitespace_key(self):
        result = json.loads(secure_input_tool(key="   ", callback=lambda k, p: "val"))
        assert "error" in result

    def test_invalid_key_name(self):
        result = json.loads(secure_input_tool(key="123INVALID", callback=lambda k, p: "val"))
        assert "error" in result
        assert "Invalid env var name" in result["error"]

    def test_valid_key_stored(self):
        hermes_home = Path(os.environ["HERMES_HOME"])
        result = json.loads(
            secure_input_tool(
                key="TEST_API_KEY",
                confirm=False,
                callback=lambda k, p: "secret123",
            )
        )
        assert result["stored"] is True
        assert result["key"] == "TEST_API_KEY"

        env_path = hermes_home / ".env"
        assert env_path.exists()
        content = env_path.read_text()
        assert "TEST_API_KEY=secret123" in content

    def test_confirmation_mismatch(self):
        call_count = [0]

        def callback(k, p):
            call_count[0] += 1
            return "first" if call_count[0] == 1 else "second"

        result = json.loads(secure_input_tool(key="TEST_KEY", callback=callback))
        assert "error" in result
        assert "do not match" in result["error"].lower()

    def test_confirmation_match(self):
        hermes_home = Path(os.environ["HERMES_HOME"])
        result = json.loads(
            secure_input_tool(key="TEST_KEY", confirm=True, callback=lambda k, p: "same_value")
        )
        assert result["stored"] is True
        env_path = hermes_home / ".env"
        assert "TEST_KEY=same_value" in env_path.read_text()

    def test_empty_value_rejected(self):
        result = json.loads(secure_input_tool(key="TEST_KEY", callback=lambda k, p: "   "))
        assert "error" in result
        assert "No value entered" in result["error"]

    def test_agent_never_sees_secret(self):
        result_raw = secure_input_tool(
            key="SECRET_KEY", confirm=False, callback=lambda k, p: "super-secret-value-123"
        )
        assert "super-secret-value-123" not in result_raw
        result = json.loads(result_raw)
        assert "super-secret-value-123" not in json.dumps(result)

    def test_callback_receives_key_and_prompt(self):
        received = []

        def callback(k, p):
            received.append((k, p))
            return "val"

        secure_input_tool(key="MY_KEY", prompt="Enter pls", confirm=False, callback=callback)
        assert len(received) == 1
        assert received[0][0] == "MY_KEY"
        assert "Enter pls" in received[0][1]

    def test_custom_prompt(self):
        received_prompt = []

        def callback(k, p):
            received_prompt.append(p)
            return "val"

        secure_input_tool(key="K", prompt="Custom prompt", confirm=False, callback=callback)
        assert "Custom prompt" in received_prompt[0]

    def test_interrupted_input(self):
        def callback(k, p):
            raise KeyboardInterrupt()

        result = json.loads(secure_input_tool(key="K", callback=callback))
        assert "error" in result
        assert "interrupted" in result["error"].lower()

    def test_confirm_disabled(self):
        """When confirm=False, callback is only called once."""
        calls = []

        def callback(k, p):
            calls.append(p)
            return "val"

        secure_input_tool(key="K", confirm=False, callback=callback)
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# _is_valid_env_name
# ---------------------------------------------------------------------------


class TestIsValidEnvName:
    def test_valid_names(self):
        assert _is_valid_env_name("OPENAI_API_KEY") is True
        assert _is_valid_env_name("_PRIVATE") is True
        assert _is_valid_env_name("A123") is True
        assert _is_valid_env_name("A") is True

    def test_invalid_names(self):
        assert _is_valid_env_name("") is False
        assert _is_valid_env_name("123ABC") is False
        assert _is_valid_env_name("KEY-NAME") is False
        assert _is_valid_env_name("KEY.NAME") is False
        assert _is_valid_env_name("KEY NAME") is False


# ---------------------------------------------------------------------------
# _append_to_env_file
# ---------------------------------------------------------------------------


class TestAppendToEnvFile:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.hermes_home = Path(os.environ["HERMES_HOME"])

    def _env_path(self):
        return self.hermes_home / ".env"

    def test_new_key(self):
        _append_to_env_file(self._env_path(), "NEW_KEY", "new_value")
        assert "NEW_KEY=new_value" in self._env_path().read_text()

    def test_upsert_existing_key(self):
        self._env_path().write_text("OLD_KEY=old_value\n")
        _append_to_env_file(self._env_path(), "OLD_KEY", "new_value")
        content = self._env_path().read_text()
        assert "OLD_KEY=new_value" in content
        assert "old_value" not in content

    def test_handles_export_prefix(self):
        self._env_path().write_text("export OLD_KEY=old_value\n")
        _append_to_env_file(self._env_path(), "OLD_KEY", "new_value")
        content = self._env_path().read_text()
        assert "OLD_KEY=new_value" in content
        assert "export OLD_KEY" not in content

    def test_preserves_other_keys(self):
        self._env_path().write_text("KEEP_ME=keep\nANOTHER=val\n")
        _append_to_env_file(self._env_path(), "NEW_KEY", "new_val")
        content = self._env_path().read_text()
        assert "KEEP_ME=keep" in content
        assert "ANOTHER=val" in content
        assert "NEW_KEY=new_val" in content

    def test_handles_missing_trailing_newline(self):
        self._env_path().write_text("EXISTING=val")
        _append_to_env_file(self._env_path(), "NEW_KEY", "new_val")
        content = self._env_path().read_text()
        assert "EXISTING=val\n" in content
        assert "NEW_KEY=new_val" in content

    def test_preserves_comments(self):
        self._env_path().write_text("# This is a comment\nKEEP=val\n")
        _append_to_env_file(self._env_path(), "NEW_KEY", "new_val")
        content = self._env_path().read_text()
        assert "# This is a comment" in content

    def test_empty_file(self):
        _append_to_env_file(self._env_path(), "KEY", "val")
        content = self._env_path().read_text().strip()
        assert content == "KEY=val"

    def test_atomic_write_permissions(self):
        _append_to_env_file(self._env_path(), "KEY", "val")
        if sys.platform != "win32":
            mode = self._env_path().stat().st_mode & 0o777
            assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_preserves_empty_lines(self):
        self._env_path().write_text("KEEP=val\n\nANOTHER=val2\n")
        _append_to_env_file(self._env_path(), "NEW", "val3")
        content = self._env_path().read_text()
        assert "\n\n" in content  # empty lines preserved


# ---------------------------------------------------------------------------
# check_secure_input_requirements
# ---------------------------------------------------------------------------


class TestCheckRequirements:
    def test_always_true(self):
        assert check_secure_input_requirements() is True


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_flat_format(self):
        assert "name" in SECURE_INPUT_SCHEMA
        assert "description" in SECURE_INPUT_SCHEMA
        assert "parameters" in SECURE_INPUT_SCHEMA
        assert "function" not in SECURE_INPUT_SCHEMA

    def test_description_length(self):
        assert len(SECURE_INPUT_SCHEMA["description"]) <= 500

    def test_required_params(self):
        required = SECURE_INPUT_SCHEMA["parameters"]["required"]
        assert "key" in required
