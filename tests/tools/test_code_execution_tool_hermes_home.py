"""Tests that code_execution_tool respects HERMES_HOME environment variable."""

import pytest


def test_execute_code_schema_respects_hermes_home(tmp_path, monkeypatch):
    """execute_code tool description should use HERMES_HOME for the .env path."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    from tools.code_execution_tool import build_execute_code_schema
    schema = build_execute_code_schema(mode="strict")
    description = schema.get("description", "")
    expected_path = str(tmp_path / ".hermes" / ".env")
    assert expected_path in description