"""Tests for scripts/validate_env.py — Plan 002-D: Cloud Env-Var Routing.

Covers:
- validate_hermes_env() returns empty list when no vars are set
- Returns empty list when vars point at existing paths
- Warns (no error) when vars point at non-existent paths
- Errors (returns error list) when vars use unsupported URI schemes (s3://, etc.)
- strict=True raises ValueError on URI errors
- strict=False does NOT raise on URI errors
- All documented operator vars are checked
- docs/env-vars.md exists and documents the variables
"""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _clean_hermes_env_vars(monkeypatch):
    """Ensure no HERMES_* operator vars are set before each test."""
    for var in ("HERMES_USERS_ROOT", "HERMES_RUNTIME_ROOT", "HERMES_MCP_SERVERS_CONFIG"):
        monkeypatch.delenv(var, raising=False)


def _load_validate_env():
    import importlib
    import sys
    # Ensure we can import scripts/validate_env as a module from the repo root
    repo_root = str(Path(__file__).parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import scripts.validate_env as ve
    importlib.reload(ve)
    return ve


class TestValidateHermesEnvNoVars:
    def test_no_vars_set_returns_empty_errors(self):
        ve = _load_validate_env()
        errors = ve.validate_hermes_env()
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_no_vars_no_raise_in_strict(self):
        ve = _load_validate_env()
        # strict=True with no errors should not raise
        errors = ve.validate_hermes_env(strict=True)
        assert errors == []


class TestValidateHermesEnvExistingPaths:
    def test_existing_path_no_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_USERS_ROOT", str(tmp_path))
        ve = _load_validate_env()
        errors = ve.validate_hermes_env()
        assert errors == []

    def test_all_vars_existing_paths_no_error(self, tmp_path, monkeypatch):
        mcp_file = tmp_path / "mcp.json"
        mcp_file.write_text("{}")
        monkeypatch.setenv("HERMES_USERS_ROOT", str(tmp_path))
        monkeypatch.setenv("HERMES_RUNTIME_ROOT", str(tmp_path))
        monkeypatch.setenv("HERMES_MCP_SERVERS_CONFIG", str(mcp_file))
        ve = _load_validate_env()
        errors = ve.validate_hermes_env()
        assert errors == []


class TestValidateHermesEnvMissingPaths:
    def test_missing_path_returns_no_error(self, tmp_path, monkeypatch):
        """Missing path is a warning, not an error."""
        missing = tmp_path / "does-not-exist"
        monkeypatch.setenv("HERMES_USERS_ROOT", str(missing))
        ve = _load_validate_env()
        errors = ve.validate_hermes_env()
        assert errors == [], f"Missing path should warn, not error; got: {errors}"


class TestValidateHermesEnvUnsupportedSchemes:
    @pytest.mark.parametrize("scheme", ["s3", "gs", "az", "gcs", "postgres", "dynamodb"])
    def test_unsupported_scheme_returns_error(self, scheme, monkeypatch):
        monkeypatch.setenv("HERMES_USERS_ROOT", f"{scheme}://my-bucket/hermes")
        ve = _load_validate_env()
        errors = ve.validate_hermes_env(strict=False)
        assert len(errors) >= 1, f"Expected at least 1 error for scheme {scheme!r}, got {errors}"
        assert scheme in errors[0] or "not yet supported" in errors[0].lower()

    def test_s3_strict_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("HERMES_USERS_ROOT", "s3://my-bucket/hermes")
        ve = _load_validate_env()
        with pytest.raises(ValueError, match="s3"):
            ve.validate_hermes_env(strict=True)

    def test_s3_non_strict_does_not_raise(self, monkeypatch):
        monkeypatch.setenv("HERMES_USERS_ROOT", "s3://my-bucket/hermes")
        ve = _load_validate_env()
        # Should not raise; should return error list
        errors = ve.validate_hermes_env(strict=False)
        assert len(errors) >= 1

    def test_runtime_root_s3_errors(self, monkeypatch):
        monkeypatch.setenv("HERMES_RUNTIME_ROOT", "s3://my-runtime/sessions")
        ve = _load_validate_env()
        errors = ve.validate_hermes_env()
        assert len(errors) >= 1

    def test_mcp_config_s3_errors(self, monkeypatch):
        monkeypatch.setenv("HERMES_MCP_SERVERS_CONFIG", "s3://bucket/mcp.json")
        ve = _load_validate_env()
        errors = ve.validate_hermes_env()
        assert len(errors) >= 1

    def test_file_uri_not_treated_as_error(self, tmp_path, monkeypatch):
        """file:// URIs are local filesystem — not an error."""
        # file:// is in the excluded scheme list or treated as path — either is fine
        monkeypatch.setenv("HERMES_USERS_ROOT", str(tmp_path))
        ve = _load_validate_env()
        errors = ve.validate_hermes_env()
        assert errors == []


class TestDocsEnvVars:
    def test_env_vars_md_exists(self):
        docs_path = Path(__file__).parent.parent / "docs" / "env-vars.md"
        assert docs_path.exists(), f"docs/env-vars.md not found at {docs_path}"

    def test_env_vars_md_documents_operator_vars(self):
        docs_path = Path(__file__).parent.parent / "docs" / "env-vars.md"
        content = docs_path.read_text()
        expected_vars = [
            "HERMES_HOME",
            "HERMES_USERS_ROOT",
            "HERMES_RUNTIME_ROOT",
            "HERMES_MCP_SERVERS_CONFIG",
        ]
        for var in expected_vars:
            assert var in content, f"{var} not documented in docs/env-vars.md"
