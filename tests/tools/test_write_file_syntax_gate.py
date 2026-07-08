"""Tests for the pre-write syntax gate in write_file().

When write_file() is called with JSON/YAML/TOML content, the
in-process syntax check must fire BEFORE _atomic_write().  A parse
failure should set result.error so that the file_tools wrapper
correctly suppresses ``files_modified`` reporting.

Fixes: #60525
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

from tools.file_operations import (
    ShellFileOperations,
    _lint_json_inproc,
    _lint_yaml_inproc,
    _lint_toml_inproc,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_env(tmp_path):
    """Create a mock terminal environment backed by a real tmp dir."""
    env = MagicMock()
    env.cwd = str(tmp_path)

    def fake_execute(cmd, cwd=None, **kwargs):
        # Simulate shell execution for file I/O commands we care about.
        # kwargs may include stdin_data, timeout, etc. — ignore them.
        if cmd.startswith("cat ") or cmd.startswith("wc -c < "):
            return {"output": "", "returncode": 1}
        if cmd.startswith("mkdir -p "):
            return {"output": "", "returncode": 0}
        return {"output": "", "returncode": 0}

    env.execute = MagicMock(side_effect=fake_execute)
    return env


@pytest.fixture()
def file_ops(mock_env):
    return ShellFileOperations(mock_env)


# ===========================================================================
# JSON syntax gate
# ===========================================================================

class TestJSONSyntaxGate:
    """write_file() must reject invalid JSON before writing to disk."""

    def test_valid_json_succeeds(self, file_ops, tmp_path):
        """Valid JSON should pass through and be written."""
        path = str(tmp_path / "config.json")
        content = '{"name": "test", "value": 42}'
        result = file_ops.write_file(path, content)
        assert result.error is None

    def test_truncated_json_aborted(self, file_ops, tmp_path):
        """Truncated JSON must be rejected before disk write."""
        path = str(tmp_path / "config.json")
        content = '{"a": 1,'
        result = file_ops.write_file(path, content)
        assert result.error is not None
        assert "Syntax check failed" in result.error
        assert "JSONDecodeError" in result.error
        # File must NOT exist on disk
        assert not os.path.exists(path)

    def test_invalid_json_type_aborted(self, file_ops, tmp_path):
        """JSON with trailing garbage must be rejected."""
        path = str(tmp_path / "data.json")
        content = '{"a": 1}garbage'
        result = file_ops.write_file(path, content)
        assert result.error is not None
        assert not os.path.exists(path)

    def test_empty_object_json_succeeds(self, file_ops, tmp_path):
        """Empty JSON object is valid syntax."""
        path = str(tmp_path / "empty.json")
        result = file_ops.write_file(path, '{}')
        assert result.error is None

    def test_json_array_succeeds(self, file_ops, tmp_path):
        """JSON array is valid syntax."""
        path = str(tmp_path / "list.json")
        result = file_ops.write_file(path, '[1, 2, 3]')
        assert result.error is None

    def test_error_response_has_lint_field(self, file_ops, tmp_path):
        """The WriteResult must carry a lint dict when the gate fires."""
        path = str(tmp_path / "bad.json")
        result = file_ops.write_file(path, '{invalid')
        assert result.error is not None
        assert result.lint is not None
        assert result.lint["status"] == "error"
        assert "JSONDecodeError" in result.lint["message"]


# ===========================================================================
# YAML syntax gate
# ===========================================================================

class TestYAMLSyntaxGate:
    """write_file() must reject invalid YAML before writing to disk."""

    def test_valid_yaml_succeeds(self, file_ops, tmp_path):
        """Valid YAML should pass through."""
        path = str(tmp_path / "config.yaml")
        content = "key: value\nlist:\n  - one\n  - two\n"
        result = file_ops.write_file(path, content)
        assert result.error is None

    def test_unclosed_quote_yaml_aborted(self, file_ops, tmp_path):
        """YAML with unclosed quote must be rejected."""
        path = str(tmp_path / "bad.yaml")
        content = 'key: "unclosed\n'
        result = file_ops.write_file(path, content)
        assert result.error is not None
        assert "Syntax check failed" in result.error
        assert not os.path.exists(path)

    def test_yml_extension_also_gated(self, file_ops, tmp_path):
        """The .yml extension must also trigger the syntax gate."""
        path = str(tmp_path / "app.yml")
        content = '{invalid: [unclosed'
        result = file_ops.write_file(path, content)
        assert result.error is not None
        assert not os.path.exists(path)


# ===========================================================================
# TOML syntax gate
# ===========================================================================

class TestTOMLSyntaxGate:
    """write_file() must reject invalid TOML before writing to disk."""

    def test_valid_toml_succeeds(self, file_ops, tmp_path):
        """Valid TOML should pass through."""
        path = str(tmp_path / "pyproject.toml")
        content = '[section]\nkey = "value"\n'
        result = file_ops.write_file(path, content)
        assert result.error is None

    def test_invalid_toml_aborted(self, file_ops, tmp_path):
        """Invalid TOML must be rejected before disk write."""
        path = str(tmp_path / "pyproject.toml")
        content = '[section\nkey = "value"'  # missing closing bracket
        result = file_ops.write_file(path, content)
        assert result.error is not None
        assert "Syntax check failed" in result.error
        assert not os.path.exists(path)


# ===========================================================================
# Non-gated extensions must NOT be affected
# ===========================================================================

class TestNonGatedExtensions:
    """Extensions outside the syntax gate must behave as before."""

    def test_python_not_gated(self, file_ops, tmp_path):
        """Python files must NOT be syntax-gated (would break test fixtures)."""
        path = str(tmp_path / "fixture.py")
        # This is not valid Python but should NOT be gated — only linted post-write.
        # The gate is intentionally scoped to JSON/YAML/TOML only.
        content = "this is not valid python syntax at all"
        # With a mock env that doesn't actually write, we only test that
        # the gate does NOT intercept .py writes.
        result = file_ops.write_file(path, content)
        # Should NOT have a syntax gate error
        assert result.error is None or "Syntax check failed" not in (result.error or "")

    def test_markdown_not_gated(self, file_ops, tmp_path):
        """Markdown files have no linter and should pass through."""
        path = str(tmp_path / "readme.md")
        content = "# Title\n\nSome **markdown** content.\n"
        result = file_ops.write_file(path, content)
        assert result.error is None or "Syntax check failed" not in (result.error or "")

    def test_txt_not_gated(self, file_ops, tmp_path):
        """Plain text files should pass through."""
        path = str(tmp_path / "notes.txt")
        content = "Just some text.\n"
        result = file_ops.write_file(path, content)
        assert result.error is None or "Syntax check failed" not in (result.error or "")


# ===========================================================================
# Regression: gate does not interfere with post-write lint delta
# ===========================================================================

class TestGateAndLintDeltaCoexist:
    """The pre-write gate and post-write lint-delta should coexist."""

    def test_valid_json_lint_still_runs(self, file_ops, tmp_path):
        """After a successful gate pass, the post-write lint should still fire."""
        path = str(tmp_path / "clean.json")
        content = '{"valid": true}'
        with patch.object(file_ops, '_check_lint_delta', wraps=file_ops._check_lint_delta) as spy:
            result = file_ops.write_file(path, content)
            # _check_lint_delta should have been called (post-write lint)
            spy.assert_called_once()

    def test_invalid_json_lint_delta_not_called(self, file_ops, tmp_path):
        """When the gate fires, _check_lint_delta should NOT be called."""
        path = str(tmp_path / "bad.json")
        content = '{broken json'
        with patch.object(file_ops, '_check_lint_delta', return_value=None) as spy:
            result = file_ops.write_file(path, content)
            spy.assert_not_called()
            assert result.error is not None


# ===========================================================================
# Linter unit sanity checks (direct function calls)
# ===========================================================================

class TestLinterFunctionsSanity:
    """Quick sanity on the in-process linter functions used by the gate."""

    def test_json_valid(self):
        ok, msg = _lint_json_inproc('{"a": 1}')
        assert ok is True
        assert msg == ""

    def test_json_invalid(self):
        ok, msg = _lint_json_inproc('{"a": 1,')
        assert ok is False
        assert "JSONDecodeError" in msg

    def test_yaml_valid(self):
        ok, msg = _lint_yaml_inproc("a: 1\n")
        assert ok is True

    def test_yaml_invalid(self):
        ok, msg = _lint_yaml_inproc('key: "unclosed\n')
        assert ok is False
        assert "YAMLError" in msg

    def test_toml_valid(self):
        ok, msg = _lint_toml_inproc("[s]\nk = 'v'\n")
        assert ok is True

    def test_toml_invalid(self):
        ok, msg = _lint_toml_inproc("[s\nk = 'v'")
        assert ok is False
