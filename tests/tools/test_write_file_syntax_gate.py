"""Tests for the fail-closed pre-write syntax gate on write_file.

Structured formats with an in-process linter (JSON/YAML/TOML) are validated
BEFORE any bytes touch disk: a candidate write that doesn't parse is refused
outright -- nothing lands on disk -- instead of being written and merely
reported afterward via the post-write lint delta.

These run against a REAL LocalEnvironment (actual shell commands / actual
files under tmp_path), matching the existing pattern in
tests/tools/test_file_write_safety.py::TestAtomicWrite.
"""

import json
from pathlib import Path

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations


@pytest.fixture
def ops(tmp_path: Path):
    env = LocalEnvironment(cwd=str(tmp_path))
    return ShellFileOperations(env, cwd=str(tmp_path))


class TestFailClosedSyntaxGate:
    def test_invalid_json_refused_file_not_created(self, ops, tmp_path: Path):
        target = tmp_path / "config.json"
        res = ops.write_file(str(target), '{"a": 1,')  # truncated / invalid
        assert res.error is not None
        assert "json" in res.error.lower()
        assert not target.exists(), "invalid JSON must NOT be written to disk"

    def test_invalid_json_refused_existing_file_not_modified(self, ops, tmp_path: Path):
        target = tmp_path / "config.json"
        target.write_text('{"a": 1}')
        res = ops.write_file(str(target), '{"a": 1,')
        assert res.error is not None
        assert target.read_text() == '{"a": 1}', (
            "existing valid file must be left untouched by a refused write"
        )

    def test_invalid_yaml_refused_file_not_created(self, ops, tmp_path: Path):
        target = tmp_path / "config.yaml"
        res = ops.write_file(str(target), 'key: "unclosed\n')
        assert res.error is not None
        assert "yaml" in res.error.lower()
        assert not target.exists(), "invalid YAML must NOT be written to disk"

    def test_invalid_yml_extension_also_refused(self, ops, tmp_path: Path):
        target = tmp_path / "config.yml"
        res = ops.write_file(str(target), 'key: "unclosed\n')
        assert res.error is not None
        assert not target.exists()

    def test_valid_json_written_exactly(self, ops, tmp_path: Path):
        target = tmp_path / "config.json"
        content = json.dumps({"a": 1, "b": [1, 2, 3]})
        res = ops.write_file(str(target), content)
        assert res.error is None, res.error
        assert target.read_text() == content

    def test_valid_yaml_written_exactly(self, ops, tmp_path: Path):
        target = tmp_path / "config.yaml"
        content = "a: 1\nb:\n  - 1\n  - 2\n"
        res = ops.write_file(str(target), content)
        assert res.error is None, res.error
        assert target.read_text() == content

    def test_non_linted_extension_with_garbage_still_written(self, ops, tmp_path: Path):
        """Behavior for extensions with NO in-process linter is unchanged --
        garbage content is written as-is, no refusal."""
        target = tmp_path / "notes.txt"
        garbage = "{{{ not json, not yaml, not anything ]]] <<<"
        res = ops.write_file(str(target), garbage)
        assert res.error is None, res.error
        assert target.read_text() == garbage

    def test_invalid_python_is_NOT_hard_refused(self, ops, tmp_path: Path):
        """Deliberate scope decision: .py keeps the pre-existing NON-BLOCKING
        lint-delta report rather than a hard refusal (see
        ``_FAIL_CLOSED_INPROC_EXTS`` in tools/file_operations.py for why --
        this codebase's own test suite writes arbitrary non-Python content
        through *.py paths as generic write-mechanics fixtures)."""
        target = tmp_path / "broken.py"
        bad_python = "def foo(:\n    pass\n"
        res = ops.write_file(str(target), bad_python)
        assert res.error is None, res.error
        assert target.read_text() == bad_python
        # Still surfaced via the (non-blocking) lint report:
        assert res.lint is not None
        assert res.lint.get("status") == "error"
        assert "SyntaxError" in res.lint.get("output", "")

    def test_invalid_toml_refused_file_not_created(self, ops, tmp_path: Path):
        target = tmp_path / "config.toml"
        res = ops.write_file(str(target), "[section\nk = 'v'")
        assert res.error is not None
        assert not target.exists()
