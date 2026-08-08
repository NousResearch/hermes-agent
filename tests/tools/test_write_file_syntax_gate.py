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


    def test_valid_json_written_exactly(self, ops, tmp_path: Path):
        target = tmp_path / "config.json"
        content = json.dumps({"a": 1, "b": [1, 2, 3]})
        res = ops.write_file(str(target), content)
        assert res.error is None, res.error
        assert target.read_text() == content


    def test_invalid_python_is_applied_but_fails_validation(self, ops, tmp_path: Path):
        """A Python write lands, but cannot be reported as a completed edit."""
        target = tmp_path / "broken.py"
        bad_python = "def foo(:\n    pass\n"
        res = ops.write_file(str(target), bad_python)
        assert target.read_text() == bad_python
        assert res.applied is True
        assert res.error is not None
        assert "VALIDATION FAILED AFTER EDIT" in res.error
        assert res.validated is False
        assert res.lint is not None
        assert res.lint.get("status") == "error"
        assert "SyntaxError" in res.lint.get("output", "")

    def test_patch_with_new_indentation_error_is_applied_but_fails(self, ops, tmp_path: Path):
        target = tmp_path / "deploy.py"
        target.write_text("def deploy():\n    return True\n")

        res = ops.patch_replace(
            str(target),
            "    return True",
            "return True",
        )

        assert target.read_text() == "def deploy():\nreturn True\n"
        assert res.success is False
        assert res.applied is True
        assert res.validated is False
        assert "VALIDATION FAILED AFTER EDIT" in res.error
        assert "IndentationError" in res.lint["output"]


    def test_custom_tagged_yaml_is_valid_and_written(self, ops, tmp_path: Path):
        """Application-defined tags (CloudFormation !Sub/!Ref, Ansible !vault)
        are valid YAML syntax; only the *consumer* defines their constructors.
        The gate is syntax-only and must let them through."""
        target = tmp_path / "template.yaml"
        content = (
            "Resources:\n"
            "  Bucket:\n"
            "    Type: AWS::S3::Bucket\n"
            "    Properties:\n"
            "      BucketName: !Sub '${AWS::StackName}-bucket'\n"
        )
        res = ops.write_file(str(target), content)
        assert res.error is None, res.error
        assert target.read_text() == content
