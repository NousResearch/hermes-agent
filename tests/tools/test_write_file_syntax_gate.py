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
import yaml

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations, _lint_yaml_inproc


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

    def test_multi_document_yaml_is_valid_and_written(self, ops, tmp_path: Path):
        """Multi-document streams (k8s manifests) are valid YAML *syntax* —
        the gate must not refuse them just because safe_load() would raise
        ComposerError on more than one document."""
        target = tmp_path / "manifests.yaml"
        content = "apiVersion: v1\nkind: Namespace\n---\napiVersion: v1\nkind: ConfigMap\n"
        res = ops.write_file(str(target), content)
        assert res.error is None, res.error
        assert target.read_text() == content

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


# Reporter's exact content (#65924) — a valid GitHub Actions workflow that the
# fail-closed gate wrongly refused when an internal PyYAML error surfaced.
_GHA_WORKFLOW = (
    "name: Deploy\n\n"
    "on:\n"
    "  push:\n"
    "    branches: [main]\n"
    "  workflow_dispatch:\n"
    "    inputs:\n"
    "      environment:\n"
    '        description: "Deployment target"\n'
    "        required: true\n"
    "        default: test\n"
    "        type: choice\n"
    "        options:\n"
    "          - test\n"
    "          - live\n\n"
    "jobs:\n"
    "  test:\n"
    "    runs-on: ubuntu-latest\n"
    "    steps:\n"
    "      - uses: actions/checkout@v4\n"
)


def _raise_typeerror(_content):
    # Mirrors the internal PyYAML failure from #65924 (a TypeError raised deep
    # in yaml.parse on certain inputs), which is NOT a YAMLError.
    raise TypeError('can only concatenate str (not "int") to str')


class TestYamlLintInternalError:
    """#65924: an *internal* linter/PyYAML error is not a syntax error, so the
    fail-closed write gate must not treat it as one and refuse a valid write."""

    def test_internal_typeerror_skips_not_refuses(self, monkeypatch):
        monkeypatch.setattr(yaml, "parse", _raise_typeerror)
        ok, msg = _lint_yaml_inproc("name: Deploy\n")
        assert ok is True
        assert msg == "__SKIP__"

    def test_real_yaml_syntax_error_still_fails(self):
        # The fix must not weaken detection of genuine syntax errors.
        ok, msg = _lint_yaml_inproc('key: "unclosed\n')
        assert ok is False
        assert msg.startswith("YAMLError")

    def test_valid_yaml_still_passes_clean(self):
        ok, msg = _lint_yaml_inproc("a: 1\nb: [1, 2, 3]\n")
        assert ok is True
        assert msg == ""

    def test_reporter_workflow_lints_as_valid(self):
        ok, msg = _lint_yaml_inproc(_GHA_WORKFLOW)
        assert ok is True, msg
        assert msg == ""


class TestWriteFileSurvivesInternalLintError:
    def test_internal_yaml_error_does_not_block_write(self, ops, tmp_path: Path, monkeypatch):
        # End-to-end: with the linter hitting an internal TypeError, write_file
        # must still write the .yml (#65924), not refuse it.
        monkeypatch.setattr(yaml, "parse", _raise_typeerror)
        target = tmp_path / "deploy.yml"
        res = ops.write_file(str(target), _GHA_WORKFLOW)
        assert res.error is None, res.error
        assert target.exists(), "internal linter error must not refuse a valid write"
        assert "actions/checkout@v4" in target.read_text()

    def test_reporter_workflow_writes_without_monkeypatch(self, ops, tmp_path: Path):
        target = tmp_path / "deploy.yml"
        res = ops.write_file(str(target), _GHA_WORKFLOW)
        assert res.error is None, res.error
        assert target.exists()
