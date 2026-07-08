"""Structured-config write must fail-closed on a syntax error.

Regression tests for the defect where ``write_file()`` ran its syntax
check *after* the atomic write and only attached the lint result to the
response, never setting the top-level ``error`` key. The wrapper in
``tools/file_tools.py`` gates ``files_modified`` on ``not result.error``,
so a corrupt JSON/YAML/TOML write was reported as a success and the
invalid file landed on disk undetected (real-world: a malformed
``cron/jobs.json`` silently disabled scheduled jobs).

The fix moves the in-process syntax check for JSON/YAML/TOML *ahead* of
the atomic write and refuses the write outright on a parse failure:

* no temp file, no rename — the original file (if any) is untouched
* the top-level ``error`` key is set so the existing ``files_modified``
  gating suppresses the success report
* scope is deliberately limited to JSON/YAML/TOML. ``.py`` is excluded:
  writing a partial/in-progress Python draft is a legitimate workflow,
  and gating it would break existing write-mechanics fixtures that push
  non-Python text through ``*.py`` paths.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations


@pytest.fixture
def ops(tmp_path: Path):
    env = LocalEnvironment(cwd=str(tmp_path))
    return ShellFileOperations(env, cwd=str(tmp_path))


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------

def test_invalid_json_refused_and_error_set(ops, tmp_path: Path):
    target = tmp_path / "config.json"
    res = ops.write_file(str(target), '{"a": 1,')  # truncated / invalid
    assert res.error is not None
    assert "json" in res.error.lower() or "JSON" in res.error


def test_invalid_json_not_written_to_disk(ops, tmp_path: Path):
    target = tmp_path / "config.json"
    ops.write_file(str(target), '{"a": 1,')
    assert not target.exists()


def test_invalid_json_preserves_existing_file(ops, tmp_path: Path):
    target = tmp_path / "cron.json"
    good = '{"jobs": [1, 2, 3]}\n'
    target.write_text(good)
    res = ops.write_file(str(target), '{"jobs": [1, 2,')  # invalid
    assert res.error is not None
    # original content must survive an atomic-write refusal
    assert target.read_text() == good


def test_valid_json_still_writes(ops, tmp_path: Path):
    target = tmp_path / "config.json"
    res = ops.write_file(str(target), '{"a": 1}\n')
    assert res.error is None, res.error
    assert target.read_text() == '{"a": 1}\n'


def test_no_temp_file_leaked_on_refusal(ops, tmp_path: Path):
    ops.write_file(str(target := tmp_path / "config.json"), '{"a": 1,')
    # The gate returns before _atomic_write, so no .hermes-tmp staging
    # file is ever created next to the target.
    leaked = [p.name for p in tmp_path.iterdir() if ".hermes-tmp" in p.name]
    assert leaked == [], f"leaked temp files: {leaked}"
    assert not target.exists()


# --------------------------------------------------------------------------
# YAML
# --------------------------------------------------------------------------

def test_invalid_yaml_refused(ops, tmp_path: Path):
    target = tmp_path / "config.yaml"
    # unclosed flow mapping — a hard YAML syntax error
    res = ops.write_file(str(target), "a: [1, 2\nb: {x\n")
    assert res.error is not None
    assert not target.exists()


def test_valid_yaml_still_writes(ops, tmp_path: Path):
    target = tmp_path / "config.yml"
    res = ops.write_file(str(target), "a: 1\nb:\n  - x\n  - y\n")
    assert res.error is None, res.error
    assert target.exists()


# --------------------------------------------------------------------------
# TOML
# --------------------------------------------------------------------------

def test_invalid_toml_refused(ops, tmp_path: Path):
    target = tmp_path / "config.toml"
    res = ops.write_file(str(target), "a = = 1\n")  # invalid TOML
    assert res.error is not None
    assert not target.exists()


def test_valid_toml_still_writes(ops, tmp_path: Path):
    target = tmp_path / "config.toml"
    res = ops.write_file(str(target), 'a = 1\n[section]\nk = "v"\n')
    assert res.error is None, res.error
    assert target.exists()


# --------------------------------------------------------------------------
# Scope guard: .py is intentionally NOT gated
# --------------------------------------------------------------------------

def test_python_syntax_error_still_writes(ops, tmp_path: Path):
    """A .py file with a syntax error must still be written — partial
    drafts are a legitimate workflow and the lint result is surfaced
    separately, not turned into a hard refusal."""
    target = tmp_path / "draft.py"
    res = ops.write_file(str(target), "def f(:\n    pass\n")  # invalid Python
    assert res.error is None
    assert target.exists()
    # the syntax problem is still reported, just not as a top-level error
    assert res.lint is not None
    assert res.lint.get("status") == "error"
