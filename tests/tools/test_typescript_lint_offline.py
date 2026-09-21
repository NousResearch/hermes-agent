"""Automatic lint must use an installed toolchain without acquiring npm packages."""

import json
import shutil

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations


@pytest.fixture
def npm_workspace(tmp_path, monkeypatch):
    if not shutil.which("npx"):
        pytest.skip("npx is not installed")
    (tmp_path / "package.json").write_text('{"name":"lint-fixture","version":"1.0.0"}')
    monkeypatch.setenv("npm_config_cache", str(tmp_path / "npm-cache"))
    monkeypatch.setenv("npm_config_registry", "http://127.0.0.1:9")
    monkeypatch.setenv("npm_config_fetch_retries", "0")
    monkeypatch.setenv("npm_config_fetch_timeout", "1000")
    source = tmp_path / "example.ts"
    source.write_text("const x: number = 1;\n")
    return tmp_path, source, ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))


def test_missing_typescript_is_skipped_without_installing(npm_workspace):
    root, source, ops = npm_workspace
    result = ops._check_lint(str(source))
    assert result.skipped, result
    assert not (root / "node_modules").exists()
    assert not (root / "package-lock.json").exists()


def test_installed_project_linter_still_reports_diagnostics(npm_workspace):
    root, source, ops = npm_workspace
    package = root / "node_modules" / "typescript"
    (package / "bin").mkdir(parents=True)
    (package / "package.json").write_text(json.dumps({
        "name": "typescript", "version": "0.0.0", "bin": {"tsc": "bin/tsc"},
    }))
    compiler = package / "bin" / "tsc"
    compiler.write_text('#!/usr/bin/env node\nconsole.log("fixture type error"); process.exit(1);\n')
    compiler.chmod(0o755)
    binaries = root / "node_modules" / ".bin"
    binaries.mkdir()
    (binaries / "tsc").symlink_to(compiler)

    result = ops._check_lint(str(source))
    assert not result.skipped
    assert not result.success
    assert "fixture type error" in result.output
