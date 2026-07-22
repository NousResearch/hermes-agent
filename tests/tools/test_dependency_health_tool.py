"""Tests for tools/dependency_health_tool.py.

Standalone and filesystem-based: only stdlib, pytest, the tool module,
and temporary fixture files are used.  Tests stay offline and never call
pip or npm.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pytest

import tools.dependency_health_tool as dep_health_tool

DependencyError = dep_health_tool.FAILURE_CODE


def _seed_pyproject(path: Path, content: str) -> None:
    (path / "pyproject.toml").write_text(dedent(content), encoding="utf-8")


def _seed_requirements(path: Path, content: str) -> None:
    (path / "requirements.txt").write_text(dedent(content), encoding="utf-8")


def _seed_package_json(path: Path, content: str) -> None:
    (path / "package.json").write_text(dedent(content), encoding="utf-8")


def _seed_package_lock(path: Path, content: str) -> None:
    (path / "package-lock.json").write_text(dedent(content), encoding="utf-8")


# ---------------------------------------------------------------------------
# check_fn behavior
# ---------------------------------------------------------------------------
class TestCheckDependencyHealthRequirements:
    def test_true_with_pyproject(self, tmp_path: Path):
        _seed_pyproject(tmp_path, " ")
        with patch.object(dep_health_tool, "_resolve_project_root", return_value=tmp_path):
            assert dep_health_tool.check_dependency_health_requirements() is True

    def test_true_with_requirements(self, tmp_path: Path):
        _seed_requirements(tmp_path, "pytest\n")
        with patch.object(dep_health_tool, "_resolve_project_root", return_value=tmp_path):
            assert dep_health_tool.check_dependency_health_requirements() is True

    def test_true_with_package_json(self, tmp_path: Path):
        _seed_package_json(tmp_path, "{}")
        with patch.object(dep_health_tool, "_resolve_project_root", return_value=tmp_path):
            assert dep_health_tool.check_dependency_health_requirements() is True

    def test_false_when_no_manifests(self, tmp_path: Path):
        with patch.object(dep_health_tool, "_resolve_project_root", return_value=tmp_path):
            assert dep_health_tool.check_dependency_health_requirements() is False

    def test_false_when_path_is_not_a_directory(self, tmp_path: Path):
        weird = tmp_path / ".git"
        weird.mkdir()
        with patch.object(dep_health_tool, "_resolve_project_root", return_value=weird):
            assert dep_health_tool.check_dependency_health_requirements() is False

    def test_true_when_only_uv_lock(self, tmp_path: Path):
        (tmp_path / "uv.lock").write_text("", encoding="utf-8")
        with patch.object(dep_health_tool, "_resolve_project_root", return_value=tmp_path):
            assert dep_health_tool.check_dependency_health_requirements() is True

    def test_failure_when_only_malformed_uv_lock(self, tmp_path: Path):
        (tmp_path / "uv.lock").write_text("this is not valid toml: [", encoding="utf-8")
        with patch.object(dep_health_tool, "_resolve_project_root", return_value=tmp_path):
            result = dep_health_tool.run()

        assert result["kind"] == "failure"
        assert result["code"] == DependencyError
        assert "uv.lock" in result["error"]

    def test_false_when_only_package_lock(self, tmp_path: Path):
        _seed_package_lock(tmp_path, '{"name":"demo"}')
        with patch.object(dep_health_tool, "_resolve_project_root", return_value=tmp_path):
            result = dep_health_tool.run()

        assert result["kind"] == "success"
        assert any("package.json" in n for n in result["data"]["notices"])


# ---------------------------------------------------------------------------
# Success envelope shape
# ---------------------------------------------------------------------------
class TestSuccessEnvelope:
    def test_success_with_pyproject_only(self, tmp_path: Path):
        _seed_pyproject(
            tmp_path,
            """
            [project]
            name = "demo"
            version = "0.1.0"
            dependencies = ["aiohttp", "click>=8", "rich"]
            optional-dependencies.dev = ["pytest"]
            """,
        )
        with patch.object(dep_health_tool, "_resolve_project_root", return_value=tmp_path):
            result = dep_health_tool.run()

        assert result["kind"] == "success"
        assert isinstance(result["data"], dict)
        assert result["data"]["python_dependencies_count"] == 4
        assert result["data"]["node_dependencies_count"] == 0
        assert isinstance(result["data"]["lockfile_status"], dict)
        assert isinstance(result["data"]["notices"], list)

    def test_success_with_package_json_and_lock(self, tmp_path: Path):
        _seed_package_json(
            tmp_path,
            """
            {
              "dependencies": {"express": "^4.0"},
              "devDependencies": {"jest": "^29.0"}
            }
            """,
        )
        _seed_package_lock(
            tmp_path,
            """
            {
              "packages": {
                "": {"name": "demo"},
                "node_modules/express": {"version": "4.0.1"},
                "node_modules/jest": {"version": "29.0.1"}
              }
            }
            """,
        )
        with patch.object(dep_health_tool, "_resolve_project_root", return_value=tmp_path):
            result = dep_health_tool.run()

        assert result["kind"] == "success"
        # package-lock.json `packages` count includes the root mapping entry.
        assert result["data"]["node_dependencies_count"] == 3
        assert result["data"]["python_dependencies_count"] == 0
        assert result["data"]["lockfile_status"]["package-lock.json"]["freshness"] == "ok"
        assert result["data"]["lockfile_status"]["package.json"]["present"] is True

    def test_success_with_requirements_txt(self, tmp_path: Path):
        _seed_requirements(tmp_path, "requests==2.31.0\n# comment\nhttpx>=0.27\n")
        with patch.object(dep_health_tool, "_resolve_project_root", return_value=tmp_path):
            result = dep_health_tool.run()

        assert result["kind"] == "success"
        assert result["data"]["python_dependencies_count"] == 2
        assert result["data"]["node_dependencies_count"] == 0

    def test_success_lockfile_status_map_shape(self, tmp_path: Path):
        _seed_package_json(tmp_path, "{}")
        with patch.object(dep_health_tool, "_resolve_project_root", return_value=tmp_path):
            result = dep_health_tool.run()

        assert result["kind"] == "success"
        assert result["data"]["lockfile_status"]
        assert "package.json" in result["data"]["lockfile_status"] or "package-lock.json" in result["data"]["lockfile_status"]

    def test_notices_list_is_capped(self, tmp_path: Path):
        many_deps = "\n".join([f"unpinned-{i}\n" for i in range(50)])
        _seed_requirements(tmp_path, many_deps)
        with patch.object(dep_health_tool, "_resolve_project_root", return_value=tmp_path):
            result = dep_health_tool.run()

        assert result["kind"] == "success"
        assert len(result["data"]["notices"]) <= 6


# ---------------------------------------------------------------------------
# Failure on malformed manifest
# ---------------------------------------------------------------------------
class TestMalformedManifestFailure:
    def test_malformed_pyproject_toml(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("this is not valid toml: [", encoding="utf-8")
        with patch.object(dep_health_tool, "_resolve_project_root", return_value=tmp_path):
            result = dep_health_tool.run()

        assert result["kind"] == "failure"
        assert result["code"] == DependencyError
        assert "pyproject.toml" in result["error"]

    def test_malformed_package_json(self, tmp_path: Path):
        (tmp_path / "package.json").write_text("{bad json", encoding="utf-8")
        with patch.object(dep_health_tool, "_resolve_project_root", return_value=tmp_path):
            result = dep_health_tool.run()

        assert result["kind"] == "failure"
        assert result["code"] == DependencyError
        assert "package.json" in result["error"]


# ---------------------------------------------------------------------------
# Notice content / outdated markers
# ---------------------------------------------------------------------------
class TestOutdatedMarkers:
    def test_unpinned_requirements_notice(self, tmp_path: Path):
        _seed_requirements(tmp_path, "requests==2.31.0\nhttpx\nrich\n")
        with patch.object(dep_health_tool, "_resolve_project_root", return_value=tmp_path):
            result = dep_health_tool.run()

        assert result["kind"] == "success"
        notice_text = "\n".join(result["data"]["notices"])
        assert "unpinned" in notice_text

    def test_vcs_requirement_notice(self, tmp_path: Path):
        _seed_requirements(tmp_path, "-e git+https://github.com/example/example.git#egg=example\n")
        with patch.object(dep_health_tool, "_resolve_project_root", return_value=tmp_path):
            result = dep_health_tool.run()

        assert result["kind"] == "success"
        assert any("VCS dependency" in n for n in result["data"]["notices"])

    def test_pyproject_toml_present_notice(self, tmp_path: Path):
        _seed_requirements(tmp_path, "pytest\n")
        # no uv.lock created
        with patch.object(dep_health_tool, "_resolve_project_root", return_value=tmp_path):
            result = dep_health_tool.run()

        assert result["kind"] == "success"
        assert any("without uv.lock" in n for n in result["data"]["notices"])

    def test_package_lock_without_package_json_notice(self, tmp_path: Path):
        _seed_package_lock(tmp_path, '{"name":"demo"}')
        with patch.object(dep_health_tool, "_resolve_project_root", return_value=tmp_path):
            result = dep_health_tool.run()

        assert result["kind"] == "success"
        assert any("package.json" in n for n in result["data"]["notices"])


# ---------------------------------------------------------------------------
# check_fn negative behavior in non-dir path
# ---------------------------------------------------------------------------
class TestCheckFnNegative:
    def test_check_fn_false_outside_valid_project(self, tmp_path: Path):
        weird = tmp_path / ".git"
        weird.mkdir()
        with patch.object(dep_health_tool, "_resolve_project_root", return_value=weird):
            assert dep_health_tool.check_dependency_health_requirements() is False
