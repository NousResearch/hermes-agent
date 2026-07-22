"""Tests for tools/repo_health_tool.py.

Standalone and dependency-light: only pytest, stdlib, and the tool module
itself are exercised.  Tests need temporary repos but stay offline by
patching the internal ``_git`` helper instead of executing real git commands.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.repo_health_tool import (
    _git,
    _snapshot,
    check_repo_health_requirements,
    register,
    relay_failure,
    relay_success,
    repo_health_tool,
)
from tools.validation_helper import Success


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_repo(path: Path, files=None):
    """Create a minimal repo-like directory tree. Not a real git repo."""
    path.mkdir(parents=True, exist_ok=True)
    for rel, content in (files or {}).items():
        child = path / rel
        child.parent.mkdir(parents=True, exist_ok=True)
        child.write_text(content, encoding="utf-8")


def _make_git_fake(
    responses: dict[tuple[str, ...], str | None],
    default: str | None = None,
):
    """Return a mock for ``tools.repo_health_tool._git``.

    Caller provides a mapping of exact argument lists to stdout strings.
    Because mock callables receive both ``args`` and ``repo`` positional
    args, the lookup key is derived from the first positional arg only.
    """
    def _fake(args, repo):  # noqa: ARG001
        key = tuple(args)
        if key in responses:
            return responses[key]
        return default

    return _fake


# ---------------------------------------------------------------------------
# check_repo_health_requirements
# ---------------------------------------------------------------------------


class TestCheckRepoHealthRequirements:
    def test_true_when_git_available_and_inside_repo(self, tmp_path: Path):
        _seed_repo(tmp_path)
        fake = _make_git_fake({
            ("rev-parse", "--show-toplevel"): str(tmp_path),
        }, default=None)
        with patch("tools.repo_health_tool._git_available", return_value=True), \
             patch("tools.repo_health_tool._git", side_effect=fake), \
             patch("pathlib.Path.cwd", return_value=tmp_path):
            assert check_repo_health_requirements() is True

    def test_false_when_git_unavailable(self, tmp_path: Path):
        with patch("tools.repo_health_tool._git_available", return_value=False):
            assert check_repo_health_requirements() is False

    def test_false_outside_repo_when_git_available(self, tmp_path: Path):
        _seed_repo(tmp_path)
        fake = _make_git_fake({
            ("rev-parse", "--show-toplevel"): None,
        }, default=None)
        with patch("tools.repo_health_tool._git_available", return_value=True), \
             patch("tools.repo_health_tool._git", side_effect=fake), \
             patch("pathlib.Path.cwd", return_value=tmp_path):
            assert check_repo_health_requirements() is False


# ---------------------------------------------------------------------------
# Success envelope shape + cleanliness propagation
# ---------------------------------------------------------------------------


class TestSuccessEnvelope:
    def test_clean_repo_shape(self, tmp_path: Path):
        _seed_repo(tmp_path, {"README.txt": "hello"})
        fake = _make_git_fake({
            ("rev-parse", "--show-toplevel"): str(tmp_path),
            ("branch", "--show-current"): "main\n",
            ("status", "--porcelain"): "",
            ("rev-parse", "--abbrev-ref", "main@{upstream}"): None,
        })
        with patch("tools.repo_health_tool._git", side_effect=fake), \
             patch("pathlib.Path.cwd", return_value=tmp_path):
            result = repo_health_tool({})

        assert result["kind"] == "success"
        assert isinstance(result["data"], dict)
        assert result["data"]["clean"] is True
        assert result["data"]["dirty_count"] == 0
        assert result["data"]["branch"] == "main"
        assert result["data"]["untracked_count"] == 0
        assert result["data"]["tracks_upstream"] is False
        assert "ahead" not in result["data"]
        assert "behind" not in result["data"]


# ---------------------------------------------------------------------------
# Dirty-state counts
# ---------------------------------------------------------------------------


class TestDirtyCounts:
    def test_modified_staged_file_counts_as_dirty(self, tmp_path: Path):
        _seed_repo(tmp_path, {"README.txt": "hello"})
        fake = _make_git_fake({
            ("rev-parse", "--show-toplevel"): str(tmp_path),
            ("branch", "--show-current"): "main\n",
            ("status", "--porcelain"): "M README.txt\n",
            ("rev-parse", "--abbrev-ref", "main@{upstream}"): None,
        })
        with patch("tools.repo_health_tool._git", side_effect=fake), \
             patch("pathlib.Path.cwd", return_value=tmp_path):
            result = repo_health_tool({})

        assert result["kind"] == "success"
        assert result["data"]["clean"] is False
        assert result["data"]["dirty_count"] == 1
        assert result["data"]["untracked_count"] == 0

    def test_untracked_file_counts_as_dirty(self, tmp_path: Path):
        _seed_repo(tmp_path)
        fake = _make_git_fake({
            ("rev-parse", "--show-toplevel"): str(tmp_path),
            ("branch", "--show-current"): "main\n",
            ("status", "--porcelain"): "?? new_file.txt\n",
            ("rev-parse", "--abbrev-ref", "main@{upstream}"): None,
        })
        with patch("tools.repo_health_tool._git", side_effect=fake), \
             patch("pathlib.Path.cwd", return_value=tmp_path):
            result = repo_health_tool({})

        assert result["kind"] == "success"
        assert result["data"]["clean"] is False
        assert result["data"]["dirty_count"] == 1
        assert result["data"]["untracked_count"] == 1

    def test_combined_dirty_and_untracked(self, tmp_path: Path):
        _seed_repo(tmp_path, {"README.txt": "hello"})
        fake = _make_git_fake({
            ("rev-parse", "--show-toplevel"): str(tmp_path),
            ("branch", "--show-current"): "main\n",
            ("status", "--porcelain"): (
                "M README.txt\n"
                "A new.txt\n"
                "?? delta.txt\n"
            ),
            ("rev-parse", "--abbrev-ref", "main@{upstream}"): None,
        })
        with patch("tools.repo_health_tool._git", side_effect=fake), \
             patch("pathlib.Path.cwd", return_value=tmp_path):
            result = repo_health_tool({})

        assert result["kind"] == "success"
        assert result["data"]["dirty_count"] == 3
        assert result["data"]["untracked_count"] == 1


# ---------------------------------------------------------------------------
# Branch-tracking behavior
# ---------------------------------------------------------------------------


class TestBranchTracking:
    def test_ahead_behind_vs_origin(self, tmp_path: Path):
        _seed_repo(tmp_path, {"README.txt": "hello"})
        fake = _make_git_fake({
            ("rev-parse", "--show-toplevel"): str(tmp_path),
            ("branch", "--show-current"): "main\n",
            ("rev-parse", "--abbrev-ref", "main@{upstream}"): "origin/main\n",
            ("rev-list", "--left-right", "--count", "main...origin/main"): "1\t0\n",
            ("status", "--porcelain"): "",
        })
        with patch("tools.repo_health_tool._git", side_effect=fake), \
             patch("pathlib.Path.cwd", return_value=tmp_path):
            result = repo_health_tool({})

        assert result["kind"] == "success"
        assert result["data"]["tracks_upstream"] is True
        assert "ahead" in result["data"]
        assert "behind" in result["data"]
        assert result["data"]["ahead"] == 1
        assert result["data"]["behind"] == 0

    def test_ahead_ahead_of_upstream(self, tmp_path: Path):
        _seed_repo(tmp_path)
        fake = _make_git_fake({
            ("rev-parse", "--show-toplevel"): str(tmp_path),
            ("branch", "--show-current"): "feature\n",
            ("rev-parse", "--abbrev-ref", "feature@{upstream}"): "origin/feature\n",
            ("rev-list", "--left-right", "--count", "feature...origin/feature"): "3\t0\n",
            ("status", "--porcelain"): "",
        })
        with patch("tools.repo_health_tool._git", side_effect=fake), \
             patch("pathlib.Path.cwd", return_value=tmp_path):
            result = repo_health_tool({})

        assert result["kind"] == "success"
        assert result["data"]["tracks_upstream"] is True
        assert result["data"]["ahead"] == 3
        assert result["data"]["behind"] == 0

    def test_missing_upstream_omits_counts(self, tmp_path: Path):
        _seed_repo(tmp_path)
        fake = _make_git_fake({
            ("rev-parse", "--show-toplevel"): str(tmp_path),
            ("branch", "--show-current"): "main\n",
            ("rev-parse", "--abbrev-ref", "main@{upstream}"): None,
            ("status", "--porcelain"): "",
        })
        with patch("tools.repo_health_tool._git", side_effect=fake), \
             patch("pathlib.Path.cwd", return_value=tmp_path):
            result = repo_health_tool({})

        assert result["kind"] == "success"
        assert result["data"]["tracks_upstream"] is False
        assert "ahead" not in result["data"]
        assert "behind" not in result["data"]


# ---------------------------------------------------------------------------
# Missing-repo failure behavior
# ---------------------------------------------------------------------------


class TestMissingRepoFailure:
    def test_not_a_git_repo_failure_envelope(self, tmp_path: Path):
        _seed_repo(tmp_path)
        fake = _make_git_fake({
            ("rev-parse", "--show-toplevel"): None,
        })
        with patch("tools.repo_health_tool._git", side_effect=fake), \
             patch("pathlib.Path.cwd", return_value=tmp_path):
            result = repo_health_tool({})

        assert result["kind"] == "failure"
        assert result["code"] == "repo_health_error"
        assert "not inside a git repository" in result["error"]
