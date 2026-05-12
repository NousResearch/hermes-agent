"""Tests for diff_analyze tool."""

import json
import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock


class TestDiffAnalyze:
    def test_check_requirements(self):
        from tools.diff_analyze import check_diff_analyze_requirements
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = check_diff_analyze_requirements()
            assert result is True

    def test_check_requirements_git_not_available(self):
        from tools.diff_analyze import check_diff_analyze_requirements
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            result = check_diff_analyze_requirements()
            assert result is False

    def test_diff_analyze_basic(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.chdir(tmpdir)
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    stdout="diff --git a/test.py b/test.py\n--- a/test.py\n+++ b/test.py\n+print('hello')",
                    returncode=0,
                    stderr="",
                )
                from tools.diff_analyze import diff_analyze
                output = diff_analyze()
                data = json.loads(output)
                assert data["success"] is True

    def test_diff_analyze_with_base_target(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.chdir(tmpdir)
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    stdout="diff --git a/test.py b/test.py\n+new line",
                    returncode=0,
                    stderr="",
                )
                from tools.diff_analyze import diff_analyze
                output = diff_analyze(base="main", target="feature")
                data = json.loads(output)
                assert data["success"] is True
                assert data["base"] == "main"
                assert data["target"] == "feature"

    def test_diff_analyze_summary_mode(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.chdir(tmpdir)
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    stdout="1 file changed, 10 insertions, 5 deletions",
                    returncode=0,
                    stderr="",
                )
                from tools.diff_analyze import diff_analyze
                output = diff_analyze(summary=True)
                data = json.loads(output)
                assert data["success"] is True
                assert "summary" in data

    def test_diff_analyze_full_mode(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.chdir(tmpdir)
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    stdout="diff --git a/test.py b/test.py\n+print('hello')",
                    returncode=0,
                    stderr="",
                )
                from tools.diff_analyze import diff_analyze
                output = diff_analyze(summary=False)
                data = json.loads(output)
                assert data["success"] is True
                assert "diff" in data

    def test_diff_analyze_file_filter(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.chdir(tmpdir)
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    stdout="diff --git a/main.py b/main.py\n+code",
                    returncode=0,
                    stderr="",
                )
                from tools.diff_analyze import diff_analyze
                output = diff_analyze(file_filter="*.py")
                data = json.loads(output)
                assert data["success"] is True

    def test_parse_diff_stats(self):
        from tools.diff_analyze import _parse_diff_stats
        diff_output = "1 file changed, 10 insertions(+), 5 deletions(-)"
        stats = _parse_diff_stats(diff_output)
        assert stats["files_changed"] == 1
        assert stats["insertions"] == 10
        assert stats["deletions"] == 5


class TestDiffAnalyzeSchema:
    def test_schema_has_required_fields(self):
        from tools.diff_analyze import DIFF_ANALYZE_SCHEMA
        assert DIFF_ANALYZE_SCHEMA["name"] == "diff_analyze"
        assert "parameters" in DIFF_ANALYZE_SCHEMA
        props = DIFF_ANALYZE_SCHEMA["parameters"]["properties"]
        assert "base" in props
        assert "target" in props
        assert "file_filter" in props
        assert "summary" in props
        assert "task_id" in props