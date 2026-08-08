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

    def test_parse_numstat_basic(self):
        from tools.diff_analyze import _parse_numstat
        numstat = "10\t5\tfile.py\n3\t1\tanother.py\n"
        stats = _parse_numstat(numstat)
        assert stats["files_changed"] == 2
        assert stats["insertions"] == 13
        assert stats["deletions"] == 6

    def test_parse_numstat_binary_file(self):
        from tools.diff_analyze import _parse_numstat
        numstat = "-\t-\timage.png\n4\t2\tsrc/main.py\n"
        stats = _parse_numstat(numstat)
        assert stats["files_changed"] == 2
        assert stats["insertions"] == 4
        assert stats["deletions"] == 2

    def test_parse_numstat_empty(self):
        from tools.diff_analyze import _parse_numstat
        assert _parse_numstat("") == {"files_changed": 0, "insertions": 0, "deletions": 0}

    def test_parse_numstat_single_file(self):
        from tools.diff_analyze import _parse_numstat
        stats = _parse_numstat("42\t0\tpackage.json")
        assert stats == {"files_changed": 1, "insertions": 42, "deletions": 0}

    def test_diff_analyze_uses_numstat_for_stats(self, monkeypatch):
        from tools.diff_analyze import diff_analyze
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.chdir(tmpdir)
            call_count = 0
            def mock_run(args, **kwargs):
                nonlocal call_count
                call_count += 1
                if "--numstat" in args:
                    return MagicMock(stdout="10\t5\tfile.py", returncode=0, stderr="")
                return MagicMock(stdout="diff --git a/file.py b/file.py\n+line", returncode=0, stderr="")
            with patch("subprocess.run", side_effect=mock_run):
                data = json.loads(diff_analyze())
                assert data["success"] is True
                assert data["stats"]["files_changed"] == 1
                assert data["stats"]["insertions"] == 10
                assert data["stats"]["deletions"] == 5


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