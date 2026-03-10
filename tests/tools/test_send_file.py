"""Tests for the send_file tool and file_transfer layer.

All tests use mocks -- no real Docker/SSH/Modal environments are needed.
"""

import gzip
import base64
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.file_transfer import (
    validate_file_path,
    detect_mime_type,
    get_file_cache_dir,
    cleanup_file_cache,
    extract_file_from_sandbox,
    MAX_FILE_SIZE,
)
from tools.send_file_tool import send_file_tool, SEND_FILE_SCHEMA


# =========================================================================
# Validation
# =========================================================================

class TestFileTransferValidation:
    """Tests for validate_file_path()."""

    def test_valid_absolute_path(self):
        assert validate_file_path("/workspace/report.csv") is None

    def test_empty_path(self):
        assert validate_file_path("") is not None

    def test_null_bytes(self):
        assert validate_file_path("/workspace/file\x00.csv") is not None

    def test_path_traversal_dotdot(self):
        result = validate_file_path("/workspace/../etc/passwd")
        assert result is not None
        assert "traversal" in result.lower()

    def test_relative_path_rejected(self):
        result = validate_file_path("relative/path.csv")
        assert result is not None
        assert "absolute" in result.lower()

    def test_valid_deep_path(self):
        assert validate_file_path("/workspace/reports/2024/q1/data.csv") is None

    def test_shell_injection_patterns_rejected(self):
        """Paths with shell injection patterns should be rejected."""
        dangerous_paths = [
            "/workspace/$(whoami).csv",
            "/workspace/`id`.csv",
            "/workspace/file;rm -rf /.csv",
            "/workspace/file|cat /etc/passwd.csv",
            "/workspace/file&&whoami.csv",
            "/workspace/file||echo pwned.csv",
        ]
        for p in dangerous_paths:
            result = validate_file_path(p)
            assert result is not None, f"Should reject: {p}"
            assert "dangerous" in result.lower()

    def test_path_with_spaces_accepted(self):
        """Paths with spaces (but no injection patterns) should be accepted."""
        assert validate_file_path("/workspace/my report.csv") is None
        assert validate_file_path("/workspace/data files/report 2024.csv") is None

    def test_legitimate_special_filenames_accepted(self):
        """Parentheses, $, >, < in filenames should be accepted when not
        forming injection patterns."""
        assert validate_file_path("/workspace/report (final).csv") is None
        assert validate_file_path("/workspace/results_$summary.csv") is None
        assert validate_file_path("/workspace/data > 100.csv") is None
        assert validate_file_path("/workspace/notes & ideas.txt") is None


# =========================================================================
# MIME Detection
# =========================================================================

class TestMimeDetection:
    """Tests for detect_mime_type()."""

    def test_csv(self):
        assert detect_mime_type("data.csv") == "text/csv"

    def test_json(self):
        assert detect_mime_type("config.json") == "application/json"

    def test_python(self):
        assert detect_mime_type("script.py") == "text/x-python"

    def test_markdown(self):
        assert detect_mime_type("README.md") == "text/markdown"

    def test_unknown_extension(self):
        result = detect_mime_type("file.xyz123")
        assert result == "application/octet-stream"

    def test_pdf(self):
        result = detect_mime_type("report.pdf")
        assert "pdf" in result.lower()


# =========================================================================
# File Cache
# =========================================================================

class TestFileCache:
    """Tests for file cache management."""

    def test_get_file_cache_dir_creates_dir(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "file_cache"
        monkeypatch.setattr("tools.file_transfer.FILE_CACHE_DIR", cache_dir)
        result = get_file_cache_dir()
        assert result.exists()
        assert result == cache_dir

    def test_cleanup_removes_old_files(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "file_cache"
        cache_dir.mkdir()
        monkeypatch.setattr("tools.file_transfer.FILE_CACHE_DIR", cache_dir)

        # Create an old file
        old_file = cache_dir / "old_file.txt"
        old_file.write_text("old")
        # Set mtime to 48 hours ago
        old_time = time.time() - (48 * 3600)
        os.utime(old_file, (old_time, old_time))

        # Create a recent file
        new_file = cache_dir / "new_file.txt"
        new_file.write_text("new")

        removed = cleanup_file_cache(max_age_hours=24)
        assert removed == 1
        assert not old_file.exists()
        assert new_file.exists()


# =========================================================================
# Local Extraction
# =========================================================================

class TestLocalExtraction:
    """Tests for local backend file extraction."""

    @patch.dict(os.environ, {"TERMINAL_ENV": "local"})
    def test_extract_existing_file(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr("tools.file_transfer.FILE_CACHE_DIR", cache_dir)

        # Create source file
        source = tmp_path / "data.csv"
        source.write_text("col1,col2\n1,2\n")

        result = extract_file_from_sandbox(str(source), "test-task")
        assert result["success"] is True
        assert result["filename"] == "data.csv"
        assert result["mime_type"] == "text/csv"
        assert result["size"] > 0
        assert os.path.exists(result["host_path"])

    @patch.dict(os.environ, {"TERMINAL_ENV": "local"})
    def test_extract_missing_file(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr("tools.file_transfer.FILE_CACHE_DIR", cache_dir)

        result = extract_file_from_sandbox("/nonexistent/file.csv", "test-task")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @patch.dict(os.environ, {"TERMINAL_ENV": "local"})
    def test_extract_too_large(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr("tools.file_transfer.FILE_CACHE_DIR", cache_dir)
        monkeypatch.setattr("tools.file_transfer.MAX_FILE_SIZE", 10)

        source = tmp_path / "big.bin"
        source.write_bytes(b"x" * 100)

        result = extract_file_from_sandbox(str(source), "test-task")
        assert result["success"] is False
        assert "too large" in result["error"].lower()

    @patch.dict(os.environ, {"TERMINAL_ENV": "local"})
    def test_extract_directory(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr("tools.file_transfer.FILE_CACHE_DIR", cache_dir)

        dir_path = tmp_path / "mydir"
        dir_path.mkdir()

        result = extract_file_from_sandbox(str(dir_path), "test-task")
        assert result["success"] is False
        assert "directory" in result["error"].lower()


# =========================================================================
# Docker Extraction (mocked)
# =========================================================================

class TestDockerExtraction:
    """Tests for Docker backend extraction with mocked subprocess."""

    @patch.dict(os.environ, {"TERMINAL_ENV": "docker"})
    def test_docker_cp_success(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr("tools.file_transfer.FILE_CACHE_DIR", cache_dir)

        mock_env = SimpleNamespace(_container_id="abc123")
        mock_envs = {"task1": mock_env}

        def side_effect(cmd, **kwargs):
            dest = cmd[-1]
            Path(dest).write_text("docker file content")
            return SimpleNamespace(returncode=0, stderr="")

        with patch("tools.terminal_tool._active_environments", mock_envs), \
             patch("tools.file_transfer.subprocess.run", side_effect=side_effect):
            result = extract_file_from_sandbox("/workspace/out.csv", "task1")

        assert result["success"] is True
        assert result["filename"] == "out.csv"

    @patch.dict(os.environ, {"TERMINAL_ENV": "docker"})
    def test_docker_no_environment(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr("tools.file_transfer.FILE_CACHE_DIR", cache_dir)

        with patch("tools.terminal_tool._active_environments", {}):
            result = extract_file_from_sandbox("/workspace/out.csv", "nonexistent")

        assert result["success"] is False
        assert "no active" in result["error"].lower()

    @patch.dict(os.environ, {"TERMINAL_ENV": "docker"})
    def test_docker_cp_failure(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr("tools.file_transfer.FILE_CACHE_DIR", cache_dir)

        mock_env = SimpleNamespace(_container_id="abc123")
        mock_envs = {"task1": mock_env}
        mock_run = MagicMock(return_value=SimpleNamespace(returncode=1, stderr="No such file"))

        with patch("tools.terminal_tool._active_environments", mock_envs), \
             patch("tools.file_transfer.subprocess.run", mock_run):
            result = extract_file_from_sandbox("/workspace/missing.csv", "task1")

        assert result["success"] is False


# =========================================================================
# SSH Extraction (mocked)
# =========================================================================

class TestSSHExtraction:
    """Tests for SSH backend extraction with mocked subprocess."""

    @patch.dict(os.environ, {"TERMINAL_ENV": "ssh"})
    def test_scp_success(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr("tools.file_transfer.FILE_CACHE_DIR", cache_dir)

        mock_env = SimpleNamespace(
            control_socket="/tmp/ctrl.sock",
            host="remote.host",
            user="testuser",
            port=22,
        )
        mock_envs = {"task1": mock_env}

        def side_effect(cmd, **kwargs):
            dest = cmd[-1]
            Path(dest).write_text("remote file content")
            return SimpleNamespace(returncode=0, stderr="")

        with patch("tools.terminal_tool._active_environments", mock_envs), \
             patch("tools.file_transfer.subprocess.run", side_effect=side_effect):
            result = extract_file_from_sandbox("/home/user/data.csv", "task1")

        assert result["success"] is True
        assert result["filename"] == "data.csv"

    @patch.dict(os.environ, {"TERMINAL_ENV": "ssh"})
    def test_scp_failure(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr("tools.file_transfer.FILE_CACHE_DIR", cache_dir)

        mock_env = SimpleNamespace(
            control_socket="/tmp/ctrl.sock",
            host="remote.host",
            user="testuser",
            port=22,
        )
        mock_envs = {"task1": mock_env}
        mock_run = MagicMock(return_value=SimpleNamespace(returncode=1, stderr="Connection refused"))

        with patch("tools.terminal_tool._active_environments", mock_envs), \
             patch("tools.file_transfer.subprocess.run", mock_run):
            result = extract_file_from_sandbox("/home/user/data.csv", "task1")

        assert result["success"] is False
        assert "scp failed" in result["error"].lower()

    @patch.dict(os.environ, {"TERMINAL_ENV": "ssh"})
    def test_scp_quotes_remote_path(self, tmp_path, monkeypatch):
        """SCP remote path should be quoted for spaces/special chars."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr("tools.file_transfer.FILE_CACHE_DIR", cache_dir)

        mock_env = SimpleNamespace(
            control_socket="/tmp/ctrl.sock",
            host="remote.host",
            user="testuser",
            port=22,
        )
        mock_envs = {"task1": mock_env}
        captured_cmds = []

        def side_effect(cmd, **kwargs):
            captured_cmds.append(cmd)
            dest = cmd[-1]
            Path(dest).write_text("content")
            return SimpleNamespace(returncode=0, stderr="")

        with patch("tools.terminal_tool._active_environments", mock_envs), \
             patch("tools.file_transfer.subprocess.run", side_effect=side_effect):
            result = extract_file_from_sandbox("/home/user/my report (final).csv", "task1")

        assert result["success"] is True
        # The remote path in scp command should be properly quoted
        scp_cmd = captured_cmds[0]
        # Find the arg that contains user@host:
        remote_arg = [a for a in scp_cmd if "testuser@remote.host:" in a]
        assert len(remote_arg) == 1, f"Expected user@host: in scp cmd: {scp_cmd}"
        remote_arg = remote_arg[0]
        # Verify quoting: path with spaces/parens should be shell-quoted
        assert "'" in remote_arg or "\\" in remote_arg


# =========================================================================
# Base64 Extraction (mocked)
# =========================================================================

class TestBase64Extraction:
    """Tests for gzip+base64 fallback extraction."""

    @patch.dict(os.environ, {"TERMINAL_ENV": "modal"})
    def test_base64_roundtrip(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr("tools.file_transfer.FILE_CACHE_DIR", cache_dir)

        original_data = b"Hello, this is test data for base64 extraction!"
        compressed = gzip.compress(original_data)
        encoded = base64.b64encode(compressed).decode("ascii")

        mock_env = MagicMock()
        mock_env.execute.side_effect = [
            {"output": str(len(original_data)), "returncode": 0},  # stat
            {"output": encoded, "returncode": 0},  # gzip+base64
        ]
        mock_envs = {"task1": mock_env}

        with patch("tools.terminal_tool._active_environments", mock_envs):
            result = extract_file_from_sandbox("/workspace/test.txt", "task1")

        assert result["success"] is True
        with open(result["host_path"], "rb") as f:
            assert f.read() == original_data

    @patch.dict(os.environ, {"TERMINAL_ENV": "modal"})
    def test_base64_file_not_found(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr("tools.file_transfer.FILE_CACHE_DIR", cache_dir)

        mock_env = MagicMock()
        mock_env.execute.return_value = {"output": "", "returncode": 1}
        mock_envs = {"task1": mock_env}

        with patch("tools.terminal_tool._active_environments", mock_envs):
            result = extract_file_from_sandbox("/workspace/missing.txt", "task1")

        assert result["success"] is False


# =========================================================================
# Send File Handler
# =========================================================================

class TestSendFileHandler:
    """Tests for the send_file tool handler."""

    def test_missing_path(self):
        result = json.loads(send_file_tool({}))
        assert "error" in result

    def test_invalid_path(self):
        result = json.loads(send_file_tool({"path": "relative/path.csv"}))
        assert "error" in result

    @patch.dict(os.environ, {"TERMINAL_ENV": "local"})
    def test_success_cli_mode(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr("tools.file_transfer.FILE_CACHE_DIR", cache_dir)
        monkeypatch.delenv("HERMES_GATEWAY", raising=False)

        # Create source file
        source = tmp_path / "report.md"
        source.write_text("# Report\nSome data")

        # Change CWD to tmp_path so the file is copied there
        original_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            result = json.loads(send_file_tool(
                {"path": str(source), "message": "Here's your report"},
                task_id="test",
            ))
            assert result["success"] is True
            assert result["delivered_via"] == "cli"
            assert os.path.exists(result["path"])
        finally:
            os.chdir(original_cwd)

    @patch.dict(os.environ, {"TERMINAL_ENV": "local", "HERMES_GATEWAY": "true"})
    def test_success_gateway_mode(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr("tools.file_transfer.FILE_CACHE_DIR", cache_dir)

        source = tmp_path / "data.csv"
        source.write_text("col1,col2\n1,2\n")

        raw = send_file_tool({"path": str(source)}, task_id="test")
        # Should contain FILE:<...> tag with angle brackets
        assert "FILE:<" in raw
        assert ">" in raw.split("FILE:<")[1]
        # Parse the JSON part
        json_part = raw.split("\n")[0]
        result = json.loads(json_part)
        assert result["success"] is True
        assert result["delivered_via"] == "gateway"

    @patch.dict(os.environ, {"TERMINAL_ENV": "local", "HERMES_GATEWAY": "true"})
    def test_gateway_mode_includes_caption(self, tmp_path, monkeypatch):
        """Caption should be carried in the FILE: tag via pipe separator."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr("tools.file_transfer.FILE_CACHE_DIR", cache_dir)

        source = tmp_path / "report.csv"
        source.write_text("a,b\n1,2\n")

        raw = send_file_tool(
            {"path": str(source), "message": "Your monthly report"},
            task_id="test",
        )
        # Caption should appear after pipe in the tag
        assert "|Your monthly report>" in raw

    @patch.dict(os.environ, {"TERMINAL_ENV": "local", "HERMES_GATEWAY": "true"})
    def test_gateway_caption_gt_escaped(self, tmp_path, monkeypatch):
        """Caption containing > must be escaped as \\> in the FILE tag."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr("tools.file_transfer.FILE_CACHE_DIR", cache_dir)

        source = tmp_path / "report.csv"
        source.write_text("a,b\n1,2\n")

        raw = send_file_tool(
            {"path": str(source), "message": "Results > 100"},
            task_id="test",
        )
        # > in caption must be escaped so the tag parser doesn't break
        assert r"|Results \> 100>" in raw
        # Unescaped > must NOT appear between pipe and closing bracket
        assert "|Results > 100>" not in raw

    @patch.dict(os.environ, {"TERMINAL_ENV": "local", "HERMES_GATEWAY": "true"})
    def test_gateway_path_gt_roundtrip(self):
        """Filename containing > must survive FILE tag roundtrip.

        On Linux > is valid in filenames; on Windows it isn't, so we mock
        the extraction result to simulate a cache path with > in the name.
        """
        fake_host = "/tmp/cache/abc123_report > final.csv"
        with patch("tools.file_transfer.extract_file_from_sandbox",
                   return_value={
                       "success": True,
                       "host_path": fake_host,
                       "filename": "report > final.csv",
                       "mime_type": "text/csv",
                       "size": 42,
                   }):
            raw = send_file_tool(
                {"path": "/workspace/report > final.csv", "message": "Report"},
                task_id="test",
            )

        # The FILE tag must have > in path escaped
        assert "\\>" in raw

        # Parse the FILE tag back with extract_files (bypass allowlist for unit test)
        from gateway.platforms.base import BasePlatformAdapter

        class _FakePath:
            def __init__(self, p): self._p = p
            def is_file(self): return True
            def __str__(self): return self._p

        with patch("tools.file_transfer.is_safe_file_path",
                   side_effect=lambda p: _FakePath(p)):
            files, cleaned = BasePlatformAdapter.extract_files(raw)
        assert len(files) == 1
        path, caption = files[0]
        # Path must contain the original > after unescape
        assert path == fake_host
        assert caption == "Report"

    def test_extraction_failure(self):
        with patch("tools.file_transfer.extract_file_from_sandbox",
                   return_value={"success": False, "error": "Docker not running"}):
            result = json.loads(send_file_tool({"path": "/workspace/file.csv"}, task_id="test"))
        assert "error" in result
        assert "docker" in result["error"].lower()


# =========================================================================
# Schema
# =========================================================================

class TestSendFileSchema:
    """Tests for the send_file schema structure."""

    def test_schema_name(self):
        assert SEND_FILE_SCHEMA["name"] == "send_file"

    def test_required_params(self):
        assert "path" in SEND_FILE_SCHEMA["parameters"]["required"]

    def test_has_message_param(self):
        props = SEND_FILE_SCHEMA["parameters"]["properties"]
        assert "message" in props

    def test_registry_registered(self):
        from tools.registry import registry
        # Import the module to trigger registration
        import tools.send_file_tool  # noqa: F401
        assert "send_file" in registry.get_all_tool_names()
