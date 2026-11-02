"""Tests for the send_file tool (sandbox & local extraction, size limits, security guards)."""

import base64
import json
import os
import stat
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.registry import registry
from tools.send_file_tool import (
    SEND_FILE_SCHEMA,
    _DEFAULT_MAX_SEND_BYTES,
    _check_remote_sensitive_path,
    _extract_local_file,
    _extract_remote_file,
    _format_size,
    send_file_tool,
)


class TestSendFileSchemaAndRegistry:
    def test_schema_structure(self):
        assert SEND_FILE_SCHEMA["name"] == "send_file"
        assert "path" in SEND_FILE_SCHEMA["parameters"]["properties"]
        assert "message" in SEND_FILE_SCHEMA["parameters"]["properties"]
        assert "path" in SEND_FILE_SCHEMA["parameters"]["required"]

    def test_registered_in_registry(self):
        tool = registry.get_entry("send_file")
        assert tool is not None
        assert tool.name == "send_file"
        assert tool.toolset == "file"
        assert tool.schema == SEND_FILE_SCHEMA
        assert tool.emoji == "📤"

    def test_format_size(self):
        assert _format_size(500) == "500 B"
        assert _format_size(2048) == "2.0 KB"
        assert _format_size(5 * 1024 * 1024) == "5.0 MB"


class TestSendFileLocal:
    def test_missing_path_argument(self):
        res = send_file_tool("")
        assert "missing required field 'path'" in res

        res_none = send_file_tool("   ")
        assert "missing required field 'path'" in res_none

    def test_blocked_device_path(self):
        res = send_file_tool("/dev/urandom")
        assert "access denied for device or proc path" in res

    def test_send_local_existing_file(self, tmp_path):
        sample_file = tmp_path / "chart.png"
        sample_data = b"\x89PNG\r\n\x1a\nfake-png-binary-data"
        sample_file.write_bytes(sample_data)

        with patch("tools.send_file_tool._get_file_ops") as mock_get_ops:
            mock_ops = MagicMock()
            mock_ops.env = MagicMock()
            mock_ops.env.is_local = True
            mock_get_ops.return_value = mock_ops

            res = send_file_tool(str(sample_file), message="Here is your requested chart")

            assert "File ready for delivery: chart.png" in res
            assert "Here is your requested chart" in res
            assert "MEDIA:" in res

            # Verify cached file was written
            media_path = res.split("MEDIA:")[1].strip()
            assert os.path.exists(media_path)
            assert Path(media_path).read_bytes() == sample_data

    def test_send_local_file_not_found(self, tmp_path):
        missing = tmp_path / "does_not_exist.pdf"
        with patch("tools.send_file_tool._get_file_ops") as mock_get_ops:
            mock_ops = MagicMock()
            mock_ops.env = MagicMock()
            mock_ops.env.is_local = True
            mock_get_ops.return_value = mock_ops

            res = send_file_tool(str(missing))
            assert "File not found" in res

    def test_send_local_directory_rejected(self, tmp_path):
        with patch("tools.send_file_tool._get_file_ops") as mock_get_ops:
            mock_ops = MagicMock()
            mock_ops.env = MagicMock()
            mock_ops.env.is_local = True
            mock_get_ops.return_value = mock_ops

            res = send_file_tool(str(tmp_path))
            assert "is a directory, not a regular file" in res

    def test_send_local_fifo_rejected(self, tmp_path):
        """Regression test for Point 1: FIFOs/named pipes must be rejected before blocking read."""
        fifo_path = tmp_path / "stream.fifo"
        with patch("tools.send_file_tool._special_file_kind", return_value="a FIFO (named pipe)"):
            data, err = _extract_local_file(str(fifo_path), _DEFAULT_MAX_SEND_BYTES)
            assert data is None
            assert "is a FIFO (named pipe), not a regular file" in err

    def test_send_local_socket_rejected(self, tmp_path):
        sock_path = tmp_path / "app.sock"
        with patch("tools.send_file_tool._special_file_kind", return_value="a socket"):
            data, err = _extract_local_file(str(sock_path), _DEFAULT_MAX_SEND_BYTES)
            assert data is None
            assert "is a socket, not a regular file" in err

    def test_send_local_oversized_file(self, tmp_path):
        large_file = tmp_path / "huge.dat"
        large_file.write_bytes(b"x" * 100)
        with patch("os.fstat") as mock_fstat:
            st = MagicMock()
            st.st_mode = stat.S_IFREG
            st.st_size = _DEFAULT_MAX_SEND_BYTES + 1024
            mock_fstat.return_value = st

            data, err = _extract_local_file(str(large_file), _DEFAULT_MAX_SEND_BYTES)
            assert data is None
            assert "exceeds the maximum allowed transfer size" in err

    @patch("tools.send_file_tool.get_read_block_error")
    def test_send_local_sensitive_path_blocked(self, mock_block_err, tmp_path):
        mock_block_err.return_value = "Sensitive configuration file blocked"

        with patch("tools.send_file_tool._get_file_ops") as mock_get_ops:
            mock_ops = MagicMock()
            mock_ops.env = MagicMock()
            mock_ops.env.is_local = True
            mock_get_ops.return_value = mock_ops

            res = send_file_tool("/home/user/.env")
            assert "access denied for protected path" in res
            assert "Sensitive configuration file blocked" in res


class TestSendFileRemoteSandbox:
    def test_send_remote_file_success(self):
        mock_ops = MagicMock()
        mock_ops.env = MagicMock()
        mock_ops.env.is_local = False
        mock_ops.cwd = "/workspace"

        raw_content = b"PDF-1.5 fake invoice report data"
        b64_content = base64.b64encode(raw_content).decode("ascii")

        def fake_exec(cmd: str):
            res = MagicMock()
            res.returncode = 0
            res.output = (
                f"__HERMES_REALPATH__:/workspace/output/invoice.pdf\n"
                f"__HERMES_SIZE__:{len(raw_content)}\n"
                f"__HERMES_DATA__\n"
                f"{b64_content}\n"
            )
            return res

        mock_ops._exec.side_effect = fake_exec

        with patch("tools.send_file_tool._get_file_ops", return_value=mock_ops):
            res = send_file_tool("output/invoice.pdf", message="Your weekly invoice")

            assert "File ready for delivery: invoice.pdf" in res
            assert "Your weekly invoice" in res
            assert "MEDIA:" in res

            media_path = res.split("MEDIA:")[1].strip()
            assert os.path.exists(media_path)
            assert Path(media_path).read_bytes() == raw_content

    def test_send_remote_file_not_found(self):
        mock_ops = MagicMock()
        mock_ops.env = MagicMock()
        mock_ops.env.is_local = False

        def fake_exec(cmd: str):
            res = MagicMock()
            res.returncode = 0
            res.output = "__HERMES_NOT_FOUND__"
            return res

        mock_ops._exec.side_effect = fake_exec

        with patch("tools.send_file_tool._get_file_ops", return_value=mock_ops):
            res = send_file_tool("/workspace/missing.csv")
            assert "File not found in sandbox" in res

    def test_send_remote_directory_rejected(self):
        mock_ops = MagicMock()
        mock_ops.env = MagicMock()
        mock_ops.env.is_local = False

        def fake_exec(cmd: str):
            res = MagicMock()
            res.returncode = 0
            res.output = "__HERMES_DIR__"
            return res

        mock_ops._exec.side_effect = fake_exec

        with patch("tools.send_file_tool._get_file_ops", return_value=mock_ops):
            res = send_file_tool("/workspace/src")
            assert "is a directory in the sandbox" in res

    def test_send_remote_special_file_rejected(self):
        mock_ops = MagicMock()
        mock_ops.env = MagicMock()
        mock_ops.env.is_local = False

        def fake_exec(cmd: str):
            res = MagicMock()
            res.returncode = 0
            res.output = "__HERMES_SPECIAL__"
            return res

        mock_ops._exec.side_effect = fake_exec

        data, err = _extract_remote_file(mock_ops, "/workspace/my.fifo", _DEFAULT_MAX_SEND_BYTES)
        assert data is None
        assert "special (non-regular) file or FIFO" in err

    def test_send_remote_symlink_to_sensitive_target_blocked(self):
        """Regression test for Point 2: Symlinks targeting remote ~/.ssh/id_rsa or /etc/shadow must be denied."""
        mock_ops = MagicMock()
        mock_ops.env = MagicMock()
        mock_ops.env.is_local = False

        def fake_exec(cmd: str):
            res = MagicMock()
            res.returncode = 0
            # Path appears innocent (/workspace/innocent.txt) but realpath is /root/.ssh/id_rsa
            res.output = (
                "__HERMES_REALPATH__:/root/.ssh/id_rsa\n"
                "__HERMES_SIZE__:1024\n"
                "__HERMES_DATA__\n"
                "UklGRg==\n"
            )
            return res

        mock_ops._exec.side_effect = fake_exec

        data, err = _extract_remote_file(mock_ops, "innocent.txt", _DEFAULT_MAX_SEND_BYTES)
        assert data is None
        assert "Access denied for protected sandbox target" in err
        assert ".ssh" in err

    def test_send_remote_file_oversized_at_probe(self):
        mock_ops = MagicMock()
        mock_ops.env = MagicMock()
        mock_ops.env.is_local = False

        def fake_exec(cmd: str):
            res = MagicMock()
            res.returncode = 0
            res.output = f"__HERMES_SIZE__:{60 * 1024 * 1024}\n__HERMES_TOO_LARGE__:{60 * 1024 * 1024}"
            return res

        mock_ops._exec.side_effect = fake_exec

        data, err = _extract_remote_file(mock_ops, "/workspace/dump.tar", _DEFAULT_MAX_SEND_BYTES)
        assert data is None
        assert "exceeds the maximum allowed transfer size" in err

    def test_send_remote_decoded_payload_size_cap_postcondition(self):
        """Regression test for Point 2: Small at probe, but large at read -> final decoded cap catches it."""
        mock_ops = MagicMock()
        mock_ops.env = MagicMock()
        mock_ops.env.is_local = False

        # Generates > 100 bytes payload when max is 50 bytes
        oversized_data = b"A" * 1000
        b64_oversized = base64.b64encode(oversized_data).decode("ascii")

        def fake_exec(cmd: str):
            res = MagicMock()
            res.returncode = 0
            # Pretends size is small (10 bytes), but payload returns 1000 bytes
            res.output = (
                "__HERMES_REALPATH__:/workspace/dynamic.log\n"
                "__HERMES_SIZE__:10\n"
                "__HERMES_DATA__\n"
                f"{b64_oversized}\n"
            )
            return res

        mock_ops._exec.side_effect = fake_exec

        data, err = _extract_remote_file(mock_ops, "dynamic.log", max_bytes=500)
        assert data is None
        assert "exceeds the maximum allowed transfer size" in err

    def test_send_remote_tilde_expansion_check(self):
        assert _check_remote_sensitive_path("~/.ssh/id_rsa") is not None
        assert _check_remote_sensitive_path("/home/user/.aws/credentials") is not None
        assert _check_remote_sensitive_path("/etc/shadow") is not None
        assert _check_remote_sensitive_path("/workspace/report.pdf") is None

    def test_send_remote_extraction_failure(self):
        mock_ops = MagicMock()
        mock_ops.env = MagicMock()
        mock_ops.env.is_local = False

        def fake_exec(cmd: str):
            res = MagicMock()
            res.returncode = 1
            res.output = "base64: error reading file: I/O error"
            return res

        mock_ops._exec.side_effect = fake_exec

        data, err = _extract_remote_file(mock_ops, "/workspace/corrupt.bin", _DEFAULT_MAX_SEND_BYTES)
        assert data is None
        assert "Failed to extract file" in err
