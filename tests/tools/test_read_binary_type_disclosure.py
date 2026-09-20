"""Tests for magic-byte type disclosure in the binary-file refusal.

"Binary file — use appropriate tools" names a recovery the model may not
have (file-only toolsets), so it thrashes. Naming the TYPE answers
what-is-this in one read: "Binary file (PNG image data, 4.0 KB)".
"""

import json
import random

import pytest

from tools.file_operations import describe_binary_file, identify_binary_bytes
from tools.file_tools import read_file_tool

_RNG = random.Random(20260810)


def _noise(n: int) -> bytes:
    return bytes(_RNG.getrandbits(8) for _ in range(n))


class TestIdentifyBinaryBytes:
    @pytest.mark.parametrize(
        "prefix,expected",
        [
            (b"\x89PNG\r\n\x1a\n", "PNG image data"),
            (b"\xff\xd8\xff", "JPEG image data"),
            (b"%PDF-", "PDF document"),
            (b"PK\x03\x04", "ZIP archive"),
            (b"\x1f\x8b", "gzip compressed data"),
            (b"\x7fELF", "ELF executable"),
            (b"MZ", "Windows PE executable"),
            (b"SQLite format 3\x00", "SQLite database"),
        ],
    )
    def test_known_signatures(self, prefix, expected):
        assert expected in identify_binary_bytes(prefix + _noise(64))

    def test_unknown_binary(self):
        assert identify_binary_bytes(b"\x00\x01\x02" * 100) == "unknown binary"

    def test_empty_sample(self):
        assert identify_binary_bytes(b"") == "unknown binary"

    def test_iso_media_requires_ftyp(self):
        # Three NULs alone are too weak; require ftyp at offset 4.
        assert identify_binary_bytes(b"\x00\x00\x00\x18ftypmp42" + _noise(32)).startswith("ISO media")
        assert identify_binary_bytes(b"\x00\x00\x00\x18AAAA" + _noise(32)) == "unknown binary"


class TestDescribeBinaryFile:
    def test_size_units(self):
        assert "512 bytes" in describe_binary_file(b"\x7fELF", 512)
        assert "4.0 KB" in describe_binary_file(b"\x7fELF", 4096)
        assert "2.0 MB" in describe_binary_file(b"\x7fELF", 2 * 1024 * 1024)

    def test_none_sample(self):
        assert "unknown binary" in describe_binary_file(None, 100)


class TestReadFileBinaryDisclosure:
    def test_lying_extension_names_real_type(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        p = tmp_path / "data.txt"  # extension says text; bytes say PNG
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + _noise(4096))
        result = json.loads(read_file_tool(str(p)))
        assert "PNG image data" in result.get("error", "")
        assert "4.1 KB" in result["error"] or "4.0 KB" in result["error"]

    def test_extensionless_binary_named(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        p = tmp_path / "mystery"
        p.write_bytes(b"\x7fELF" + _noise(1024))
        result = json.loads(read_file_tool(str(p)))
        assert "ELF executable" in result.get("error", "")

    def test_unknown_binary_still_refused(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        p = tmp_path / "junk.out"
        p.write_bytes(b"\x00\x01\x02" * 500)
        result = json.loads(read_file_tool(str(p)))
        assert "unknown binary" in result.get("error", "")

    def test_text_file_unaffected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        p = tmp_path / "ok.txt"
        p.write_text("hello world\n")
        result = json.loads(read_file_tool(str(p)))
        assert "hello world" in result.get("content", "")
