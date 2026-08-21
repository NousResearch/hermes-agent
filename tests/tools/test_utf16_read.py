"""Tests for UTF-16 text file reading (transcode to UTF-8).

Ported from MoonshotAI/kimi-code#2647: UTF-16 text files (Windows Notepad
.txt, PowerShell `>` redirects) previously tripped the binary-file guard
because the terminal env decodes stdout as UTF-8 with errors="replace",
mangling the content with U+FFFD. ShellFileOperations now probes raw bytes
via the backend's Python and transcodes UTF-16 (BOM or zero-byte parity
heuristic) to UTF-8.

These run against a real LocalEnvironment so the actual shell + subprocess
path executes (E2E, no mocks).
"""

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations


@pytest.fixture
def fops(tmp_path):
    env = LocalEnvironment(cwd=str(tmp_path))
    return ShellFileOperations(env, cwd=str(tmp_path))


def _write(tmp_path, name: str, text: str, encoding: str) -> str:
    p = tmp_path / name
    p.write_bytes(text.encode(encoding))
    return str(p)


class TestUtf16Read:
    def test_utf16le_bom(self, fops, tmp_path):
        path = _write(tmp_path, "notepad.txt", "hello\nworld\n", "utf-16-le")
        # prepend BOM manually via utf-16 (writes native-endian BOM); use explicit
        raw = "\ufefflíne one\nlíne two\n".encode("utf-16-le")
        (tmp_path / "bom-le.txt").write_bytes(raw)
        result = fops.read_file(str(tmp_path / "bom-le.txt"))
        assert result.error is None
        assert "líne one" in result.content
        assert "\ufeff" not in result.content
        assert "Transcoded from UTF-16-LE" in (result.hint or "")

    def test_utf16be_bom(self, fops, tmp_path):
        raw = "\ufeffbig endian text\nsecond line\n".encode("utf-16-be")
        (tmp_path / "bom-be.txt").write_bytes(raw)
        result = fops.read_file(str(tmp_path / "bom-be.txt"))
        assert result.error is None
        assert "big endian text" in result.content
        assert "UTF-16-BE" in (result.hint or "")

    def test_utf16le_bomless(self, fops, tmp_path):
        path = _write(tmp_path, "bomless.txt", "plain ascii saved as utf16\n", "utf-16-le")
        result = fops.read_file(path)
        assert result.error is None
        assert "plain ascii saved as utf16" in result.content

    def test_utf16le_mixed_cjk(self, fops, tmp_path):
        # Mixed Latin/CJK: CJK UTF-16 units carry no zero byte — the parity
        # heuristic must still detect from the Latin characters present.
        path = _write(tmp_path, "mixed.txt", "log: 你好世界 done\n", "utf-16-le")
        result = fops.read_file(path)
        assert result.error is None
        assert "你好世界" in result.content

    def test_crlf_normalized(self, fops, tmp_path):
        path = _write(tmp_path, "crlf.txt", "\ufeffa\r\nb\r\nc", "utf-16-le")
        result = fops.read_file(path)
        assert result.error is None
        assert result.total_lines == 3
        assert "1|a" in result.content and "2|b" in result.content

    def test_pagination(self, fops, tmp_path):
        text = "\ufeff" + "\n".join(f"line{i}" for i in range(1, 21)) + "\n"
        path = str(tmp_path / "paged.txt")
        (tmp_path / "paged.txt").write_bytes(text.encode("utf-16-le"))
        result = fops.read_file(path, offset=5, limit=3)
        assert result.error is None
        assert "5|line5" in result.content
        assert "7|line7" in result.content
        assert "line8" not in result.content
        assert result.truncated is True
        assert "offset=8" in (result.hint or "")

    def test_real_binary_still_refused(self, fops, tmp_path):
        # Zero bytes at BOTH parities → not UTF-16 → stays binary.
        p = tmp_path / "blob.dat"
        p.write_bytes(bytes([0x00, 0x01, 0x02, 0x00, 0xFF, 0x00, 0x00, 0xFE]) * 40)
        result = fops.read_file(str(p))
        assert result.is_binary is True
        assert result.error is not None

    def test_binary_extension_not_rescued(self, fops, tmp_path):
        # A .png is never probed for UTF-16 even if its bytes look like it.
        p = tmp_path / "img.png"
        p.write_bytes("fake image".encode("utf-16-le"))
        result = fops.read_file(str(p))
        assert result.is_binary is True or result.error is not None

    def test_utf8_file_unaffected(self, fops, tmp_path):
        p = tmp_path / "normal.txt"
        p.write_text("just utf-8\n", encoding="utf-8")
        result = fops.read_file(str(p))
        assert result.error is None
        assert "just utf-8" in result.content
        assert "Transcoded" not in (result.hint or "")
