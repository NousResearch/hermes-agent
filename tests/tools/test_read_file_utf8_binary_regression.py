"""End-to-end regression tests for the UTF-8 'flagged as binary' class.

Covers the dupe-swarm cluster (#76886, #77047, #77842, #80221, #80251,
#80308, #80922) through the REAL local terminal backend — the transport
whose ``errors="replace"`` decode manufactured the U+FFFD that the old
text-layer heuristic misread as binary.

Byte-layer detection (``_sample_file_bytes`` + ``_is_likely_binary_bytes``)
must classify:

* valid UTF-8 cut mid-multibyte-character at the 1000-byte sample boundary
  (CJK, Cyrillic, emoji) → text
* UTF-8 with a BOM (utf-8-sig) → text
* genuine binaries (PNG/ELF magic, NUL bytes) → binary
* empty files → text
* UTF-16 (either endianness, NUL-heavy) → binary (read-only; a lossy
  errors="replace" round-trip would corrupt it — see #80717 for the
  transcode work that would lift this)
"""

import os
import shutil

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations

pytestmark = pytest.mark.skipif(
    shutil.which("head") is None or shutil.which("base64") is None,
    reason="requires POSIX shell utilities",
)


@pytest.fixture
def ops(tmp_path):
    return ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))


def _write(tmp_path, name, data: bytes):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


class TestReadFileBinaryClassification:
    def test_cjk_cut_mid_character_reads_as_text(self, ops, tmp_path):
        # 3-byte chars; byte 1000 is not a multiple of 3 → sample cuts a char.
        path = _write(tmp_path, "cjk.md", ("漢字テキスト" * 200).encode("utf-8"))
        r = ops.read_file(path)
        assert r.is_binary is False and r.error is None
        assert "漢字テキスト" in r.content

    def test_cyrillic_cut_mid_character_reads_as_text(self, ops, tmp_path):
        # 1 ASCII byte offsets the 2-byte Cyrillic chars so byte 1000 splits one.
        path = _write(tmp_path, "cyr.md", ("x" + "Привет мир\n" * 100).encode("utf-8"))
        r = ops.read_file(path)
        assert r.is_binary is False and r.error is None

    def test_utf8_sig_bom_cyrillic_reads_as_text(self, ops, tmp_path):
        # 80922: utf-8-sig BOM + Cyrillic body.
        path = _write(
            tmp_path, "bom.txt", ("Привет мир\n" * 100).encode("utf-8-sig")
        )
        r = ops.read_file(path)
        assert r.is_binary is False and r.error is None
        assert "Привет" in r.content

    def test_empty_file_reads_as_text(self, ops, tmp_path):
        path = _write(tmp_path, "empty.txt", b"")
        r = ops.read_file(path)
        assert r.is_binary is False and r.error is None

    def test_png_magic_stays_binary(self, ops, tmp_path):
        path = _write(
            tmp_path, "blob.dat", b"\x89PNG\r\n\x1a\n" + os.urandom(2048)
        )
        r = ops.read_file(path)
        assert r.is_binary is True

    def test_elf_magic_stays_binary(self, ops, tmp_path):
        path = _write(
            tmp_path, "a.out.dat", b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 64 + b"code"
        )
        r = ops.read_file(path)
        assert r.is_binary is True

    def test_nul_byte_in_text_stays_binary(self, ops, tmp_path):
        path = _write(tmp_path, "nul.txt", b"hello\x00world" + b"a" * 128)
        r = ops.read_file(path)
        assert r.is_binary is True

    @pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be"])
    def test_utf16_stays_read_only(self, ops, tmp_path, encoding):
        # UTF-16 is full of NUL bytes; a lossy replace-decode round-trip
        # would corrupt it, so it must remain flagged until a transcode
        # path (#80717) lands. This is a do-not-regress pin, either endian.
        path = _write(
            tmp_path, f"{encoding}.txt", ("hello world\n" * 50).encode(encoding)
        )
        r = ops.read_file(path)
        assert r.is_binary is True


class TestSiblingSites:
    def test_read_file_raw_cjk_cut_is_text(self, ops, tmp_path):
        # 80221: patch/V4A goes through read_file_raw — same sampling site.
        path = _write(tmp_path, "cjk.md", ("漢字テキスト" * 200).encode("utf-8"))
        r = ops.read_file_raw(path)
        assert r.is_binary is False and r.error is None

    def test_patch_replace_on_boundary_cut_cjk_file(self, ops, tmp_path):
        path = _write(
            tmp_path, "doc.md", ("漢字テキスト" * 200 + "\nEND-MARKER\n").encode("utf-8")
        )
        r = ops.patch_replace(path, "END-MARKER", "END-PATCHED")
        assert r.success is True
        assert "END-PATCHED" in open(path, encoding="utf-8").read()

    def test_search_finds_cjk_content(self, ops, tmp_path):
        # 80308: content search must not skip valid CJK files.
        _write(tmp_path, "notes.md", ("漢字テキスト\n" * 400).encode("utf-8"))
        r = ops.search("漢字", path=str(tmp_path), target="content")
        assert r.error is None
        assert r.matches
