"""Tests for tools.binary_extensions — has_binary_extension and BINARY_EXTENSIONS."""

from __future__ import annotations

import pytest

from tools.binary_extensions import BINARY_EXTENSIONS, has_binary_extension


# ============================================================================
# has_binary_extension
# ============================================================================
class TestHasBinaryExtension:
    def test_png_is_binary(self):
        assert has_binary_extension("photo.png") is True

    def test_jpg_is_binary(self):
        assert has_binary_extension("image.jpg") is True

    def test_mp4_is_binary(self):
        assert has_binary_extension("video.mp4") is True

    def test_zip_is_binary(self):
        assert has_binary_extension("archive.zip") is True

    def test_exe_is_binary(self):
        assert has_binary_extension("program.exe") is True

    def test_pdf_is_not_binary(self):
        """PDF is intentionally excluded — agents may want to inspect."""
        assert has_binary_extension("document.pdf") is False

    def test_txt_is_not_binary(self):
        assert has_binary_extension("readme.txt") is False

    def test_py_is_not_binary(self):
        assert has_binary_extension("main.py") is False

    def test_md_is_not_binary(self):
        assert has_binary_extension("README.md") is False

    def test_json_is_not_binary(self):
        assert has_binary_extension("config.json") is False

    def test_no_extension(self):
        assert has_binary_extension("Makefile") is False

    def test_dotfile_no_extension(self):
        assert has_binary_extension(".gitignore") is False

    def test_hidden_file_with_extension(self):
        """Hidden .png file — still binary."""
        assert has_binary_extension(".hidden.png") is True

    def test_case_insensitive(self):
        assert has_binary_extension("IMAGE.PNG") is True
        assert has_binary_extension("Image.Jpg") is True
        assert has_binary_extension("video.MP4") is True

    def test_empty_string(self):
        assert has_binary_extension("") is False

    def test_only_dot(self):
        """Just a dot, no extension."""
        assert has_binary_extension(".") is False

    def test_multiple_dots_takes_last(self):
        """archive.tar.gz → .gz is binary. backup.data.old → .old is not."""
        assert has_binary_extension("archive.tar.gz") is True
        assert has_binary_extension("backup.data.old") is False

    def test_all_registered_extensions(self):
        """Every extension in BINARY_EXTENSIONS should be detected."""
        for ext in BINARY_EXTENSIONS:
            assert has_binary_extension(f"file{ext}") is True, f"Failed for {ext}"

    def test_path_with_dirs(self):
        assert has_binary_extension("/path/to/file.png") is True
        assert has_binary_extension("a/b/c/file.py") is False


# ============================================================================
# BINARY_EXTENSIONS constant
# ============================================================================
class TestBinaryExtensionsSet:
    def test_is_frozenset(self):
        assert isinstance(BINARY_EXTENSIONS, frozenset)

    def test_all_start_with_dot(self):
        for ext in BINARY_EXTENSIONS:
            assert ext.startswith("."), f"{ext} does not start with dot"

    def test_all_lowercase(self):
        for ext in BINARY_EXTENSIONS:
            assert ext == ext.lower(), f"{ext} is not lowercase"

    def test_minimum_count(self):
        """Sanity check: there should be at least 50 extensions."""
        assert len(BINARY_EXTENSIONS) > 50

    def test_no_duplicates(self):
        assert len(BINARY_EXTENSIONS) == len(set(BINARY_EXTENSIONS))
