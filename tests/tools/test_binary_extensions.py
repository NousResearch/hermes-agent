"""Unit tests for has_binary_extension (tools/binary_extensions.py).

has_binary_extension is a pure, I/O-free helper that file_tools uses to skip
binary files during text-based operations. These tests pin its contract:
case-insensitive extension matching against BINARY_EXTENSIONS, the deliberate
.pdf exclusion, and correct handling of paths with no extension.
"""

import pytest

from tools.binary_extensions import has_binary_extension


class TestHasBinaryExtension:
    @pytest.mark.parametrize("path", [
        "photo.png", "archive.zip", "program.exe", "clip.mp4",
        "song.mp3", "font.ttf", "lib.so", "data.sqlite",
    ])
    def test_known_binary_extensions_true(self, path):
        assert has_binary_extension(path) is True

    @pytest.mark.parametrize("path", ["IMAGE.PNG", "Clip.MP4", "Archive.ZiP"])
    def test_matching_is_case_insensitive(self, path):
        assert has_binary_extension(path) is True

    @pytest.mark.parametrize("path", [
        "script.py", "notes.txt", "README.md", "config.json", "style.css",
    ])
    def test_text_extensions_false(self, path):
        assert has_binary_extension(path) is False

    def test_pdf_is_deliberately_excluded(self):
        # .pdf is text-inspectable â€” intentionally NOT in BINARY_EXTENSIONS.
        assert has_binary_extension("report.pdf") is False

    @pytest.mark.parametrize("path", ["README", "Makefile", "LICENSE"])
    def test_no_extension_false(self, path):
        assert has_binary_extension(path) is False

    def test_multiple_dots_uses_last_extension(self):
        assert has_binary_extension("archive.tar.gz") is True

    def test_dot_in_directory_not_treated_as_extension(self):
        # Last dot is in the directory segment; the file itself has no extension.
        assert has_binary_extension("my.dir/file") is False
