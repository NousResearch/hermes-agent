"""Resolve image paths whose Unicode whitespace was flattened to U+0020.

macOS names screenshots ``Screenshot 2026-07-10 at 9.23.58<U+202F>AM.png``, using a
NARROW NO-BREAK SPACE before AM/PM. A model that lists the directory and echoes the
name back almost always emits a plain space, so ``vision_analyze`` reported
"image file not found" for a file it had just been shown.

The retry is deliberately narrow: same directory, unique match only, local backend
only. It must never resolve a genuinely missing file, and never pick between
ambiguous candidates.
"""
import asyncio

import pytest

from tools.image_source import (
    SourceNotFound,
    _flatten_unicode_spaces,
    _fuzzy_whitespace_match,
    ResolveContext,
    resolve_image_source,
)

NNBSP = "\u202f"  # NARROW NO-BREAK SPACE (macOS screenshot names)
NBSP = "\u00a0"   # NO-BREAK SPACE
PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _resolve(path):
    return asyncio.run(resolve_image_source(str(path), ResolveContext()))


class TestFlattenUnicodeSpaces:
    def test_folds_narrow_no_break_space(self):
        assert _flatten_unicode_spaces(f"9.23.58{NNBSP}AM.png") == "9.23.58 AM.png"

    def test_folds_no_break_space(self):
        assert _flatten_unicode_spaces(f"a{NBSP}b") == "a b"

    def test_leaves_ordinary_names_alone(self):
        assert _flatten_unicode_spaces("plain name.png") == "plain name.png"


class TestFuzzyWhitespaceMatch:
    def test_finds_sibling_differing_only_by_narrow_space(self, tmp_path):
        real = tmp_path / f"Screenshot at 9.23.58{NNBSP}AM.png"
        real.write_bytes(PNG)
        asked = tmp_path / "Screenshot at 9.23.58 AM.png"
        assert _fuzzy_whitespace_match(asked) == real

    def test_returns_none_when_no_sibling_matches(self, tmp_path):
        (tmp_path / "other.png").write_bytes(PNG)
        assert _fuzzy_whitespace_match(tmp_path / "missing.png") is None

    def test_returns_none_when_ambiguous(self, tmp_path):
        """Two candidates that both flatten to the request -> refuse to guess."""
        (tmp_path / f"a{NNBSP}b.png").write_bytes(PNG)
        (tmp_path / f"a{NBSP}b.png").write_bytes(PNG)
        assert _fuzzy_whitespace_match(tmp_path / "a b.png") is None

    def test_returns_none_when_parent_missing(self, tmp_path):
        assert _fuzzy_whitespace_match(tmp_path / "nope" / "x.png") is None

    def test_does_not_match_a_directory(self, tmp_path):
        (tmp_path / f"a{NNBSP}b").mkdir()
        assert _fuzzy_whitespace_match(tmp_path / "a b") is None


class TestResolveImageSource:
    def test_resolves_flattened_macos_screenshot_name(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "local")
        (tmp_path / f"Screenshot at 9.23.58{NNBSP}AM.png").write_bytes(PNG)

        resolved = _resolve(tmp_path / "Screenshot at 9.23.58 AM.png")

        assert resolved.mime == "image/png"
        assert resolved.origin == "file"
        assert resolved.data == PNG

    def test_exact_match_is_preferred_and_unaffected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "local")
        exact = tmp_path / "plain.png"
        exact.write_bytes(PNG)
        assert _resolve(exact).data == PNG

    def test_genuinely_missing_file_still_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "local")
        (tmp_path / f"Screenshot at 9.23.58{NNBSP}AM.png").write_bytes(PNG)

        with pytest.raises(SourceNotFound):
            _resolve(tmp_path / "Screenshot at 9.99.99 AM.png")

    def test_ambiguous_candidates_do_not_resolve(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "local")
        (tmp_path / f"a{NNBSP}b.png").write_bytes(PNG)
        (tmp_path / f"a{NBSP}b.png").write_bytes(PNG)

        with pytest.raises(SourceNotFound):
            _resolve(tmp_path / "a b.png")
