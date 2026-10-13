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

    def test_leading_and_trailing_unicode_space_is_stripped_before_lookup(
        self, tmp_path, monkeypatch,
    ):
        """``resolve_image_source`` strips the source first; U+202F is whitespace.

        Pinned so nobody mistakes the strip for the fuzzy retry: a trailing NNBSP
        never reaches ``_fuzzy_whitespace_match`` at all.
        """
        monkeypatch.setenv("TERMINAL_ENV", "local")
        exact = tmp_path / "plain.png"
        exact.write_bytes(PNG)
        assert _resolve(f"{NNBSP}{exact}{NNBSP}").data == PNG

    def test_internal_narrow_space_survives_the_strip(self, tmp_path, monkeypatch):
        """The macOS case is an INTERNAL U+202F; only the retry can rescue it."""
        monkeypatch.setenv("TERMINAL_ENV", "local")
        (tmp_path / f"shot at 9.23.58{NNBSP}AM.png").write_bytes(PNG)
        assert _resolve(tmp_path / "shot at 9.23.58 AM.png").data == PNG


class TestCredentialGuardStillApplies:
    def test_guard_receives_the_fuzzy_resolved_path(self, tmp_path, monkeypatch):
        """The retry's target must go through raise_if_read_blocked, not around it."""
        import agent.file_safety as fs

        monkeypatch.setenv("TERMINAL_ENV", "local")
        real = tmp_path / f"a{NNBSP}b.png"
        real.write_bytes(PNG)

        seen = []
        monkeypatch.setattr(fs, "raise_if_read_blocked", lambda p: seen.append(p))

        _resolve(tmp_path / "a b.png")

        assert seen == [str(real)], (
            "the guard must be handed the path the retry actually resolved to"
        )

    def test_guard_can_still_block_a_fuzzy_resolved_path(self, tmp_path, monkeypatch):
        import agent.file_safety as fs
        from tools.image_source import SourceUnsafe

        monkeypatch.setenv("TERMINAL_ENV", "local")
        (tmp_path / f"a{NNBSP}b.png").write_bytes(PNG)

        def deny(_path):
            raise ValueError("Access denied: test")

        monkeypatch.setattr(fs, "raise_if_read_blocked", deny)

        with pytest.raises(SourceUnsafe):
            _resolve(tmp_path / "a b.png")

    def test_no_blocked_basename_can_be_fabricated_by_flattening(self):
        """A blocked name has no preimage but itself.

        ``_flatten_unicode_spaces`` only maps Zs -> U+0020. No blocked basename
        contains a Zs codepoint, so no *other* filename can flatten onto one and
        be picked up by the retry.
        """
        from agent.file_safety import _BLOCKED_PROJECT_ENV_BASENAMES as blocked

        assert blocked, "sanity: the denylist is non-empty"
        for name in blocked:
            assert _flatten_unicode_spaces(name) == name, name
