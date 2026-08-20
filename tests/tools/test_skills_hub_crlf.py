"""Focused regression tests for quarantine_bundle CRLF symmetry.

Demonstrates the Windows text-mode newline translation bug and the fix:
the quarantine write path must prevent implicit CRLF translation so the
bytes on disk equal the bytes whose content hash is expected.
"""

from __future__ import annotations

import pytest

from tools.skills_hub import quarantine_bundle


@pytest.fixture
def skills_root(tmp_path, monkeypatch):
    """Isolate the skills dir (and thus the quarantine root) under tmp_path."""
    import tools.skills_hub as sh

    root = tmp_path / "skills"
    root.mkdir()
    monkeypatch.setattr(sh, "_skills_dir", lambda: root)
    return root


def _bundle(files):
    from tools.skills_hub import SkillBundle

    return SkillBundle(
        name="crlf-demo",
        files=files,
        source="github",
        identifier="owner/repo/crlf-demo",
        trust_level="community",
    )


class TestQuarantineCRLFSymmetry:
    def test_quarantine_write_preserves_lf_bytes(self, skills_root):
        """On Windows, writing through a text-mode file would translate
        ``\\n`` to ``\\r\\n``; the quarantine write path must preserve the
        exact in-memory bytes so the on-disk hash matches the bundle hash."""
        files = {"SKILL.md": "line one\nline two\n"}
        bundle = _bundle(files)
        dest = quarantine_bundle(bundle)

        # The written bytes must match the in-memory bytes exactly.
        written = (dest / "SKILL.md").read_bytes()
        assert written == b"line one\nline two\n", (
            f"quarantine wrote {written!r}; expected LF-only bytes"
        )
        assert b"\r\n" not in written

    def test_quarantine_hash_matches_bundle_hash(self, skills_root):
        """The on-disk quarantine content hash must equal the bundle content
        hash — otherwise an update check after quarantine would see a
        mismatch and report a false update."""
        from tools.skills_hub import bundle_content_hash
        from tools.skills_guard import content_hash

        files = {"SKILL.md": "a\nb\nc\n", "notes.txt": "x\ny\n"}
        bundle = _bundle(files)
        dest = quarantine_bundle(bundle)

        assert content_hash(dest) == bundle_content_hash(bundle)

    def test_crlf_input_bytes_preserved_as_is(self, skills_root):
        """If the bundle itself carries CRLF bytes, they are preserved —
        the fix only disables *implicit* translation, it never rewrites
        content."""
        files = {"SKILL.md": "a\r\nb\r\n"}
        bundle = _bundle(files)
        dest = quarantine_bundle(bundle)
        written = (dest / "SKILL.md").read_bytes()
        assert written == b"a\r\nb\r\n"
