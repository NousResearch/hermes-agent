"""A ``skill_manage`` write leaves the bytes it did not touch alone (#153752).

A SKILL.md written with a Windows editor (or copied from a Windows install) uses CRLF endings and
may carry a UTF-8 BOM. Rewriting one line of it used to rewrite every line ending — the whole file
became LF and lost its BOM — because the write path normalised both on the way through. The style of
the file on disk is now kept: CRLF stays CRLF, a BOM stays a BOM, and only the edited text changes.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from tools.skill_manager_tool import _edit_skill, _patch_skill, _write_file

BOM = b"\xef\xbb\xbf"


@pytest.fixture
def skills(tmp_path):
    """Search only the temp skills root — never the real ~/.hermes/skills/."""
    with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
        yield tmp_path


def _skill_text(name: str = "crlf-skill") -> str:
    return (
        "---\n"
        f"name: {name}\n"
        "description: Probe skill for a byte-preserving write.\n"
        "---\n"
        "\n"
        "# Probe\n"
        "\n"
        "Line one.\n"
        "Line two.\n"
        "Line three.\n"
    )


def _make(skills: Path, *, crlf: bool = True, bom: bool = False) -> Path:
    """Write the probe skill with the requested on-disk style; return its SKILL.md path."""
    skill_dir = skills / "crlf-skill"
    skill_dir.mkdir()
    body = _skill_text()
    if crlf:
        body = body.replace("\n", "\r\n")
    path = skill_dir / "SKILL.md"
    path.write_bytes((BOM if bom else b"") + body.encode("utf-8"))
    return path


def _expected_after_patch() -> bytes:
    return _skill_text().replace("Line two.", "Line 2.").encode("utf-8")


def test_patch_keeps_crlf_line_endings_and_changes_only_the_patched_line(skills):
    path = _make(skills)

    result = _patch_skill("crlf-skill", "Line two.", "Line 2.")

    assert result["success"] is True
    raw = path.read_bytes()
    assert raw.count(b"\r\n") == 10
    assert raw.count(b"\n") == 10, "a CRLF file must not gain bare-LF lines"
    assert raw.replace(b"\r\n", b"\n") == _expected_after_patch()


def test_patch_keeps_the_utf8_bom(skills):
    path = _make(skills, bom=True)

    result = _patch_skill("crlf-skill", "Line two.", "Line 2.")

    assert result["success"] is True
    raw = path.read_bytes()
    assert raw.startswith(BOM)
    assert raw.count(b"\r\n") == 10
    assert raw[len(BOM):].replace(b"\r\n", b"\n") == _expected_after_patch()


def test_patch_leaves_an_lf_file_lf(skills):
    """The mirror control: keeping the on-disk style must not invent CRLF in an LF skill."""
    path = _make(skills, crlf=False)

    result = _patch_skill("crlf-skill", "Line two.", "Line 2.")

    assert result["success"] is True
    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert raw == _expected_after_patch()


def test_full_rewrite_keeps_crlf_endings(skills):
    path = _make(skills)

    result = _edit_skill("crlf-skill", _skill_text().replace("Line three.", "Line 3."))

    assert result["success"] is True
    raw = path.read_bytes()
    assert raw.replace(b"\r\n", b"\n") == _skill_text().replace("Line three.", "Line 3.").encode("utf-8")
    assert raw.count(b"\r\n") == 10


def test_write_file_keeps_the_existing_supporting_files_style(skills):
    skill_md = _make(skills)
    references = skill_md.parent / "references"
    references.mkdir()
    notes = references / "notes.md"
    notes.write_bytes(b"one\r\ntwo\r\n")

    result = _write_file("crlf-skill", "references/notes.md", "alpha\nbeta\n")

    assert result["success"] is True
    assert notes.read_bytes() == b"alpha\r\nbeta\r\n"


def test_a_new_skill_is_written_exactly_as_supplied(skills):
    from tools.skill_manager_tool import _create_skill

    result = _create_skill("fresh-skill", _skill_text("fresh-skill"))

    assert result["success"] is True
    raw = (skills / "fresh-skill" / "SKILL.md").read_bytes()
    assert raw == _skill_text("fresh-skill").encode("utf-8")
