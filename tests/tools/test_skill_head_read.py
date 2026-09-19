"""Head-only SKILL.md reads: correctness, escalation ladder, fallback, and the write ceiling.

Skill discovery scans every SKILL.md once per pass and only needs the routing fields in the
frontmatter (name/description/platforms/environments) — over 131 installed skills the last of them
ended within 422 bytes. The old path read the whole file and sliced it (`read_text()[:4000]`), so
cost scaled with file size: 3.03ms vs 0.76ms over 131 skills. These tests pin the head read, its
escalation for long frontmatter, the whole-file fallback for unclosed frontmatter, and the
frontmatter ceiling enforced at the write boundary.
"""

import pathlib

import pytest

from agent.skill_utils import (
    SKILL_FRONTMATTER_MAX_BYTES,
    SKILL_HEAD_MAX_BYTES,
    parse_frontmatter,
    read_skill_head,
)
from tools.skill_manager_tool import _validate_frontmatter
from tools.skill_usage import _read_skill_name

BIG_BODY = "\n" + ("filler line for the body\n" * 5000)  # ~120 KB, deliberately larger than the head


def _make_skill(tmp_path, extra_fm: str = "", body: str = BIG_BODY, name: str = "head-sample",
                bom: bool = False, raw: str | None = None):
    p = tmp_path / "sample" / "SKILL.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    if raw is None:
        raw = f"---\nname: {name}\ndescription: sample skill\n{extra_fm}---\n{body}"
    if bom:
        raw = "\ufeff" + raw
    p.write_text(raw, encoding="utf-8")
    return p


def _forbid_whole_file_read(monkeypatch):
    """Any whole-file read on the head path is a regression, not an implementation detail."""
    real = pathlib.Path.read_text

    def boom(self, *a, **k):
        raise AssertionError(f"whole-file read on the head path: {self}")

    monkeypatch.setattr(pathlib.Path, "read_text", boom)
    return real


def test_head_read_short_frontmatter_never_reads_the_whole_file(tmp_path, monkeypatch):
    p = _make_skill(tmp_path)
    _forbid_whole_file_read(monkeypatch)
    head = read_skill_head(p)
    fm, _body = parse_frontmatter(head)
    assert fm.get("name") == "head-sample"
    assert len(head) < p.stat().st_size  # only the head was fetched


def test_head_read_escalates_for_frontmatter_beyond_the_first_window(tmp_path, monkeypatch):
    extra = "metadata:\n" + "".join(f"  key{i}: {'x' * 200}\n" for i in range(40))  # ~8 KB frontmatter
    p = _make_skill(tmp_path, extra_fm=extra)
    _forbid_whole_file_read(monkeypatch)
    head = read_skill_head(p)
    fm, _body = parse_frontmatter(head)
    assert fm.get("name") == "head-sample"
    assert fm.get("metadata"), "escalated read must expose the long metadata block"


def test_head_read_falls_back_to_whole_file_when_frontmatter_never_closes(tmp_path, monkeypatch):
    p = _make_skill(tmp_path, raw="---\nname: broken\n" + ("y" * (SKILL_HEAD_MAX_BYTES + 5000)))
    calls = []
    real = pathlib.Path.read_text

    def spy(self, *a, **k):
        calls.append(str(self))
        return real(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "read_text", spy)
    out = read_skill_head(p)
    assert calls, "unclosed frontmatter must still resolve via a whole-file read"
    assert len(out) > SKILL_HEAD_MAX_BYTES


def test_head_read_tolerates_a_bom(tmp_path, monkeypatch):
    p = _make_skill(tmp_path, bom=True)
    _forbid_whole_file_read(monkeypatch)
    fm, _body = parse_frontmatter(read_skill_head(p))
    assert fm.get("name") == "head-sample"


def test_head_read_without_frontmatter_returns_head_without_raising(tmp_path, monkeypatch):
    p = _make_skill(tmp_path, raw="# Just a document\n\n" + BIG_BODY)
    _forbid_whole_file_read(monkeypatch)
    head = read_skill_head(p)
    assert head.startswith("# Just a document")
    assert len(head) < p.stat().st_size


def test_head_read_missing_file_is_empty_not_an_error(tmp_path):
    assert read_skill_head(tmp_path / "nope" / "SKILL.md") == ""


def test_read_skill_name_uses_the_head_path_and_keeps_its_fallback(tmp_path, monkeypatch):
    p = _make_skill(tmp_path, name="named-skill")
    _forbid_whole_file_read(monkeypatch)
    assert _read_skill_name(p, fallback="dir-name") == "named-skill"

    no_name = _make_skill(tmp_path, raw="---\ndescription: no name field\n---\n" + BIG_BODY)
    assert _read_skill_name(no_name, fallback="dir-name") == "dir-name"


def test_head_read_matches_a_whole_file_parse(tmp_path):
    """Equivalence: whatever the full-file parse yields, the head read must yield too."""
    p = _make_skill(tmp_path, extra_fm="platforms: [linux]\nenvironments: [wsl]\n")
    full_fm, _full_body = parse_frontmatter(p.read_text(encoding="utf-8-sig", errors="replace"))
    head_fm, _head_body = parse_frontmatter(read_skill_head(p))
    for field in ("name", "description", "platforms", "environments"):
        assert head_fm.get(field) == full_fm.get(field)


def test_validate_frontmatter_rejects_oversized_frontmatter(tmp_path):
    extra = "metadata:\n" + "".join(f"  key{i}: {'x' * 300}\n" for i in range(20))  # > 4096 B frontmatter
    content = f"---\nname: bloated\ndescription: too much metadata\n{extra}---\nbody"
    assert len(content[: content.index("\n---", 4) + 4].encode()) > SKILL_FRONTMATTER_MAX_BYTES
    err = _validate_frontmatter(content, new_skill=True)
    assert err and "Frontmatter is" in err and str(SKILL_FRONTMATTER_MAX_BYTES) in err.replace(",", "")


def test_validate_frontmatter_still_allows_editing_an_existing_oversized_skill():
    """The ceiling is a creation-time rule: refusing edits would deadlock existing skills —
    their author could not even shrink the frontmatter back under the limit."""
    extra = "metadata:\n" + "".join(f"  key{i}: {'x' * 300}\n" for i in range(20))
    content = f"---\nname: legacy-bloated\ndescription: legacy\n{extra}---\nbody"
    assert _validate_frontmatter(content) is None


def test_validate_frontmatter_accepts_a_normal_skill():
    assert _validate_frontmatter("---\nname: fine\ndescription: short and fine\n---\nbody") is None
