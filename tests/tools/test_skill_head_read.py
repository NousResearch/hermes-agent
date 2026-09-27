"""Head-only SKILL.md reads: correctness, escalation ladder, fallback, and the write ceiling.

Skill discovery scans every SKILL.md once per pass and only needs the routing fields in the
frontmatter (name/description/platforms/environments) — over 155 installed skills every frontmatter
closed inside the first window (median 277 B, max 3,188 B). The old path read the whole file and
sliced it (`read_text()[:4000]`), so cost scaled with file size: read time 5.8 ms vs 2.4 ms and
2,286 KB vs 593 KB read over those 155 SKILL.md. These tests pin the head read, its
escalation for long frontmatter, the whole-file fallback for unclosed frontmatter, the CRLF case
that must NOT fall back (a Windows-written SKILL.md is the population the utf-8-sig decoding exists
for), the discovery call site that must keep using it, and the frontmatter ceiling enforced at the
write boundary.
"""

import builtins
import pathlib

from agent.skill_utils import (
    SKILL_FRONTMATTER_MAX_BYTES,
    SKILL_HEAD_MAX_BYTES,
    _frontmatter_closed,
    parse_frontmatter,
    read_skill_head,
)
from tools.skill_manager_tool import _validate_frontmatter
from tools.skill_usage import _read_skill_name

BIG_BODY = "\n" + ("filler line for the body\n" * 5000)  # ~120 KB, deliberately larger than the head
CRLF_BODY = "\r\n" + ("filler line for the body\r\n" * 5000)  # same, Windows line endings


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


def _make_crlf_skill(tmp_path, extra_fm: str = "", name: str = "crlf-sample"):
    """The same skill saved with CRLF line endings (Windows editor, Hermes-managed external dir)."""
    raw = f"---\r\nname: {name}\r\ndescription: sample skill\r\n{extra_fm}---\r\n{CRLF_BODY}"
    return _make_skill(tmp_path, raw=raw)


def _forbid_whole_file_read(monkeypatch):
    """Any whole-file read on the head path is a regression, not an implementation detail."""

    def boom(self, *a, **k):
        raise AssertionError(f"whole-file read on the head path: {self}")

    monkeypatch.setattr(pathlib.Path, "read_text", boom)


def _count_bytes_read(monkeypatch, path, sink):
    """Count bytes returned by ``read`` for *path* only.

    The whole-file fallback goes through ``Path.read_text`` → ``io.open``, so it stays out of the
    count and the assertion below measures the head read alone.
    """
    real_open = builtins.open

    class _Counting:
        __slots__ = ("_fh",)

        def __init__(self, fh):
            self._fh = fh

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._fh.close()
            return False

        def read(self, n=-1):
            data = self._fh.read(n)
            sink.append(len(data))
            return data

    def counting_open(file, *a, **k):
        fh = real_open(file, *a, **k)
        return _Counting(fh) if str(file) == str(path) else fh

    monkeypatch.setattr(builtins, "open", counting_open)


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


def test_frontmatter_closure_predicate_agrees_with_the_parser_on_crlf():
    """A CRLF file closes with ``---\\r\\n``. ``parse_frontmatter`` accepts that (``\\s*``), so the
    closure predicate must too — otherwise every CRLF skill is reported unclosed forever and takes
    the whole-file fallback this module removes."""
    crlf = "---\r\nname: a\ndescription: d\r\n---\r\nbody\r\n"
    assert _frontmatter_closed(crlf) is True
    assert parse_frontmatter(crlf)[0].get("name") == "a"
    assert _frontmatter_closed("\ufeff---\r\nname: a\r\n---\r\nbody") is True
    assert _frontmatter_closed("---\nname: a\n---\nbody\n") is True
    assert _frontmatter_closed("no frontmatter at all") is True
    assert _frontmatter_closed("---\nname: a\n") is False  # still unclosed: escalation stays live


def test_head_read_stops_at_the_fence_on_a_crlf_skill(tmp_path, monkeypatch):
    p = _make_crlf_skill(tmp_path)
    _forbid_whole_file_read(monkeypatch)  # a CRLF fence must not trigger the whole-file fallback
    head = read_skill_head(p)
    assert parse_frontmatter(head)[0].get("name") == "crlf-sample"
    assert len(head) < p.stat().st_size


def test_head_read_crlf_escalates_but_never_reads_past_the_cap(tmp_path, monkeypatch):
    """The cap is a ceiling on bytes read, so the fallback warning quotes what was really read."""
    raw = "---\r\nname: broken\r\n" + ("y" * (SKILL_HEAD_MAX_BYTES + 5000))
    p = tmp_path / "huge" / "SKILL.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(raw.encode("utf-8"))

    read_bytes = []
    _count_bytes_read(monkeypatch, p, read_bytes)
    out = read_skill_head(p)

    assert len(read_bytes) > 1, "an unclosed frontmatter must still escalate"
    assert sum(read_bytes) <= SKILL_HEAD_MAX_BYTES, f"read {sum(read_bytes)} B, past the cap"
    assert len(out) > SKILL_HEAD_MAX_BYTES, "unclosed frontmatter must still resolve in full"


def test_validate_frontmatter_accepts_crlf_content():
    """The write boundary and the read path must agree about CRLF."""
    assert _validate_frontmatter("---\r\nname: crlf\ndescription: short and fine\r\n---\r\nbody\r\n") is None


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


def test_discovery_call_site_uses_the_head_read(tmp_path, monkeypatch):
    """The wiring, not only the helper: reverting the call site to ``[:4000]`` must fail here.

    ``read_skill_head``'s own tests cover the helper, but a whole-file read parked at the
    discovery call site (`_read_skill_text(skill_md)[:4000]`) would leave them all green —
    the change would silently stop applying while the helper stayed correct.
    """
    from tools import skills_tool

    skill = tmp_path / "skills" / "wired-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: wired-skill\ndescription: wired through discovery\n---\n" + BIG_BODY, encoding="utf-8")
    monkeypatch.setattr(skills_tool, "_skill_search_dirs", lambda: ([], [tmp_path / "skills"], None))
    monkeypatch.setattr(skills_tool, "_SKILLS_CACHE", {})
    _forbid_whole_file_read(monkeypatch)  # a whole-file read on the discovery path IS the regression

    found = skills_tool._find_all_skills(skip_disabled=True)
    assert [s["name"] for s in found] == ["wired-skill"]
