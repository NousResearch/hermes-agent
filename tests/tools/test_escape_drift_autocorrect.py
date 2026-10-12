"""Tests for escape-drift auto-correction on ``patch`` (tools/escape_drift_autocorrect.py).

Contract: a verified doubled anchor is corrected only for a backslash-free replacement;
replacement backslashes require a clean resend and leave the file unchanged. Quote
correction retains its separate checks, and ``note`` reports when a correction fired.
"""

import json

import pytest

from tools.escape_drift_autocorrect import AmbiguousEscapeDriftError, maybe_correct_escape_drift


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    return tmp_path


def _patch_tool(**kwargs):
    from tools.file_tools import patch_tool
    return json.loads(patch_tool(**kwargs))


# (file content, drifted old_string, new_string that also adds a legitimately escaped line)
_MIXED_CALLS = {
    "quote_new_line_really_escaped": (
        "x = 'it'\n",
        r"x = \'it\'",
        r"x = \'it\'" + "\n" + r'y = "say \"hi\""',
    ),
}


def test_mixed_quote_escaping_is_not_corrected():
    content, old, new = _MIXED_CALLS["quote_new_line_really_escaped"]
    assert maybe_correct_escape_drift(old, new, content) == (old, new, None)


# The anchor proves only apostrophe drift; the replacement's \" has no evidence either way.
_UNPROVEN_QUOTE_KIND = (
    'message = "it' + "'" + 's fine"\n',
    r"it\'s fine",
    r"it\'s \"great\"",
)


def test_quote_kind_escaped_only_in_replacement_is_rejected():
    content, old, new = _UNPROVEN_QUOTE_KIND
    with pytest.raises(AmbiguousEscapeDriftError):
        maybe_correct_escape_drift(old, new, content)
    from tools.fuzzy_match import fuzzy_find_and_replace
    out, count, _strategy, err = fuzzy_find_and_replace(content, old, new)
    assert count == 0 and out == content
    assert "intended escaping" in err


_SHARED_CASES = [
    ('pat = r"\\d+"\n', r'pat = r"\\d+"', 'pat = None', 'pat = None\n'),
    ("x = 'it'\n", r"x = \'it\'", r"x = \'its\'", "x = 'its'\n"),
    ('pat = r"\\d+"\n', r'pat = r"\\d+"', r'pat = r"\\w+"', None),
]


@pytest.mark.parametrize("engine", ["replace", "v4a", "skill"])
@pytest.mark.parametrize("content,old,new,expected", _SHARED_CASES,
                         ids=["backslash_free", "quote", "ambiguous"])
def test_shared_correction_policy_across_tools(workdir, monkeypatch, engine, content, old, new, expected):
    target = workdir / "example.txt"
    target.write_text(content, encoding="utf-8")
    if engine == "skill":
        from tools import skill_manager_tool
        skill_dir = workdir / "skills" / "example"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: example\ndescription: Test skill.\n---\n", encoding="utf-8")
        target = skill_dir / "references" / "example.txt"
        target.parent.mkdir()
        target.write_text(content, encoding="utf-8")
        monkeypatch.setattr(skill_manager_tool, "SKILLS_DIR", skill_dir.parent)
        monkeypatch.setattr("agent.skill_utils.get_all_skills_dirs", lambda: [skill_dir.parent])
        result = json.loads(skill_manager_tool.skill_manage(
            action="patch", name="example", file_path=str(target.relative_to(skill_dir)),
            old_string=old, new_string=new, task_id="t-shared"))
    elif engine == "v4a":
        patch = (f"*** Begin Patch\n*** Update File: {target}\n@@\n"
                 + "\n".join("-" + line for line in old.splitlines()) + "\n"
                 + "\n".join("+" + line for line in new.splitlines()) + "\n*** End Patch")
        result = _patch_tool(mode="patch", patch=patch, task_id="t-shared")
    else:
        result = _patch_tool(path=str(target), old_string=old, new_string=new, task_id="t-shared")
    if expected is None:
        assert result.get("success") is not True
        assert "intentional backslashes" in result["error"]
        assert target.read_text(encoding="utf-8-sig") == content
    else:
        assert result["success"] is True
        assert target.read_text(encoding="utf-8-sig") == expected
        assert "escape-drift auto-corrected" in result["note"]


@pytest.mark.parametrize("content,old,new,expected", _SHARED_CASES,
                         ids=["backslash_free", "quote", "ambiguous"])
def test_approval_preview_agrees_with_patch(workdir, content, old, new, expected):
    from acp_adapter.edit_approval import build_edit_proposal
    target = workdir / "preview.txt"
    target.write_text(content, encoding="utf-8")
    arguments = dict(path=str(target), old_string=old, new_string=new)
    if expected is None:
        with pytest.raises(ValueError, match="intentional backslashes"):
            build_edit_proposal("patch", arguments)
        result = _patch_tool(**arguments, task_id="t-preview")
        assert result.get("success") is not True
        assert target.read_text(encoding="utf-8-sig") == content
    else:
        proposal = build_edit_proposal("patch", arguments)
        assert target.read_text(encoding="utf-8-sig") == content
        result = _patch_tool(**arguments, task_id="t-preview")
        assert result["success"] is True
        assert proposal.new_text == target.read_text(encoding="utf-8-sig") == expected


class TestPatchEndToEnd:
    @pytest.mark.parametrize("replacement", [
        r'sep = r"a\c"',
        r'sep = r"a\\c"' + "\n" + r'other = "C:\\backup"',
        r'sep = r"a\\c"' + "\n" + r'print("done\n")',
        r'sep = r"a\\c"',
    ], ids=["clean_replacement", "windows_path", "odd_escape", "all_runs_even"])
    def test_doubled_anchor_with_replacement_backslashes_is_rejected(self, workdir, replacement):
        f = workdir / "anchor.py"
        original = 'sep = r"a\\b"\n'
        f.write_text(original, encoding="utf-8")
        result = _patch_tool(path=str(f), old_string=r'sep = r"a\\b"',
                             new_string=replacement, task_id="t-drift")
        assert result.get("success") is not True
        assert "intentional backslashes" in result["error"]
        assert f.read_text(encoding="utf-8-sig") == original

    def test_backslash_mixed_call_rejected_not_mangled(self, workdir):
        # A multi-line doubled anchor also requires a clean resend of mixed new code.
        f = workdir / "a.py"
        original = 'pat = re.compile(r"\\d+")\nsep = r"a\\b"\n'
        f.write_text(original, encoding="utf-8")
        replacement = r'pat = re.compile(r"\\w+")' + "\n" + r'sep = r"a\\b"' + "\n" + r'print("done\n")'
        r = _patch_tool(path=str(f), old_string=r'pat = re.compile(r"\\d+")' + "\n" + r'sep = r"a\\b"',
                        new_string=replacement,
                        task_id="t-drift")
        assert r.get("success") is not True
        assert f.read_text(encoding="utf-8-sig") == original

    def test_quote_kind_escaped_only_in_replacement_is_rejected(self, workdir):
        content, old, new = _UNPROVEN_QUOTE_KIND
        f = workdir / "e.py"
        f.write_text(content, encoding="utf-8")
        r = _patch_tool(path=str(f), old_string=old, new_string=new, task_id="t-drift")
        assert r.get("success") is not True
        assert f.read_text(encoding="utf-8-sig") == content

    def test_quote_mixed_call_rejected_not_mangled(self, workdir):
        content, old, new = _MIXED_CALLS["quote_new_line_really_escaped"]
        f = workdir / "b.py"
        f.write_text(content, encoding="utf-8")
        r = _patch_tool(path=str(f), old_string=old, new_string=new, task_id="t-drift")
        assert r.get("success") is not True
        assert f.read_text(encoding="utf-8-sig") == content

    @pytest.mark.parametrize("content,old,new,expected", [
        ('pat = re.compile(r"\\d+")\n', r'pat = re.compile(r"\\d+")', 'pat = None',
         'pat = None\n'),
        ("x = 'it'\n", r"x = \'it\'", r"x = \'its\'" + "\n" + r"y = \'ok\'",
         "x = 'its'\ny = 'ok'\n"),
    ], ids=["backslash_free_replacement", "quote_escaped"])
    def test_verified_correction_with_note(self, workdir, content, old, new, expected):
        f = workdir / "c.py"
        f.write_text(content, encoding="utf-8")
        r = _patch_tool(path=str(f), old_string=old, new_string=new, task_id="t-drift")
        assert r["success"] is True
        assert "escape-drift auto-corrected" in r["note"]
        assert f.read_text(encoding="utf-8-sig") == expected

    @pytest.mark.parametrize("content,old,new", [
        ('s = "a\\\\b"\n', r's = "a\\b"', r's = "a\\c"'),
        ("s = 'it\\'s'\n", r"s = 'it\'s'", r"s = 'it\'s ok'"),
        ('sep = r"a\\b"\n', r'sep = r"a\b"',
         r'sep = r"a\c"' + "\n" + r'other = "C:\\backup"'),
    ], ids=["real_double_backslash", "real_escaped_quote", "clean_resend"])
    def test_file_really_containing_escapes_is_left_alone(self, workdir, content, old, new):
        f = workdir / "d.py"
        f.write_text(content, encoding="utf-8")
        r = _patch_tool(path=str(f), old_string=old, new_string=new, task_id="t-drift")
        assert r["success"] is True
        assert r.get("note") is None
        assert f.read_text(encoding="utf-8-sig") == content.replace(old, new)

    @pytest.mark.parametrize("escaped_lines", [
        [r'    pattern  = r"\\d+"'],
        [r'    pattern  = r"\\d+"', r'    pattern   = r"\\d+"'],
    ], ids=["one_escaped_candidate", "two_escaped_candidates"])
    def test_correction_refuses_when_the_sent_anchor_also_matches(self, workdir, escaped_lines):
        # The escaped text the caller sent is in the file a whitespace difference away, so the
        # de-escaped exact match is not proof of drift: refuse instead of editing either line.
        content = "".join(f"def e{i}():\n{line}\n" for i, line in enumerate(escaped_lines))
        content += "def plain():\n" + r'    pattern = r"\d+"' + "\n"
        f = workdir / "f.py"
        f.write_text(content, encoding="utf-8")
        r = _patch_tool(path=str(f), old_string=r'pattern = r"\\d+"',
                        new_string="pattern = None", task_id="t-drift")
        assert r.get("success") is not True
        assert f.read_text(encoding="utf-8-sig") == content
