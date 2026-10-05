"""Tests for escape-drift auto-correction on ``patch`` (tools/escape_drift_autocorrect.py).

Contract: a verified doubled anchor is corrected only for a backslash-free replacement;
replacement backslashes require a clean resend and leave the file unchanged. Quote
correction retains its separate checks, and ``note`` reports when a correction fired.
"""

import json

import pytest

from tools.escape_drift_autocorrect import maybe_correct_escape_drift


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
        f.write_text(original)
        result = _patch_tool(path=str(f), old_string=r'sep = r"a\\b"',
                             new_string=replacement, task_id="t-drift")
        assert result.get("success") is not True
        assert "intentional backslashes" in result["error"]
        assert f.read_text() == original

    def test_backslash_mixed_call_rejected_not_mangled(self, workdir):
        # A multi-line doubled anchor also requires a clean resend of mixed new code.
        f = workdir / "a.py"
        original = 'pat = re.compile(r"\\d+")\nsep = r"a\\b"\n'
        f.write_text(original)
        replacement = r'pat = re.compile(r"\\w+")' + "\n" + r'sep = r"a\\b"' + "\n" + r'print("done\n")'
        r = _patch_tool(path=str(f), old_string=r'pat = re.compile(r"\\d+")' + "\n" + r'sep = r"a\\b"',
                        new_string=replacement,
                        task_id="t-drift")
        assert r.get("success") is not True
        assert f.read_text() == original

    def test_quote_mixed_call_rejected_not_mangled(self, workdir):
        content, old, new = _MIXED_CALLS["quote_new_line_really_escaped"]
        f = workdir / "b.py"
        f.write_text(content)
        r = _patch_tool(path=str(f), old_string=old, new_string=new, task_id="t-drift")
        assert r.get("success") is not True
        assert f.read_text() == content

    @pytest.mark.parametrize("content,old,new,expected", [
        ('pat = re.compile(r"\\d+")\n', r'pat = re.compile(r"\\d+")', 'pat = None',
         'pat = None\n'),
        ("x = 'it'\n", r"x = \'it\'", r"x = \'its\'" + "\n" + r"y = \'ok\'",
         "x = 'its'\ny = 'ok'\n"),
    ], ids=["backslash_free_replacement", "quote_escaped"])
    def test_verified_correction_with_note(self, workdir, content, old, new, expected):
        f = workdir / "c.py"
        f.write_text(content)
        r = _patch_tool(path=str(f), old_string=old, new_string=new, task_id="t-drift")
        assert r["success"] is True
        assert "escape-drift auto-corrected" in r["note"]
        assert f.read_text() == expected

    @pytest.mark.parametrize("content,old,new", [
        ('s = "a\\\\b"\n', r's = "a\\b"', r's = "a\\c"'),
        ("s = 'it\\'s'\n", r"s = 'it\'s'", r"s = 'it\'s ok'"),
        ('sep = r"a\\b"\n', r'sep = r"a\b"',
         r'sep = r"a\c"' + "\n" + r'other = "C:\\backup"'),
    ], ids=["real_double_backslash", "real_escaped_quote", "clean_resend"])
    def test_file_really_containing_escapes_is_left_alone(self, workdir, content, old, new):
        f = workdir / "d.py"
        f.write_text(content)
        r = _patch_tool(path=str(f), old_string=old, new_string=new, task_id="t-drift")
        assert r["success"] is True
        assert r.get("note") is None
        assert f.read_text() == content.replace(old, new)
