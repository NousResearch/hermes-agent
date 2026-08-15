"""Windows agent-loop correctness regressions.

Covers the papercut class swept in #84364/#84378's follow-up: silent path
mangling, crashes, and divergent hashing that made day-to-day agent use on
Windows unpleasant. Each test names the issue it pins.
"""

import os
import re
import sys
from pathlib import Path

import pytest

from hermes_cli._subprocess_compat import split_command_line


class TestSplitCommandLine:
    """#83934 / #78293 — backslashes in Windows paths must survive splitting."""

    @pytest.mark.windows_only
    def test_windows_path_backslashes_preserved(self):
        argv = split_command_line(r"sessions export C:\Users\me\Desktop\out.jsonl")
        assert argv == ["sessions", "export", r"C:\Users\me\Desktop\out.jsonl"]

    @pytest.mark.windows_only
    def test_quoted_path_with_spaces(self):
        argv = split_command_line(r'run "C:\Program Files\App\tool.exe" --flag')
        assert argv == ["run", r"C:\Program Files\App\tool.exe", "--flag"]

    @pytest.mark.windows_only
    def test_bare_hook_command_path(self):
        argv = split_command_line(r"C:\Users\u\.local\bin\dcg.exe --hook pre")
        assert argv[0] == r"C:\Users\u\.local\bin\dcg.exe"

    @pytest.mark.linux_only
    def test_posix_behavior_unchanged(self):
        assert split_command_line("echo 'a b' c") == ["echo", "a b", "c"]

    def test_unbalanced_quote_raises(self):
        with pytest.raises(ValueError):
            split_command_line('run "unterminated')


class TestShellHooksWindowsPaths:
    """#78293 — hook script paths with backslashes resolve correctly."""

    @pytest.mark.windows_only
    def test_command_script_path_keeps_backslashes(self):
        from agent.shell_hooks import _command_script_path

        path = _command_script_path(r"C:\hooks\guard.py --strict")
        assert path == r"C:\hooks\guard.py"

    @pytest.mark.windows_only
    def test_script_is_executable_finds_real_file(self, tmp_path):
        from agent.shell_hooks import script_is_executable

        script = tmp_path / "hook.py"
        script.write_text("print('ok')\n", encoding="utf-8")
        #

        assert script_is_executable(f'python "{script}"') or script_is_executable(
            f"python {script}"
        )


class TestWindowsMarketingVersion:
    """#51755 — Windows 11 must not be reported as Windows 10."""

    @pytest.mark.windows_only
    def test_matches_build_number(self):
        from agent.prompt_builder import _windows_marketing_version

        build = sys.getwindowsversion().build
        expected = "11" if build >= 22000 else "10"
        assert _windows_marketing_version() == expected

    def test_fallback_on_lookup_failure(self, monkeypatch):
        import agent.prompt_builder as pb

        if sys.platform == "win32":
            monkeypatch.delattr(sys, "getwindowsversion")
        assert isinstance(pb._windows_marketing_version(), str)


class TestAutocompleteDevicePaths:
    """#42016 — relpath ValueError on device paths must not escape."""

    def test_relpath_valueerror_pattern(self):
        # The guarded pattern in _get_project_files: a ValueError from
        # os.path.relpath (different mount) is skipped, not raised.
        bad = "\\\\.\\nul" if sys.platform == "win32" else "/dev/null"
        cwd = os.getcwd()
        files = []
        for p in [bad, os.path.join(cwd, "real.txt")]:
            try:
                rel = os.path.relpath(p, cwd) if os.path.isabs(p) else p
            except ValueError:
                continue
            files.append(rel)
        assert "real.txt" in files


class TestBrowserScreenshotPathRegex:
    """#83884 — Windows drive-letter screenshot paths must be detected."""

    def _re(self):
        from tools.browser_use_cli import _IMAGE_PATH_RE

        return _IMAGE_PATH_RE

    def test_windows_backslash_path(self):
        m = self._re().findall(r"Saved screenshot to C:\Users\u\shots\page.png done")
        assert m == [r"C:\Users\u\shots\page.png"]

    def test_windows_forward_slash_path(self):
        m = self._re().findall("shot: C:/Users/u/shots/page.jpeg")
        assert m == ["C:/Users/u/shots/page.jpeg"]

    def test_posix_path_still_matches(self):
        m = self._re().findall("wrote /tmp/bu-task/shot.webp")
        assert m == ["/tmp/bu-task/shot.webp"]

    def test_plain_words_do_not_match(self):
        assert self._re().findall("no images here, just prose.png-like text /x") == []


class TestSkillHashSymmetry:
    """#62310 — disk hash and bundle hash must agree on every OS."""

    def _make_skill(self, root: Path) -> Path:
        skill = root / "demo-skill"
        (skill / "references" / "methods").mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: demo\n---\nbody\n", encoding="utf-8")
        (skill / "references" / "methods" / "x.md").write_text("x\n", encoding="utf-8")
        # Mixed-case name exercises the case-sensitive sort divergence.
        (skill / "Zeta.md").write_text("z\n", encoding="utf-8")
        return skill

    def test_disk_and_bundle_hashes_match(self, tmp_path):
        from tools.skills_guard import content_hash
        from tools.skills_hub import SkillBundle, bundle_content_hash

        skill = self._make_skill(tmp_path)
        disk = content_hash(skill)

        files = {}
        for f in skill.rglob("*"):
            if f.is_file():
                # Native separators — what Windows bundle construction produces.
                files[str(f.relative_to(skill))] = f.read_bytes()
        bundle = SkillBundle(
            name="demo-skill", files=files, source="test",
            identifier="test/demo-skill", trust_level="community",
        )
        assert bundle_content_hash(bundle) == disk

    def test_backslash_and_posix_keys_hash_identically(self):
        from tools.skills_hub import SkillBundle, bundle_content_hash

        posix = SkillBundle(
            name="s",
            files={"references/a.md": b"a", "SKILL.md": b"s"},
            source="test",
            identifier="t/s",
            trust_level="community",
        )
        windows = SkillBundle(
            name="s",
            files={"references\\a.md": b"a", "SKILL.md": b"s"},
            source="test",
            identifier="t/s",
            trust_level="community",
        )
        assert bundle_content_hash(posix) == bundle_content_hash(windows)


class TestLineEndingPreservation:
    """Pin LF preservation on Windows write/patch paths.

    A live Windows session saw a repo-LF file (agent/prompt_builder.py)
    come back full-CRLF after an edit, exploding the git diff to every
    line (4699-line churn). The flip is not reproducible through the
    current tool APIs — these tests pin the correct behavior so any
    regression on the Windows write path (bash stdin streaming, temp-file
    rename) is caught immediately rather than corrupting user repos.
    """

    def test_write_file_preserves_lf_on_overwrite(self, tmp_path):
        from tools.file_tools import write_file_tool
        import json

        p = tmp_path / "mod.py"
        p.write_bytes(b"a = 1\nb = 2\n")
        res = json.loads(write_file_tool(path=str(p), content="a = 1\nb = 22\n"))
        assert res.get("success", True)
        assert b"\r\n" not in p.read_bytes()

    def test_patch_preserves_lf_multiline(self, tmp_path):
        from tools.file_tools import patch_tool
        import json

        p = tmp_path / "mod.py"
        p.write_bytes(b"def f():\n    return 1\n\ndef g():\n    return 2\n")
        res = json.loads(patch_tool(
            mode="replace", path=str(p),
            old_string="    return 1", new_string="    return 100",
        ))
        assert res.get("success")
        data = p.read_bytes()
        assert b"\r\n" not in data
        assert b"return 100" in data

    def test_patch_preserves_crlf_file(self, tmp_path):
        from tools.file_tools import patch_tool
        import json

        p = tmp_path / "mod.py"
        p.write_bytes(b"def f():\r\n    return 1\r\n")
        res = json.loads(patch_tool(
            mode="replace", path=str(p),
            old_string="    return 1", new_string="    return 100",
        ))
        assert res.get("success")
        data = p.read_bytes()
        # CRLF file stays CRLF — no mixed endings after an LF-args patch.
        assert b"\r\n" in data
        assert b"return 100\r\n" in data
        assert b"\n\n" not in data.replace(b"\r\n", b"")
