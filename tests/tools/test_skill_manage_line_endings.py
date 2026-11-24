"""Regression: skill_manage and atomic_write_text must not rewrite line endings.

On native Windows, ``atomic_write_text`` opened its temp file in text mode without
``newline=""``, so every "\\n" became "\\r\\n". ``skill_manage`` reads targets with
universal newlines, so a one-word patch to an LF file rewrote the whole file as CRLF.

The same universal-newline read means a patch to a CRLF file is written back as LF
wherever text mode does not translate (POSIX, or Windows once the helper is verbatim),
so skill_manage must restore the target's own line ending.

Plain ``unittest`` so it runs without extra dependencies; pytest collects it too.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

FRONTMATTER = "---\nname: {name}\ndescription: Use when testing line endings. Checks bytes.\n---\n"
BODY = "\n# Title\n\nalpha line\nbravo line\ncharlie line\n\n## Notes\n\ndelta line\necho line\n"


def _skill_text(name: str, eol: str) -> bytes:
    return (FRONTMATTER.format(name=name) + BODY).replace("\n", eol).encode("utf-8")


class _HermesHomeCase(unittest.TestCase):
    """Isolated HERMES_HOME so nothing outside the temp dir is touched."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="eol-regress-")
        self.home = Path(self._tmp) / "hermes-home"
        (self.home / "skills").mkdir(parents=True)
        self._old_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = str(self.home)

    def tearDown(self):
        if self._old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = self._old_home
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_skill(self, name: str, eol: str) -> Path:
        skill_dir = self.home / "skills" / "testing" / name
        skill_dir.mkdir(parents=True)
        path = skill_dir / "SKILL.md"
        path.write_bytes(_skill_text(name, eol))
        return path


class AtomicWriteTextNewlines(_HermesHomeCase):
    def test_content_is_written_verbatim(self):
        from utils import atomic_write_text

        for payload in ("a\nb\nc\n", "a\r\nb\r\nc\r\n", "mixed\nline\r\nend"):
            target = self.home / "verbatim.txt"
            atomic_write_text(target, payload)
            self.assertEqual(target.read_bytes(), payload.encode("utf-8"), repr(payload))


class SkillManagePreservesLineEndings(_HermesHomeCase):
    def _patch(self, name: str):
        from tools import skill_manager_tool as smt

        result = smt._patch_skill(name, "bravo line", "BRAVO line")
        self.assertTrue(result.get("success"), result)

    def _assert_only_hunk_changed(self, before: bytes, after: bytes):
        old, new = b"bravo line", b"BRAVO line"
        self.assertEqual(before.count(old), 1)
        idx = before.index(old)
        # Bytes outside the hunk are unchanged, byte for byte.
        self.assertEqual(after[:idx], before[:idx])
        self.assertEqual(after[idx:idx + len(new)], new)
        self.assertEqual(after[idx + len(new):], before[idx + len(old):])
        self.assertEqual(len(after), len(before))

    def test_patch_lf_file_keeps_lf(self):
        path = self._make_skill("eol-lf", "\n")
        before = path.read_bytes()
        self._patch("eol-lf")
        after = path.read_bytes()
        self.assertNotIn(b"\r\n", after)
        self._assert_only_hunk_changed(before, after)

    def test_patch_crlf_file_keeps_crlf(self):
        path = self._make_skill("eol-crlf", "\r\n")
        before = path.read_bytes()
        self._patch("eol-crlf")
        after = path.read_bytes()
        self.assertEqual(after.count(b"\r\n"), after.count(b"\n"))
        self._assert_only_hunk_changed(before, after)

    def test_edit_crlf_file_keeps_crlf(self):
        from tools import skill_manager_tool as smt

        path = self._make_skill("eol-edit", "\r\n")
        new_text = FRONTMATTER.format(name="eol-edit") + BODY.replace("echo", "foxtrot")
        result = smt._edit_skill("eol-edit", new_text)  # model content arrives as LF
        self.assertTrue(result.get("success"), result)
        self.assertEqual(path.read_bytes(), new_text.replace("\n", "\r\n").encode("utf-8"))

    def test_write_file_new_file_is_verbatim_lf(self):
        from tools import skill_manager_tool as smt

        self._make_skill("eol-new", "\n")
        script = "#!/usr/bin/env bash\necho ok\n"
        result = smt._write_file("eol-new", "scripts/run.sh", script)
        self.assertTrue(result.get("success"), result)
        target = self.home / "skills" / "testing" / "eol-new" / "scripts" / "run.sh"
        self.assertEqual(target.read_bytes(), script.encode("utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
