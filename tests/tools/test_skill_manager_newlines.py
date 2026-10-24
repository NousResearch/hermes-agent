"""Byte-level skill write contracts, including native Windows newline translation.

Also runnable without pytest: python tests/tools/test_skill_manager_newlines.py
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch


CONTENT = "---\nname: newline-contract\ndescription: Use when testing newline preservation.\n---\n\nKeep café content.\n"


@contextmanager
def _isolated_manager(tmp_path):
    home = tmp_path / "home"
    with patch.dict(os.environ, {"HERMES_HOME": str(home)}):
        from tools import skill_manager_tool as manager
        from tools.registry import registry

        # Keep resolution isolated even if another test imported the module first.
        with patch.object(manager, "SKILLS_DIR", home / "skills"):
            token = manager._skill_gate_bypass.set(True)
            try:
                yield manager, registry, home / "skills" / "newline-contract"
            finally:
                manager._skill_gate_bypass.reset(token)


def _dispatch(registry, **operation):
    return json.loads(registry.dispatch(
        "skill_manage", {"operations": [{"name": "newline-contract", **operation}]}))


def test_skill_mutations_preserve_supplied_lf_bytes(tmp_path):
    with _isolated_manager(tmp_path) as (_, registry, root):
        observed = {}
        expected = {}

        def check(label, relative, text, **operation):
            result = _dispatch(registry, **operation)
            assert result["success"], result
            observed[label] = (root / relative).read_bytes()
            expected[label] = text.encode("utf-8")

        check("create", "SKILL.md", CONTENT, action="create", content=CONTENT)
        changed = CONTENT.replace("Keep café content.", "Keep changed content.\nKeep LF.")
        check("patch", "SKILL.md", changed, action="patch",
              old_string="Keep café content.", new_string="Keep changed content.\nKeep LF.")
        check("rewrite", "SKILL.md", CONTENT, action="patch", content=CONTENT)
        support = "First café line.\nSecond line.\n"
        check("support-write", "references/newlines.md", support, action="write_file",
              file_path="references/newlines.md", file_content=support)
        check("support-patch", "references/newlines.md", support.replace("Second", "Last"),
              action="patch", file_path="references/newlines.md",
              old_string="Second", new_string="Last")
        assert observed == expected


def test_rejected_skill_write_restores_original_lf_bytes(tmp_path):
    with _isolated_manager(tmp_path) as (manager, registry, root):
        root.mkdir(parents=True)
        skill_md = root / "SKILL.md"
        skill_md.write_bytes(CONTENT.encode("utf-8"))
        supporting = root / "references" / "newlines.md"
        supporting.parent.mkdir()
        supporting.write_bytes(b"Original\nSupporting\n")
        before = {p: p.read_bytes() for p in (skill_md, supporting)}
        # Exercise the per-file scan rollback, not just the batch's byte-copy rollback.
        with patch.object(manager, "_security_scan_skill", return_value="Rejected by test scan"):
            for operation in (
                {"action": "patch", "old_string": "Keep", "new_string": "Replace"},
                {"action": "write_file", "file_path": "references/newlines.md",
                 "file_content": "Replacement\nLines\n"},
            ):
                result = json.loads(registry.dispatch(
                    "skill_manage", {"name": "newline-contract", **operation}))
                assert not result["success"], result
                assert "Rejected by test scan" in result["error"]
        assert {p: p.read_bytes() for p in before} == before


if __name__ == "__main__":
    import sys
    import unittest

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    def run_case(test):
        with tempfile.TemporaryDirectory(prefix="skill_newline_") as directory:
            test(Path(directory))

    suite = unittest.TestSuite(
        unittest.FunctionTestCase(lambda test=test: run_case(test), description=test.__name__)
        for test in (test_skill_mutations_preserve_supplied_lf_bytes,
                     test_rejected_skill_write_restores_original_lf_bytes)
    )
    sys.exit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
