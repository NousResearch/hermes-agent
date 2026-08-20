#!/usr/bin/env python3
"""Regression test: the skill security scan must fail CLOSED.

``_security_scan_skill`` no-ops when the guard is disabled (the default).
When the guard is ENABLED, a scanner exception used to be caught, logged, and
return ``None``.  Every mutating caller treats ``None`` as "scan passed", so a
broken scanner silently allowed an UNSCANNED skill write to persist.

The fix (commit 9e10a630) returns a non-empty error string from the except
branch instead, which the 4 mutating callers already treat as "blocked" — they
roll back the write.  The disabled-guard path is unchanged (still returns
None).

These tests are mechanical (no live LLM, no real scanner):

  TEST B.1  guard ON + scan_skill raises  -> _security_scan_skill returns a
            NON-EMPTY string (fail closed), not None.
  TEST B.2  end-to-end _create_skill with guard ON + scanner raising -> the
            create is BLOCKED (success False) AND the skill dir is rolled
            back (removed / not persisted).
  TEST B.3  DEFAULT-OFF regression: guard disabled -> returns None even if
            scan_skill would raise (it must never be called).
"""

import unittest
from unittest.mock import patch

from tools.skill_manager_tool import _security_scan_skill, _create_skill


VALID_SKILL_CONTENT = """\
---
name: test-skill
description: A test skill for unit testing.
---

# Test Skill

Step 1: Do the thing.
"""


class TestSecurityScanFailClosed(unittest.TestCase):
    # ------------------------------------------------------------------ B.1
    def test_scanner_exception_fails_closed_when_guard_on(self):
        """guard ON + scan_skill raises -> non-empty error string, NOT None.

        Pre-fix this returned None ('scan passed'); callers would persist the
        unscanned skill.  The fix must surface a blocking error string.
        """
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td)
            with patch(
                "tools.skill_manager_tool._guard_agent_created_enabled",
                return_value=True,
            ), patch(
                "tools.skill_manager_tool._GUARD_AVAILABLE", True
            ), patch(
                "tools.skill_manager_tool.scan_skill",
                side_effect=RuntimeError("scanner blew up"),
            ):
                result = _security_scan_skill(skill_dir)

        self.assertIsNotNone(
            result,
            "scanner exception with guard ON returned None — that is the "
            " silent-allow bug; it must fail closed.",
        )
        self.assertIsInstance(result, str)
        self.assertTrue(result, "fail-closed error string must be non-empty")
        # The scanner failure should be surfaced as the reason.
        self.assertIn("fail-closed", result.lower())

    def test_should_allow_install_exception_also_fails_closed(self):
        """The except branch must also catch failures from should_allow_install
        (the call after scan_skill), not just scan_skill itself."""
        import tempfile
        from pathlib import Path
        from tools.skills_guard import ScanResult

        fake = ScanResult(
            skill_name="t",
            source="agent-created",
            trust_level="agent-created",
            verdict="safe",
            findings=[],
            summary="ok",
        )
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td)
            with patch(
                "tools.skill_manager_tool._guard_agent_created_enabled",
                return_value=True,
            ), patch(
                "tools.skill_manager_tool._GUARD_AVAILABLE", True
            ), patch(
                "tools.skill_manager_tool.scan_skill", return_value=fake
            ), patch(
                "tools.skill_manager_tool.should_allow_install",
                side_effect=ValueError("verdict logic broke"),
            ):
                result = _security_scan_skill(skill_dir)

        self.assertIsNotNone(result, "post-scan failure must also fail closed")
        self.assertIn("fail-closed", result.lower())

    # ------------------------------------------------------------------ B.2
    def test_create_skill_rolls_back_when_scanner_raises(self):
        """End-to-end: a raising scanner (guard ON) blocks the create AND
        removes the skill directory — no unscanned skill persists on disk."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            skills_root = Path(td)
            with patch("tools.skill_manager_tool.SKILLS_DIR", skills_root), \
                 patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills_root]), \
                 patch("tools.skill_manager_tool._guard_agent_created_enabled", return_value=True), \
                 patch("tools.skill_manager_tool._GUARD_AVAILABLE", True), \
                 patch("tools.skill_manager_tool.scan_skill",
                       side_effect=RuntimeError("scanner blew up")):
                result = _create_skill("danger-skill", VALID_SKILL_CONTENT)

                # 1) The create is blocked.
                self.assertFalse(
                    result.get("success"),
                    f"create succeeded despite a broken scanner: {result!r}",
                )
                self.assertIn("error", result)

                # 2) The skill directory was rolled back (not persisted).
                skill_dir = skills_root / "danger-skill"
                self.assertFalse(
                    skill_dir.exists(),
                    f"skill dir was NOT rolled back — unscanned skill persisted "
                    f"at {skill_dir}",
                )
                # And nothing else leaked into the skills root.
                self.assertEqual(
                    list(skills_root.iterdir()),
                    [],
                    "skills root is not empty after a blocked create",
                )

    # ------------------------------------------------------------------ B.3
    def test_default_off_returns_none_and_never_scans(self):
        """DEFAULT-OFF regression: guard disabled -> None, scan_skill never
        called even if it would raise.  No behavioral regression for the
        common path."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td)
            with patch(
                "tools.skill_manager_tool._guard_agent_created_enabled",
                return_value=False,
            ), patch(
                "tools.skill_manager_tool.scan_skill",
                side_effect=AssertionError("scan_skill must not run when guard is off"),
            ) as mock_scan:
                result = _security_scan_skill(skill_dir)

        self.assertIsNone(result, "guard-off path must return None (no-op)")
        mock_scan.assert_not_called()

    def test_guard_unavailable_returns_none(self):
        """If the scanner module failed to import (_GUARD_AVAILABLE False),
        _security_scan_skill is a no-op returning None — the guard simply
        isn't installed.  (Distinct from the fail-closed except branch, which
        only triggers once the guard is both available AND enabled.)"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td)
            with patch("tools.skill_manager_tool._GUARD_AVAILABLE", False), \
                 patch("tools.skill_manager_tool._guard_agent_created_enabled",
                       return_value=True):
                result = _security_scan_skill(skill_dir)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
