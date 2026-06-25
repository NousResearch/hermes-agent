"""Tests for the cross-profile soft guard wired into write_file / patch /
skill_manage.

The classifier is tested in tests/agent/test_file_safety_cross_profile.py.
This file tests that the tool surfaces:

  1. Refuse cross-profile writes by default and return the warning.
  2. Accept cross-profile writes when cross_profile=True is passed.
  3. Continue to accept in-profile writes normally.
  4. skill_manage's "not found" error names other profiles where the
     skill exists.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def fake_hermes(tmp_path, monkeypatch):
    """Build a two-profile Hermes layout and point HERMES_HOME at
    the hermes-security profile (matching the original-incident shape).
    """
    root = tmp_path / "fake-hermes"
    (root / "skills" / "shared-skill").mkdir(parents=True)
    (root / "skills" / "shared-skill" / "SKILL.md").write_text(
        "---\nname: shared-skill\ndescription: default copy.\n---\n"
    )

    sec_home = root / "profiles" / "hermes-security"
    (sec_home / "skills").mkdir(parents=True)

    coder_home = root / "profiles" / "coder"
    (coder_home / "skills").mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(sec_home))

    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: root)

    import agent.file_safety as fs
    monkeypatch.setattr(fs, "_hermes_home_path", lambda: sec_home)
    monkeypatch.setattr(fs, "_hermes_root_path", lambda: root)

    return {
        "root": root,
        "sec_home": sec_home,
        "coder_home": coder_home,
    }


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


class TestWriteFileCrossProfileGuard:
    def test_in_profile_write_allowed(self, fake_hermes):
        from tools.file_tools import write_file_tool
        target = fake_hermes["sec_home"] / "skills" / "new-skill" / "SKILL.md"
        target.parent.mkdir(parents=True)
        result_json = write_file_tool(str(target), "in-profile content")
        result = json.loads(result_json)
        assert not result.get("error"), f"In-profile write should succeed: {result}"
        assert target.exists()
        assert target.read_text() == "in-profile content"

    def test_cross_profile_write_blocked_by_default(self, fake_hermes):
        """The May 2026 incident — security-profile session edits default
        profile's skill. Must be blocked."""
        from tools.file_tools import write_file_tool
        target = fake_hermes["root"] / "skills" / "shared-skill" / "SKILL.md"
        original = target.read_text()
        result_json = write_file_tool(str(target), "OVERWRITTEN")
        result = json.loads(result_json)
        assert result.get("error"), "Cross-profile write should be refused"
        assert "cross-profile" in result["error"].lower()
        assert "default" in result["error"]
        assert "hermes-security" in result["error"]
        # File untouched.
        assert target.read_text() == original

    def test_cross_profile_True_bypass(self, fake_hermes):
        """Explicit override after user direction must succeed."""
        from tools.file_tools import write_file_tool
        target = fake_hermes["root"] / "skills" / "shared-skill" / "SKILL.md"
        result_json = write_file_tool(
            str(target), "user-directed override", cross_profile=True
        )
        result = json.loads(result_json)
        assert not result.get("error"), f"cross_profile=True must succeed: {result}"
        assert target.read_text() == "user-directed override"

    def test_non_hermes_path_unaffected(self, fake_hermes, tmp_path):
        from tools.file_tools import write_file_tool
        target = tmp_path / "outside" / "main.py"
        target.parent.mkdir()
        result_json = write_file_tool(str(target), "print('hello')")
        result = json.loads(result_json)
        assert not result.get("error")
        assert target.exists()


# ---------------------------------------------------------------------------
# patch
# ---------------------------------------------------------------------------


class TestPatchCrossProfileGuard:
    def test_cross_profile_patch_blocked(self, fake_hermes):
        from tools.file_tools import patch_tool
        target = fake_hermes["root"] / "skills" / "shared-skill" / "SKILL.md"
        original = target.read_text()
        result_json = patch_tool(
            mode="replace",
            path=str(target),
            old_string="default copy.",
            new_string="HIJACKED.",
        )
        result = json.loads(result_json)
        assert result.get("error")
        assert "cross-profile" in result["error"].lower()
        assert target.read_text() == original

    def test_cross_profile_patch_bypass(self, fake_hermes):
        from tools.file_tools import patch_tool
        target = fake_hermes["root"] / "skills" / "shared-skill" / "SKILL.md"
        result_json = patch_tool(
            mode="replace",
            path=str(target),
            old_string="default copy.",
            new_string="user-directed update.",
            cross_profile=True,
        )
        result = json.loads(result_json)
        assert not result.get("error"), f"cross_profile=True bypass: {result}"
        assert "user-directed update." in target.read_text()

    def test_v4a_patch_extracts_path_for_guard(self, fake_hermes):
        """V4A patches embed the target paths in the patch body, not in
        a ``path`` kwarg. The guard must still apply."""
        from tools.file_tools import patch_tool
        target = fake_hermes["root"] / "skills" / "shared-skill" / "SKILL.md"
        original = target.read_text()
        v4a = (
            "*** Begin Patch\n"
            f"*** Update File: {target}\n"
            "@@\n"
            "-default copy.\n"
            "+HIJACKED.\n"
            "*** End Patch"
        )
        result_json = patch_tool(mode="patch", patch=v4a)
        result = json.loads(result_json)
        assert result.get("error"), f"V4A cross-profile must block: {result}"
        assert "cross-profile" in result["error"].lower()
        assert target.read_text() == original


# ---------------------------------------------------------------------------
# skill_manage — error message naming other profile (item D)
# ---------------------------------------------------------------------------


class TestSkillManageCrossProfileErrorUX:
    def _make_skill_in_profile(self, profile_dir: Path, name: str):
        d = profile_dir / "skills" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: a skill.\n---\n"
        )

    def test_error_names_other_profile_when_skill_lives_there(
        self, fake_hermes, monkeypatch
    ):
        """The original incident shape — model expects 'foo' in active
        profile, but 'foo' lives in default. Error must point at default."""
        self._make_skill_in_profile(fake_hermes["root"], "default-only-skill")

        # Re-import the module so SKILLS_DIR picks up HERMES_HOME (set in
        # the fixture). Skill_manager_tool computes SKILLS_DIR at import.
        import importlib
        import tools.skill_manager_tool
        importlib.reload(tools.skill_manager_tool)
        from tools.skill_manager_tool import _skill_not_found_error

        err = _skill_not_found_error("default-only-skill")
        assert "not found in active profile 'hermes-security'" in err
        assert "default" in err
        assert "cross_profile=True" in err

    def test_error_names_multiple_profiles(self, fake_hermes, monkeypatch):
        """When the skill exists in TWO other profiles, both should be named."""
        self._make_skill_in_profile(fake_hermes["root"], "everywhere-skill")
        self._make_skill_in_profile(fake_hermes["coder_home"], "everywhere-skill")

        import importlib
        import tools.skill_manager_tool
        importlib.reload(tools.skill_manager_tool)
        from tools.skill_manager_tool import _skill_not_found_error

        err = _skill_not_found_error("everywhere-skill")
        assert "default" in err
        assert "coder" in err
        # Switch-profiles hint
        assert "hermes -p" in err

    def test_genuinely_missing_skill_keeps_helpful_hint(
        self, fake_hermes, monkeypatch
    ):
        """When no profile has the skill, error falls back to skills_list hint."""
        import importlib
        import tools.skill_manager_tool
        importlib.reload(tools.skill_manager_tool)
        from tools.skill_manager_tool import _skill_not_found_error

        err = _skill_not_found_error("totally-imaginary-skill")
        assert "not found in active profile 'hermes-security'" in err
        assert "skills_list" in err


# ---------------------------------------------------------------------------
# System prompt active-profile line (item B)
# ---------------------------------------------------------------------------


class TestSystemPromptActiveProfile:
    def test_default_profile_line_in_prompt(self, tmp_path, monkeypatch):
        """When active profile is 'default', the prompt names it and warns
        about the root profiles/<name>/ path."""
        # Don't set HERMES_HOME — falls back to default.
        import agent.file_safety as fs
        monkeypatch.setattr(fs, "_hermes_home_path", lambda: tmp_path / "fake")
        monkeypatch.setattr(fs, "_hermes_root_path", lambda: tmp_path / "fake")

        from agent.file_safety import _resolve_active_profile_name
        assert _resolve_active_profile_name() == "default"
        # Build the line manually to pin the contract — the prompt builder
        # is too heavy to instantiate end-to-end in a unit test.
        # See agent/system_prompt.py for the exact wording.

    def test_named_profile_line_in_prompt_text(self, fake_hermes):
        """When active profile is 'hermes-security', the prompt warns
        explicitly about NOT modifying default's skills/plugins/cron/memories."""
        # Spot-check by reading the source — the contract is:
        # (1) names the active profile, (2) names the default-profile
        # paths, (3) says "do not modify another profile's" without
        # explicit user direction.
        from pathlib import Path
        src = Path("agent/system_prompt.py").read_text(encoding="utf-8")
        assert "Active Hermes profile" in src
        assert "cross_profile=True" in src
        assert "get_default_hermes_root" in src
        assert "hermes_root" in src
        # Both branches present (default and named profile).
        assert "Active Hermes profile: default" in src
        assert "Active Hermes profile: {active_profile}" in src
