"""Tests for the author-to-shared create path in tools/skill_manager_tool.py.

Covers the spec (2026-06-29 author-to-shared prevention):
  - shared-by-default destination resolution + group validation
  - the local=true escape hatch + profile-aware reminder
  - upstream-safety (no skills-shared tree -> legacy local behavior)
  - the skills.author_to_shared config override
  - name-validator hardening (re.fullmatch, rejects trailing newline) — RC-2
  - rollback cleans the shared tree on scan-block — RC-6
  - success-path create metrics — AC-15
"""

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

import tools.skill_manager_tool as smt
from tools.skill_manager_tool import (
    _validate_name,
    _valid_shared_groups,
    _shared_authoring_enabled,
    _resolve_skill_dir,
    _create_skill,
    skill_manage,
    SkillCreateError,
)


VALID_CONTENT = """\
---
name: probe-skill
description: A probe skill for author-to-shared unit testing.
---

# Probe Skill

Step 1: do the thing.
"""


@contextmanager
def _fleet_home(tmp_path, groups=("devops", "general", "research")):
    """A fleet-shaped HERMES_HOME: a skills-shared/ tree with the given groups
    plus an empty local skills/ tree. Patches the module globals + _find_skill's
    search roots so nothing touches the real ~/.hermes."""
    home = tmp_path / "home"
    local = home / "skills"
    shared = home / "skills-shared"
    local.mkdir(parents=True, exist_ok=True)
    for g in groups:
        (shared / g).mkdir(parents=True, exist_ok=True)
    with patch.object(smt, "HERMES_HOME", home), \
         patch.object(smt, "SKILLS_DIR", local), \
         patch("agent.skill_utils.get_all_skills_dirs",
               return_value=[local] + [shared / g for g in groups]):
        yield home, local, shared


@contextmanager
def _vanilla_home(tmp_path):
    """An upstream-shaped HERMES_HOME: NO skills-shared/ tree at all."""
    home = tmp_path / "home"
    local = home / "skills"
    local.mkdir(parents=True, exist_ok=True)
    with patch.object(smt, "HERMES_HOME", home), \
         patch.object(smt, "SKILLS_DIR", local), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[local]):
        yield home, local


@contextmanager
def _profile_home(tmp_path, profile="athena", root_name="root",
                  groups=("devops", "general", "research")):
    """A PROFILE-shaped HERMES_HOME: ``<root>/profiles/<name>`` with the shared
    tree at the TOP-LEVEL root (``<root>/skills-shared``) — the real fleet
    layout for a specialist gateway. This is the regression fixture for the
    profile-home bug: resolving the shared tree HERMES_HOME-relative pointed at
    the nonexistent ``profiles/<name>/skills-shared`` and every specialist
    create hard-errored."""
    root = tmp_path / root_name
    home = root / "profiles" / profile
    local = home / "skills"
    shared = root / "skills-shared"
    local.mkdir(parents=True, exist_ok=True)
    for g in groups:
        (shared / g).mkdir(parents=True, exist_ok=True)
    with patch.object(smt, "HERMES_HOME", home), \
         patch.object(smt, "SKILLS_DIR", local), \
         patch("agent.skill_utils.get_all_skills_dirs",
               return_value=[local] + [shared / g for g in groups]):
        yield root, home, local, shared


# ---------------------------------------------------------------------------
# RC-2: name validator hardening (fullmatch — rejects trailing newline etc.)
# ---------------------------------------------------------------------------

class TestNameValidatorHardening:
    def test_trailing_newline_rejected(self):
        assert _validate_name("foo\n") is not None

    def test_embedded_newline_rejected(self):
        assert _validate_name("foo\nbar") is not None

    def test_control_char_rejected(self):
        assert _validate_name("foo\tbar") is not None
        assert _validate_name("foo\x00") is not None

    def test_traversal_rejected(self):
        assert _validate_name("../evil") is not None
        assert _validate_name("/etc/x") is not None

    def test_valid_names_still_pass(self):
        assert _validate_name("my-skill") is None
        assert _validate_name("skill123") is None
        assert _validate_name("my_skill.v2") is None


# ---------------------------------------------------------------------------
# _valid_shared_groups / _shared_authoring_enabled
# ---------------------------------------------------------------------------

class TestGroupResolution:
    def test_lists_live_groups(self, tmp_path):
        with _fleet_home(tmp_path) as (home, local, shared):
            (shared / ".archive").mkdir()          # excluded
            (shared / "__pycache__").mkdir()       # excluded
            (shared / "notes.txt").write_text("x")  # not a dir
            assert _valid_shared_groups() == ["devops", "general", "research"]

    def test_empty_when_no_shared_tree(self, tmp_path):
        with _vanilla_home(tmp_path):
            assert _valid_shared_groups() == []

    def test_authoring_enabled_autodetect(self, tmp_path):
        with _fleet_home(tmp_path / "fleet"):
            with patch.object(smt, "_author_to_shared_setting", return_value=None):
                assert _shared_authoring_enabled() is True
        with _vanilla_home(tmp_path / "vanilla"):
            with patch.object(smt, "_author_to_shared_setting", return_value=None):
                assert _shared_authoring_enabled() is False

    def test_config_override_wins(self, tmp_path):
        with _fleet_home(tmp_path / "fleet"):
            with patch.object(smt, "_author_to_shared_setting", return_value=False):
                assert _shared_authoring_enabled() is False
        with _vanilla_home(tmp_path / "vanilla"):
            with patch.object(smt, "_author_to_shared_setting", return_value=True):
                assert _shared_authoring_enabled() is True


# ---------------------------------------------------------------------------
# _resolve_skill_dir — the core redirect
# ---------------------------------------------------------------------------

class TestResolveSkillDir:
    def test_shared_default(self, tmp_path):
        with _fleet_home(tmp_path) as (home, local, shared):
            with patch.object(smt, "_author_to_shared_setting", return_value=None):
                d = _resolve_skill_dir("foo", "devops")
                assert d == shared / "devops" / "foo"

    def test_local_flag_uses_legacy_tree(self, tmp_path):
        with _fleet_home(tmp_path) as (home, local, shared):
            with patch.object(smt, "_author_to_shared_setting", return_value=None):
                d = _resolve_skill_dir("foo", "devops", local=True)
                assert d == local / "devops" / "foo"

    def test_unknown_group_raises(self, tmp_path):
        with _fleet_home(tmp_path):
            with patch.object(smt, "_author_to_shared_setting", return_value=None):
                with pytest.raises(smt.SkillCreateError) as ei:
                    _resolve_skill_dir("foo", "devops-stuff")
                assert "devops-stuff" in str(ei.value)
                assert "devops" in str(ei.value)  # lists valid groups

    def test_missing_category_raises(self, tmp_path):
        with _fleet_home(tmp_path):
            with patch.object(smt, "_author_to_shared_setting", return_value=None):
                with pytest.raises(smt.SkillCreateError):
                    _resolve_skill_dir("foo", None)

    def test_vanilla_falls_to_local(self, tmp_path):
        with _vanilla_home(tmp_path) as (home, local):
            with patch.object(smt, "_author_to_shared_setting", return_value=None):
                d = _resolve_skill_dir("foo", "devops")
                assert d == local / "devops" / "foo"


# ---------------------------------------------------------------------------
# AC-1 / AC-2 / AC-4: _create_skill end to end
# ---------------------------------------------------------------------------

class TestCreateSkill:
    def test_ac1_shared_by_default(self, tmp_path):
        with _fleet_home(tmp_path) as (home, local, shared):
            with patch.object(smt, "_author_to_shared_setting", return_value=None):
                r = _create_skill("probe-skill", VALID_CONTENT, "devops")
        assert r["success"] is True
        assert r["authored_to"] == "shared"
        landed = shared / "devops" / "probe-skill" / "SKILL.md"
        assert landed.exists()
        assert (shared / "devops" / "probe-skill") in landed.parents or landed.parent == shared / "devops" / "probe-skill"

    def test_ac2_unknown_group_fails_with_list(self, tmp_path):
        with _fleet_home(tmp_path) as (home, local, shared):
            with patch.object(smt, "_author_to_shared_setting", return_value=None):
                r = _create_skill("probe-skill", VALID_CONTENT, "nonsense")
        assert r["success"] is False
        assert "nonsense" in r["error"]
        assert "devops" in r["error"]
        # nothing created
        assert not (shared / "nonsense").exists()

    def test_ac2_missing_category_fails(self, tmp_path):
        with _fleet_home(tmp_path) as (home, local, shared):
            with patch.object(smt, "_author_to_shared_setting", return_value=None):
                r = _create_skill("probe-skill", VALID_CONTENT, None)
        assert r["success"] is False
        assert "required" in r["error"].lower() or "group" in r["error"].lower()

    def test_ac3_local_escape_hatch_warns(self, tmp_path):
        with _fleet_home(tmp_path) as (home, local, shared):
            with patch.object(smt, "_author_to_shared_setting", return_value=None), \
                 patch.object(smt, "_active_profile_name", return_value="default"):
                r = _create_skill("probe-skill", VALID_CONTENT, "devops", local=True)
        assert r["success"] is True
        assert r["authored_to"] == "local"
        assert (local / "devops" / "probe-skill" / "SKILL.md").exists()
        assert "warning" in r
        assert "default/probe-skill" in r["warning"]
        assert "local-skills-allowlist.txt" in r["warning"]

    def test_ac4_vanilla_upstream_safe(self, tmp_path):
        with _vanilla_home(tmp_path) as (home, local):
            with patch.object(smt, "_author_to_shared_setting", return_value=None):
                r = _create_skill("probe-skill", VALID_CONTENT, "devops")
        assert r["success"] is True
        assert r["authored_to"] == "local"
        assert (local / "devops" / "probe-skill" / "SKILL.md").exists()

    def test_ac10_traversal_name_fails_no_dir(self, tmp_path):
        with _fleet_home(tmp_path) as (home, local, shared):
            with patch.object(smt, "_author_to_shared_setting", return_value=None):
                r = _create_skill("../evil", VALID_CONTENT, "devops")
        assert r["success"] is False
        # nothing escaped the group dir
        assert not (home / "evil").exists()
        assert not (shared.parent / "evil").exists()

    def test_ac10_trailing_newline_name_fails(self, tmp_path):
        with _fleet_home(tmp_path) as (home, local, shared):
            with patch.object(smt, "_author_to_shared_setting", return_value=None):
                r = _create_skill("foo\n", VALID_CONTENT, "devops")
        assert r["success"] is False

    def test_ac11_rollback_cleans_shared_tree_on_scan_block(self, tmp_path):
        with _fleet_home(tmp_path) as (home, local, shared):
            with patch.object(smt, "_author_to_shared_setting", return_value=None), \
                 patch.object(smt, "_security_scan_skill", return_value="BLOCKED: secret found"):
                r = _create_skill("probe-skill", VALID_CONTENT, "devops")
        assert r["success"] is False
        # the empty skill dir must be removed (no git-tracked cruft)
        assert not (shared / "devops" / "probe-skill").exists()

    def test_ac15_metrics_recorded(self, tmp_path):
        calls = []
        with _fleet_home(tmp_path) as (home, local, shared):
            with patch.object(smt, "_author_to_shared_setting", return_value=None), \
                 patch.object(smt, "_record_create_metric",
                              side_effect=lambda *a, **k: calls.append((a, k))):
                _create_skill("probe-skill", VALID_CONTENT, "devops")
        assert len(calls) == 1
        assert calls[0][1].get("shared") is True


# ---------------------------------------------------------------------------
# skill_manage dispatch threads local + AC-5 config override end-to-end
# ---------------------------------------------------------------------------

class TestSkillManageDispatch:
    def test_local_param_threaded(self, tmp_path):
        with _fleet_home(tmp_path) as (home, local, shared):
            with patch.object(smt, "_author_to_shared_setting", return_value=None), \
                 patch.object(smt, "_active_profile_name", return_value="default"):
                out = skill_manage("create", "probe-skill", content=VALID_CONTENT,
                                   category="devops", local=True)
        res = json.loads(out)
        assert res["success"] is True
        assert res["authored_to"] == "local"

    def test_ac5_config_false_forces_local(self, tmp_path):
        with _fleet_home(tmp_path) as (home, local, shared):
            with patch.object(smt, "_author_to_shared_setting", return_value=False):
                out = skill_manage("create", "probe-skill", content=VALID_CONTENT,
                                   category="devops")
        res = json.loads(out)
        assert res["success"] is True
        assert res["authored_to"] == "local"
        assert (local / "devops" / "probe-skill" / "SKILL.md").exists()

    def test_ac5_config_true_forces_shared(self, tmp_path):
        with _vanilla_home(tmp_path) as (home, local):
            # no shared tree, but flag true -> shared mode -> no groups -> hard error
            with patch.object(smt, "_author_to_shared_setting", return_value=True):
                out = skill_manage("create", "probe-skill", content=VALID_CONTENT,
                                   category="devops")
        res = json.loads(out)
        assert res["success"] is False  # shared mode forced, but no groups exist


# ---------------------------------------------------------------------------
# Profile-home resolution (the 2026-07-01 profile bug regression suite)
# ---------------------------------------------------------------------------

class TestProfileHomeSharedRoot:
    """_shared_skills_root() must resolve the TOP-LEVEL shared tree from a
    profile HERMES_HOME (<root>/profiles/<p>), not a profile-relative path."""

    def test_profile_home_resolves_top_level_shared(self, tmp_path):
        with _profile_home(tmp_path) as (root, home, local, shared):
            assert smt._shared_skills_root() == shared
            assert smt._shared_skills_root().is_dir()

    def test_profile_home_finds_groups(self, tmp_path):
        with _profile_home(tmp_path) as (root, home, local, shared):
            assert _valid_shared_groups() == ["devops", "general", "research"]

    def test_profile_home_autodetect_enabled(self, tmp_path):
        with _profile_home(tmp_path) as (root, home, local, shared):
            with patch.object(smt, "_author_to_shared_setting", return_value=None):
                assert _shared_authoring_enabled() is True

    def test_profile_home_create_lands_in_top_level_shared(self, tmp_path):
        with _profile_home(tmp_path) as (root, home, local, shared):
            with patch.object(smt, "_author_to_shared_setting", return_value=None):
                r = _create_skill("probe-skill", VALID_CONTENT, "devops")
        assert r["success"] is True
        assert r["authored_to"] == "shared"
        assert (shared / "devops" / "probe-skill" / "SKILL.md").exists()
        # and NOT under the profile dir
        assert not (home / "skills-shared").exists()

    def test_docker_profile_layout(self, tmp_path):
        # /opt/data/profiles/<p> -> /opt/data (root_name plays /opt/data here)
        with _profile_home(tmp_path, profile="coder", root_name="opt-data") as (
                root, home, local, shared):
            assert smt._shared_skills_root() == shared
            assert _valid_shared_groups() == ["devops", "general", "research"]

    def test_red_proof_old_expression_missed_the_tree(self, tmp_path):
        """Documents the bug: the OLD HERMES_HOME-relative expression points at
        a nonexistent profile-relative dir in profile mode. If this ever starts
        passing (profile trees growing their own skills-shared/), re-examine
        the resolution policy rather than silently deleting this test."""
        with _profile_home(tmp_path) as (root, home, local, shared):
            old_expression = smt.HERMES_HOME / "skills-shared"
            assert not old_expression.is_dir()          # the bug's root cause
            assert old_expression != smt._shared_skills_root()  # fix diverges

    def test_vanilla_home_unchanged_by_fix(self, tmp_path):
        """INV-1: a non-profile HERMES_HOME resolves exactly as before."""
        with _vanilla_home(tmp_path) as (home, local):
            assert smt._shared_skills_root() == home / "skills-shared"

    def test_fleet_home_unchanged_by_fix(self, tmp_path):
        """INV-2: the default-profile (top-level home) layout is identical."""
        with _fleet_home(tmp_path) as (home, local, shared):
            assert smt._shared_skills_root() == shared

    def test_fail_open_on_resolution_error(self, tmp_path):
        """A pathological HERMES_HOME (parent access raising) falls back to the
        HERMES_HOME-relative path instead of crashing the create."""
        class _BadPath:
            @property
            def parent(self):
                raise RuntimeError("boom")
            def __truediv__(self, other):
                return Path("/tmp/fallback") / other
        with patch.object(smt, "HERMES_HOME", _BadPath()):
            # must not raise; must return the fallback HERMES_HOME-relative join
            out = smt._shared_skills_root()
            assert str(out).endswith("skills-shared")
