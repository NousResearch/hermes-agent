"""Regression tests for #30119: TUI Gateway sidebar must surface skills
that come from ``skills.external_dirs`` so operators can confirm at a
glance that their config was picked up.

The fix covers three closely-related gaps:

* ``_find_all_skills`` now tags each emitted skill with a ``source`` of
  ``"local"`` or ``"external"`` — without this, no downstream surface
  could tell the two apart.
* ``skills_list`` no longer short-circuits to an empty payload when
  ``~/.hermes/skills/`` is missing.  External-only deployments (fresh
  profiles, locked-down ``$HERMES_HOME``) used to lose every external
  skill from the agent listing even though ``skill_view`` would happily
  serve them by name.
* ``hermes_cli.banner`` exposes the new
  ``get_external_skill_categories`` helper, which the TUI Gateway and
  branding sidebar use to pin operator-managed categories ahead of the
  bundled tree so they survive the ``SKILLS_MAX`` truncation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


# ─── fixtures ────────────────────────────────────────────────────────────


def _write_skill(dir_path: Path, *, name: str, description: str = "") -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    body = description or f"{name} body"
    (dir_path / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {body}\n---\n\n{body}\n",
        encoding="utf-8",
    )


@pytest.fixture
def hermes_home(tmp_path: Path) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    return home


@pytest.fixture
def external_dir(tmp_path: Path) -> Path:
    ext = tmp_path / "common-skills"
    ext.mkdir()
    return ext


@pytest.fixture(autouse=True)
def _isolate_external_dirs_cache():
    """Prevent cross-test bleed of the mtime-keyed cache."""
    from agent import skill_utils
    skill_utils._external_dirs_cache_clear()
    yield
    skill_utils._external_dirs_cache_clear()


def _activate(home: Path, external: Path) -> None:
    (home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {external}\n",
        encoding="utf-8",
    )


# ─── source tagging on _find_all_skills ──────────────────────────────────


class TestFindAllSkillsSourceTag:
    def test_local_skill_tagged_local(self, hermes_home, external_dir):
        local = hermes_home / "skills"
        _write_skill(local / "devops" / "deploy", name="local-deploy")
        _activate(hermes_home, external_dir)

        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local),
        ):
            from tools.skills_tool import _find_all_skills
            skills = _find_all_skills()

        by_name = {s["name"]: s for s in skills}
        assert "local-deploy" in by_name
        assert by_name["local-deploy"]["source"] == "local"

    def test_external_skill_tagged_external(self, hermes_home, external_dir):
        local = hermes_home / "skills"
        local.mkdir()  # exists but empty so SKILLS_DIR is scanned
        _write_skill(external_dir / "ops-tools", name="external-only")
        _activate(hermes_home, external_dir)

        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local),
        ):
            from tools.skills_tool import _find_all_skills
            skills = _find_all_skills()

        by_name = {s["name"]: s for s in skills}
        assert by_name["external-only"]["source"] == "external"

    def test_mixed_tree_tags_each_skill_independently(
        self, hermes_home, external_dir
    ):
        local = hermes_home / "skills"
        _write_skill(local / "devops" / "local", name="local-skill")
        _write_skill(external_dir / "shared" / "external", name="external-skill")
        _activate(hermes_home, external_dir)

        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local),
        ):
            from tools.skills_tool import _find_all_skills
            skills = _find_all_skills()

        sources = {s["name"]: s["source"] for s in skills}
        assert sources["local-skill"] == "local"
        assert sources["external-skill"] == "external"

    def test_collision_keeps_local_source_label(
        self, hermes_home, external_dir
    ):
        """Local wins on name collision and KEEPS its ``source=local``
        label — the external duplicate must not silently overwrite the
        provenance the operator relies on.
        """
        local = hermes_home / "skills"
        _write_skill(local / "creative" / "art", name="duplicate")
        _write_skill(external_dir / "art" / "duplicate", name="duplicate")
        _activate(hermes_home, external_dir)

        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local),
        ):
            from tools.skills_tool import _find_all_skills
            skills = _find_all_skills()

        matching = [s for s in skills if s["name"] == "duplicate"]
        assert len(matching) == 1
        assert matching[0]["source"] == "local"


# ─── skills_list no longer short-circuits when local SKILLS_DIR missing ──


class TestSkillsListExternalDirsWhenLocalMissing:
    def test_external_skills_visible_without_local_dir(
        self, hermes_home, external_dir
    ):
        _write_skill(external_dir / "fresh-skill", name="fresh-skill")
        _activate(hermes_home, external_dir)
        local = hermes_home / "skills"
        assert not local.exists()

        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local),
        ):
            from tools.skills_tool import skills_list
            payload = json.loads(skills_list())

        assert payload["success"] is True
        names = [s["name"] for s in payload.get("skills", [])]
        assert "fresh-skill" in names, payload

    def test_read_only_hermes_home_does_not_hide_externals(
        self, hermes_home, external_dir, monkeypatch
    ):
        """Even if ``mkdir`` fails (read-only mount, locked-down
        profile), external skills must still be listed."""
        _write_skill(external_dir / "shared-skill", name="shared-skill")
        _activate(hermes_home, external_dir)
        local = hermes_home / "skills"

        original_mkdir = Path.mkdir

        def _refuse_mkdir(self, *args, **kwargs):
            if self == local:
                raise OSError("read-only filesystem")
            return original_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", _refuse_mkdir)

        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local),
        ):
            from tools.skills_tool import skills_list
            payload = json.loads(skills_list())

        assert payload["success"] is True
        names = [s["name"] for s in payload.get("skills", [])]
        assert "shared-skill" in names

    def test_empty_when_no_skills_anywhere(self, hermes_home):
        (hermes_home / "config.yaml").write_text("skills: {}\n", encoding="utf-8")
        local = hermes_home / "skills"

        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local),
        ):
            from tools.skills_tool import skills_list
            payload = json.loads(skills_list())

        assert payload["success"] is True
        assert payload.get("skills", []) == []


# ─── banner helper: get_external_skill_categories ────────────────────────


class TestGetExternalSkillCategories:
    def test_returns_sorted_external_categories(self):
        mock_skills = [
            {"name": "a", "description": "", "category": "devops", "source": "local"},
            {"name": "b", "description": "", "category": "private", "source": "external"},
            {"name": "c", "description": "", "category": "general", "source": "external"},
        ]
        with patch("tools.skills_tool._find_all_skills", return_value=mock_skills):
            from hermes_cli.banner import get_external_skill_categories
            result = get_external_skill_categories()
        assert result == ["general", "private"]

    def test_no_externals_returns_empty(self):
        mock_skills = [
            {"name": "a", "description": "", "category": "devops", "source": "local"},
        ]
        with patch("tools.skills_tool._find_all_skills", return_value=mock_skills):
            from hermes_cli.banner import get_external_skill_categories
            assert get_external_skill_categories() == []

    def test_uncategorized_external_buckets_under_general(self):
        mock_skills = [
            {"name": "loose", "description": "", "category": None, "source": "external"},
        ]
        with patch("tools.skills_tool._find_all_skills", return_value=mock_skills):
            from hermes_cli.banner import get_external_skill_categories
            assert get_external_skill_categories() == ["general"]

    def test_mixed_category_reported_when_at_least_one_external(self):
        """A category that contains both a local and an external skill
        still counts as external for sidebar-pinning purposes — the
        operator added that external skill and expects to see it."""
        mock_skills = [
            {"name": "a", "description": "", "category": "shared", "source": "local"},
            {"name": "b", "description": "", "category": "shared", "source": "external"},
        ]
        with patch("tools.skills_tool._find_all_skills", return_value=mock_skills):
            from hermes_cli.banner import get_external_skill_categories
            assert get_external_skill_categories() == ["shared"]

    def test_handles_find_all_failure_gracefully(self):
        with patch("tools.skills_tool._find_all_skills", side_effect=RuntimeError("boom")):
            from hermes_cli.banner import get_external_skill_categories
            assert get_external_skill_categories() == []


# ─── end-to-end: gateway session.info payload pins external categories ───


class TestGatewaySessionInfoExposesExternalCategories:
    def test_session_info_includes_external_skill_categories(
        self, hermes_home, external_dir
    ):
        _write_skill(external_dir / "private-skill", name="private-skill")
        _activate(hermes_home, external_dir)
        local = hermes_home / "skills"
        _write_skill(local / "devops" / "deploy", name="bundled-deploy")

        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local),
        ):
            from hermes_cli.banner import (
                get_available_skills,
                get_external_skill_categories,
            )
            by_cat = get_available_skills()
            external_cats = get_external_skill_categories()

        # Built-in category is present, external category is present,
        # and the external categories list singles out only the
        # external-sourced category for sidebar pinning.
        assert "devops" in by_cat
        assert "general" in by_cat  # uncategorised external bucket
        assert "general" in external_cats
        assert "devops" not in external_cats


# ─── back-compat: existing payload shape is unchanged ────────────────────


class TestGetAvailableSkillsShapeUnchanged:
    def test_existing_dict_shape_preserved(self):
        mock_skills = [
            {"name": "a", "description": "", "category": "devops", "source": "local"},
            {"name": "b", "description": "", "category": "private", "source": "external"},
        ]
        with patch("tools.skills_tool._find_all_skills", return_value=mock_skills):
            from hermes_cli.banner import get_available_skills
            result = get_available_skills()
        assert result == {"devops": ["a"], "private": ["b"]}
