"""Skill readiness must honor the active profile secret scope.

#112082: on a multiplexed gateway the cron worker installs the routed profile's
scope (``set_secret_scope(build_profile_secret_scope(...))`` after hydrating
external secret sources), but ``_is_env_var_persisted`` resolved required vars
only against ``<profile>/.env`` and raw ``os.getenv`` — so a var that exists
solely in the profile's external secret snapshot read as missing and cron
preflight blocked the job as ``blocked_config``.

Fail-closed parity: under multiplexing an ambient value belongs to the launch
profile, so it must NOT satisfy another profile's skill requirement.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent import secret_scope
from tools.skills_tool import skill_view


def _make_skill(skills_dir, name, env_var):
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Scope-gated skill.\n"
        f"required_environment_variables:\n  - name: {env_var}\n    prompt: Service token\n---\n\n# {name}\n",
        encoding="utf-8",
    )
    return skill_dir


@pytest.fixture()
def multiplex_scope():
    """Multiplexing on with a scope installed, as the cron worker does."""
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"SCOPED_SERVICE_TOKEN": "profile-token"})
    try:
        yield token
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)


class TestReadinessResolvesScope:
    def test_scope_only_var_is_available(self, tmp_path, monkeypatch, multiplex_scope):
        """Var lives only in the profile scope (external secret source) —
        .env and ambient env lack it — yet readiness must pass."""
        monkeypatch.delenv("SCOPED_SERVICE_TOKEN", raising=False)
        skills_dir = tmp_path / "skills"
        _make_skill(skills_dir, "scoped-gated", "SCOPED_SERVICE_TOKEN")

        with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
            result = json.loads(skill_view("scoped-gated"))

        assert result["success"] is True
        assert result["setup_needed"] is False
        assert result["missing_required_environment_variables"] == []
        assert result["readiness_status"] == "available"

    def test_scope_only_var_registers_sandbox_passthrough(
        self, tmp_path, monkeypatch, multiplex_scope
    ):
        """The available name must still flow into env passthrough so sandboxed
        execution resolves the value from the active scope."""
        from tools import env_passthrough

        monkeypatch.delenv("SCOPED_SERVICE_TOKEN", raising=False)
        skills_dir = tmp_path / "skills"
        _make_skill(skills_dir, "scoped-gated", "SCOPED_SERVICE_TOKEN")

        env_passthrough.clear_env_passthrough()
        try:
            with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
                json.loads(skill_view("scoped-gated"))
            assert env_passthrough.is_env_passthrough("SCOPED_SERVICE_TOKEN") is True
            assert (
                env_passthrough.resolve_passthrough_value("SCOPED_SERVICE_TOKEN")
                == "profile-token"
            )
        finally:
            env_passthrough.clear_env_passthrough()


class TestReadinessFailsClosedUnderMultiplex:
    def test_ambient_value_does_not_satisfy_other_profile(self, tmp_path, monkeypatch):
        """Multiplexing active, no scope: the ambient env belongs to the launch
        profile, so its value must not mark the routed profile's skill ready."""
        monkeypatch.setenv("LAUNCH_PROFILE_TOKEN", "launch-token")
        secret_scope.set_multiplex_active(True)
        try:
            skills_dir = tmp_path / "skills"
            _make_skill(skills_dir, "ambient-gated", "LAUNCH_PROFILE_TOKEN")

            with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
                result = json.loads(skill_view("ambient-gated"))

            assert result["setup_needed"] is True
            assert result["missing_required_environment_variables"] == [
                "LAUNCH_PROFILE_TOKEN"
            ]
        finally:
            secret_scope.set_multiplex_active(False)

    def test_scope_miss_does_not_fall_back_to_process_env(
        self, tmp_path, monkeypatch, multiplex_scope
    ):
        """A scope is installed (routed profile) but lacks the var while ambient
        env carries the launch profile's value — still missing."""
        monkeypatch.setenv("SIBLING_PROFILE_TOKEN", "sibling-token")
        skills_dir = tmp_path / "skills"
        _make_skill(skills_dir, "sibling-gated", "SIBLING_PROFILE_TOKEN")

        with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
            result = json.loads(skill_view("sibling-gated"))

        assert result["setup_needed"] is True
        assert result["missing_required_environment_variables"] == [
            "SIBLING_PROFILE_TOKEN"
        ]


class TestCronPreflight:
    def test_scoped_var_job_passes_preflight(
        self, tmp_path, monkeypatch, multiplex_scope
    ):
        """End-to-end: with the var resolved by the routed profile's scope the
        cron skills preflight must not block the job."""
        from cron.scheduler_preflight import _preflight_check_skills

        monkeypatch.delenv("SCOPED_SERVICE_TOKEN", raising=False)
        skills_dir = tmp_path / "skills"
        _make_skill(skills_dir, "scoped-gated", "SCOPED_SERVICE_TOKEN")

        env_passthrough_restore = patch("tools.skills_tool.SKILLS_DIR", skills_dir)
        with env_passthrough_restore:
            assert _preflight_check_skills({"skills": ["scoped-gated"]}) is None

    def test_missing_var_job_still_blocks(self, tmp_path, monkeypatch, multiplex_scope):
        from cron.scheduler_preflight import _preflight_check_skills

        monkeypatch.delenv("NOWHERE_PROVIDED_TOKEN", raising=False)
        _make_skill(tmp_path / "skills", "ungated-elsewhere", "NOWHERE_PROVIDED_TOKEN")

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"):
            reason = _preflight_check_skills({"skills": ["ungated-elsewhere"]})

        assert reason is not None
        assert "NOWHERE_PROVIDED_TOKEN" in reason


class TestEmptyScopeValueParity:
    def test_empty_scope_value_stays_missing(self, tmp_path, monkeypatch):
        """Parity with the empty-``.env``-value rule: an empty string in the
        scope does not satisfy the requirement."""
        token = secret_scope.set_secret_scope({"BLANK_SCOPE_TOKEN": ""})
        try:
            monkeypatch.delenv("BLANK_SCOPE_TOKEN", raising=False)
            skills_dir = tmp_path / "skills"
            _make_skill(skills_dir, "blank-gated", "BLANK_SCOPE_TOKEN")

            with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
                result = json.loads(skill_view("blank-gated"))

            assert result["setup_needed"] is True
            assert result["missing_required_environment_variables"] == [
                "BLANK_SCOPE_TOKEN"
            ]
        finally:
            secret_scope.reset_secret_scope(token)
