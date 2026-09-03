"""Tests for cron.required_skills config enforcement (issue #79797).

Covers the create/update gate in tools/cronjob_tools.py: the agent tool and
the hermes cron CLI both route through cronjob(), so these tests exercise the
unified tool surface with cron.* config patched via load_config().
"""

import json

import pytest

from tools.cronjob_tools import cronjob


def _config(required_skills=None, enforce=True, agent_only=True):
    cfg = {"cron": {}}
    if required_skills is not None:
        cfg["cron"]["required_skills"] = required_skills
    cfg["cron"]["required_skills_enforce"] = enforce
    cfg["cron"]["required_skills_agent_only"] = agent_only
    return cfg


class TestCronRequiredSkills:
    @pytest.fixture(autouse=True)
    def _isolate_cron_store(self, tmp_path, monkeypatch):
        monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
        monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
        monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
        # Point the script-path validator at the temp dir so no_agent jobs can
        # reference a real (relative) script without touching the real home.
        import hermes_constants

        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "watch.sh").write_text("echo hi\n")

    def _set_required_skills(self, monkeypatch, **kwargs):
        import hermes_cli.config as hcfg

        monkeypatch.setattr(hcfg, "load_config", lambda: _config(**kwargs))

    def _create(self, **kwargs):
        return json.loads(cronjob(action="create", schedule="every 1h", **kwargs))

    def test_create_rejected_when_missing_required_skill(self, monkeypatch):
        self._set_required_skills(monkeypatch, required_skills=["cron-output"])
        result = self._create(prompt="send me the daily brief")
        assert result["success"] is False
        assert "cron.required_skills" in result["error"]
        assert "cron-output" in result["error"]

    def test_create_ok_when_required_skill_present(self, monkeypatch):
        self._set_required_skills(monkeypatch, required_skills=["cron-output"])
        result = self._create(prompt="brief", skills=["cron-output"])
        assert result["success"] is True

    def test_create_ok_when_no_required_skills_configured(self, monkeypatch):
        self._set_required_skills(monkeypatch, required_skills=[])
        result = self._create(prompt="brief")
        assert result["success"] is True

    def test_create_no_agent_exempt_by_default(self, monkeypatch):
        self._set_required_skills(monkeypatch, required_skills=["cron-output"])
        result = self._create(prompt=None, script="watch.sh", no_agent=True)
        assert result["success"] is True

    def test_create_no_agent_checked_when_agent_only_false(self, monkeypatch):
        self._set_required_skills(
            monkeypatch, required_skills=["cron-output"], agent_only=False
        )
        result = self._create(prompt=None, script="watch.sh", no_agent=True)
        assert result["success"] is False
        assert "cron-output" in result["error"]

    def test_update_clear_skills_rejected(self, monkeypatch):
        self._set_required_skills(monkeypatch, required_skills=["cron-output"])
        created = self._create(prompt="brief", skills=["cron-output"])
        result = json.loads(
            cronjob(action="update", job_id=created["job_id"], skills=[])
        )
        assert result["success"] is False
        assert "cron-output" in result["error"]

    def test_update_replacing_skills_with_missing_required_rejected(self, monkeypatch):
        self._set_required_skills(monkeypatch, required_skills=["cron-output"])
        created = self._create(prompt="brief", skills=["cron-output"])
        result = json.loads(
            cronjob(
                action="update",
                job_id=created["job_id"],
                skills=["other-skill"],
            )
        )
        assert result["success"] is False
        assert "cron-output" in result["error"]

    def test_update_adding_required_skill_ok(self, monkeypatch):
        # Job created before the policy existed (nothing required at the time).
        self._set_required_skills(monkeypatch, required_skills=[])
        created = self._create(prompt="brief")
        # Policy lands later; an update that adds the skill satisfies it.
        self._set_required_skills(monkeypatch, required_skills=["cron-output"])
        result = json.loads(
            cronjob(
                action="update",
                job_id=created["job_id"],
                skills=["cron-output"],
            )
        )
        assert result["success"] is True

    def test_update_unrelated_field_allowed_on_noncompliant_job(self, monkeypatch):
        # A legacy job created before the policy is not trapped: updates that
        # leave the skills axis untouched still go through, so it can be
        # remediated field-by-field.
        self._set_required_skills(monkeypatch, required_skills=[])
        created = self._create(prompt="brief")
        self._set_required_skills(monkeypatch, required_skills=["cron-output"])
        result = json.loads(
            cronjob(action="update", job_id=created["job_id"], name="renamed")
        )
        assert result["success"] is True
        assert result["job"]["name"] == "renamed"

    def test_update_flip_no_agent_off_checked(self, monkeypatch):
        self._set_required_skills(monkeypatch, required_skills=["cron-output"])
        created = self._create(prompt=None, script="watch.sh", no_agent=True)
        # Flipping to agent mode without the required skill is rejected.
        result = json.loads(
            cronjob(action="update", job_id=created["job_id"], no_agent=False)
        )
        assert result["success"] is False
        assert "cron-output" in result["error"]

    def test_enforce_false_allows_with_warning(self, monkeypatch):
        self._set_required_skills(
            monkeypatch, required_skills=["cron-output"], enforce=False
        )
        result = self._create(prompt="brief")
        assert result["success"] is True

    def test_force_bypasses_gate(self, monkeypatch):
        self._set_required_skills(monkeypatch, required_skills=["cron-output"])
        result = self._create(prompt="brief", force=True)
        assert result["success"] is True


class TestConfigAccessors:
    def test_string_form_split(self):
        from hermes_cli.config import cron_required_skills

        assert cron_required_skills(
            {"cron": {"required_skills": "cron-output, plain-english"}}
        ) == ["cron-output", "plain-english"]

    def test_list_form(self):
        from hermes_cli.config import cron_required_skills

        assert cron_required_skills(
            {"cron": {"required_skills": ["cron-output"]}}
        ) == ["cron-output"]

    def test_garbage_yields_empty(self):
        from hermes_cli.config import cron_required_skills

        assert cron_required_skills({"cron": {"required_skills": 42}}) == []
        assert cron_required_skills({"cron": {}}) == []
        assert cron_required_skills({}) == []

    def test_only_literal_false_disables(self):
        from hermes_cli.config import (
            cron_required_skills_agent_only,
            cron_required_skills_enforce,
        )

        assert (
            cron_required_skills_enforce(
                {"cron": {"required_skills_enforce": False}}
            )
            is False
        )
        assert (
            cron_required_skills_enforce(
                {"cron": {"required_skills_enforce": "false"}}
            )
            is True
        )
        assert (
            cron_required_skills_agent_only(
                {"cron": {"required_skills_agent_only": False}}
            )
            is False
        )
        assert cron_required_skills_agent_only({"cron": {}}) is True
