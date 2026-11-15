"""Cron create/update must not store a ``--skill`` name that resolves to nothing (#151012).

``--script`` is validated against the scripts directory at create time; ``--skill`` was not, so a
typo (or a stale name after the skill was uninstalled) was saved silently and the job then ran
without the skill's context — the failure only surfaced when the job fired.

The validator mirrors the run-time loader (``cron.scheduler_prompt._load_cron_skill_parts``): a
bundle key shadows a same-slug skill, otherwise ``skill_view`` on the slash-normalized name decides.
So nothing that WOULD load at run time may be refused here.
"""

import json

import pytest

from cron.jobs import get_job
from tools.cronjob_tools import cronjob


@pytest.fixture(autouse=True)
def _cron_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")


@pytest.fixture
def installed_skill():
    """A real skill and a real bundle in the sandboxed home, so both resolve like a live install."""
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    skill_dir = home / "skills" / "probe-skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: probe-skill\ndescription: Fixture skill for cron validation tests.\n---\n\nBody.\n",
        encoding="utf-8",
    )
    bundles = home / "skill-bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    (bundles / "probe-bundle.yaml").write_text(
        "name: probe-bundle\ndescription: Fixture bundle for cron validation tests.\n"
        "skills:\n  - probe-skill\n",
        encoding="utf-8",
    )
    return "probe-skill"


def test_create_rejects_unknown_skill():
    result = json.loads(
        cronjob(
            action="create",
            schedule="every 1h",
            prompt="probe",
            name="probe-skill-create",
            skill="nonexistent-skill-xyz",
        )
    )
    assert result["success"] is False
    assert "nonexistent-skill-xyz" in result["error"]
    # The rejected call must not leave a job behind.
    assert json.loads(cronjob(action="list"))["count"] == 0


def test_create_accepts_installed_skill(installed_skill):
    result = json.loads(
        cronjob(
            action="create",
            schedule="every 1h",
            prompt="probe",
            name="probe-skill-create",
            skill=installed_skill,
        )
    )
    assert result["success"] is True, result.get("error")
    assert result["skills"] == [installed_skill]
    stored = get_job(result["job_id"])
    assert stored is not None and stored["skills"] == [installed_skill]


def test_create_accepts_absolute_skill_path(installed_skill):
    # A trusted absolute path (the form slash commands and older cron rows store) normalizes to
    # the relative name at run time, so create must accept it too.
    from hermes_constants import get_hermes_home

    absolute = str(get_hermes_home() / "skills" / installed_skill)
    result = json.loads(
        cronjob(
            action="create",
            schedule="every 1h",
            prompt="probe",
            name="probe-skill-absolute",
            skill=absolute,
        )
    )
    assert result["success"] is True, result.get("error")


def test_create_accepts_bundle_name(installed_skill):
    # A bundle shadows a same-slug skill at run time; attaching one by name stays valid.
    result = json.loads(
        cronjob(
            action="create",
            schedule="every 1h",
            prompt="probe",
            name="probe-bundle-create",
            skills=["probe-bundle"],
        )
    )
    assert result["success"] is True, result.get("error")
    assert result["skills"] == ["probe-bundle"]


def test_update_rejects_unknown_skill(installed_skill):
    created = json.loads(
        cronjob(
            action="create",
            schedule="every 1h",
            prompt="probe",
            name="probe-skill-update",
            skill=installed_skill,
        )
    )
    assert created["success"] is True, created.get("error")
    job_id = created["job_id"]

    updated = json.loads(cronjob(action="update", job_id=job_id, skills=["nonexistent-skill-xyz"]))
    assert updated["success"] is False
    assert "nonexistent-skill-xyz" in updated["error"]
    # Refused, so the stored skill list is untouched.
    stored = get_job(job_id)
    assert stored is not None and stored["skills"] == [installed_skill]
