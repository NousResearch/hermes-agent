"""Schedule-update request policy is typed, atomic and separate from stored job state."""
import copy
import json
from pathlib import Path

import pytest

from cron import jobs
from tools import cronjob_tools  # registers the real handler


def _home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    return tmp_path / ".hermes"


def _dispatch(**args):
    return json.loads(cronjob_tools.registry.dispatch("cronjob_manage", args))


def _create():
    result = _dispatch(action="create", prompt="Check status", schedule="every 1h", preserve_lifecycle=True)
    assert result["success"] is True
    job = jobs.get_job(result["job_id"])
    assert job["enabled"] is True and job["state"] == "scheduled"
    assert "preserve_lifecycle" not in job
    return job["id"]


@pytest.mark.parametrize("surface", ["tool", "cli"])
@pytest.mark.parametrize("lifecycle", ["disabled", "paused"])
@pytest.mark.parametrize("preserve", [True, False, None, "omitted"])
def test_schedule_edit_preserves_or_rearms_as_requested(tmp_path, monkeypatch, surface, lifecycle, preserve):
    _home(tmp_path, monkeypatch)
    job_id = _create()
    if lifecycle == "paused":
        assert _dispatch(action="pause", job_id=job_id)["success"] is True
    else:
        jobs.update_job(job_id, {"enabled": False, "state": "scheduled"})
    if surface == "tool":
        args = {"action": "update", "job_id": job_id, "schedule": "every 2h", "prompt": "Revised status"}
        if preserve != "omitted":
            args["preserve_lifecycle"] = preserve
        assert _dispatch(**args)["success"] is True
    else:
        from hermes_cli import main
        parser, _ = main._build_cli_parser()
        argv = ["cron", "edit", job_id, "--schedule", "every 2h", "--prompt", "Revised status"]
        if preserve is True:
            argv.append("--preserve-lifecycle")
        args = parser.parse_args(argv)
        assert args.func(args) == 0
    job = jobs.get_job(job_id)
    assert job["schedule"]["minutes"] == 120
    assert job["prompt"] == "Revised status"
    assert job["enabled"] is (False if preserve is True or lifecycle == "paused" else True)
    assert job["state"] == ("paused" if lifecycle == "paused" else "scheduled")
    assert "preserve_lifecycle" not in job


@pytest.mark.parametrize("value", ["false", "true", "", 0, 1, [], {}])
def test_nonboolean_flag_refuses_the_entire_update(tmp_path, monkeypatch, value):
    home = _home(tmp_path, monkeypatch)
    job_id = _create()
    jobs.update_job(job_id, {"enabled": False, "state": "scheduled"})
    before = copy.deepcopy(jobs.get_job(job_id))
    store = home / "cron" / "jobs.json"
    original_bytes = store.read_bytes()
    result = _dispatch(action="update", job_id=job_id, schedule="every 2h",
                        prompt="Revised status", preserve_lifecycle=value)
    actual = jobs.get_job(job_id)
    assert result["success"] is False, f"value={value!r}; stored enabled={actual['enabled']!r}; result={result}"
    assert "boolean" in result["error"]
    assert actual == before
    assert store.read_bytes() == original_bytes
