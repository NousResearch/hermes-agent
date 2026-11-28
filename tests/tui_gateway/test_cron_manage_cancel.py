"""cron.manage ``cancel``: a gateway client stops a job's live run.

``hermes cron cancel``, the dashboard's ``POST /api/cron/jobs/{id}/cancel`` and
``cronjob(action="cancel")`` all reach ``cron.cancellation.cancel_run``; the RPC forwards to the
same tool action, so the cancel is per run and the job keeps its schedule.
"""

import json
from contextlib import contextmanager

from tui_gateway import server


@contextmanager
def _home(path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(path)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _profile_with_job(tmp_path, monkeypatch):
    home = tmp_path / "profiles" / "botA"
    (home / "cron").mkdir(parents=True)
    (home / "cron" / "jobs.json").write_text(json.dumps({"jobs": [
        {"id": "job-live", "name": "nightly-report", "prompt": "hello", "enabled": True}]}), encoding="utf-8")
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: home)
    return home


def _manage(**params):
    return server.handle_request({"id": "1", "method": "cron.manage", "params": params})


def test_cancel_stops_that_profiles_live_run_with_the_callers_reason(tmp_path, monkeypatch):
    from cron.cancellation import CancelRequest
    from cron.executions import create_execution, mark_execution_running

    home = _profile_with_job(tmp_path, monkeypatch)
    with _home(home):
        record = create_execution("job-live", source="test")
        mark_execution_running(record["id"])

    resp = _manage(action="cancel", name="nightly-report", reason="stopped from the app", profile="botA")

    assert "result" in resp, resp
    assert resp["result"]["cancelled"] is True
    assert resp["result"]["execution_id"] == record["id"]
    assert resp["result"]["job"]["job_id"] == "job-live"
    with _home(home):  # the run's own cancel check reads this profile's marker
        request = CancelRequest(record["id"])
        assert request.is_set()
        assert request.reason == "stopped from the app"
    assert not CancelRequest(record["id"]).is_set()  # nothing landed in the launch profile
    # Per run, not per schedule: the job is still enabled.
    listed = _manage(action="list", profile="botA")["result"]["jobs"]
    assert [job["enabled"] for job in listed if job["job_id"] == "job-live"] == [True]


def test_cancel_with_nothing_in_flight_is_a_noop_not_an_error(tmp_path, monkeypatch):
    _profile_with_job(tmp_path, monkeypatch)

    resp = _manage(action="cancel", name="job-live", profile="botA")

    assert "result" in resp, resp
    assert resp["result"]["success"] is True
    assert resp["result"]["cancelled"] is False
    assert resp["result"]["execution_id"] is None
