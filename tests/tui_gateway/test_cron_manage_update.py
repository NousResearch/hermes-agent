"""cron.manage ``update`` — two invariant tests (adapter forwarding + contract/scope).

Exercises the real backend (``server.handle_request`` -> ``cronjob()`` /
``cron.jobs``) against a temp ``HERMES_HOME``: no mocks of the cron store, no
source reads. Mirrors ``tests/cron/test_cron_manage_profile_scope.py``'s harness.
"""

import pytest

from tui_gateway import server

LONG_PROMPT = "P" * 150
OTHER_PROMPT = "Q" * 160


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so jobs.json never touches the real store."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def _call(params, rid="1"):
    return server.handle_request({"id": rid, "method": "cron.manage", "params": params})


def _add(name="job", schedule="every 5m", prompt=LONG_PROMPT, **extra):
    resp = _call({"action": "add", "name": name, "schedule": schedule,
                   "prompt": prompt, **extra})
    assert "result" in resp, resp
    assert resp["result"]["success"] is True, resp
    return resp["result"]["job_id"]


def test_cron_manage_update_forwards_allowed_fields(temp_home):
    from cron.jobs import get_job

    jid = _add(name="before")
    resp = _call({"action": "update", "job_id": jid, "name": "after",
                   "prompt": OTHER_PROMPT, "schedule": "every 10m",
                   "repeat": "forever", "continuity": True, "deliver": "all"})
    assert resp["result"]["success"] is True, resp
    job = resp["result"]["job"]
    assert job["name"] == "after"
    assert job["schedule"] == "every 10m"
    assert job["repeat"] == "forever"
    assert job["deliver"] == "all"
    assert job.get("continuity") is True

    stored = get_job(jid)
    assert stored["name"] == "after"
    assert stored["prompt"] == OTHER_PROMPT
    assert (stored.get("repeat") or {}).get("times") is None
    assert stored["deliver"] == "all"


def test_cron_manage_update_contract_and_profile_scope(temp_home, monkeypatch):
    assert _call({"action": "update", "name": "x"})["error"]["code"] == 4063
    assert _call({"action": "list", "bogus_field": 123})["error"]["code"] == 4000

    missing = _call({"action": "update", "job_id": "does-not-exist", "name": "x"})
    assert missing["result"]["success"] is False, missing
    assert "not found" in missing["result"]["error"], missing

    job_a = _add(name="job-in-A")
    profile_b = temp_home / "profiles" / "scopeB"
    profile_b.mkdir(parents=True)

    import hermes_cli.profiles as profiles

    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: profile_b)

    job_b = _add(name="job-in-B", profile="scopeB")
    scoped = _call({"action": "list", "profile": "scopeB"})
    assert scoped["result"]["scoped"] == "scopeB", scoped
    names_b = [j["name"] for j in scoped["result"]["jobs"]]
    assert "job-in-B" in names_b
    assert "job-in-A" not in names_b

    # B is invisible without its profile: the update finds nothing.
    assert _call({"action": "update", "job_id": job_b,
                   "name": "renamed-elsewhere"})["result"]["success"] is False

    updated = _call({"action": "update", "job_id": job_b,
                      "name": "job-in-B-renamed", "profile": "scopeB"})
    assert updated["result"]["success"] is True, updated
    assert updated["result"]["job"]["name"] == "job-in-B-renamed", updated

    again = _call({"action": "list"})
    names_a = [j["name"] for j in again["result"]["jobs"]]
    assert "job-in-A" in names_a
    assert "job-in-B" not in names_a
    assert "job-in-B-renamed" not in names_a

    from hermes_constants import get_hermes_home_override

    assert get_hermes_home_override() is None
    assert job_a != job_b
