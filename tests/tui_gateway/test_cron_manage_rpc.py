"""``cron.manage`` adapts ``tools.cronjob_tools.cronjob`` JSON into the closed ``CronManageResult``.

Each action fills a different subset of the result keys, and ``_format_job`` only emits the
optional row keys (script, workdir, continuity, ...) when they are set, so the models must
default them — otherwise every action after ``list`` on an empty store answers ``5023``.
"""

import json

import pytest


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tui_gateway import server

    return server


def _manage(server, **params):
    return server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "cron.manage", "params": params})


def test_cron_manage_every_action_answers_a_result(server):
    added = _manage(server, action="add", name="probe", schedule="every 1h", prompt="hi")
    assert "result" in added, added
    job_id = added["result"]["job_id"]
    assert job_id

    listed = _manage(server, action="list")
    assert "result" in listed, listed
    assert [row["job_id"] for row in listed["result"]["jobs"]] == [job_id]

    for action in ("pause", "resume", "remove"):
        answer = _manage(server, action=action, name=job_id)
        assert "result" in answer and answer["result"]["success"] is True, (action, answer)

    assert json.loads(json.dumps(_manage(server, action="list")))["result"]["count"] == 0
