"""Trigger sync writes webhook routes the existing adapter already reloads."""

from workflow.store import save_documents
from workflow.triggers import route_name, sync_webhook_routes


def test_webhook_trigger_registers_a_dynamic_route(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    save_documents(
        [
            {
                "id": "ship",
                "name": "Ship",
                "scenario": {
                    "steps": [
                        {
                            "id": "go",
                            "kind": "trigger",
                            "config": {"title": "Hook", "on": {"type": "webhook", "spec": ""}},
                        }
                    ],
                    "edges": [],
                },
            }
        ],
        "ship",
    )
    secrets = sync_webhook_routes()
    assert "ship" in secrets
    assert secrets["ship"]["url"].endswith(f"/webhooks/{secrets['ship']['route']}")
    assert secrets["ship"]["route"].startswith("wf-")
    assert secrets["ship"]["route"] != "wf:ship"
    subs = (home / "webhook_subscriptions.json").read_text(encoding="utf-8")
    assert route_name("ship") in subs
    assert '"workflow": "ship"' in subs
    assert '"hermes_workflow": true' in subs


def test_manual_trigger_does_not_register_a_route(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    save_documents(
        [
            {
                "id": "hand",
                "name": "Hand",
                "scenario": {
                    "steps": [
                        {
                            "id": "go",
                            "kind": "trigger",
                            "config": {"title": "Play", "on": {"type": "manual", "spec": ""}},
                        }
                    ],
                    "edges": [],
                },
            }
        ],
        "hand",
    )
    assert sync_webhook_routes() == {}
    path = home / "webhook_subscriptions.json"
    if path.exists():
        assert route_name("hand") not in path.read_text(encoding="utf-8")


def test_each_cron_trigger_is_its_own_job_that_publishes_its_start(tmp_path, monkeypatch):
    import runpy

    from cron.jobs import list_jobs
    from workflow import events
    from workflow.triggers import sync_cron_jobs

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    steps = [
        {"id": "daily", "kind": "trigger", "config": {"on": {"type": "cron", "spec": "0 9 * * *"}}},
        {"id": "weekly", "kind": "trigger", "config": {"on": {"type": "cron", "spec": "0 9 * * 1"}}},
    ]
    save_documents([{"id": "digest", "name": "Digest", "scenario": {"steps": steps, "edges": []}}], "digest")
    assert len(sync_cron_jobs()) == 2
    jobs = {job["origin"]["node_id"]: job for job in list_jobs(include_disabled=True)}
    assert set(jobs) == {"daily", "weekly"}

    runpy.run_path(str(home / "scripts" / jobs["weekly"]["script"]))
    (start,) = events.drain()
    assert start["name"] == events.START
    assert start["payload"] == {"workflowId": "digest", "trigger": "weekly", "source": "cron"}
