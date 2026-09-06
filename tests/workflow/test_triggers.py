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
