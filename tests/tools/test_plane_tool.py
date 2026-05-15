import json
from pathlib import Path

import pytest

import tools.plane_tool as plane_tool
from toolsets import TOOLSETS, resolve_toolset


@pytest.fixture
def stub_client(monkeypatch):
    class StubClient:
        workspace_slug = "ai_factory"
        project_id = "project-123"

        class config:
            workspace_slug = "ai_factory"
            project_id = "project-123"

        def get_project_identifier(self):
            return "AIFACTORY"

        def get_current_user(self):
            return {"id": "user-1", "email": "emeric@example.com", "display_name": "Emeric"}

        def get_project(self):
            return {"id": "project-123", "name": "AI_Factory", "identifier": "AIFACTORY"}

        def list_states(self):
            return [
                {"id": "s1", "name": "Todo"},
                {"id": "s2", "name": "Done"},
            ]

        def list_labels(self):
            return [
                {"id": "l1", "name": "backend"},
                {"id": "l2", "name": "research"},
            ]

        def list_work_items(self, **kwargs):
            return [
                {
                    "id": "w1",
                    "sequence_id": 1,
                    "name": "Test modifié par Hermes",
                    "priority": "medium",
                    "state": {"id": "s1", "name": "Todo"},
                    "labels": [{"id": "l1", "name": "backend"}],
                    "assignees": [{"id": "u1", "display_name": "Emeric"}],
                    "description_html": "<p>First task</p>",
                },
                {
                    "id": "w2",
                    "sequence_id": 2,
                    "name": "Deuxième carte créée par Hermes",
                    "priority": "high",
                    "state": {"id": "s2", "name": "Done"},
                    "labels": [{"id": "l2", "name": "research"}],
                    "assignees": [{"id": "u2", "display_name": "Nova"}],
                    "description_html": "<p>Second task</p>",
                },
            ]

        def get_work_item(self, **kwargs):
            for item in self.list_work_items():
                if kwargs.get("work_item_id") == item["id"]:
                    return item
                if kwargs.get("sequence_id") == item["sequence_id"]:
                    return item
                rid = f"AIFACTORY-{item['sequence_id']}"
                if kwargs.get("readable_id") == rid:
                    return item
            raise ValueError("not found")

        def resolve_state_id(self, value):
            return {"Todo": "s1", "Done": "s2", "s1": "s1", "s2": "s2"}[value]

        def resolve_label_ids(self, values):
            mapping = {"backend": "l1", "research": "l2", "l1": "l1", "l2": "l2"}
            return [mapping[v] for v in values]

        def create_work_item(self, payload):
            return {
                "id": "w3",
                "sequence_id": 3,
                "name": payload["name"],
                "priority": payload.get("priority"),
                "state": {"id": payload.get("state"), "name": "Todo"},
                "labels": [{"id": lid, "name": lid} for lid in payload.get("labels", [])],
                "assignees": [{"id": aid, "display_name": aid} for aid in payload.get("assignees", [])],
                "description_html": payload.get("description_html"),
            }

        def update_work_item(self, work_item_id, payload):
            self.last_update = {"work_item_id": work_item_id, "payload": payload}
            return {
                "id": work_item_id,
                "sequence_id": 1,
                "name": payload.get("name", "Updated"),
                "priority": payload.get("priority"),
                "state": {"id": payload.get("state"), "name": "Done"},
                "labels": [{"id": lid, "name": lid} for lid in payload.get("labels", [])],
                "assignees": [{"id": aid, "display_name": aid} for aid in payload.get("assignees", [])],
                "description_html": payload.get("description_html"),
            }

        def add_comment(self, work_item_id, comment_html):
            self.last_comment = {"work_item_id": work_item_id, "comment_html": comment_html}
            return {"id": "c1", "issue": work_item_id, "comment_html": comment_html}

    client = StubClient()
    monkeypatch.setattr(plane_tool, "get_plane_client", lambda: client)
    return client


def test_plane_toolset_registered():
    assert "plane" in TOOLSETS
    resolved = resolve_toolset("plane")
    assert "plane_ping" in resolved
    assert "plane_board_snapshot" in resolved
    assert "plane_add_comment" in resolved
    assert "plane_sync_progress" in resolved
    assert "plane_import_to_kanban" in resolved


def test_plane_ping_reports_user_project_and_latency(stub_client):
    data = json.loads(plane_tool._handle_ping({}))
    assert data["ok"] is True
    assert data["latency_ms"] >= 0
    assert data["user_email"] == "emeric@example.com"
    assert data["user"]["display_name"] == "Emeric"
    assert data["workspace"] == "ai_factory"
    assert data["project_name"] == "AI_Factory"
    assert data["project"]["identifier"] == "AIFACTORY"


def test_plane_board_snapshot_summarizes_counts(stub_client):
    data = json.loads(plane_tool._handle_board_snapshot({"include_items_per_state": True, "per_state_limit": 1}))
    assert data["project"] == {"id": "project-123", "name": "AI_Factory", "identifier": "AIFACTORY"}
    assert data["total_items"] == 2
    assert data["counts_by_state"]["Todo"] == 1
    assert data["states"][0] == {"id": "s1", "name": "Todo", "group": None, "count": 1}
    assert data["items_by_state"]["Todo"][0]["readable_id"] == "AIFACTORY-1"
    assert "description_html" not in data["items"][0]
    assert "project_payload" not in data


def test_plane_board_snapshot_verbose_includes_raw_payloads(stub_client):
    data = json.loads(plane_tool._handle_board_snapshot({"verbose": True}))
    assert data["project_payload"]["identifier"] == "AIFACTORY"
    assert data["states_payload"][0]["id"] == "s1"
    assert data["items_payload"][0]["description_html"] == "<p>First task</p>"


def test_plane_list_work_items_filters_by_state_and_query(stub_client):
    data = json.loads(plane_tool._handle_list_work_items({"state": "Done", "query": "Deuxième"}))
    assert data["count"] == 1
    assert data["items"][0]["readable_id"] == "AIFACTORY-2"
    assert data["items"][0]["state_name"] == "Done"
    assert data["items"][0]["assignees_names"] == ["Nova"]
    assert data["items"][0]["url"].endswith("/issues/AIFACTORY-2")
    assert "description_html" not in data["items"][0]


def test_plane_list_work_items_verbose_includes_raw_payloads(stub_client):
    data = json.loads(plane_tool._handle_list_work_items({"state": "Done", "verbose": True}))
    assert data["items_payload"][0]["description_html"] == "<p>Second task</p>"


def test_plane_get_work_item_supports_sequence_id(stub_client):
    data = json.loads(plane_tool._handle_get_work_item({"sequence_id": 1}))
    assert data["item"]["readable_id"] == "AIFACTORY-1"
    assert data["item"]["state_name"] == "Todo"
    assert "description_html" not in data["item"]


def test_plane_get_work_item_verbose_includes_raw_and_enriched_payload(stub_client):
    data = json.loads(plane_tool._handle_get_work_item({"sequence_id": 1, "verbose": True}))
    assert data["item"]["readable_id"] == "AIFACTORY-1"
    assert data["payload"]["description_html"] == "<p>First task</p>"
    assert data["enriched_item"]["state_name"] == "Todo"


def test_plane_create_work_item_normalizes_state_labels_and_markdown(stub_client):
    data = json.loads(plane_tool._handle_create_work_item({
        "name": "Nouvelle tâche",
        "description_markdown": "Bonjour\n\nMonde",
        "state": "Todo",
        "labels": ["backend"],
        "assignees": ["u1"],
    }))
    assert data["item"]["name"] == "Nouvelle tâche"
    assert data["item"]["state"]["id"] == "s1"
    assert data["already_existed"] is False
    assert data["external_source"] == "nova-hermes"
    assert data["external_id"].startswith("plane-create:")
    assert data["external_id_generated"] is True
    assert "Bonjour" in data["item"]["description_html"]


def test_plane_create_work_item_returns_existing_for_same_external_id(monkeypatch):
    class IdempotentClient:
        workspace_slug = "ai_factory"
        project_id = "project-123"

        class config:
            workspace_slug = "ai_factory"
            project_id = "project-123"

        def __init__(self):
            self.created_payloads = []

        def get_project(self):
            return {"id": "project-123", "name": "AI_Factory", "identifier": "AIFACTORY"}

        def find_work_item_by_external_id(self, *, external_source, external_id):
            assert external_source == "nova-hermes"
            assert external_id == "retry-key-1"
            return {
                "id": "w-existing",
                "sequence_id": 7,
                "name": "Already there",
                "external_source": external_source,
                "external_id": external_id,
                "state": {"id": "s1", "name": "Todo"},
            }

        def create_work_item(self, payload):
            self.created_payloads.append(payload)
            raise AssertionError("create_work_item should not be called")

        def resolve_state_id(self, value):
            return None

    client = IdempotentClient()
    monkeypatch.setattr(plane_tool, "get_plane_client", lambda: client)

    data = json.loads(plane_tool._handle_create_work_item({
        "name": "Already there",
        "external_id": "retry-key-1",
    }))

    assert data["already_existed"] is True
    assert data["created"] is None
    assert data["item"]["id"] == "w-existing"
    assert data["external_id_generated"] is False
    assert client.created_payloads == []


def test_plane_create_work_item_generates_same_external_id_from_normalized_name(stub_client):
    first = json.loads(plane_tool._handle_create_work_item({"name": "  Same   Name  "}))
    second = json.loads(plane_tool._handle_create_work_item({"name": "same name"}))

    assert first["external_id"] == second["external_id"]
    assert first["external_id_generated"] is True


def test_plane_create_work_item_recovers_from_409_when_lookup_misses(monkeypatch):
    from tools.plane_client import PlaneAPIError

    class FlakyLookupClient:
        workspace_slug = "ai_factory"
        project_id = "project-123"

        class config:
            workspace_slug = "ai_factory"
            project_id = "project-123"

        def __init__(self):
            self.find_calls = 0
            self.create_calls = 0

        def get_project(self):
            return {"id": "project-123", "name": "AI_Factory", "identifier": "AIFACTORY"}

        def get_project_identifier(self):
            return "AIFACTORY"

        def find_work_item_by_external_id(self, *, external_source, external_id):
            self.find_calls += 1
            if self.find_calls == 1:
                return None
            return {
                "id": "w-existing",
                "sequence_id": 42,
                "name": "Already there",
                "external_source": external_source,
                "external_id": external_id,
                "state": {"id": "s1", "name": "Todo"},
            }

        def create_work_item(self, payload):
            self.create_calls += 1
            raise PlaneAPIError(
                409,
                '{"error":"duplicate external id"}',
                "https://api.plane.so/work-items/",
            )

        def resolve_state_id(self, value):
            return None

    client = FlakyLookupClient()
    monkeypatch.setattr(plane_tool, "get_plane_client", lambda: client)

    data = json.loads(plane_tool._handle_create_work_item({
        "name": "Already there",
        "external_id": "retry-key-409",
    }))

    assert data["success"] is True
    assert data["already_existed"] is True
    assert data["created"] is None
    assert data["item"]["id"] == "w-existing"
    assert data["external_id"] == "retry-key-409"
    assert data["external_id_generated"] is False
    assert client.find_calls == 2
    assert client.create_calls == 1


def test_plane_update_work_item_requires_changes(stub_client):
    raw = plane_tool._handle_update_work_item({"work_item_id": "w1"})
    assert "at least one updatable field is required" in raw


def test_plane_update_work_item_nominal_path(stub_client):
    data = json.loads(plane_tool._handle_update_work_item({
        "work_item_id": "w1",
        "name": "Renamed via Hermes",
        "state": "Done",
        "labels": ["backend"],
        "priority": "high",
    }))
    assert data["success"] is True
    assert data["item"]["id"] == "w1"
    payload = stub_client.last_update["payload"]
    assert stub_client.last_update["work_item_id"] == "w1"
    assert payload["name"] == "Renamed via Hermes"
    assert payload["state_id"] == "s2"
    assert payload["labels"] == ["l1"]
    assert payload["priority"] == "high"


def test_plane_update_work_item_ignores_none_fields_in_patch_payload(stub_client):
    data = json.loads(plane_tool._handle_update_work_item({
        "work_item_id": "w1",
        "name": "New name only",
        "priority": None,
        "state": None,
        "labels": None,
        "assignees": None,
        "target_date": None,
        "description_markdown": None,
    }))
    assert data["success"] is True
    payload = stub_client.last_update["payload"]
    assert payload == {"name": "New name only"}


def test_plane_update_work_item_empty_list_clears_labels_and_assignees(stub_client):
    json.loads(plane_tool._handle_update_work_item({
        "work_item_id": "w1",
        "labels": [],
        "assignees": [],
    }))
    payload = stub_client.last_update["payload"]
    assert payload.get("labels") == []
    assert payload.get("assignees") == []


def test_plane_add_comment_supports_sequence_id_and_default_nova_prefix(stub_client):
    data = json.loads(plane_tool._handle_add_comment({
        "sequence_id": 1,
        "body_markdown": "Terminé\nProchaine étape: revue",
    }))
    assert data["success"] is True
    assert data["comment"]["issue"] == "w1"
    assert data["item"]["readable_id"] == "AIFACTORY-1"
    assert "[Nova] Terminé" in data["comment_html"]
    assert "<br>" in data["comment_html"]


def test_plane_add_comment_can_disable_prefix(stub_client):
    data = json.loads(plane_tool._handle_add_comment({
        "work_item_id": "w1",
        "body_markdown": "Message brut",
        "prefix": False,
    }))
    assert "[Nova]" not in data["comment_html"]
    assert "Message brut" in data["comment_html"]


def test_parse_plane_linkage_from_imported_kanban_body():
    body = """Imported from Plane AIFACTORY-12

plane_workspace_slug: ai_factory
plane_project_id: project-123
plane_work_item_id: w1
plane_sequence_id: 12
plane_url: https://app.plane.so/ai_factory/projects/project-123/issues/AIFACTORY-12
plane_state_id: s1
"""
    linkage = plane_tool._parse_plane_linkage_from_body(body)
    assert linkage["plane_work_item_id"] == "w1"
    assert linkage["plane_sequence_id"] == 12
    assert linkage["plane_url"].endswith("AIFACTORY-12")


def test_plane_sync_progress_comments_and_updates_status(monkeypatch, stub_client):
    monkeypatch.setattr(
        plane_tool,
        "_lookup_plane_link_from_kanban_task",
        lambda hermes_card_id: {
            "hermes_card_id": hermes_card_id,
            "plane_work_item_id": "w1",
            "plane_sequence_id": 1,
            "plane_url": "https://app.plane.so/ai_factory/projects/project-123/issues/AIFACTORY-1",
        },
    )

    data = json.loads(plane_tool._handle_sync_progress({
        "hermes_card_id": "t_imported",
        "summary": "Implémentation terminée",
        "status": "Done",
    }))

    assert data["success"] is True
    assert data["hermes_card_id"] == "t_imported"
    assert data["item"]["readable_id"] == "AIFACTORY-1"
    assert data["status"] == "Done"
    assert data["status_updated"] is True
    assert stub_client.last_update == {"work_item_id": "w1", "payload": {"state_id": "s2", "state": "s2"}}
    assert stub_client.last_comment["work_item_id"] == "w1"
    assert "[Nova] Implémentation terminée" in stub_client.last_comment["comment_html"]


def test_plane_sync_progress_requires_linked_hermes_card(monkeypatch, stub_client):
    def missing_link(card_id):
        raise ValueError(f"Hermes kanban task {card_id} is not linked to a Plane work item")

    monkeypatch.setattr(plane_tool, "_lookup_plane_link_from_kanban_task", missing_link)

    data = json.loads(plane_tool._handle_sync_progress({
        "hermes_card_id": "t_plain",
        "summary": "Avancement",
    }))

    assert "not linked to a Plane work item" in data["error"]


def test_plane_prepare_workdir_creates_expected_structure(tmp_path):
    data = json.loads(plane_tool._handle_prepare_workdir({
        "sequence_id": 12,
        "title": "RAG benchmark",
        "base_dir": str(tmp_path),
        "project_key": "AIFACTORY",
    }))
    task_dir = Path(data["workdir"])
    assert task_dir.name.startswith("AIFACTORY-12_")
    assert task_dir.exists()
    assert (task_dir / "work").is_dir()
    assert (task_dir / "deliverables").is_dir()
    assert (task_dir / "README.md").exists()
    readme = (task_dir / "README.md").read_text(encoding="utf-8")
    assert "AIFACTORY-12" in readme


def test_plane_prepare_workdir_honours_custom_project_key(tmp_path):
    data = json.loads(plane_tool._handle_prepare_workdir({
        "sequence_id": 7,
        "title": "Test",
        "base_dir": str(tmp_path),
        "project_key": "novax",
    }))
    task_dir = Path(data["workdir"])
    assert task_dir.name.startswith("NOVAX-7_")
    readme = (task_dir / "README.md").read_text(encoding="utf-8")
    assert "NOVAX-7" in readme


def test_plane_import_to_kanban_workdir_uses_real_project_key(
    monkeypatch, tmp_path
):
    class CustomKeyClient:
        workspace_slug = "ai_factory"
        project_id = "project-123"

        class config:
            workspace_slug = "ai_factory"
            project_id = "project-123"

        def get_project_identifier(self):
            return "NOVAX"

        def get_project(self):
            return {"id": "project-123", "name": "Nova X", "identifier": "NOVAX"}

        def list_work_items(self, **kwargs):
            return [{
                "id": "w42",
                "sequence_id": 42,
                "name": "Carte custom",
                "state": {"id": "s1", "name": "Todo"},
            }]

        def get_work_item(self, **kwargs):
            return self.list_work_items()[0]

    client = CustomKeyClient()
    monkeypatch.setattr(plane_tool, "get_plane_client", lambda: client)

    class FakeConn:
        def execute(self, sql, params=()):
            class _Cur:
                def fetchone(self):
                    return None

            return _Cur()

        def close(self):
            pass

    monkeypatch.setattr("hermes_cli.kanban_db.init_db", lambda: None)
    monkeypatch.setattr("hermes_cli.kanban_db.connect", lambda: FakeConn())
    monkeypatch.setattr(
        "hermes_cli.kanban_db.create_task",
        lambda conn, **kwargs: "task-1",
    )

    data = json.loads(plane_tool._handle_import_to_kanban({
        "sequence_ids": [42],
        "assignee": "emeric",
        "create_workdir": True,
        "workdir_base_dir": str(tmp_path),
    }))
    workdir = Path(data["created_tasks"][0]["workdir"])
    assert workdir.name.startswith("NOVAX-42_")


def test_plane_import_to_kanban_creates_kanban_task_with_plane_linkage(
    monkeypatch, stub_client, tmp_path
):
    class FakeConn:
        def execute(self, sql, params=()):
            class _Cur:
                def fetchone(self):
                    return None

            return _Cur()

        def close(self):
            pass

    captured: dict[str, list] = {"calls": []}

    def fake_create_task(conn, **kwargs):
        captured["calls"].append(kwargs)
        return "task-newly-imported"

    monkeypatch.setattr("hermes_cli.kanban_db.init_db", lambda: None)
    monkeypatch.setattr("hermes_cli.kanban_db.connect", lambda: FakeConn())
    monkeypatch.setattr("hermes_cli.kanban_db.create_task", fake_create_task)

    data = json.loads(plane_tool._handle_import_to_kanban({
        "sequence_ids": [1],
        "assignee": "emeric",
        "create_workdir": True,
        "workdir_base_dir": str(tmp_path),
    }))

    assert data["success"] is True
    assert len(data["created_tasks"]) == 1
    created_task = data["created_tasks"][0]
    assert created_task["task_id"] == "task-newly-imported"
    assert created_task["plane_work_item_id"] == "w1"
    assert created_task["plane_sequence_id"] == 1
    assert created_task["already_imported"] is False
    assert Path(created_task["workdir"]).exists()

    assert len(captured["calls"]) == 1
    create_kwargs = captured["calls"][0]
    assert create_kwargs["title"] == "[Plane AIFACTORY-1] Test modifié par Hermes"
    assert create_kwargs["assignee"] == "emeric"
    assert create_kwargs["idempotency_key"] == "plane:ai_factory:project-123:w1"
    assert "plane_work_item_id: w1" in create_kwargs["body"]
    assert "plane_sequence_id: 1" in create_kwargs["body"]
    assert "plane_url:" in create_kwargs["body"]


def test_plane_import_to_kanban_flags_already_imported_when_idempotency_hits(
    monkeypatch, stub_client
):
    class FakeConn:
        def execute(self, sql, params=()):
            class _Cur:
                def fetchone(self):
                    return {"id": "task-existing-row"}

            return _Cur()

        def close(self):
            pass

    monkeypatch.setattr("hermes_cli.kanban_db.init_db", lambda: None)
    monkeypatch.setattr("hermes_cli.kanban_db.connect", lambda: FakeConn())
    monkeypatch.setattr(
        "hermes_cli.kanban_db.create_task",
        lambda conn, **kwargs: "task-existing-row",
    )

    data = json.loads(plane_tool._handle_import_to_kanban({
        "sequence_ids": [1],
        "assignee": "emeric",
    }))

    assert data["success"] is True
    created_task = data["created_tasks"][0]
    assert created_task["task_id"] == "task-existing-row"
    assert created_task["already_imported"] is True
    assert created_task["workdir"] is None
