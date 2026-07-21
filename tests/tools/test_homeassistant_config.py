import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.homeassistant_client import HomeAssistantClient, HomeAssistantClientError
from tools.homeassistant_config import HomeAssistantResources


def test_client_rejects_unsafe_or_credentialed_base_urls():
    for url in (
        "file:///tmp/ha",
        "http://user:pass@ha.local:8123",
        "http://ha.local:8123/api?token=secret",
        "http://ha.local:8123/#fragment",
    ):
        with pytest.raises(ValueError):
            HomeAssistantClient(url, "token")


def test_client_requires_token():
    with pytest.raises(ValueError, match="token"):
        HomeAssistantClient("http://ha.local:8123", "")


@pytest.mark.asyncio
async def test_resource_capabilities_are_explicit_and_bounded():
    client = MagicMock()
    client.rest = AsyncMock(
        return_value={"version": "2026.7.0", "components": ["automation", "timer"]}
    )
    resources = HomeAssistantResources(client)

    capabilities = await resources.capabilities()

    assert capabilities["home_assistant_version"] == "2026.7.0"
    assert capabilities["mutable"]["automation"] == ["create", "update"]
    assert capabilities["mutable"]["timer"] == ["create", "update"]
    assert "input_boolean" in capabilities["unavailable"]
    assert capabilities["mutable"]["entity"] == ["update"]
    assert "dashboard" in capabilities["excluded"]
    assert "integration" in capabilities["excluded"]


@pytest.mark.asyncio
async def test_rest_resource_get_and_apply_use_official_config_endpoint():
    client = MagicMock()
    client.rest = AsyncMock(side_effect=[{"alias": "Old"}, {"result": "ok"}])
    resources = HomeAssistantResources(client)

    current = await resources.get("automation", "morning")
    result = await resources.apply("automation", "morning", "update", {"alias": "New"})

    assert current == {"alias": "Old"}
    assert result == {"alias": "New"}
    assert client.rest.await_args_list[0].args == ("GET", "/api/config/automation/config/morning")
    assert client.rest.await_args_list[1].args == (
        "POST", "/api/config/automation/config/morning", {"alias": "New"}
    )


@pytest.mark.asyncio
async def test_helper_uses_websocket_collection_commands():
    client = MagicMock()
    client.websocket = AsyncMock(
        side_effect=[
            [{"id": "tea", "name": "Tea"}],
            {"id": "tea", "name": "Tea timer"},
        ]
    )
    resources = HomeAssistantResources(client)

    assert await resources.get("timer", "tea") == {"id": "tea", "name": "Tea"}
    result = await resources.apply("timer", "tea", "update", {"name": "Tea timer"})

    assert result == {"id": "tea", "name": "Tea timer"}
    assert client.websocket.await_args_list[0].args == ({"type": "timer/list"},)
    assert client.websocket.await_args_list[1].args == (
        {"type": "timer/update", "timer_id": "tea", "name": "Tea timer"},
    )


@pytest.mark.asyncio
async def test_rollback_only_deletes_a_resource_hermes_created():
    client = MagicMock()
    client.websocket = AsyncMock(return_value=None)
    resources = HomeAssistantResources(client)

    with pytest.raises(ValueError, match="created by Hermes"):
        await resources.rollback(
            {"resource_type": "timer", "resource_id": "tea", "operation": "create",
             "before": None, "created_by_hermes": False}
        )

    await resources.rollback(
        {"resource_type": "timer", "resource_id": "tea", "operation": "create",
         "before": None, "created_by_hermes": True}
    )
    client.websocket.assert_awaited_once_with({"type": "timer/delete", "timer_id": "tea"})


@pytest.mark.asyncio
async def test_helper_rollback_strips_response_only_fields():
    client = MagicMock()
    client.websocket = AsyncMock(return_value={"id": "tea", "name": "Tea"})
    resources = HomeAssistantResources(client)

    await resources.rollback({
        "resource_type": "timer", "resource_id": "tea", "operation": "update",
        "before": {"id": "tea", "entity_id": "timer.tea", "editable": True,
                   "name": "Tea", "duration": "00:05:00"},
    })

    client.websocket.assert_awaited_once_with({
        "type": "timer/update", "timer_id": "tea", "name": "Tea",
        "duration": "00:05:00",
    })


@pytest.mark.asyncio
async def test_registry_update_rejects_immutable_only_definition():
    resources = HomeAssistantResources(MagicMock())
    with pytest.raises(ValueError, match="no mutable fields"):
        await resources.apply(
            "entity", "light.kitchen", "update",
            {"entity_id": "light.renamed", "unique_id": "immutable"},
        )


@pytest.mark.asyncio
async def test_unknown_resource_type_fails_closed():
    resources = HomeAssistantResources(MagicMock())
    with pytest.raises(ValueError, match="unsupported"):
        await resources.get("dashboard", "lovelace")


@pytest.mark.asyncio
async def test_resource_id_cannot_escape_config_endpoint():
    client = MagicMock()
    resources = HomeAssistantResources(client)
    with pytest.raises(ValueError, match="invalid.*resource_id"):
        await resources.apply(
            "automation", "../../services/persistent_notification/create", "update",
            {"alias": "Nope"},
        )
    client.rest.assert_not_called()


@pytest.mark.asyncio
async def test_group_create_uses_config_entry_flow():
    client = MagicMock()
    client.rest = AsyncMock(side_effect=[
        {"flow_id": "flow-1", "type": "menu"},
        {"flow_id": "flow-1", "type": "form"},
        {"type": "create_entry", "result": {"entry_id": "entry-1", "domain": "group"}},
    ])
    resources = HomeAssistantResources(client)

    result = await resources.apply(
        "group", "downstairs", "create",
        {"group_type": "light", "name": "Downstairs", "entities": ["light.kitchen"],
         "hide_members": False, "all": False},
    )

    assert result["entry_id"] == "entry-1"
    assert client.rest.await_args_list[0].args == (
        "POST", "/api/config/config_entries/flow",
        {"handler": "group", "show_advanced_options": False},
    )
    assert client.rest.await_args_list[1].args[-1] == {"next_step_id": "light"}


@pytest.mark.asyncio
async def test_group_get_snapshots_editable_options_and_aborts_flow():
    client = MagicMock()
    client.websocket = AsyncMock(return_value=[
        {"entry_id": "entry-1", "domain": "group", "title": "Downstairs"}
    ])
    client.rest = AsyncMock(side_effect=[
        {
            "flow_id": "options-1",
            "data_schema": [
                {"name": "entities", "default": ["light.kitchen"]},
                {"name": "hide_members", "default": False},
                {"name": "all", "default": True},
            ],
        },
        None,
    ])
    resources = HomeAssistantResources(client)

    result = await resources.get("group", "entry-1")

    assert result == {
        "entry_id": "entry-1", "name": "Downstairs",
        "entities": ["light.kitchen"], "hide_members": False, "all": True,
    }
    assert client.rest.await_args_list[-1].args == (
        "DELETE", "/api/config/config_entries/options/flow/options-1"
    )


def test_generated_area_id_is_used_for_audit_and_rollback_target():
    from tools.homeassistant_config_tool import _resource_id_from_result

    assert _resource_id_from_result(
        "area", {"area_id": "01JAREA"}, "requested-name"
    ) == "01JAREA"


def test_preview_apply_and_rollback_handlers_enforce_strict_approval(tmp_path):
    from tools import homeassistant_config_tool as tool
    from tools.homeassistant_store import HomeAssistantChangeStore

    manager = MagicMock()
    manager.get = AsyncMock(
        side_effect=[
            {"alias": "Old"}, {"alias": "Old"},
            {"alias": "New"}, {"alias": "New"},
        ]
    )
    manager.apply = AsyncMock(return_value={"alias": "New"})
    manager.rollback = AsyncMock(return_value={"alias": "Old"})
    store = HomeAssistantChangeStore(tmp_path / "changes.sqlite3")

    with (
        patch.object(tool, "_get_runtime", return_value=(manager, store, 900)),
        patch.object(
            tool, "request_tool_approval", return_value={"approved": True, "message": None}
        ) as approval,
    ):
        preview = json.loads(tool._handle_manage({
            "action": "preview", "resource_type": "automation", "resource_id": "morning",
            "operation": "update", "definition": {"alias": "New"},
        }))["result"]
        applied = json.loads(tool._handle_manage({
            "action": "apply", "proposal_id": preview["proposal_id"]
        }))["result"]
        rolled_back = json.loads(tool._handle_manage({
            "action": "rollback", "change_id": applied["change_id"]
        }))["result"]

    assert applied["status"] == "applied"
    assert rolled_back["status"] == "rolled_back"
    assert approval.call_count == 2
    for call in approval.call_args_list:
        assert call.kwargs["allow_yolo"] is False
        assert call.kwargs["allow_headless"] is False
        assert call.kwargs["allow_permanent"] is False


@pytest.mark.parametrize("approval_result", [
    {"approved": False, "message": "denied"},
    {"approved": False, "message": "approval_required", "approval_id": "pending-1"},
])
def test_denied_or_deferred_apply_never_mutates_home_assistant(
    tmp_path, approval_result
):
    from tools import homeassistant_config_tool as tool
    from tools.homeassistant_store import HomeAssistantChangeStore

    manager = MagicMock()
    manager.get = AsyncMock(return_value={"alias": "Old"})
    manager.apply = AsyncMock()
    store = HomeAssistantChangeStore(tmp_path / "changes.sqlite3")
    proposal = store.create_proposal(
        resource_type="automation", resource_id="morning", operation="update",
        before={"alias": "Old"}, desired={"alias": "New"},
    )

    with (
        patch.object(tool, "_get_runtime", return_value=(manager, store, 900)),
        patch.object(tool, "request_tool_approval", return_value=approval_result),
    ):
        result = json.loads(tool._handle_manage({
            "action": "apply", "proposal_id": proposal["id"],
        }))

    assert result["error"] == approval_result["message"]
    manager.get.assert_not_called()
    manager.apply.assert_not_called()
    assert store.get_proposal(proposal["id"])["status"] == "pending"


def test_denied_rollback_never_mutates_home_assistant(tmp_path):
    from tools import homeassistant_config_tool as tool
    from tools.homeassistant_store import HomeAssistantChangeStore

    manager = MagicMock()
    manager.get = AsyncMock()
    manager.rollback = AsyncMock()
    store = HomeAssistantChangeStore(tmp_path / "changes.sqlite3")
    proposal = store.create_proposal(
        resource_type="automation", resource_id="morning", operation="update",
        before={"alias": "Old"}, desired={"alias": "New"},
    )
    store.claim_proposal(proposal["id"], proposal["before_fingerprint"])
    change = store.record_applied(
        proposal["id"], after={"alias": "New"}, created_by_hermes=False,
    )

    with (
        patch.object(tool, "_get_runtime", return_value=(manager, store, 900)),
        patch.object(
            tool, "request_tool_approval",
            return_value={"approved": False, "message": "denied"},
        ),
    ):
        result = json.loads(tool._handle_manage({
            "action": "rollback", "change_id": change["id"],
        }))

    assert result["error"] == "denied"
    manager.get.assert_not_called()
    manager.rollback.assert_not_called()
    assert store.get_change(change["id"])["status"] == "applied"


def test_interrupted_apply_is_audited_and_reconciled_from_remote_state(tmp_path):
    from tools import homeassistant_config_tool as tool
    from tools.homeassistant_store import HomeAssistantChangeStore

    manager = MagicMock()
    manager.get = AsyncMock(side_effect=[{"alias": "Old"}, {"alias": "New"}])
    manager.apply = AsyncMock(side_effect=RuntimeError("response lost"))
    store = HomeAssistantChangeStore(tmp_path / "changes.sqlite3")
    proposal = store.create_proposal(
        resource_type="automation", resource_id="morning", operation="update",
        before={"alias": "Old"}, desired={"alias": "New"},
    )

    with (
        patch.object(tool, "_get_runtime", return_value=(manager, store, 900)),
        patch.object(
            tool, "request_tool_approval",
            return_value={"approved": True, "message": None},
        ),
    ):
        failed = json.loads(tool._handle_manage({
            "action": "apply", "proposal_id": proposal["id"],
        }))
        history = json.loads(tool._handle_inspect({"action": "history"}))["result"]

    assert "response lost" in failed["error"]
    assert history["reconciliation"][0]["status"] == "applied"
    assert history["changes"][0]["status"] == "applied"
    assert history["changes"][0]["before"] == {"alias": "Old"}
    assert history["changes"][0]["after"] == {"alias": "New"}


def test_preview_redacts_secret_values_from_tool_output(tmp_path):
    from tools import homeassistant_config_tool as tool
    from tools.homeassistant_store import HomeAssistantChangeStore

    manager = MagicMock()
    manager.get = AsyncMock(return_value=None)
    store = HomeAssistantChangeStore(tmp_path / "changes.sqlite3")
    with patch.object(tool, "_get_runtime", return_value=(manager, store, 900)):
        result = tool._handle_manage({
            "action": "preview", "resource_type": "script", "resource_id": "notify",
            "operation": "create", "definition": {"api_token": "super-secret", "alias": "Notify"},
        })

    assert "super-secret" not in result
    assert "[REDACTED]" in result
