"""Capability-gated Home Assistant configuration resource adapters."""

from __future__ import annotations

from typing import Any


REST_RESOURCES = {"automation", "script", "scene"}
GROUP_RESOURCE = "group"
HELPER_RESOURCES = {
    "input_boolean", "input_number", "input_select", "input_text",
    "input_datetime", "counter", "timer", "schedule",
}
REGISTRY_RESOURCES = {
    "entity": "config/entity_registry",
    "device": "config/device_registry",
    "area": "config/area_registry",
}
EXCLUDED_RESOURCES = [
    "addon", "dashboard", "device_removal", "integration", "pairing",
]


class HomeAssistantResources:
    def __init__(self, client):
        self.client = client

    def _ensure_supported(self, resource_type: str) -> None:
        if resource_type not in REST_RESOURCES | HELPER_RESOURCES | set(REGISTRY_RESOURCES) | {GROUP_RESOURCE}:
            raise ValueError(f"unsupported Home Assistant resource type: {resource_type}")

    async def capabilities(self) -> dict[str, Any]:
        config = await self.client.rest("GET", "/api/config")
        mutable = {
            resource: ["create", "update"]
            for resource in sorted(REST_RESOURCES | HELPER_RESOURCES | {GROUP_RESOURCE})
        }
        mutable.update({"entity": ["update"], "device": ["update"], "area": ["create", "update"]})
        return {
            "home_assistant_version": config.get("version"),
            "mutable": mutable,
            "excluded": EXCLUDED_RESOURCES,
            "rollback": "update if unchanged; delete only exact Hermes-created resources",
        }

    async def list(self, resource_type: str) -> list[dict[str, Any]]:
        self._ensure_supported(resource_type)
        if resource_type in REST_RESOURCES:
            entries = await self.client.websocket({"type": "config/entity_registry/list"})
            platform = "homeassistant" if resource_type == "scene" else resource_type
            ids = [
                entry.get("unique_id")
                for entry in entries
                if entry.get("platform") == platform and entry.get("unique_id")
            ]
            result = []
            for resource_id in ids:
                item = await self.get(resource_type, resource_id)
                if item is not None:
                    result.append(item)
        elif resource_type == GROUP_RESOURCE:
            entries = await self.client.websocket({"type": "config_entries/get", "domain": "group"})
            result = [entry for entry in entries if entry.get("domain") == "group"]
        elif resource_type in HELPER_RESOURCES:
            result = await self.client.websocket({"type": f"{resource_type}/list"})
        else:
            result = await self.client.websocket(
                {"type": f"{REGISTRY_RESOURCES[resource_type]}/list"}
            )
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return list(result.values())
        return []

    @staticmethod
    def _matches(resource_type: str, item: dict[str, Any], resource_id: str) -> bool:
        candidates = (
            item.get("id"), item.get(f"{resource_type}_id"), item.get("entity_id"),
            item.get("device_id"), item.get("area_id"), item.get("unique_id"),
        )
        return resource_id in candidates

    async def get(self, resource_type: str, resource_id: str) -> dict[str, Any] | None:
        self._ensure_supported(resource_type)
        if not resource_id:
            raise ValueError("resource_id is required")
        if resource_type in REST_RESOURCES:
            try:
                return await self.client.rest(
                    "GET", f"/api/config/{resource_type}/config/{resource_id}"
                )
            except Exception as exc:
                if "HTTP 404" in str(exc):
                    return None
                raise
        if resource_type == GROUP_RESOURCE:
            entries = await self.list(resource_type)
            return next((entry for entry in entries if entry.get("entry_id") == resource_id), None)
        items = await self.list(resource_type)
        return next(
            (item for item in items if self._matches(resource_type, item, resource_id)), None
        )

    async def apply(
        self,
        resource_type: str,
        resource_id: str,
        operation: str,
        definition: dict[str, Any],
    ) -> dict[str, Any]:
        self._ensure_supported(resource_type)
        if operation not in {"create", "update"}:
            raise ValueError("operation must be create or update")
        if operation == "create" and resource_type in {"entity", "device"}:
            raise ValueError(f"Home Assistant does not support creating {resource_type} metadata")
        if resource_type in REST_RESOURCES:
            await self.client.rest(
                "POST", f"/api/config/{resource_type}/config/{resource_id}", definition
            )
            return definition
        if resource_type == GROUP_RESOURCE:
            if operation == "create":
                group_type = definition.get("group_type")
                if not group_type:
                    raise ValueError("group create requires group_type")
                flow = await self.client.rest(
                    "POST", "/api/config/config_entries/flow",
                    {"handler": "group", "show_advanced_options": False},
                )
                flow_id = flow["flow_id"]
                await self.client.rest(
                    "POST", f"/api/config/config_entries/flow/{flow_id}",
                    {"next_step_id": group_type},
                )
                result = await self.client.rest(
                    "POST", f"/api/config/config_entries/flow/{flow_id}",
                    {key: value for key, value in definition.items() if key != "group_type"},
                )
                return result.get("result", result)
            flow = await self.client.rest(
                "POST", "/api/config/config_entries/options/flow", {"handler": resource_id}
            )
            options = {
                key: value for key, value in definition.items()
                if key not in {"name", "group_type"}
            }
            result = await self.client.rest(
                "POST", f"/api/config/config_entries/options/flow/{flow['flow_id']}", options
            )
            if "name" in definition:
                await self.client.websocket(
                    {"type": "config_entries/update", "entry_id": resource_id,
                     "title": definition["name"]}
                )
            return result.get("result", result)
        if resource_type in HELPER_RESOURCES:
            command = {"type": f"{resource_type}/{operation}", **definition}
            if operation == "update":
                command[f"{resource_type}_id"] = resource_id
            result = await self.client.websocket(command)
            return result if isinstance(result, dict) else definition
        prefix = REGISTRY_RESOURCES[resource_type]
        command = {"type": f"{prefix}/{operation}", **definition}
        if operation == "update":
            command[f"{resource_type}_id"] = resource_id
        result = await self.client.websocket(command)
        return result if isinstance(result, dict) else definition

    async def rollback(self, change: dict[str, Any]) -> Any:
        resource_type = change["resource_type"]
        resource_id = change["resource_id"]
        self._ensure_supported(resource_type)
        if change["operation"] == "update":
            return await self.apply(
                resource_type, resource_id, "update", change["before"]
            )
        if not change.get("created_by_hermes"):
            raise ValueError("created resource was not recorded as created by Hermes")
        if resource_type in REST_RESOURCES:
            return await self.client.rest(
                "DELETE", f"/api/config/{resource_type}/config/{resource_id}"
            )
        if resource_type == GROUP_RESOURCE:
            return await self.client.websocket(
                {"type": "config_entries/delete", "entry_id": resource_id}
            )
        if resource_type in HELPER_RESOURCES:
            return await self.client.websocket(
                {"type": f"{resource_type}/delete", f"{resource_type}_id": resource_id}
            )
        if resource_type == "area":
            return await self.client.websocket(
                {"type": "config/area_registry/delete", "area_id": resource_id}
            )
        raise ValueError(f"Home Assistant does not support deleting {resource_type} metadata")
