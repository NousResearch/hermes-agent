"""Procedural, read-only agent tools for the Roadmaps orchestration surface.

Roadmaps is an orchestration layer over the native Hermes stores.  This first
slice deliberately exposes reads only: durable mutations will be added behind
service-side authorization, optimistic versions, and idempotent events.
"""

from __future__ import annotations

import json
from typing import Any

from hermes_cli.roadmaps_service import RoadmapsService, RoadmapsUnavailable
from hermes_cli.roadmaps_writer import (
    InvalidRoadmapTransitionError,
    RoadmapNodeNotFoundError,
    RoadmapNotFoundError,
    RoadmapsWriteError,
    RoadmapsWriter,
    StaleRoadmapVersionError,
)
from hermes_constants import get_hermes_home
from tools.registry import registry


def _required(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    value = value.strip()
    if not value or len(value) > RoadmapsService.MAX_IDENTIFIER_LENGTH:
        raise ValueError(f"{name} must be a non-empty string of at most 128 characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{name} contains control characters")
    return value


def _service() -> RoadmapsService:
    """Return a read-only service for the active profile only."""
    return RoadmapsService(get_hermes_home() / "projects.db")


def _writer() -> RoadmapsWriter:
    """Return an authorized writer for the active profile only."""
    return RoadmapsWriter(get_hermes_home() / "projects.db")


def _mutation_error(exc: Exception) -> str:
    if isinstance(exc, StaleRoadmapVersionError):
        return json.dumps({"success": False, "error": "stale_roadmap_version", "detail": str(exc)})
    if isinstance(exc, (RoadmapNotFoundError, RoadmapNodeNotFoundError)):
        return json.dumps({"success": False, "error": "roadmap_not_found", "detail": str(exc)})
    if isinstance(exc, InvalidRoadmapTransitionError):
        return json.dumps({"success": False, "error": "invalid_transition", "detail": str(exc)})
    return json.dumps({"success": False, "error": "roadmap_write_failed", "detail": str(exc)})


def roadmap_list(profile_id: str, project_id: str | None = None) -> str:
    """List Roadmaps in the active profile, optionally scoped to one project."""
    profile_id = _required(profile_id, "profile_id")
    if project_id is not None:
        project_id = _required(project_id, "project_id")
    try:
        result = _service().list(profile_id, project_id)
    except RoadmapsUnavailable as exc:
        return json.dumps({"success": False, "error": "roadmaps unavailable", "detail": str(exc)})
    return json.dumps(result, ensure_ascii=False)


def roadmap_context(profile_id: str, project_id: str, roadmap_id: str) -> str:
    """Read the complete durable snapshot for one explicitly scoped Roadmap."""
    profile_id = _required(profile_id, "profile_id")
    project_id = _required(project_id, "project_id")
    roadmap_id = _required(roadmap_id, "roadmap_id")
    try:
        result = _service().get_snapshot(profile_id, project_id, roadmap_id)
    except RoadmapsUnavailable as exc:
        return json.dumps({"success": False, "error": "roadmaps unavailable", "detail": str(exc)})
    return json.dumps(result, ensure_ascii=False)


registry.register(
    name="roadmap_list",
    toolset="roadmaps",
    schema={
        "name": "roadmap_list",
        "description": (
            "List Roadmaps available in the active Hermes profile. The profile_id "
            "and optional project_id are mandatory scope identifiers; never infer "
            "another profile or silently fall back to a default project. Read-only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "profile_id": {"type": "string", "description": "Explicit Hermes profile scope"},
                "project_id": {"type": "string", "description": "Optional explicit Project scope"},
            },
            "required": ["profile_id"],
        },
    },
    handler=lambda args, **kw: roadmap_list(
        profile_id=args.get("profile_id", ""), project_id=args.get("project_id")
    ),
    max_result_size_chars=100_000,
)

registry.register(
    name="roadmap_context",
    toolset="roadmaps",
    schema={
        "name": "roadmap_context",
        "description": (
            "Read the complete durable Roadmap snapshot for an explicitly scoped "
            "profile, Project, and roadmap. Use this before acting on an assigned "
            "orchestrated task. Read-only; does not mutate Hermes state."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "profile_id": {"type": "string", "description": "Explicit Hermes profile scope"},
                "project_id": {"type": "string", "description": "Explicit Project id"},
                "roadmap_id": {"type": "string", "description": "Explicit Roadmap id"},
            },
            "required": ["profile_id", "project_id", "roadmap_id"],
        },
    },
    handler=lambda args, **kw: roadmap_context(
        profile_id=args.get("profile_id", ""),
        project_id=args.get("project_id", ""),
        roadmap_id=args.get("roadmap_id", ""),
    ),
    max_result_size_chars=200_000,
)


def _mutation_args(args: dict) -> dict:
    """Extract the shared scope/actor/version params from a mutation call."""
    return {
        "profile_id": args.get("profile_id", ""),
        "project_id": args.get("project_id", ""),
        "roadmap_id": args.get("roadmap_id", ""),
        "node_id": args.get("node_id", ""),
        "actor": args.get("actor", ""),
        "expected_version": args.get("expected_version"),
    }


def roadmap_claim_node(profile_id: str, project_id: str, roadmap_id: str, node_id: str, actor: str, expected_version: Any) -> str:
    try:
        result = _writer().claim_node(profile_id, project_id, roadmap_id, node_id, actor, expected_version)
    except (ValueError, RoadmapsWriteError) as exc:
        return _mutation_error(exc)
    return json.dumps(result, ensure_ascii=False)


def roadmap_update_progress(profile_id: str, project_id: str, roadmap_id: str, node_id: str, actor: str, progress: Any, expected_version: Any) -> str:
    try:
        result = _writer().update_progress(profile_id, project_id, roadmap_id, node_id, actor, progress, expected_version)
    except (ValueError, RoadmapsWriteError) as exc:
        return _mutation_error(exc)
    return json.dumps(result, ensure_ascii=False)


def roadmap_complete_node(profile_id: str, project_id: str, roadmap_id: str, node_id: str, actor: str, expected_version: Any) -> str:
    try:
        result = _writer().complete_node(profile_id, project_id, roadmap_id, node_id, actor, expected_version)
    except (ValueError, RoadmapsWriteError) as exc:
        return _mutation_error(exc)
    return json.dumps(result, ensure_ascii=False)


def roadmap_block_node(profile_id: str, project_id: str, roadmap_id: str, node_id: str, actor: str, reason: str, expected_version: Any) -> str:
    try:
        result = _writer().block_node(profile_id, project_id, roadmap_id, node_id, actor, reason, expected_version)
    except (ValueError, RoadmapsWriteError) as exc:
        return _mutation_error(exc)
    return json.dumps(result, ensure_ascii=False)


def roadmap_unblock_node(profile_id: str, project_id: str, roadmap_id: str, node_id: str, actor: str, expected_version: Any) -> str:
    try:
        result = _writer().unblock_node(profile_id, project_id, roadmap_id, node_id, actor, expected_version)
    except (ValueError, RoadmapsWriteError) as exc:
        return _mutation_error(exc)
    return json.dumps(result, ensure_ascii=False)


def _mutation_schema(name: str, description: str, extra_properties: dict | None = None) -> dict:
    properties = {
        "profile_id": {"type": "string", "description": "Explicit Hermes profile scope"},
        "project_id": {"type": "string", "description": "Explicit Project id"},
        "roadmap_id": {"type": "string", "description": "Explicit Roadmap id"},
        "node_id": {"type": "string", "description": "Explicit node id within the active roadmap version"},
        "actor": {"type": "string", "description": "Identity performing the mutation (profile or agent id)"},
        "expected_version": {"type": "integer", "description": "Roadmap active version the caller observed; mutation is rejected if the plan was revised"},
    }
    if extra_properties:
        properties.update(extra_properties)
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": ["profile_id", "project_id", "roadmap_id", "node_id", "actor", "expected_version"],
        },
    }


registry.register(
    name="roadmap_claim_node",
    toolset="roadmaps",
    schema=_mutation_schema(
        "roadmap_claim_node",
        "Claim an explicitly scoped roadmap node: moves a 'ready' node to 'in_progress' and assigns the actor as owner. Rejected if the roadmap was revised since the caller read it (pass the observed expected_version).",
    ),
    handler=lambda args, **kw: roadmap_claim_node(**_mutation_args(args)),
    max_result_size_chars=20_000,
)

registry.register(
    name="roadmap_update_progress",
    toolset="roadmaps",
    schema=_mutation_schema(
        "roadmap_update_progress",
        "Record execution progress (0-100) on an 'in_progress' roadmap node. Requires the explicit scope, actor, and the observed expected_version.",
        {"progress": {"type": "integer", "description": "Progress percentage, 0-100"}},
    ),
    handler=lambda args, **kw: roadmap_update_progress(
        progress=args.get("progress"), **_mutation_args(args)
    ),
    max_result_size_chars=20_000,
)

registry.register(
    name="roadmap_complete_node",
    toolset="roadmaps",
    schema=_mutation_schema(
        "roadmap_complete_node",
        "Mark an explicitly scoped roadmap node completed (from 'in_progress' or 'blocked'); progress becomes 100 and any block reason is cleared.",
    ),
    handler=lambda args, **kw: roadmap_complete_node(**_mutation_args(args)),
    max_result_size_chars=20_000,
)

registry.register(
    name="roadmap_block_node",
    toolset="roadmaps",
    schema=_mutation_schema(
        "roadmap_block_node",
        "Block an 'in_progress' roadmap node with a required typed reason; the reason is persisted on the node.",
        {"reason": {"type": "string", "description": "Non-empty block reason"}},
    ),
    handler=lambda args, **kw: roadmap_block_node(
        reason=args.get("reason", ""), **_mutation_args(args)
    ),
    max_result_size_chars=20_000,
)

registry.register(
    name="roadmap_unblock_node",
    toolset="roadmaps",
    schema=_mutation_schema(
        "roadmap_unblock_node",
        "Return a 'blocked' roadmap node to 'in_progress' and clear its block reason.",
    ),
    handler=lambda args, **kw: roadmap_unblock_node(**_mutation_args(args)),
    max_result_size_chars=20_000,
)
