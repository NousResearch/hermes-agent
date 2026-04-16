"""Generic helpers for table-driven control-plane dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class ActionRoute:
    actions: set[str]
    handler: Callable[[dict], Any]
    transform: Callable[[dict], dict] | None = None


@dataclass(frozen=True)
class ControlRouteSpec:
    actions: set[str]
    handler: Callable[[dict], Any]
    transform: Callable[[dict], dict] | None = None


@dataclass(frozen=True)
class ControlPlaneSpec:
    name: str
    description: str
    aliases: dict[str, str]
    route_specs_factory: Callable[[], list[ControlRouteSpec]]
    properties: dict[str, Any]
    extra_actions: set[str]
    normalize_args: Callable[[dict], dict] | None = None
    postprocess_result: Callable[[dict, Any], Any] | None = None


def normalize_control_action(value, aliases: dict[str, str]) -> str:
    action = str(value or "").strip().lower()
    return aliases.get(action, action)


def build_routes(specs: Iterable[ControlRouteSpec]) -> list[ActionRoute]:
    return [
        ActionRoute(actions=set(spec.actions), handler=spec.handler, transform=spec.transform)
        for spec in specs
    ]


def build_control_schema(
    *,
    name: str,
    description: str,
    route_specs: Iterable[ControlRouteSpec],
    properties: dict[str, Any],
    extra_actions: set[str] | None = None,
) -> dict[str, Any]:
    action_enum: set[str] = set(extra_actions or set())
    for spec in route_specs:
        action_enum.update(spec.actions)
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": sorted(action_enum),
                    "description": f"{name} action to perform.",
                },
                **properties,
            },
            "required": ["action"],
        },
    }


def build_control_schema_from_spec(spec: ControlPlaneSpec) -> dict[str, Any]:
    return build_control_schema(
        name=spec.name,
        description=spec.description,
        route_specs=spec.route_specs_factory(),
        properties=spec.properties,
        extra_actions=spec.extra_actions,
    )


def normalize_control_payload(args: dict, *, spec: ControlPlaneSpec) -> dict:
    normalized = dict(args)
    if spec.normalize_args is not None:
        normalized = spec.normalize_args(normalized)
    normalized["action"] = normalize_control_action(normalized.get("action"), spec.aliases)
    return normalized


def dispatch_control_action(
    args: dict,
    *,
    routes: Iterable[ActionRoute],
    unsupported: Callable[[dict], Any] | None = None,
):
    action = str(args.get("action") or "").strip().lower()
    for route in routes:
        if action not in route.actions:
            continue
        payload = route.transform(dict(args)) if route.transform else dict(args)
        return route.handler(payload)
    if unsupported is not None:
        return unsupported(dict(args))
    raise ValueError(f"Unsupported control action: {action}")


def run_control_plane(
    args: dict,
    *,
    spec: ControlPlaneSpec,
    unsupported: Callable[[dict], Any] | None = None,
):
    normalized_args = normalize_control_payload(dict(args), spec=spec)
    result = dispatch_control_action(
        normalized_args,
        routes=build_routes(spec.route_specs_factory()),
        unsupported=unsupported,
    )
    if spec.postprocess_result is not None:
        return spec.postprocess_result(normalized_args, result)
    return result
