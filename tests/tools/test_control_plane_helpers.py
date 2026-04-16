"""Tests for generic control-plane dispatch helpers."""

from tools.control_plane_helpers import (
    ActionRoute,
    ControlPlaneSpec,
    ControlRouteSpec,
    build_control_schema,
    build_control_schema_from_spec,
    build_routes,
    dispatch_control_action,
    normalize_control_payload,
    normalize_control_action,
    run_control_plane,
)


def test_normalize_control_action_applies_aliases():
    assert normalize_control_action("Kick", {"kick": "kick_user"}) == "kick_user"


def test_dispatch_control_action_routes_first_matching_handler():
    calls: list[dict] = []

    def handler(args):
        calls.append(args)
        return "ok"

    result = dispatch_control_action(
        {"action": "kick_user", "target": "group:1"},
        routes=[
            ActionRoute(actions={"mute_user"}, handler=lambda _args: "wrong"),
            ActionRoute(actions={"kick_user"}, handler=handler),
        ],
    )

    assert result == "ok"
    assert calls == [{"action": "kick_user", "target": "group:1"}]


def test_dispatch_control_action_supports_transform_before_handler():
    result = dispatch_control_action(
        {"action": "list_files", "target": "group:1"},
        routes=[
            ActionRoute(
                actions={"list_files"},
                transform=lambda args: {**args, "action": "list"},
                handler=lambda args: args,
            ),
        ],
    )

    assert result == {"action": "list", "target": "group:1"}


def test_build_routes_turns_specs_into_dispatchable_routes():
    routes = build_routes(
        [
            ControlRouteSpec(
                actions={"list_files"},
                transform=lambda args: {**args, "action": "list"},
                handler=lambda args: args,
            )
        ]
    )

    assert isinstance(routes[0], ActionRoute)
    assert routes[0].actions == {"list_files"}


def test_build_control_schema_collects_sorted_action_enum():
    schema = build_control_schema(
        name="qq_control",
        description="desc",
        route_specs=[
            ControlRouteSpec(actions={"b_action", "a_action"}, handler=lambda _args: None),
            ControlRouteSpec(actions={"c_action"}, handler=lambda _args: None),
        ],
        properties={"target": {"type": "string"}},
        extra_actions={"alias_action"},
    )

    assert schema["name"] == "qq_control"
    assert schema["parameters"]["properties"]["action"]["enum"] == [
        "a_action",
        "alias_action",
        "b_action",
        "c_action",
    ]


def test_normalize_control_payload_uses_spec_aliases_and_normalizer():
    spec = ControlPlaneSpec(
        name="qq_control",
        description="desc",
        aliases={"kick": "kick_user"},
        route_specs_factory=lambda: [],
        properties={},
        extra_actions=set(),
        normalize_args=lambda args: {**args, "source": "normalized"},
    )

    payload = normalize_control_payload({"action": "kick"}, spec=spec)

    assert payload == {"action": "kick_user", "source": "normalized"}


def test_build_control_schema_from_spec_uses_route_factory():
    spec = ControlPlaneSpec(
        name="qq_control",
        description="desc",
        aliases={"alias_send": "send_message"},
        route_specs_factory=lambda: [
            ControlRouteSpec(actions={"send_message", "list_requests"}, handler=lambda _args: None)
        ],
        properties={"target": {"type": "string"}},
        extra_actions={"alias_send"},
    )

    schema = build_control_schema_from_spec(spec)

    assert schema["name"] == "qq_control"
    assert schema["parameters"]["properties"]["action"]["enum"] == [
        "alias_send",
        "list_requests",
        "send_message",
    ]


def test_run_control_plane_applies_spec_postprocess_result():
    spec = ControlPlaneSpec(
        name="qq_control",
        description="desc",
        aliases={"kick": "kick_user"},
        route_specs_factory=lambda: [
            ControlRouteSpec(actions={"kick_user"}, handler=lambda args: {"payload": args})
        ],
        properties={},
        extra_actions=set(),
        postprocess_result=lambda args, result: {
            "normalized_action": args["action"],
            "result": result,
        },
    )

    result = run_control_plane({"action": "kick"}, spec=spec)

    assert result == {
        "normalized_action": "kick_user",
        "result": {"payload": {"action": "kick_user"}},
    }
