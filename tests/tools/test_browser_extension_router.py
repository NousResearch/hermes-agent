import json

import pytest

from tools.browser_extension_router import route_browser_tool, routed_browser_handler
from tools import browser_tool_install as bt_install


class FakeBroker:
    def __init__(self, *, scope=None, selected=None, result=None, error=None,
                 registered=True):
        self.scope = scope
        self.selected = selected
        self.result = result
        self.error = error
        self.registered = registered
        self.calls = []

    def scope_for_session(self, **identity):
        self.calls.append(("scope", identity))
        return self.scope

    def lane_registered(self, **identity):
        self.calls.append(("lane_registered", identity))
        return self.registered

    def select(self, scope, action):
        self.calls.append(("select", scope, action))
        return self.selected

    def dispatch(self, scope, *, action, arguments, tool_call_id=""):
        self.calls.append(("dispatch", scope, action, arguments, tool_call_id))
        if self.error:
            raise self.error
        return self.result


def test_feature_off_calls_existing_backend_once_without_touching_broker():
    broker = FakeBroker()
    fallbacks = []
    args = {"url": "https://example.test"}

    result = route_browser_tool(
        "browser_navigate",
        args,
        fallback=lambda: fallbacks.append(args.copy()) or "legacy-result",
        broker=broker,
        enabled=False,
        session_id="session-fixture",
        task_id="task-fixture",
        tool_call_id="tool-call-fixture",
    )

    assert result == "legacy-result"
    assert fallbacks == [{"url": "https://example.test"}]
    assert broker.calls == []


@pytest.mark.parametrize(
    "scope,selected",
    [(None, None), ("scope-fixture", None)],
)
def test_bound_request_without_exact_capable_controller_fails_closed(scope, selected):
    from gateway.browser_control_broker import ControllerUnavailable

    broker = FakeBroker(scope=scope, selected=selected)
    fallbacks = []

    with pytest.raises(ControllerUnavailable, match="browser_navigate"):
        route_browser_tool(
            "browser_navigate",
            {"url": "https://example.test"},
            fallback=lambda: fallbacks.append(True) or "unsafe-legacy-result",
            broker=broker,
            enabled=True,
            session_id="session-fixture",
            task_id="task-fixture",
            principal_id="principal-fixture",
            transport_family="local-api",
            tool_call_id="tool-call-fixture",
        )

    assert fallbacks == []
    assert not any(call[0] == "dispatch" for call in broker.calls)


def test_stamped_identity_without_registered_lane_keeps_legacy_backend():
    """Transport auth alone must not make the extension lane authoritative.

    A dashboard/API session carries a server-stamped principal for every
    authenticated request, but until a controller actually REGISTERS for the
    lane, browser tools keep the legacy backend (regression: flag ON +
    authenticated session + no extension bricked every browser_* call).
    """
    broker = FakeBroker(scope=None, registered=False)
    fallbacks = []

    result = route_browser_tool(
        "browser_navigate",
        {"url": "https://example.test"},
        fallback=lambda: fallbacks.append(True) or "legacy-result",
        broker=broker,
        enabled=True,
        session_id="session-fixture",
        principal_id="principal-fixture",
        transport_family="cloud-ticket-ws",
        tool_call_id="tool-call-fixture",
    )

    assert result == "legacy-result"
    assert fallbacks == [True]
    assert not any(call[0] == "dispatch" for call in broker.calls)


def test_registered_lane_with_offline_controller_still_fails_closed():
    """Once a controller registered, its absence is fail-closed, not fallback."""
    from gateway.browser_control_broker import ControllerUnavailable

    broker = FakeBroker(scope=None, registered=True)
    fallbacks = []

    with pytest.raises(ControllerUnavailable, match="browser_navigate"):
        route_browser_tool(
            "browser_navigate",
            {"url": "https://example.test"},
            fallback=lambda: fallbacks.append(True) or "unsafe-legacy-result",
            broker=broker,
            enabled=True,
            session_id="session-fixture",
            principal_id="principal-fixture",
            transport_family="cloud-ticket-ws",
            tool_call_id="tool-call-fixture",
        )

    assert fallbacks == []


def test_real_broker_lane_registered_tracks_registration_lifecycle():
    """lane_registered: False before attach, True after, True while offline."""
    from gateway.browser_control_broker import BrowserControlBroker, ControllerScope

    broker = BrowserControlBroker(command_timeout=0.1)
    identity = dict(
        session_id="sess-1",
        principal_id="principal-1",
        transport_family="cloud-ticket-ws",
    )
    assert broker.lane_registered(**identity) is False

    scope = ControllerScope(
        principal_id="principal-1",
        profile_id="default",
        session_id="sess-1",
        controller_id="ctrl-1",
        browser_profile_id="bp-1",
        transport_family="cloud-ticket-ws",
        capabilities=frozenset({"browser_navigate"}),
    )
    owner = object()
    broker.attach(scope, lambda frame: None, owner=owner)
    assert broker.lane_registered(**identity) is True

    broker.disconnect(scope, owner=owner)
    # Offline controller: lane stays bound (fail closed), never legacy.
    assert broker.lane_registered(**identity) is True
    bound_scope = broker.scope_for_session(**identity)
    assert bound_scope is not None
    assert broker.select(bound_scope, "browser_navigate") is None


def test_selected_controller_receives_immutable_arguments_and_context():
    broker = FakeBroker(
        scope="scope-fixture",
        selected="connection-fixture",
        result='{"ok": true, "source": "browser-extension"}',
    )
    args = {"url": "https://example.test"}

    result = route_browser_tool(
        "browser_navigate",
        args,
        fallback=lambda: pytest.fail("selected controller must not call fallback"),
        broker=broker,
        enabled=True,
        session_id="session-fixture",
        task_id="task-fixture",
        principal_id="principal-fixture",
        transport_family="local-api",
        tool_call_id="tool-call-fixture",
    )

    assert result == '{"ok": true, "source": "browser-extension"}'
    assert args == {"url": "https://example.test"}
    assert broker.calls == [
        (
            "scope",
            {
                "session_id": "session-fixture",
                "task_id": "task-fixture",
                "principal_id": "principal-fixture",
                "transport_family": "local-api",
            },
        ),
        ("select", "scope-fixture", "browser_navigate"),
        (
            "dispatch",
            "scope-fixture",
            "browser_navigate",
            {"url": "https://example.test"},
            "tool-call-fixture",
        ),
    ]


def test_selected_controller_dict_result_is_serialized_for_registry_contract():
    broker = FakeBroker(
        scope="scope-fixture",
        selected="connection-fixture",
        result={"ok": True, "title": "Example Domain", "refs": []},
    )

    result = route_browser_tool(
        "browser_snapshot",
        {},
        fallback=lambda: pytest.fail("selected controller must not call fallback"),
        broker=broker,
        enabled=True,
        session_id="session-fixture",
        principal_id="principal-fixture",
        transport_family="local-api",
    )

    assert result == '{"ok": true, "title": "Example Domain", "refs": []}'


def test_selected_controller_failure_never_retries_through_existing_backend():
    broker = FakeBroker(
        scope="scope-fixture",
        selected="connection-fixture",
        error=TimeoutError("controller timed out"),
    )
    fallbacks = []

    with pytest.raises(TimeoutError, match="controller timed out"):
        route_browser_tool(
            "browser_navigate",
            {"url": "https://example.test"},
            fallback=lambda: fallbacks.append(True) or "unsafe-retry",
            broker=broker,
            enabled=True,
            session_id="session-fixture",
            task_id="task-fixture",
            principal_id="principal-fixture",
            transport_family="local-api",
            tool_call_id="tool-call-fixture",
        )

    assert fallbacks == []


def test_missing_server_bound_identity_falls_back_without_querying_broker():
    broker = FakeBroker(scope="attacker-scope", selected="attacker-controller")
    fallbacks = []

    result = route_browser_tool(
        "browser_navigate",
        {"url": "https://example.test"},
        fallback=lambda: fallbacks.append(True) or "legacy-result",
        broker=broker,
        enabled=True,
        session_id="session-fixture",
    )

    assert result == "legacy-result"
    assert fallbacks == [True]
    assert broker.calls == []


def test_routed_handler_reads_server_bound_identity_from_session_context(monkeypatch):
    from gateway import browser_control_broker
    from gateway.session_context import clear_session_vars, set_session_vars

    broker = FakeBroker(
        scope="scope-fixture",
        selected="connection-fixture",
        result="controller-result",
    )
    monkeypatch.setattr(browser_control_broker, "browser_control_enabled", lambda: True)
    monkeypatch.setattr(
        browser_control_broker, "get_browser_control_broker", lambda: broker
    )
    tokens = set_session_vars(
        session_id="session-fixture",
        browser_control_principal="principal-fixture",
        browser_control_transport_family="cloud-ticket-ws",
    )
    try:
        result = routed_browser_handler(
            "browser_navigate",
            {"url": "https://example.test"},
            fallback=lambda: pytest.fail("bound controller must be selected"),
            tool_call_id="tool-call-fixture",
        )
    finally:
        clear_session_vars(tokens)

    assert result == "controller-result"
    assert broker.calls[0] == (
        "scope",
        {
            "session_id": "session-fixture",
            "task_id": None,
            "principal_id": "principal-fixture",
            "transport_family": "cloud-ticket-ws",
        },
    )


def test_routeable_browser_tools_are_available_for_a_session_that_can_bind_a_controller(monkeypatch):
    """The extension route must not be stripped by legacy Browser Use checks, and whether it is offered does
    not depend on what is attached: a controller that attaches or leaves must not rewrite the tool block."""
    from tools import browser_tool

    monkeypatch.setattr(bt_install, "check_browser_requirements", lambda: False)
    monkeypatch.setattr(browser_tool, "extension_lane_configured", lambda: True)

    assert browser_tool.check_browser_snapshot_requirements() is True
    assert browser_tool.check_browser_click_requirements() is True


def test_extension_availability_requires_exact_scope_and_capability(monkeypatch):
    from gateway import browser_control_broker
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import browser_extension_router

    broker = FakeBroker(scope="scope-fixture", selected="connection-fixture")
    monkeypatch.setattr(browser_control_broker, "browser_control_enabled", lambda: True)
    monkeypatch.setattr(
        browser_control_broker, "get_browser_control_broker", lambda: broker
    )
    tokens = set_session_vars(
        session_id="session-fixture",
        browser_control_principal="principal-fixture",
        browser_control_transport_family="local-api",
    )
    try:
        assert browser_extension_router.extension_controller_available("browser_snapshot") is True
    finally:
        clear_session_vars(tokens)

    assert broker.calls == [
        (
            "scope",
            {
                "session_id": "session-fixture",
                "principal_id": "principal-fixture",
                "transport_family": "local-api",
            },
        ),
        ("select", "scope-fixture", "browser_snapshot"),
    ]


def test_bound_controller_disappearing_after_schema_build_never_falls_back(monkeypatch):
    from gateway.browser_control_broker import (
        BrowserControlBroker,
        ControllerScope,
        ControllerUnavailable,
    )
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import browser_extension_router

    broker = BrowserControlBroker(command_timeout=0.1)
    scope = ControllerScope(
        principal_id="principal-fixture",
        profile_id="default",
        session_id="session-fixture",
        controller_id="controller-fixture",
        browser_profile_id="browser-profile-fixture",
        transport_family="local-api",
        capabilities=frozenset({"browser_snapshot"}),
    )
    broker.attach(scope, lambda _frame: None, owner="socket-fixture")
    monkeypatch.setattr(
        "gateway.browser_control_broker.browser_control_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "gateway.browser_control_broker.get_browser_control_broker",
        lambda: broker,
    )
    tokens = set_session_vars(
        session_id="session-fixture",
        browser_control_principal="principal-fixture",
        browser_control_transport_family="local-api",
    )
    fallbacks = []
    try:
        assert browser_extension_router.extension_controller_available(
            "browser_snapshot"
        ) is True
        assert broker.disconnect_owner("socket-fixture") == 1
        with pytest.raises(ControllerUnavailable, match="browser_snapshot"):
            routed_browser_handler(
                "browser_snapshot",
                {},
                fallback=lambda: fallbacks.append(True) or "unsafe-legacy-result",
            )
    finally:
        clear_session_vars(tokens)
        broker.reset()

    assert fallbacks == []


def test_routeable_browser_tools_preserve_legacy_gate_without_bound_identity(monkeypatch):
    """A feature flag alone must not advertise tools outside a bound request."""
    from gateway import browser_control_broker
    from tools import browser_tool

    monkeypatch.setattr(browser_control_broker, "browser_control_enabled", lambda: True)
    monkeypatch.setattr(bt_install, "check_browser_requirements", lambda: False)

    assert browser_tool.check_browser_snapshot_requirements() is False


def test_bound_browser_request_bypasses_availability_caches(monkeypatch):
    from gateway import browser_control_broker
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.registry import CHECK_FN_CACHE_BYPASS, check_fn_cache_scope

    monkeypatch.setattr(browser_control_broker, "browser_control_enabled", lambda: True)
    tokens = set_session_vars(
        session_id="session-fixture",
        browser_control_principal="principal-fixture",
        browser_control_transport_family="local-api",
    )
    try:
        assert check_fn_cache_scope() == CHECK_FN_CACHE_BYPASS
    finally:
        clear_session_vars(tokens)


def test_bound_identity_without_extension_control_keeps_cache_scope(monkeypatch):
    """api_server binds a principal on every request; without the feature flag
    that identity must not bypass the availability caches (#79047)."""
    from gateway import browser_control_broker
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.registry import CHECK_FN_CACHE_BYPASS, check_fn_cache_scope

    monkeypatch.setattr(browser_control_broker, "browser_control_enabled", lambda: False)
    tokens = set_session_vars(
        session_id="session-fixture",
        browser_control_principal="principal-fixture",
        browser_control_transport_family="local-api",
    )
    try:
        assert check_fn_cache_scope() != CHECK_FN_CACHE_BYPASS
    finally:
        clear_session_vars(tokens)


def test_registry_advertises_browser_tools_to_a_bound_session_when_legacy_backend_is_down_and_nothing_is_attached(
    monkeypatch,
):
    from gateway import browser_control_broker
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.registry import registry

    monkeypatch.setattr(bt_install, "check_browser_requirements", lambda: False)
    monkeypatch.setattr(browser_control_broker, "browser_control_enabled", lambda: True)
    tokens = set_session_vars(
        session_id="session-fixture",
        browser_control_principal="principal-fixture",
        browser_control_transport_family="local-api",
    )
    try:
        definitions = registry.get_definitions({"browser_snapshot", "browser_click"}, quiet=True)
    finally:
        clear_session_vars(tokens)

    assert sorted(definition["function"]["name"] for definition in definitions) == [
        "browser_click", "browser_snapshot",
    ]


def test_batch_steps_reach_the_controller_in_one_dispatch_and_are_never_mutated():
    steps = [
        {"action": "click", "ref": "@e1"},
        {"action": "type", "ref": "@e1", "text": "hello"},
        {"action": "press", "key": "Enter"},
    ]
    args = {"actions": steps, "stop_on_error": False}
    snapshot = {"actions": [dict(step) for step in steps], "stop_on_error": False}
    broker = FakeBroker(scope="scope-fixture", selected="connection-fixture", result={"status": "completed"})

    result = route_browser_tool(
        "browser_batch", args, fallback=lambda: pytest.fail("a selected controller never falls back"),
        broker=broker, enabled=True, session_id="session-fixture", principal_id="principal-fixture",
        transport_family="local-api", tool_call_id="tool-call-fixture",
    )

    assert result == '{"status": "completed"}'
    assert args == snapshot
    dispatches = [call for call in broker.calls if call[0] == "dispatch"]
    assert dispatches == [("dispatch", "scope-fixture", "browser_batch", snapshot, "tool-call-fixture")]


def test_bound_controller_without_batch_fails_closed_instead_of_running_steps_elsewhere():
    from gateway.browser_control_broker import ControllerUnavailable

    broker = FakeBroker(scope="scope-fixture", selected=None)
    with pytest.raises(ControllerUnavailable, match="browser_batch"):
        route_browser_tool(
            "browser_batch", {"actions": [{"action": "back"}]},
            fallback=lambda: pytest.fail("the extension lane fails closed"),
            broker=broker, enabled=True, session_id="session-fixture", principal_id="principal-fixture",
            transport_family="local-api",
        )


@pytest.mark.parametrize(
    "action, args",
    [
        ("browser_click", {"x": 10, "y": 20.5}),
        ("browser_scroll", {"direction": "right", "amount": 300}),
        ("browser_press", {"key": "a", "modifiers": ["Control", "Shift"]}),
        ("browser_type", {"ref": "@e1", "text": "hello", "clear": False}),
        ("browser_fill", {"ref": "@e1", "value": "hello"}),
        ("browser_select", {"ref": "@e2", "value": "Two"}),
        ("browser_hover", {"ref": "@e3"}),
        ("browser_drag", {"ref": "@e3", "to_ref": "@e4"}),
        ("browser_scroll_to", {"ref": "@e9"}),
        ("browser_tabs", {}),
        ("browser_tab_activate", {"tab_id": 7}),
        ("browser_tab_create", {"url": "https://example.test"}),
        ("browser_tab_close", {"tab_id": 7}),
        ("browser_tab_group", {"tab_ids": [7, 8]}),
        ("browser_tab_ungroup", {"tab_ids": [7, 8]}),
        ("browser_screenshot", {"ref": "@e5"}),
    ],
)
def test_every_extension_action_reaches_the_controller_with_exactly_the_model_arguments(action, args):
    sent = dict(args)
    broker = FakeBroker(scope="scope-fixture", selected="connection-fixture", result={"status": "ok"})

    route_browser_tool(
        action, args, fallback=lambda: pytest.fail("a selected controller never falls back"), broker=broker,
        enabled=True, session_id="session-fixture", principal_id="principal-fixture", transport_family="local-api",
        tool_call_id="tool-call-fixture",
    )

    assert args == sent
    assert ("dispatch", "scope-fixture", action, sent, "tool-call-fixture") in broker.calls


def test_a_controller_that_does_not_offer_an_action_fails_closed_with_a_remedy():
    from gateway.browser_control_broker import ControllerUnavailable

    broker = FakeBroker(scope="scope-fixture", selected=None)
    with pytest.raises(ControllerUnavailable) as raised:
        route_browser_tool(
            "browser_hover", {"ref": "@e1"}, fallback=lambda: pytest.fail("the extension lane fails closed"),
            broker=broker, enabled=True, session_id="session-fixture", principal_id="principal-fixture",
            transport_family="local-api",
        )

    message = str(raised.value)
    assert "browser_hover" in message and "update the extension" in message


@pytest.mark.parametrize(
    "action, args",
    [
        ("browser_click", {"ref": "@e1", "tab_id": 12}),
        ("browser_snapshot", {"tab_id": 12}),
        ("browser_batch", {"actions": [{"action": "click", "ref": "@e1"}], "tab_id": 12}),
    ],
)
def test_tab_id_travels_in_the_command_arguments_untouched(action, args):
    sent = json.loads(json.dumps(args))
    broker = FakeBroker(scope="scope-fixture", selected="connection-fixture", result={"status": "ok"})

    route_browser_tool(
        action, args, fallback=lambda: pytest.fail("a selected controller never falls back"), broker=broker,
        enabled=True, session_id="session-fixture", principal_id="principal-fixture", transport_family="local-api",
    )

    assert args == sent
    dispatched = [call for call in broker.calls if call[0] == "dispatch"]
    assert [call[3] for call in dispatched] == [sent]
