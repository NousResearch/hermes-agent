import pytest

from tools.browser_extension_router import route_browser_tool, routed_browser_handler


class FakeBroker:
    def __init__(self, *, scope=None, selected=None, result=None, error=None):
        self.scope = scope
        self.selected = selected
        self.result = result
        self.error = error
        self.calls = []

    def scope_for_session(self, **identity):
        self.calls.append(("scope", identity))
        return self.scope

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
def test_no_exact_capable_controller_preserves_existing_backend(scope, selected):
    broker = FakeBroker(scope=scope, selected=selected)
    fallbacks = []

    result = route_browser_tool(
        "browser_navigate",
        {"url": "https://example.test"},
        fallback=lambda: fallbacks.append(True) or "legacy-result",
        broker=broker,
        enabled=True,
        session_id="session-fixture",
        task_id="task-fixture",
        tool_call_id="tool-call-fixture",
    )

    assert result == "legacy-result"
    assert fallbacks == [True]
    assert not any(call[0] == "dispatch" for call in broker.calls)


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


def test_routeable_browser_tools_are_available_for_bound_extension_controller(monkeypatch):
    """The extension route must not be stripped by legacy Browser Use checks."""
    from tools import browser_tool

    monkeypatch.setattr(browser_tool, "check_browser_requirements", lambda: False)
    monkeypatch.setattr(
        browser_tool,
        "extension_controller_available",
        lambda action: action == "browser_snapshot",
    )

    assert browser_tool.check_browser_snapshot_requirements() is True
    assert browser_tool.check_browser_click_requirements() is False


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


def test_routeable_browser_tools_preserve_legacy_gate_without_bound_identity(monkeypatch):
    """A feature flag alone must not advertise tools outside a bound request."""
    from gateway import browser_control_broker
    from tools import browser_tool

    monkeypatch.setattr(browser_control_broker, "browser_control_enabled", lambda: True)
    monkeypatch.setattr(browser_tool, "check_browser_requirements", lambda: False)

    assert browser_tool.check_browser_snapshot_requirements() is False


def test_bound_browser_request_bypasses_availability_caches():
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.registry import CHECK_FN_CACHE_BYPASS, check_fn_cache_scope

    tokens = set_session_vars(
        session_id="session-fixture",
        browser_control_principal="principal-fixture",
        browser_control_transport_family="local-api",
    )
    try:
        assert check_fn_cache_scope() == CHECK_FN_CACHE_BYPASS
    finally:
        clear_session_vars(tokens)


def test_registry_advertises_snapshot_through_extension_when_legacy_backend_is_down(
    monkeypatch,
):
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import browser_tool
    from tools.registry import registry

    monkeypatch.setattr(browser_tool, "check_browser_requirements", lambda: False)
    monkeypatch.setattr(
        browser_tool,
        "extension_controller_available",
        lambda action: action == "browser_snapshot",
    )
    tokens = set_session_vars(
        session_id="session-fixture",
        browser_control_principal="principal-fixture",
        browser_control_transport_family="local-api",
    )
    try:
        definitions = registry.get_definitions({"browser_snapshot", "browser_click"}, quiet=True)
    finally:
        clear_session_vars(tokens)

    assert [definition["function"]["name"] for definition in definitions] == [
        "browser_snapshot"
    ]
