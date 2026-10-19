"""Who is offered the browser extension's tool surface, and that it does not change back inside a session.

A browser extension session (``browser.extension_control.enabled``, the server-bound principal and transport,
and a controller the extension registered for the session) is offered the extension surface: the tab and action
tools only the extension runs, ``tab_id`` and the extension wording on the long-standing tools. Every other
session gets the browser tools as they were, including an API client of the same profile that never registers a
controller. Once a session has it, the surface stays whatever is attached later: a tool block that changes
between two requests of one conversation loses the provider's prompt cache.
"""

import json
import re

import pytest

from gateway import browser_control_broker as broker_mod
from gateway.session_context import clear_session_vars, set_session_vars
from model_tools import _apply_dynamic_schemas, _clear_tool_defs_cache, get_tool_definitions
from tools import browser_tool
from tools import browser_tool_extension_schemas as lane
from tools import browser_tool_install as bt_install
from tools.registry import invalidate_check_fn_cache, registry

LONG_STANDING = (
    "browser_navigate", "browser_snapshot", "browser_click", "browser_type", "browser_scroll", "browser_back",
    "browser_press",
)
# What the browser toolset offers a stock session here (get_images and console wait on the built-in CLI itself).
STOCK_BROWSER_TOOLS = {*LONG_STANDING, "browser_vault"}
EXTENSION_WORDING = re.compile(r"extension|approv|background tab|tab_id|Hermes Control", re.IGNORECASE)
STALE_ADVICE = re.compile(r"Requires browser_navigate|called before other browser tools|to be called first|navigate first")
IDENTITY = dict(browser_control_principal="principal-fixture", browser_control_transport_family="local-api")


@pytest.fixture
def broker(monkeypatch):
    """A real broker standing in for the process-wide one, extension control on, built-in browser available."""
    fresh = broker_mod.BrowserControlBroker(command_timeout=0.1)
    monkeypatch.setattr(broker_mod, "browser_control_enabled", lambda *a: True)
    monkeypatch.setattr(broker_mod, "get_browser_control_broker", lambda: fresh)
    monkeypatch.setattr(bt_install, "check_browser_requirements", lambda: True)
    # a bound request bypasses both caches; the unbound requests below must not see this test's answers either
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    yield fresh
    fresh.reset()
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()


def _definitions(session_id="session-1", **identity):
    tokens = set_session_vars(session_id=session_id, **identity)
    try:
        return get_tool_definitions(enabled_toolsets=["browser"], quiet_mode=True, skip_tool_search_assembly=True)
    finally:
        clear_session_vars(tokens)


def _bound_definitions(session_id="session-1"):
    return _definitions(session_id, **IDENTITY)


def _by_name(definitions):
    return {d["function"]["name"]: d["function"] for d in definitions}


def _attach(broker, session_id="session-1", capabilities=None):
    scope = broker_mod.ControllerScope(
        principal_id="principal-fixture", profile_id="default", session_id=session_id,
        controller_id="controller-fixture", browser_profile_id="browser-fixture", transport_family="local-api",
        capabilities=frozenset(capabilities if capabilities is not None else broker_mod.BROWSER_CONTROL_CAPABILITIES),
    )
    broker.attach(scope, lambda frame: None, owner="socket-fixture")
    return scope


# --- sessions without the extension lane get the stock surface --------------------------------------------------


@pytest.mark.parametrize(
    "enabled, identity",
    [
        pytest.param(False, IDENTITY, id="control-off-bound-request"),  # api_server binds an identity on every request
        pytest.param(True, {}, id="control-on-unbound-request"),  # a CLI session
        pytest.param(False, {}, id="control-off-unbound-request"),
    ],
)
def test_a_session_that_cannot_bind_a_controller_gets_the_stock_browser_surface(broker, monkeypatch, enabled, identity):
    monkeypatch.setattr(broker_mod, "browser_control_enabled", lambda *a: enabled)
    _attach(broker)  # even with a controller attached, only the session decides

    tools = _by_name(_definitions(**identity))

    assert set(tools) == STOCK_BROWSER_TOOLS
    stock = {s["name"]: s for s in browser_tool.BROWSER_TOOL_SCHEMAS}
    for name in LONG_STANDING:  # browser_type also carries the Vault note, as on main
        assert tools[name]["parameters"] == stock[name]["parameters"], name
        assert tools[name]["description"].startswith(stock[name]["description"]), name
    for name, function in tools.items():
        assert "tab_id" not in json.dumps(function), name
    assert [n for n in LONG_STANDING if EXTENSION_WORDING.search(tools[n]["description"])] == []
    assert set(tools["browser_click"]["parameters"]["properties"]) == {"ref"}
    assert set(tools["browser_type"]["parameters"]["properties"]) == {"ref", "text"}
    assert tools["browser_scroll"]["parameters"]["properties"]["direction"]["enum"] == ["up", "down"]


def test_an_api_client_the_extension_never_registered_for_gets_the_stock_surface(broker):
    """With extension control on, api_server binds a principal and transport on every request, so the flag and
    the identity alone would hand the extension surface to a chat UI that never attaches a browser."""
    _attach(broker, session_id="session-other")  # the extension works in another session of the same principal

    tools = _by_name(_bound_definitions())

    assert set(tools) == STOCK_BROWSER_TOOLS
    assert [n for n in LONG_STANDING if EXTENSION_WORDING.search(tools[n]["description"])] == []


def test_without_a_built_in_browser_an_unregistered_api_client_is_offered_no_routed_browser_tools(broker, monkeypatch):
    monkeypatch.setattr(bt_install, "check_browser_requirements", lambda: False)

    tools = _by_name(_bound_definitions())

    assert set(tools).isdisjoint({*LONG_STANDING, *lane.EXTENSION_ONLY_TOOLS})


def test_none_of_the_extension_tools_is_offered_without_the_lane(broker, monkeypatch):
    monkeypatch.setattr(broker_mod, "browser_control_enabled", lambda *a: False)

    assert lane.EXTENSION_ONLY_TOOLS.isdisjoint(_by_name(_definitions(**IDENTITY)))
    # the same holds when the toolset is asked for by name rather than through the browser toolset
    tokens = set_session_vars(session_id="session-1", **IDENTITY)
    try:
        everything = get_tool_definitions(quiet_mode=True, skip_tool_search_assembly=True)
    finally:
        clear_session_vars(tokens)
    assert lane.EXTENSION_ONLY_TOOLS.isdisjoint(_by_name(everything))


# --- sessions with the lane get the whole extension surface -------------------------------------------------------


def test_a_session_the_extension_registered_for_gets_the_extension_surface(broker):
    _attach(broker)

    tools = _by_name(_bound_definitions())

    assert set(tools) == STOCK_BROWSER_TOOLS | lane.EXTENSION_ONLY_TOOLS
    for name in lane.TAB_TARGETING_TOOLS:
        assert tools[name]["parameters"]["properties"]["tab_id"]["type"] == "integer", name
    assert {"x", "y", "tab_id"} <= set(tools["browser_click"]["parameters"]["properties"])
    assert tools["browser_scroll"]["parameters"]["properties"]["direction"]["enum"] == ["up", "down", "left", "right"]
    assert "approve" in tools["browser_click"]["description"]
    assert [n for n in LONG_STANDING if STALE_ADVICE.search(tools[n]["description"])] == []


def test_the_tool_block_is_identical_while_the_controller_goes_offline_leaves_and_comes_back(broker):
    """The first reviewer's probe: pausing Hermes Control or a reconnect must not rewrite the tools of a session."""
    scope = _attach(broker)
    registered = _bound_definitions()
    assert broker.disconnect_owner("socket-fixture") == 1
    offline = _bound_definitions()
    broker.detach(scope)
    detached = _bound_definitions()
    _attach(broker)
    back = _bound_definitions()

    for later in (offline, detached, back):
        assert json.dumps(later, sort_keys=True) == json.dumps(registered, sort_keys=True)


def test_a_chat_that_started_before_hermes_control_changes_once_and_never_back(broker):
    """A session's surface switches on the first time the extension registers for it (the person turned Hermes
    Control on in that chat), and stays on."""
    before = _bound_definitions()
    scope = _attach(broker)
    after = _bound_definitions()
    broker.detach(scope)

    assert set(_by_name(before)) == STOCK_BROWSER_TOOLS
    assert set(_by_name(after)) == STOCK_BROWSER_TOOLS | lane.EXTENSION_ONLY_TOOLS
    assert _bound_definitions() == after


def test_a_controller_that_advertises_fewer_actions_gets_the_same_tool_block(broker):
    _attach(broker, session_id="session-full")
    full = _bound_definitions("session-full")
    _attach(broker, capabilities={"browser_snapshot"})

    assert _bound_definitions() == full


def test_one_process_serves_a_lane_session_and_stock_sessions_without_leaking_either(broker):
    _attach(broker, session_id="session-bound")
    stock = _definitions("session-cli")
    assert set(_by_name(stock)) == STOCK_BROWSER_TOOLS
    chat_ui = _bound_definitions("session-chat-ui")
    assert set(_by_name(chat_ui)) == STOCK_BROWSER_TOOLS
    laned = _bound_definitions("session-bound")
    assert set(_by_name(laned)) == STOCK_BROWSER_TOOLS | lane.EXTENSION_ONLY_TOOLS

    assert _definitions("session-cli") == stock
    assert _bound_definitions("session-chat-ui") == chat_ui
    assert _bound_definitions("session-bound") == laned


# --- the lane forms ---------------------------------------------------------------------------------------------


def test_every_long_standing_browser_tool_has_an_extension_form_that_applies_to_its_current_text():
    stock = {s["name"]: s for s in browser_tool.BROWSER_TOOL_SCHEMAS}
    table = {name for name, _emoji, check_fn, *_ in browser_tool._BROWSER_TOOL_TABLE if check_fn is None}

    assert set(LONG_STANDING) <= table
    assert lane.LANE_TOOLS == set(LONG_STANDING)
    for name in LONG_STANDING:
        form = lane.lane_schema(stock[name])  # raises when a rewritten sentence is no longer in the main text
        assert form["description"] != stock[name]["description"], name
        assert not STALE_ADVICE.search(form["description"]), name
        assert form["description"].strip() == form["description"] and "  " not in form["description"], name


def test_a_reworded_main_description_fails_loudly_instead_of_keeping_stale_advice():
    base = {"name": "browser_navigate", "description": "Open a page.", "parameters": {"type": "object", "properties": {}}}

    with pytest.raises(ValueError, match="browser_navigate"):
        lane.lane_schema(base)


def test_the_lane_form_leaves_the_stock_schema_untouched():
    stock = next(s for s in browser_tool.BROWSER_TOOL_SCHEMAS if s["name"] == "browser_click")
    before = json.dumps(stock, sort_keys=True)

    lane.lane_schema(stock)

    assert json.dumps(stock, sort_keys=True) == before


def test_extension_only_tools_are_registered_allowlisted_and_never_in_the_stock_schema_list():
    stock_names = {s["name"] for s in browser_tool.BROWSER_TOOL_SCHEMAS}

    assert lane.EXTENSION_ONLY_TOOLS.isdisjoint(stock_names)
    for name in lane.EXTENSION_ONLY_TOOLS:
        assert registry.get_entry(name) is not None, name
        assert name in broker_mod.BROWSER_CONTROL_CAPABILITIES, name


def test_the_vault_note_and_the_retrieval_hint_still_compose_with_the_lane_form(broker):
    _attach(broker)
    stock = {s["name"]: s for s in browser_tool.BROWSER_TOOL_SCHEMAS}
    defs = [{"type": "function", "function": dict(stock[n])} for n in ("browser_navigate", "browser_type")]
    defs += [{"type": "function", "function": {"name": n, "description": "", "parameters": {}}}
             for n in ("web_search", "browser_vault")]
    tokens = set_session_vars(session_id="session-1", **IDENTITY)
    try:
        rendered = _by_name(_apply_dynamic_schemas(defs))
    finally:
        clear_session_vars(tokens)

    assert "Available lightweight retrieval tool: web_search." in rendered["browser_navigate"]["description"]
    assert "already open" in rendered["browser_navigate"]["description"]
    assert "Never type a password" in rendered["browser_type"]["description"]
    assert "tab_id" in rendered["browser_type"]["parameters"]["properties"]


@pytest.mark.parametrize("tool", ("browser_type", "browser_fill", "browser_batch"))
def test_every_tool_that_types_text_carries_the_vault_note_when_the_vault_is_available(broker, tool):
    _attach(broker)
    schemas = {s["name"]: s for s in (*browser_tool.BROWSER_TOOL_SCHEMAS, *lane.EXTENSION_ONLY_SCHEMAS)}
    defs = [{"type": "function", "function": dict(schemas[tool])},
            {"type": "function", "function": {"name": "browser_vault", "description": "", "parameters": {}}}]
    tokens = set_session_vars(session_id="session-1", **IDENTITY)
    try:
        with_vault = _by_name(_apply_dynamic_schemas(defs))[tool]["description"]
        without_vault = _by_name(_apply_dynamic_schemas(defs[:1]))[tool]["description"]
    finally:
        clear_session_vars(tokens)

    assert "Never type a password" in with_vault
    assert "Never type a password" not in without_vault


# --- nothing attached: calls fail closed and say what to ask for, or keep the built-in browser ---------------------


def _call(tool, args=None):
    tokens = set_session_vars(session_id="session-1", **IDENTITY)
    try:
        return json.loads(registry.dispatch(tool, args or {}, task_id="task-1"))
    finally:
        clear_session_vars(tokens)


def test_a_call_after_the_controller_went_offline_fails_closed_and_asks_for_hermes_control(broker):
    _attach(broker)
    broker.disconnect_owner("socket-fixture")

    out = _call("browser_snapshot")

    assert "ControllerUnavailable" in out["error"]
    assert "turn on Hermes Control" in out["error"]


@pytest.mark.parametrize("built_in", (True, False), ids=("built-in-browser", "no-built-in-browser"))
def test_a_session_the_extension_registered_for_never_falls_back_to_the_built_in_browser(broker, monkeypatch, built_in):
    """Its tools describe the extension (the page already open, refs from the attached snapshot), which a headless
    browser starting on about:blank does not match, so with the controller gone the call fails closed."""
    monkeypatch.setattr(bt_install, "check_browser_requirements", lambda: built_in)
    ran = []
    monkeypatch.setattr(browser_tool, "browser_snapshot", lambda **kw: ran.append(kw) or json.dumps({"success": True}))
    scope = _attach(broker)
    _bound_definitions()
    broker.detach(scope)  # the person turned Hermes Control off: the lane has no controller at all now

    out = _call("browser_snapshot")

    assert "ControllerUnavailable" in out["error"] and "turn on Hermes Control" in out["error"]
    assert ran == []


def test_the_built_in_browser_still_answers_a_bound_session_the_extension_never_registered_for(broker, monkeypatch):
    ran = []
    monkeypatch.setattr(browser_tool, "browser_snapshot", lambda **kw: ran.append(kw) or json.dumps({"success": True}))

    assert _call("browser_snapshot") == {"success": True}
    assert len(ran) == 1


# --- string values for the lane arguments of the long-standing tools ---------------------------------------------


def test_string_values_for_the_lane_arguments_are_coerced_on_the_long_standing_tools(monkeypatch):
    """The registry coerces against the registered schema, which for these seven tools is main's (no tab_id,
    clear, amount, ...), so their handler coerces against the extension form the model was shown."""
    sent = []
    monkeypatch.setattr(browser_tool, "routed_browser_handler",
                        lambda name, args, **kw: sent.append((name, args)) or json.dumps({"success": True}))
    calls = (
        ("browser_click", {"ref": "@e1", "tab_id": "57"}, {"ref": "@e1", "tab_id": 57}),
        ("browser_type", {"ref": "@e1", "text": "x", "clear": "false"}, {"ref": "@e1", "text": "x", "clear": False}),
        ("browser_scroll", {"direction": "down", "amount": "300"}, {"direction": "down", "amount": 300}),
        ("browser_press", {"key": "a", "modifiers": '["Control"]'}, {"key": "a", "modifiers": ["Control"]}),
    )
    for name, args, _ in calls:
        assert json.loads(registry.dispatch(name, args, task_id="task-1")) == {"success": True}, name

    assert sent == [(name, coerced) for name, _, coerced in calls]


# --- tool search can defer the rarely used extension tools ---------------------------------------------------------


def test_the_rarely_used_extension_tools_are_deferred_behind_tool_search(broker):
    from tools.tool_search import ToolSearchConfig

    cold = {"browser_drag", "browser_tab_activate", "browser_tab_group", "browser_tab_ungroup"}
    assert cold <= ToolSearchConfig.from_raw(None).effective_defer_tools
    _attach(broker)

    tokens = set_session_vars(session_id="session-1", **IDENTITY)
    try:
        laned = get_tool_definitions(enabled_toolsets=["browser"], quiet_mode=True)
    finally:
        clear_session_vars(tokens)
    stock = get_tool_definitions(enabled_toolsets=["browser"], quiet_mode=True)

    names = set(_by_name(laned))
    assert "tool_search" in names and cold.isdisjoint(names)
    assert (lane.EXTENSION_ONLY_TOOLS - cold) <= names
    assert not [name for name in lane.EXTENSION_ONLY_TOOLS if name in json.dumps(stock)]
