"""MCP legs of manage_connections (the fold that retired setup_mcp).

Contracts:
- the backend owns the work: authorize mints its own URL, install writes credentials and installs,
  enable flips the flag; the card only says approved / skipped / continue
- a card claim of any other state moves nothing
- off the desktop there is no card: the work runs at once and the result carries the link
- catalog validation: install is catalog-only, enable/authorize need a configured server
- the replay shim keeps an old ``setup_mcp`` call dispatching
- deadline ownership: fixed operation deadline + sequential-deadline exemption
"""

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from hermes_cli.mcp_catalog import AuthSpec, EnvVarSpec

import tools.connectors.tool  # registers the tool
from tools.connectors.contract import Actor, SettleReason, LegState
from tools.operations import Owner, operations
from tools.connectors.legs import mcp
from tools.connectors import operation as op
from tools.connectors.legs.mcp import apply_answer
from tools.connectors.operation import ConnectionOperation, Leg
from tools.connectors.tool import MANAGE_CONNECTIONS_SCHEMA, manage_connections
from tools.registry import registry

CATALOG = ["figma", "linear", "notion"]
CONFIGURED = {"paper": {"command": "paper-mcp"}, "linear": {"url": "https://mcp.linear.app/mcp"}}


class FakeAttempt:
    """An OAuth flow in flight, as the watcher reads it."""

    def __init__(self, auth_url):
        self.auth_url = auth_url
        self.status = "pending"
        self.error = ""
        self.tools = []

    def poll(self):
        return {"status": self.status, "error": self.error, "tools": list(self.tools)}

    def approve(self, tools):
        self.tools, self.status = list(tools), "approved"

    def fail(self, error):
        self.error, self.status = error, "error"


class FakeBackend:
    """The one fake: the catalog, the installer and the OAuth flow runner behind ``mcp.py``."""

    def __init__(self, *, missing_env=(), tools=("read", "write"), install_error="", oauth_error=""):
        self.calls = []
        self.attempts = {}
        self.missing_env = list(missing_env)
        self.tools = list(tools)
        self.install_error = install_error
        self.oauth_error = oauth_error

    def required_env(self, name):
        self.calls.append(("required_env", name))
        return [{"name": key, "prompt": f"{key}?", "required": True} for key in self.missing_env]

    def start_oauth(self, name):
        self.calls.append(("start_oauth", name))
        if self.oauth_error:
            raise RuntimeError(self.oauth_error)
        attempt = FakeAttempt(f"https://auth.example/{name}/{len(self.attempts) + 1}")
        self.attempts[name] = attempt
        return attempt

    def install(self, name, env):
        self.calls.append(("install", name, dict(env)))
        if self.install_error:
            raise RuntimeError(self.install_error)
        return list(self.tools)

    def enable(self, name):
        self.calls.append(("enable", name))


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture(autouse=True)
def _clean_live():
    operations.reset_for_tests()
    yield
    operations.reset_for_tests()


@pytest.fixture(autouse=True)
def _catalog(backend):
    # The default backend is patched too: a call that cannot be handed one (registry dispatch, the
    # inline executor) must never reach the real catalog or installer from a test.
    with patch("tools.connectors.legs.mcp._catalog_names", return_value=CATALOG), \
         patch("tools.connectors.legs.mcp._configured_names", return_value=sorted(CONFIGURED)), \
         patch("tools.connectors.legs.mcp._default_backend", return_value=backend), \
         patch("tools.connectors.legs.mcp.session_platform", return_value="desktop"):
        yield


@pytest.fixture
def changes(monkeypatch):
    """Every transition the operation emits, through the hook the gateway installs."""
    recorded = []
    monkeypatch.setattr(op.ConnectionOperation, "on_change",
                        staticmethod(lambda operation, change, snapshot: recorded.append(change)))
    return recorded


class FakeClient:
    def __init__(self):
        self.calls = []

    def list_connectors(self, **_):
        self.calls.append("list")
        return [{"connector": "gmail", "enabled": True, "connected": False}]

    def connections(self, connectors, *, reinitiate=False):
        self.calls.append(("connections", tuple(connectors), reinitiate))
        return {"results": [{"connector": c, "status": "initiated", "connect_url": f"https://x/{c}"} for c in connectors]}


def _mcp_leg(name):
    return {"name": name, "mcp": True}


def _linear(**kw):
    return {"name": "linear", "mcp": True, **kw}


# ---------------------------------------------------------------------------
# the card round-trip: the backend does the work, the card answers approved / skipped
# ---------------------------------------------------------------------------


def _answering(answer, *, session_id="s1", delay=0.01):
    """A card that emits (callback returns None) and answers the live operation a moment later,
    the way ``connection.respond`` does from the renderer."""
    seen = []

    def callback(payload):
        seen.append(payload)

        def respond():
            operation = operations.get(Owner.current(session_id), payload["op_id"])
            if operation is not None:
                apply_answer(operation, answer)

        if answer is not None:
            threading.Timer(delay, respond).start()
        return None

    callback.seen = seen
    return callback


def _mcp(args, callback, **kw):
    with patch("tools.connectors.run.WATCH_INTERVAL_SECONDS", 0.01):
        return json.loads(manage_connections(args, connection_callback=callback, session_id="s1", **kw))


def test_authorize_mints_its_own_url_into_the_card_and_connects_when_the_flow_approves(backend):
    def callback(payload):
        callback.seen.append(payload)
        # The browser hits the backend's callback route; the flow reports approval out of band.
        backend.attempts["paper"].approve(["search", "comment"])
        return None

    callback.seen = []
    out = _mcp({"action": "authorize", "connectors": [_mcp_leg("paper")]}, callback, mcp_backend=backend)

    (offered,) = callback.seen[0]["legs"]
    assert offered["state"] == LegState.initiated.value
    assert offered["connect_url"] == "https://auth.example/paper/1"
    (settled,) = out["legs"]
    assert settled["state"] == LegState.connected.value
    assert settled["tools"] == ["search", "comment"]
    assert out["settled_by"] == SettleReason.all_resolved.value


def test_a_flow_that_fails_moves_the_row_to_failed_with_the_error_text(backend, changes):
    def callback(payload):
        backend.attempts["paper"].fail("the provider rejected the client registration")
        return None

    with patch("tools.connectors.operation.OPERATION_DEADLINE_SECONDS", 0.4):
        _mcp({"action": "authorize", "connectors": [_mcp_leg("paper")]}, callback, mcp_backend=backend)

    failure = [c for c in changes if c and c["to"] == LegState.failed.value][-1]
    assert failure["detail"] == "the provider rejected the client registration"
    assert failure["actor"] == Actor.backend_watcher.value


def test_a_flow_that_never_starts_fails_the_row_before_the_card_is_drawn():
    backend = FakeBackend(oauth_error="no OAuth callback route in this process")
    callback = _answering(json.dumps({"settled_by": "continue"}))
    out = _mcp({"action": "authorize", "connectors": [_mcp_leg("paper")]}, callback, mcp_backend=backend)

    (offered,) = callback.seen[0]["legs"]
    assert offered["state"] == LegState.failed.value
    assert offered["detail"] == "no OAuth callback route in this process"
    assert "connect_url" not in offered
    assert out["settled_by"] == SettleReason.continue_.value


def test_install_waits_for_the_credentials_it_declares_and_installs_with_them():
    backend = FakeBackend(missing_env=["FIGMA_TOKEN"], tools=["get_file"])
    answer = json.dumps({"legs": [{"name": "figma", "status": "approved", "env": {"FIGMA_TOKEN": "tok-1"}}]})
    callback = _answering(answer)
    out = _mcp({"action": "install", "connectors": [_mcp_leg("figma")]}, callback, mcp_backend=backend)

    (offered,) = callback.seen[0]["legs"]
    assert offered["state"] == LegState.pending.value
    assert offered["required_env"] == [{"name": "FIGMA_TOKEN", "prompt": "FIGMA_TOKEN?", "required": True}]
    assert ("install", "figma", {"FIGMA_TOKEN": "tok-1"}) in backend.calls
    (settled,) = out["legs"]
    assert settled["state"] == LegState.connected.value
    assert settled["tools"] == ["get_file"]

    entry = SimpleNamespace(auth=AuthSpec(type="api_key", env=[EnvVarSpec(
        name="FIGMA_TOKEN", prompt="Token", required=False, secret=False, default="guest")]))
    with patch.object(mcp, "_catalog_entry", return_value=entry), \
         patch("hermes_cli.config.get_env_value", return_value=None):
        assert mcp._CatalogBackend().required_env("figma") == [{
            "name": "FIGMA_TOKEN", "prompt": "Token", "required": False, "secret": False, "default": "guest",
        }]


def test_install_failure_writes_no_credentials(monkeypatch):
    entry = SimpleNamespace(auth=AuthSpec(type="api_key", env=[EnvVarSpec(name="FIGMA_TOKEN", prompt="Token")]))
    calls = []
    monkeypatch.setattr(mcp, "_catalog_entry", lambda name: entry)
    monkeypatch.setattr("hermes_cli.mcp_catalog.install_entry", lambda entry, enable: (_ for _ in ()).throw(RuntimeError("install failed")))
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda key, value: calls.append((key, value)))
    monkeypatch.setattr("hermes_cli.mcp_config._remove_mcp_server", lambda name: calls.append(("remove", name)))

    with pytest.raises(RuntimeError, match="install failed"):
        mcp._CatalogBackend().install("figma", {"FIGMA_TOKEN": "secret"})

    assert calls == [("remove", "figma")]


def test_install_success_writes_credentials_after_probe(monkeypatch):
    entry = SimpleNamespace(auth=AuthSpec(type="api_key", env=[EnvVarSpec(name="FIGMA_TOKEN", prompt="Token")]))
    calls = []
    monkeypatch.setattr(mcp, "_catalog_entry", lambda name: entry)
    monkeypatch.setattr("hermes_cli.mcp_catalog.install_entry", lambda entry, enable: calls.append("install_entry"))
    monkeypatch.setattr(mcp, "_probe_tool_names", lambda name: calls.append("probe") or ["get_file"])
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda key, value: calls.append(("save_env_value", key, value)))

    assert mcp._CatalogBackend().install("figma", {"FIGMA_TOKEN": "secret"}) == ["get_file"]
    assert calls == ["install_entry", "probe", ("save_env_value", "FIGMA_TOKEN", "secret")]


def test_enable_connects_on_the_card_s_approval(backend):
    answer = json.dumps({"legs": [{"name": "paper", "status": "approved"}]})
    out = _mcp({"action": "enable", "connectors": [_mcp_leg("paper")]}, _answering(answer), mcp_backend=backend)

    assert ("enable", "paper") in backend.calls
    (settled,) = out["legs"]
    assert settled["state"] == LegState.connected.value


def test_a_card_claim_other_than_approved_or_skipped_moves_nothing(backend):
    answer = json.dumps({"legs": [{"name": "paper", "status": "connected", "tools": ["x"]}],
                         "settled_by": "continue"})
    out = _mcp({"action": "enable", "connectors": [_mcp_leg("paper")]}, _answering(answer), mcp_backend=backend)

    assert backend.calls == []
    (settled,) = out["legs"]
    assert settled["state"] == LegState.not_connected.value
    assert out["settled_by"] == SettleReason.continue_.value


def test_a_skip_resolves_the_row_and_the_operation(backend):
    answer = json.dumps({"legs": [{"name": "paper", "status": "skipped"}]})
    out = _mcp({"action": "enable", "connectors": [_mcp_leg("paper")]}, _answering(answer), mcp_backend=backend)

    assert out["legs"][0]["state"] == LegState.skipped.value
    assert out["settled_by"] == SettleReason.all_resolved.value


def test_no_answer_settles_by_deadline_and_marks_legs_not_connected(backend):
    with patch("tools.connectors.operation.OPERATION_DEADLINE_SECONDS", 0.05):
        out = _mcp({"action": "install", "connectors": [_linear()]}, _answering(None), mcp_backend=backend)
    assert out["settled_by"] == SettleReason.deadline.value
    assert out["legs"][0]["state"] == LegState.not_connected.value
    assert "error" not in out


def test_mcp_secrets_never_reach_the_model():
    backend = FakeBackend(missing_env=["LINEAR_API_KEY"])
    answer = json.dumps({"legs": [{"name": "linear", "status": "approved",
                                      "env": {"LINEAR_API_KEY": "sk-secret"}}]})
    out = _mcp({"action": "install", "connectors": [_linear()]}, _answering(answer), mcp_backend=backend)
    assert "sk-secret" not in json.dumps(out)


# ---------------------------------------------------------------------------
# off the desktop: no card, so the work runs at once
# ---------------------------------------------------------------------------


def _off_desktop(args, **kw):
    with patch("tools.connectors.legs.mcp.session_platform", return_value="tui"):
        return json.loads(manage_connections(args, session_id="s1", **kw))


def test_off_desktop_authorize_returns_the_link_at_once_and_opens_no_operation(backend):
    out = _off_desktop({"action": "authorize", "connectors": [_mcp_leg("paper")]}, mcp_backend=backend)

    (leg,) = out["legs"]
    assert leg["state"] == LegState.initiated.value
    assert leg["connect_url"] == "https://auth.example/paper/1"
    assert out["status"] == "initiated"
    assert operations.current(Owner.current("s1")) == []


def test_off_desktop_install_without_its_credentials_fails_and_names_them():
    backend = FakeBackend(missing_env=["FIGMA_TOKEN"])
    out = _off_desktop({"action": "install", "connectors": [_mcp_leg("figma")]}, mcp_backend=backend)

    (leg,) = out["legs"]
    assert leg["state"] == LegState.failed.value
    assert "FIGMA_TOKEN" in leg["detail"]
    assert not [c for c in backend.calls if c[0] == "install"]


def test_registry_dispatch_never_blocks_and_never_reaches_a_card(backend):
    # registry.dispatch forwards no callback; the call must return, not block.
    with patch("tools.connectors.legs.mcp.session_platform", return_value="tui"):
        out = json.loads(registry.dispatch("manage_connections", {"action": "enable", "connectors": [_linear()]}))
    assert out["legs"][0]["state"] == LegState.connected.value


def test_a_managed_action_never_accepts_mcp_legs_and_vice_versa():
    client = FakeClient()
    out = json.loads(manage_connections(
        {"action": "connect", "connectors": ["gmail", _linear()]}, client_factory=lambda: client))
    assert "managed-connector action" in out["error"]
    assert client.calls == []  # rejected before any gateway call

    out = json.loads(manage_connections({"action": "install", "connectors": ["gmail", _linear()]}))
    assert "must carry" in out["error"]


def test_a_managed_call_off_desktop_returns_a_link_per_leg():
    client = FakeClient()
    out = json.loads(manage_connections(
        {"action": "connect", "connectors": ["gmail"]}, client_factory=lambda: client))
    assert client.calls == [("connections", ("gmail",), False)]
    assert out["legs"][0]["connect_url"] == "https://x/gmail"
    assert out["status"] == "initiated"


def test_unknown_leg_fields_are_rejected():
    out = json.loads(manage_connections({"action": "install", "connectors": [_linear(url="https://evil")]}))
    assert "unknown leg field" in out["error"] and "url" in out["error"]


# ---------------------------------------------------------------------------
# catalog validation
# ---------------------------------------------------------------------------


def test_install_is_catalog_only_and_lists_the_catalog_on_a_miss():
    out = json.loads(manage_connections({"action": "install", "connectors": [{"name": "github", "mcp": True}]}))
    assert "github" in out["error"]
    assert "figma, linear, notion" in out["error"]


def test_enable_and_authorize_need_a_configured_server():
    out = json.loads(manage_connections({"action": "enable", "connectors": [{"name": "figma", "mcp": True}]}))
    assert "figma" in out["error"] and "paper" in out["error"]


# ---------------------------------------------------------------------------
# the inline executor + replay shim
# ---------------------------------------------------------------------------


def _agent(callback):
    return SimpleNamespace(session_id="s1", connection_callback=callback)


def test_inline_executor_hands_the_agent_callback_to_the_tool(backend):
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext

    callback = _answering(json.dumps({"legs": [{"name": "paper", "status": "approved"}]}))
    with patch("tools.connectors.run.WATCH_INTERVAL_SECONDS", 0.01):
        out = json.loads(INLINE_TOOL_EXECUTORS["manage_connections"](
            _agent(callback), {"action": "enable", "connectors": [_mcp_leg("paper")]}, InlineToolContext("task")))
    assert len(callback.seen) == 1
    assert out["legs"][0]["state"] == LegState.connected.value


def test_setup_mcp_replay_shim_translates_to_an_mcp_leg(backend):
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext

    callback = _answering(json.dumps({"legs": [{"name": "linear", "status": "skipped"}]}))
    with patch("tools.connectors.run.WATCH_INTERVAL_SECONDS", 0.01):
        out = json.loads(INLINE_TOOL_EXECUTORS["setup_mcp"](
            _agent(callback), {"server": "linear", "action": "install", "reason": "old convo"}, InlineToolContext("task", tool_call_id="call-9")))
    (leg,) = callback.seen[0]["legs"]
    assert (leg["name"], leg["kind"], leg["action"], leg["state"]) == ("linear", "mcp", "install", "pending")
    assert callback.seen[0]["tool_call_id"] == "call-9"
    assert out["legs"][0]["state"] == LegState.skipped.value


def test_setup_mcp_is_gone_from_every_advertised_toolset():
    from toolsets import TOOLSETS, resolve_toolset

    assert all("setup_mcp" not in resolve_toolset(name) for name in TOOLSETS)
    assert "manage_connections" in resolve_toolset("connections")
    assert "hand-edit" in MANAGE_CONNECTIONS_SCHEMA["description"]
    assert "mcp_servers" in MANAGE_CONNECTIONS_SCHEMA["description"]


# ---------------------------------------------------------------------------
# deadline ownership
# ---------------------------------------------------------------------------


def test_the_bounded_wait_owns_the_deadline_not_the_sequential_guard():
    from agent import tool_executor as te

    assert "manage_connections" in te._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS


# ---------------------------------------------------------------------------
# the surface, the actor of a repeated failure, the settle race, the worker guards
# ---------------------------------------------------------------------------


def _runner_for(action, backend, *legs):
    """A prepared runner and its operation, the way ``run_operation`` opens one."""
    operation = ConnectionOperation([Leg(n, "mcp", action) for n in legs], session_key="s1")
    runner = mcp.open_runner(action, backend)
    runner.prepare(operation)
    return runner, operation


def _observe_until(runner, operation, name, state, *, seconds=2.0):
    """Run the watch loop's read until the named leg reaches ``state``."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        runner.observe(operation)
        if operation.leg(name).state == state:
            return
        time.sleep(0.01)
    raise AssertionError(f"{name} stayed {operation.leg(name).state.value}, wanted {state.value}")


def test_the_off_desktop_note_never_sends_the_model_to_an_action_that_refuses_mcp_names():
    """``status`` (and every other managed action) refuses an MCP leg, so the note that the
    model reads off the desktop must not send it there."""
    import re

    from tools.connectors.legs.normalize import ALL_ACTIONS, validate_action

    refusing = [action for action in ALL_ACTIONS if validate_action(action, [], ["paper"])]
    assert refusing, "a managed action must refuse an MCP leg for this contract to mean anything"
    named = [a for a in refusing if re.search(rf"\b{a}\b", mcp.OFF_DESKTOP_NOTE)]
    assert named == []


def test_a_desktop_session_with_no_callback_gets_the_link_at_once_and_opens_no_operation(backend):
    """A call that arrives without the callback (registry dispatch, say from execute_code) has
    nothing to render a card, so an operation would block the tool for its whole deadline with
    nobody to answer it. The link goes to the model instead, the way it does off the desktop."""
    with patch("tools.connectors.operation.OPERATION_DEADLINE_SECONDS", 0.2), \
         patch("tools.connectors.run.WATCH_INTERVAL_SECONDS", 0.01):
        out = json.loads(manage_connections({"action": "authorize", "connectors": [_mcp_leg("paper")]},
                                            connection_callback=None, session_id="s1", mcp_backend=backend))

    assert out["status"] == "initiated"
    assert out["legs"][0]["connect_url"] == "https://auth.example/paper/1"
    assert operations.current(Owner.current("s1")) == []


def test_a_repeated_failure_is_reported_by_the_backend_watcher_not_the_user(changes):
    """Try again that fails again has no state change to emit, only newer text; the backend
    produced that text, so the frame must not claim the user did."""
    backend = FakeBackend(oauth_error="the provider rejected the client registration")
    runner, operation = _runner_for("authorize", backend, "paper")
    assert mcp.retry(operation, ["paper"]) is None
    runner.close()

    assert operation.leg("paper").state == LegState.failed
    assert [c["actor"] for c in changes] == [Actor.backend_watcher.value] * 2


def test_a_continue_during_the_read_drops_the_outcome_instead_of_raising(backend):
    """``connection.respond`` can settle the operation between the watch loop's settled check and
    the transition; a settled operation has a frozen result, so the lost move is not an error."""
    runner, operation = _runner_for("authorize", backend, "paper")
    attempt = backend.attempts["paper"]
    attempt.approve(["search"])
    poll = attempt.poll
    attempt.poll = lambda: (operation.settle(SettleReason.continue_), poll())[1]

    runner.observe(operation)
    runner.close()

    assert operation.result()["legs"][0]["state"] == LegState.not_connected.value


def test_a_worker_whose_operation_settled_first_drops_its_outcome(backend):
    """The install runs on a worker thread; when Continue settles the operation first, the result
    is already frozen and the outcome has nowhere to go."""
    runner, operation = _runner_for("install", backend, "figma")
    started, release = threading.Event(), threading.Event()
    operation.transition("figma", LegState.initiated, Actor.backend_watcher)
    runner.spawn(operation, operation.leg("figma"), lambda: (started.set(), release.wait(5), ["get_file"])[2])
    assert started.wait(5)
    worker = next(t for t in threading.enumerate() if t.name == "mcp-install-figma")

    operation.settle(SettleReason.continue_)
    release.set()
    worker.join(5)
    runner.close()

    assert not worker.is_alive()
    assert not runner.work["figma"].done.is_set()
    assert operation.result()["legs"][0]["state"] == LegState.not_connected.value


def test_the_prepare_threads_and_the_worker_carry_the_calling_thread_s_profile(tmp_path):
    """A named-profile turn binds its home through a contextvar on the tool thread. The OAuth
    flows start on prepare threads and the install runs on a worker; each must see that same
    home, or the token and the catalog read land in the process home."""
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

    backend = FakeBackend()
    homes = []
    start_oauth, install = backend.start_oauth, backend.install
    backend.start_oauth = lambda name: (homes.append((name, get_hermes_home())), start_oauth(name))[1]
    backend.install = lambda name, env: (homes.append((name, get_hermes_home())), install(name, env))[1]

    token = set_hermes_home_override(tmp_path)
    try:
        runner, operation = _runner_for("authorize", backend, "paper", "linear")
        runner.close()
        runner, operation = _runner_for("install", backend, "figma")
        apply_answer(operation, json.dumps({"legs": [{"name": "figma", "status": "approved"}]}))
        _observe_until(runner, operation, "figma", LegState.connected)
        runner.close()
    finally:
        reset_hermes_home_override(token)

    assert sorted(name for name, _ in homes) == ["figma", "linear", "paper"]
    assert [home for _, home in homes] == [tmp_path] * 3


def test_a_skip_of_a_row_that_already_connected_is_ignored_and_the_rest_of_the_answer_lands():
    """The card can send a skip for a row the watcher connected a moment earlier. That skip has
    nothing to move; the other skips and the Continue in the same answer must still apply."""
    operation = ConnectionOperation([Leg(n, "mcp", "enable") for n in ("paper", "linear", "notion")], session_key="s1")
    operation.transition("paper", LegState.initiated, Actor.backend_watcher)
    operation.transition("paper", LegState.connected, Actor.backend_watcher)
    operation.transition("notion", LegState.initiated, Actor.backend_watcher)

    apply_answer(operation, json.dumps({
        "legs": [{"name": "paper", "status": "skipped"}, {"name": "linear", "status": "skipped"}],
        "settled_by": "continue",
    }))

    assert operation.leg("paper").state == LegState.connected
    assert operation.leg("linear").state == LegState.skipped
    assert operation.settled_by == SettleReason.continue_
    assert operation.result()["legs"][2]["state"] == LegState.not_connected.value


def test_a_skip_that_loses_the_race_to_the_watcher_is_ignored_not_raised(monkeypatch):
    """The resolved check and the transition are two steps; the watcher can connect the row between
    them. The refused move is the same nothing-to-do as a row resolved before the answer arrived."""
    operation = ConnectionOperation([Leg(n, "mcp", "enable") for n in ("paper", "linear")], session_key="s1")
    operation.transition("paper", LegState.initiated, Actor.backend_watcher)
    real = operation.transition

    def racing(name, to, actor, **fields):
        if name == "paper" and to == LegState.skipped:
            real("paper", LegState.connected, Actor.backend_watcher)  # the watcher lands first
        return real(name, to, actor, **fields)

    monkeypatch.setattr(operation, "transition", racing)
    apply_answer(operation, json.dumps({
        "legs": [{"name": "paper", "status": "skipped"}, {"name": "linear", "status": "skipped"}],
        "settled_by": "continue",
    }))

    assert operation.leg("paper").state == LegState.connected
    assert operation.leg("linear").state == LegState.skipped  # the rest of the answer landed
    assert operation.all_resolved


def test_try_again_on_a_settled_operation_starts_no_work(backend):
    runner, operation = _runner_for("enable", backend, "paper")
    operation.settle(SettleReason.continue_)

    assert mcp.retry(operation, ["paper"]) is not None
    runner.close()
    assert backend.calls == []


# ---------------------------------------------------------------------------
# install: the credentials the worker would otherwise prompt for
# ---------------------------------------------------------------------------


def test_an_install_approved_without_its_required_credential_never_reaches_the_worker():
    """``install_entry`` prompts on stdin for a credential it cannot find; on a worker thread that
    call never returns, so the row waits for the card's fields instead."""
    backend = FakeBackend(missing_env=["FIGMA_TOKEN"])
    runner, operation = _runner_for("install", backend, "figma")

    apply_answer(operation, json.dumps({"legs": [{"name": "figma", "status": "approved"}]}))
    runner.close()

    leg = operation.leg("figma")
    assert leg.state == LegState.pending
    assert [spec["name"] for spec in leg.required_env] == ["FIGMA_TOKEN"]
    assert not [call for call in backend.calls if call[0] == "install"]


def test_a_retry_of_a_failed_install_reuses_the_credentials_the_card_supplied():
    """Try again sends no credentials — the card has no fields on a failed row — so the install
    that runs again is the one the user already approved."""
    backend = FakeBackend(missing_env=["FIGMA_TOKEN"], install_error="the registry was unreachable")
    runner, operation = _runner_for("install", backend, "figma")
    apply_answer(operation, json.dumps({"legs": [{"name": "figma", "status": "approved",
                                                     "env": {"FIGMA_TOKEN": "tok-1"}}]}))
    _observe_until(runner, operation, "figma", LegState.failed)

    assert mcp.retry(operation, ["figma"]) is None
    _observe_until(runner, operation, "figma", LegState.failed)
    runner.close()

    assert [c for c in backend.calls if c[0] == "install"] == [("install", "figma", {"FIGMA_TOKEN": "tok-1"})] * 2


def test_enable_writes_the_config_under_the_shared_config_lock():
    """The dashboard's ``PUT /api/mcp/servers/{name}/enabled`` writes ``mcp_servers`` under this
    lock; an unserialised read-modify-write from the tool thread drops one of the two writes."""
    import hermes_cli.web_server as web_server
    from hermes_cli.config import load_config, save_config

    save_config({**load_config(), "mcp_servers": {"paper": {"command": "paper-mcp", "enabled": False}}})
    done = threading.Event()

    def flip():
        mcp._CatalogBackend().enable("paper")
        done.set()

    with web_server._CONFIG_MUTATION_LOCK:
        threading.Thread(target=flip, daemon=True, name="enable-under-lock").start()
        assert not done.wait(0.5)
        assert load_config()["mcp_servers"]["paper"]["enabled"] is False
    assert done.wait(5)
    assert load_config()["mcp_servers"]["paper"]["enabled"] is True


# ---------------------------------------------------------------------------
# several legs in one call
# ---------------------------------------------------------------------------


def test_several_authorize_flows_start_together_and_share_one_wait():
    """Each flow blocks until its provider publishes the authorization URL; one wait per leg
    would keep the card empty for minutes on a two-server call."""
    backend = FakeBackend()
    both_in_flight = threading.Barrier(2, timeout=5)
    start_oauth = backend.start_oauth
    backend.start_oauth = lambda name: (both_in_flight.wait(), start_oauth(name))[1]

    runner, operation = _runner_for("authorize", backend, "paper", "linear")
    runner.close()

    assert [t.state for t in operation.legs] == [LegState.initiated] * 2
    assert all(t.connect_url for t in operation.legs)


def test_a_url_that_arrives_after_the_shared_wait_still_lands_on_its_row(monkeypatch):
    """The shared wait bounds how long prepare blocks, not how long a provider may take. A row whose
    URL arrives after it must still get the link, and nothing else may have written the row first."""
    monkeypatch.setattr(mcp, "PREPARE_WAIT_SECONDS", 0.05)
    backend = FakeBackend()
    start_oauth = backend.start_oauth
    backend.start_oauth = lambda name: (time.sleep(0.3) if name == "linear" else None, start_oauth(name))[1]
    thread_errors = []
    monkeypatch.setattr(threading, "excepthook", lambda args: thread_errors.append(args.exc_value))

    runner, operation = _runner_for("authorize", backend, "paper", "linear")
    deadline = time.time() + 2.0
    while operation.leg("linear").state != LegState.initiated and time.time() < deadline:
        time.sleep(0.02)
    runner.close()

    assert thread_errors == []
    assert [t.state for t in operation.legs] == [LegState.initiated] * 2
    assert operation.leg("linear").connect_url


def test_the_off_desktop_operation_emits_no_connection_update_frames(backend, changes):
    """Off the desktop the operation is in no session's registry, so a frame would reach a
    renderer that knows nothing about it."""
    out = _off_desktop({"action": "authorize", "connectors": [_mcp_leg("paper")]}, mcp_backend=backend)

    assert out["legs"][0]["state"] == LegState.initiated.value
    assert changes == []
    assert operations.current(Owner.current("s1")) == []
