"""MCP targets of manage_connections (the fold that retired setup_mcp).

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

import tools.connectors.tool  # registers the tool
from tools.connectors.contract import Actor, SettleReason, TargetState
from tools.connectors import live, mcp
from tools.connectors import operation as op
from tools.connectors.mcp import apply_answer
from tools.connectors.operation import ConnectionOperation, Target
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
    live.reset_for_tests()
    yield
    live.reset_for_tests()


@pytest.fixture(autouse=True)
def _catalog(backend):
    # The default backend is patched too: a call that cannot be handed one (registry dispatch, the
    # inline executor) must never reach the real catalog or installer from a test.
    with patch("tools.connectors.mcp._catalog_names", return_value=CATALOG), \
         patch("tools.connectors.mcp._configured_names", return_value=sorted(CONFIGURED)), \
         patch("tools.connectors.mcp._default_backend", return_value=backend), \
         patch("tools.connectors.mcp.session_platform", return_value="desktop"):
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


def _mcp_target(name):
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
            operation = live.get(session_id, payload["op_id"])
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
    out = _mcp({"action": "authorize", "connectors": [_mcp_target("paper")]}, callback, mcp_backend=backend)

    (offered,) = callback.seen[0]["targets"]
    assert offered["state"] == TargetState.initiated.value
    assert offered["connect_url"] == "https://auth.example/paper/1"
    (settled,) = out["targets"]
    assert settled["state"] == TargetState.connected.value
    assert settled["tools"] == ["search", "comment"]
    assert out["settled_by"] == SettleReason.all_resolved.value


def test_a_flow_that_fails_moves_the_row_to_failed_with_the_error_text(backend, changes):
    def callback(payload):
        backend.attempts["paper"].fail("the provider rejected the client registration")
        return None

    with patch("tools.connectors.operation.OPERATION_DEADLINE_SECONDS", 0.4):
        _mcp({"action": "authorize", "connectors": [_mcp_target("paper")]}, callback, mcp_backend=backend)

    failure = [c for c in changes if c and c["to"] == TargetState.failed.value][-1]
    assert failure["detail"] == "the provider rejected the client registration"
    assert failure["actor"] == Actor.backend_watcher.value


def test_a_flow_that_never_starts_fails_the_row_before_the_card_is_drawn():
    backend = FakeBackend(oauth_error="no OAuth callback route in this process")
    callback = _answering(json.dumps({"settled_by": "continue"}))
    out = _mcp({"action": "authorize", "connectors": [_mcp_target("paper")]}, callback, mcp_backend=backend)

    (offered,) = callback.seen[0]["targets"]
    assert offered["state"] == TargetState.failed.value
    assert offered["detail"] == "no OAuth callback route in this process"
    assert "connect_url" not in offered
    assert out["settled_by"] == SettleReason.continue_.value


def test_install_waits_for_the_credentials_it_declares_and_installs_with_them():
    backend = FakeBackend(missing_env=["FIGMA_TOKEN"], tools=["get_file"])
    answer = json.dumps({"targets": [{"name": "figma", "status": "approved", "env": {"FIGMA_TOKEN": "tok-1"}}]})
    callback = _answering(answer)
    out = _mcp({"action": "install", "connectors": [_mcp_target("figma")]}, callback, mcp_backend=backend)

    (offered,) = callback.seen[0]["targets"]
    assert offered["state"] == TargetState.pending.value
    assert offered["required_env"] == [{"name": "FIGMA_TOKEN", "prompt": "FIGMA_TOKEN?", "required": True}]
    assert ("install", "figma", {"FIGMA_TOKEN": "tok-1"}) in backend.calls
    (settled,) = out["targets"]
    assert settled["state"] == TargetState.connected.value
    assert settled["tools"] == ["get_file"]


def test_enable_connects_on_the_card_s_approval(backend):
    answer = json.dumps({"targets": [{"name": "paper", "status": "approved"}]})
    out = _mcp({"action": "enable", "connectors": [_mcp_target("paper")]}, _answering(answer), mcp_backend=backend)

    assert ("enable", "paper") in backend.calls
    (settled,) = out["targets"]
    assert settled["state"] == TargetState.connected.value


def test_a_card_claim_other_than_approved_or_skipped_moves_nothing(backend):
    answer = json.dumps({"targets": [{"name": "paper", "status": "connected", "tools": ["x"]}],
                         "settled_by": "continue"})
    out = _mcp({"action": "enable", "connectors": [_mcp_target("paper")]}, _answering(answer), mcp_backend=backend)

    assert backend.calls == []
    (settled,) = out["targets"]
    assert settled["state"] == TargetState.not_connected.value
    assert out["settled_by"] == SettleReason.continue_.value


def test_a_skip_resolves_the_row_and_the_operation(backend):
    answer = json.dumps({"targets": [{"name": "paper", "status": "skipped"}]})
    out = _mcp({"action": "enable", "connectors": [_mcp_target("paper")]}, _answering(answer), mcp_backend=backend)

    assert out["targets"][0]["state"] == TargetState.skipped.value
    assert out["settled_by"] == SettleReason.all_resolved.value


def test_no_answer_settles_by_deadline_and_marks_targets_not_connected(backend):
    with patch("tools.connectors.operation.OPERATION_DEADLINE_SECONDS", 0.05):
        out = _mcp({"action": "install", "connectors": [_linear()]}, _answering(None), mcp_backend=backend)
    assert out["settled_by"] == SettleReason.deadline.value
    assert out["targets"][0]["state"] == TargetState.not_connected.value
    assert "error" not in out


def test_mcp_secrets_never_reach_the_model():
    backend = FakeBackend(missing_env=["LINEAR_API_KEY"])
    answer = json.dumps({"targets": [{"name": "linear", "status": "approved",
                                      "env": {"LINEAR_API_KEY": "sk-secret"}}]})
    out = _mcp({"action": "install", "connectors": [_linear()]}, _answering(answer), mcp_backend=backend)
    assert "sk-secret" not in json.dumps(out)


# ---------------------------------------------------------------------------
# off the desktop: no card, so the work runs at once
# ---------------------------------------------------------------------------


def _off_desktop(args, **kw):
    with patch("tools.connectors.mcp.session_platform", return_value="tui"):
        return json.loads(manage_connections(args, session_id="s1", **kw))


def test_off_desktop_authorize_returns_the_link_at_once_and_opens_no_operation(backend):
    out = _off_desktop({"action": "authorize", "connectors": [_mcp_target("paper")]}, mcp_backend=backend)

    (target,) = out["targets"]
    assert target["state"] == TargetState.initiated.value
    assert target["connect_url"] == "https://auth.example/paper/1"
    assert out["status"] == "initiated"
    assert live.current("s1") is None


def test_off_desktop_install_without_its_credentials_fails_and_names_them():
    backend = FakeBackend(missing_env=["FIGMA_TOKEN"])
    out = _off_desktop({"action": "install", "connectors": [_mcp_target("figma")]}, mcp_backend=backend)

    (target,) = out["targets"]
    assert target["state"] == TargetState.failed.value
    assert "FIGMA_TOKEN" in target["detail"]
    assert not [c for c in backend.calls if c[0] == "install"]


def test_registry_dispatch_never_blocks_and_never_reaches_a_card(backend):
    # registry.dispatch forwards no callback; the call must return, not block.
    with patch("tools.connectors.mcp.session_platform", return_value="tui"):
        out = json.loads(registry.dispatch("manage_connections", {"action": "enable", "connectors": [_linear()]}))
    assert out["targets"][0]["state"] == TargetState.connected.value


def test_a_managed_action_never_accepts_mcp_targets_and_vice_versa():
    client = FakeClient()
    out = json.loads(manage_connections(
        {"action": "connect", "connectors": ["gmail", _linear()]}, client_factory=lambda: client))
    assert "managed-connector action" in out["error"]
    assert client.calls == []  # rejected before any gateway call

    out = json.loads(manage_connections({"action": "install", "connectors": ["gmail", _linear()]}))
    assert "must carry" in out["error"]


def test_a_managed_call_off_desktop_returns_a_link_per_target():
    client = FakeClient()
    out = json.loads(manage_connections(
        {"action": "connect", "connectors": ["gmail"]}, client_factory=lambda: client))
    assert client.calls == [("connections", ("gmail",), False)]
    assert out["targets"][0]["connect_url"] == "https://x/gmail"
    assert out["status"] == "initiated"


def test_unknown_target_fields_are_rejected():
    out = json.loads(manage_connections({"action": "install", "connectors": [_linear(url="https://evil")]}))
    assert "unknown target field" in out["error"] and "url" in out["error"]


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

    callback = _answering(json.dumps({"targets": [{"name": "paper", "status": "approved"}]}))
    with patch("tools.connectors.run.WATCH_INTERVAL_SECONDS", 0.01):
        out = json.loads(INLINE_TOOL_EXECUTORS["manage_connections"](
            _agent(callback), {"action": "enable", "connectors": [_mcp_target("paper")]}, InlineToolContext("task")))
    assert len(callback.seen) == 1
    assert out["targets"][0]["state"] == TargetState.connected.value


def test_setup_mcp_replay_shim_translates_to_an_mcp_target(backend):
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext

    callback = _answering(json.dumps({"targets": [{"name": "linear", "status": "skipped"}]}))
    with patch("tools.connectors.run.WATCH_INTERVAL_SECONDS", 0.01):
        out = json.loads(INLINE_TOOL_EXECUTORS["setup_mcp"](
            _agent(callback), {"server": "linear", "action": "install", "reason": "old convo"}, InlineToolContext("task", tool_call_id="call-9")))
    (target,) = callback.seen[0]["targets"]
    assert (target["name"], target["kind"], target["action"], target["state"]) == ("linear", "mcp", "install", "pending")
    assert callback.seen[0]["tool_call_id"] == "call-9"
    assert out["targets"][0]["state"] == TargetState.skipped.value


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


def _runner_for(action, backend, *targets):
    """A prepared runner and its operation, the way ``run_operation`` opens one."""
    operation = ConnectionOperation([Target(n, "mcp", action) for n in targets], session_key="s1")
    runner = mcp.open_runner(action, backend)
    runner.prepare(operation)
    return runner, operation


def _observe_until(runner, operation, name, state, *, seconds=2.0):
    """Run the watch loop's read until the named target reaches ``state``."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        runner.observe(operation)
        if operation.target(name).state == state:
            return
        time.sleep(0.01)
    raise AssertionError(f"{name} stayed {operation.target(name).state.value}, wanted {state.value}")


def test_the_off_desktop_note_never_sends_the_model_to_an_action_that_refuses_mcp_names():
    """``status`` (and every other managed action) refuses an MCP target, so the note that the
    model reads off the desktop must not send it there."""
    import re

    from tools.connectors.targets import ALL_ACTIONS, validate_action

    refusing = [action for action in ALL_ACTIONS if validate_action(action, [], ["paper"])]
    assert refusing, "a managed action must refuse an MCP target for this contract to mean anything"
    named = [a for a in refusing if re.search(rf"\b{a}\b", mcp.OFF_DESKTOP_NOTE)]
    assert named == []


def test_a_desktop_session_with_no_callback_gets_the_link_at_once_and_opens_no_operation(backend):
    """A call that arrives without the callback (registry dispatch, say from execute_code) has
    nothing to render a card, so an operation would block the tool for its whole deadline with
    nobody to answer it. The link goes to the model instead, the way it does off the desktop."""
    with patch("tools.connectors.operation.OPERATION_DEADLINE_SECONDS", 0.2), \
         patch("tools.connectors.run.WATCH_INTERVAL_SECONDS", 0.01):
        out = json.loads(manage_connections({"action": "authorize", "connectors": [_mcp_target("paper")]},
                                            connection_callback=None, session_id="s1", mcp_backend=backend))

    assert out["status"] == "initiated"
    assert out["targets"][0]["connect_url"] == "https://auth.example/paper/1"
    assert live.current("s1") is None


def test_a_repeated_failure_is_reported_by_the_backend_watcher_not_the_user(changes):
    """Try again that fails again has no state change to emit, only newer text; the backend
    produced that text, so the frame must not claim the user did."""
    backend = FakeBackend(oauth_error="the provider rejected the client registration")
    runner, operation = _runner_for("authorize", backend, "paper")
    assert mcp.retry(operation, ["paper"]) is None
    runner.close()

    assert operation.target("paper").state == TargetState.failed
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

    assert operation.result()["targets"][0]["state"] == TargetState.not_connected.value


def test_a_worker_whose_operation_settled_first_drops_its_outcome(backend):
    """The install runs on a worker thread; when Continue settles the operation first, the result
    is already frozen and the outcome has nowhere to go."""
    runner, operation = _runner_for("install", backend, "figma")
    started, release = threading.Event(), threading.Event()
    operation.transition("figma", TargetState.initiated, Actor.backend_watcher)
    runner.spawn(operation, operation.target("figma"), lambda: (started.set(), release.wait(5), ["get_file"])[2])
    assert started.wait(5)
    worker = next(t for t in threading.enumerate() if t.name == "mcp-install-figma")

    operation.settle(SettleReason.continue_)
    release.set()
    worker.join(5)
    runner.close()

    assert not worker.is_alive()
    assert not runner.work["figma"].done.is_set()
    assert operation.result()["targets"][0]["state"] == TargetState.not_connected.value


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
        apply_answer(operation, json.dumps({"targets": [{"name": "figma", "status": "approved"}]}))
        _observe_until(runner, operation, "figma", TargetState.connected)
        runner.close()
    finally:
        reset_hermes_home_override(token)

    assert sorted(name for name, _ in homes) == ["figma", "linear", "paper"]
    assert [home for _, home in homes] == [tmp_path] * 3


def test_a_skip_of_a_row_that_already_connected_is_ignored_and_the_rest_of_the_answer_lands():
    """The card can send a skip for a row the watcher connected a moment earlier. That skip has
    nothing to move; the other skips and the Continue in the same answer must still apply."""
    operation = ConnectionOperation([Target(n, "mcp", "enable") for n in ("paper", "linear", "notion")], session_key="s1")
    operation.transition("paper", TargetState.initiated, Actor.backend_watcher)
    operation.transition("paper", TargetState.connected, Actor.backend_watcher)
    operation.transition("notion", TargetState.initiated, Actor.backend_watcher)

    apply_answer(operation, json.dumps({
        "targets": [{"name": "paper", "status": "skipped"}, {"name": "linear", "status": "skipped"}],
        "settled_by": "continue",
    }))

    assert operation.target("paper").state == TargetState.connected
    assert operation.target("linear").state == TargetState.skipped
    assert operation.settled_by == SettleReason.continue_
    assert operation.result()["targets"][2]["state"] == TargetState.not_connected.value


def test_a_skip_that_loses_the_race_to_the_watcher_is_ignored_not_raised(monkeypatch):
    """The resolved check and the transition are two steps; the watcher can connect the row between
    them. The refused move is the same nothing-to-do as a row resolved before the answer arrived."""
    operation = ConnectionOperation([Target(n, "mcp", "enable") for n in ("paper", "linear")], session_key="s1")
    operation.transition("paper", TargetState.initiated, Actor.backend_watcher)
    real = operation.transition

    def racing(name, to, actor, **fields):
        if name == "paper" and to == TargetState.skipped:
            real("paper", TargetState.connected, Actor.backend_watcher)  # the watcher lands first
        return real(name, to, actor, **fields)

    monkeypatch.setattr(operation, "transition", racing)
    apply_answer(operation, json.dumps({
        "targets": [{"name": "paper", "status": "skipped"}, {"name": "linear", "status": "skipped"}],
        "settled_by": "continue",
    }))

    assert operation.target("paper").state == TargetState.connected
    assert operation.target("linear").state == TargetState.skipped  # the rest of the answer landed
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

    apply_answer(operation, json.dumps({"targets": [{"name": "figma", "status": "approved"}]}))
    runner.close()

    target = operation.target("figma")
    assert target.state == TargetState.pending
    assert [spec["name"] for spec in target.required_env] == ["FIGMA_TOKEN"]
    assert not [call for call in backend.calls if call[0] == "install"]


def test_a_retry_of_a_failed_install_reuses_the_credentials_the_card_supplied():
    """Try again sends no credentials — the card has no fields on a failed row — so the install
    that runs again is the one the user already approved."""
    backend = FakeBackend(missing_env=["FIGMA_TOKEN"], install_error="the registry was unreachable")
    runner, operation = _runner_for("install", backend, "figma")
    apply_answer(operation, json.dumps({"targets": [{"name": "figma", "status": "approved",
                                                     "env": {"FIGMA_TOKEN": "tok-1"}}]}))
    _observe_until(runner, operation, "figma", TargetState.failed)

    assert mcp.retry(operation, ["figma"]) is None
    _observe_until(runner, operation, "figma", TargetState.failed)
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
# several targets in one call
# ---------------------------------------------------------------------------


def test_several_authorize_flows_start_together_and_share_one_wait():
    """Each flow blocks until its provider publishes the authorization URL; one wait per target
    would keep the card empty for minutes on a two-server call."""
    backend = FakeBackend()
    both_in_flight = threading.Barrier(2, timeout=5)
    start_oauth = backend.start_oauth
    backend.start_oauth = lambda name: (both_in_flight.wait(), start_oauth(name))[1]

    runner, operation = _runner_for("authorize", backend, "paper", "linear")
    runner.close()

    assert [t.state for t in operation.targets] == [TargetState.initiated] * 2
    assert all(t.connect_url for t in operation.targets)


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
    while operation.target("linear").state != TargetState.initiated and time.time() < deadline:
        time.sleep(0.02)
    runner.close()

    assert thread_errors == []
    assert [t.state for t in operation.targets] == [TargetState.initiated] * 2
    assert operation.target("linear").connect_url


def test_the_off_desktop_operation_emits_no_connection_update_frames(backend, changes):
    """Off the desktop the operation is in no session's registry, so a frame would reach a
    renderer that knows nothing about it."""
    out = _off_desktop({"action": "authorize", "connectors": [_mcp_target("paper")]}, mcp_backend=backend)

    assert out["targets"][0]["state"] == TargetState.initiated.value
    assert changes == []
    assert live.current("s1") is None
