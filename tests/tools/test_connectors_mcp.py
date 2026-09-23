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

from tools.connectors.contract import SettleReason, TargetState
from tools.connectors import live
from tools.connectors.mcp import apply_answer
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
        self.discovery_error = ""

    def poll(self):
        return {"status": self.status, "error": self.error, "tools": list(self.tools),
                "discovery_error": self.discovery_error}

    def approve(self, tools):
        self.tools, self.status = list(tools), "approved"

    def fail(self, error):
        self.error, self.status = error, "error"


class FakeBackend:
    """The one fake: the catalog, the installer and the OAuth flow runner behind ``mcp.py``."""

    def __init__(self, *, missing_env=(), tools=("read", "write"), registered_tools=(),
                 install_error="", registration_error="", oauth_error=""):
        self.calls = []
        self.attempts = {}
        self.missing_env = list(missing_env)
        self.tools = list(tools)
        self.registered_tools = list(registered_tools)
        self.install_error = install_error
        self.registration_error = registration_error
        self.oauth_error = oauth_error

    def required_env(self, name):
        self.calls.append(("required_env", name))
        return [{"name": key, "prompt": f"{key}?", "required": True,
                 "secret": True, "default": ""} for key in self.missing_env]

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
    registered = []

    def register(runner, target, name):
        runner.backend.calls.append(("register", name))
        if runner.backend.registration_error:
            from tools.connectors.mcp import _detail

            return [], _detail(runner.backend.registration_error, runner, target)
        names = list(runner.backend.registered_tools)
        for tool_name in names:
            registry.register(
                name=tool_name,
                toolset=f"mcp-{name}",
                schema={"name": tool_name, "description": f"Registered {tool_name}", "parameters": {}},
                handler=lambda *_args, **_kwargs: "{}",
            )
            registered.append(tool_name)
        return names, ""

    with patch("tools.connectors.mcp._catalog_names", return_value=CATALOG), \
         patch("tools.connectors.mcp._configured_names", return_value=sorted(CONFIGURED)), \
         patch("tools.connectors.mcp._default_backend", return_value=backend), \
         patch("tools.connectors.mcp._register_connected", side_effect=register):
        yield
    with registry._lock:
        for tool_name in registered:
            registry._tools.pop(tool_name, None)
        if registered:
            registry._generation += 1


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


def test_install_waits_for_the_credentials_it_declares_and_installs_with_them():
    backend = FakeBackend(
        missing_env=["FIGMA_TOKEN"], tools=["probe_only"],
        registered_tools=["mcp__figma__get_file", "mcp__figma__list_files"],
    )
    answer = json.dumps({"targets": [{"name": "figma", "status": "approved", "env": {"FIGMA_TOKEN": "tok-1"}}]})
    callback = _answering(answer)
    out = _mcp({"action": "install", "connectors": [_mcp_target("figma")]}, callback, mcp_backend=backend)

    (offered,) = callback.seen[0]["targets"]
    assert offered["state"] == TargetState.pending.value
    assert offered["required_env"] == [{"name": "FIGMA_TOKEN", "prompt": "FIGMA_TOKEN?",
                                        "required": True, "secret": True, "default": ""}]
    assert ("install", "figma", {"FIGMA_TOKEN": "tok-1"}) in backend.calls
    (settled,) = out["targets"]
    assert settled["state"] == TargetState.connected.value
    assert settled["tools"] == ["mcp__figma__get_file", "mcp__figma__list_files"]
    assert all(name in settled["tools_listing"] for name in settled["tools"])
    assert "tool_describe" in settled["tools_listing"] and "tool_call" in settled["tools_listing"]


def test_a_card_claim_other_than_approved_or_skipped_moves_nothing(backend):
    answer = json.dumps({"targets": [{"name": "paper", "status": "connected", "tools": ["x"]}],
                         "settled_by": "continue"})
    out = _mcp({"action": "enable", "connectors": [_mcp_target("paper")]}, _answering(answer), mcp_backend=backend)

    assert backend.calls == []
    (settled,) = out["targets"]
    assert settled["state"] == TargetState.not_connected.value
    assert out["settled_by"] == SettleReason.continue_.value


def test_no_answer_settles_by_deadline_and_marks_targets_not_connected(backend):
    with patch("tools.connectors.operation.OPERATION_DEADLINE_SECONDS", 0.05):
        out = _mcp({"action": "install", "connectors": [_linear()]}, _answering(None), mcp_backend=backend)
    assert out["settled_by"] == SettleReason.deadline.value
    assert out["targets"][0]["state"] == TargetState.not_connected.value
    assert "error" not in out


def test_mcp_secrets_never_reach_the_model():
    backend = FakeBackend(missing_env=["LINEAR_API_KEY"], registration_error="sk-secret was rejected")
    answer = json.dumps({"targets": [{"name": "linear", "status": "approved",
                                      "env": {"LINEAR_API_KEY": "sk-secret"}}]})
    out = _mcp({"action": "install", "connectors": [_linear()]}, _answering(answer), mcp_backend=backend)
    payload = json.dumps(out)
    assert "sk-secret" not in payload
    assert "[REDACTED]" in payload
    assert out["targets"][0]["state"] == TargetState.connected.value
    assert out["targets"][0]["tools"] == []
    assert out["targets"][0]["discovery_error"] == "[REDACTED] was rejected"


# ---------------------------------------------------------------------------
# off the desktop: no card, so the work runs at once
# ---------------------------------------------------------------------------


def _off_desktop(args, **kw):
    return json.loads(manage_connections(args, session_id="s1", **kw))


def test_off_desktop_authorize_returns_the_link_at_once_and_opens_no_operation(backend):
    out = _off_desktop({"action": "authorize", "connectors": [_mcp_target("paper")]}, mcp_backend=backend)

    (target,) = out["targets"]
    assert target["state"] == TargetState.initiated.value
    assert target["connect_url"] == "https://auth.example/paper/1"
    assert out["status"] == "initiated"
    assert live.current("s1") is None


def test_registry_dispatch_never_blocks_and_never_reaches_a_card(backend):
    # registry.dispatch forwards no callback; the call must return, not block.
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



# ---------------------------------------------------------------------------
# install commits are all-or-nothing: a post-config-save failure restores both stores
# ---------------------------------------------------------------------------


def test_install_env_failure_rolls_back_config_and_env(tmp_path, monkeypatch):
    """install() promises 'a failure writes nothing'. _save_mcp_server lands before _save_env,
    so a mid-env failure must put the previous server entry and the previous .env values back."""
    import tools.connectors.mcp as mcp

    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("LINEAR_TOKEN=old-token\n")
    (home / "config.yaml").write_text("mcp_servers:\n  linear:\n    command: old-linear\n")
    monkeypatch.setenv("HERMES_HOME", str(home))

    entry = SimpleNamespace(auth=SimpleNamespace(env=[
        SimpleNamespace(name="LINEAR_TOKEN", prompt="token?", required=True, secret=True, default=""),
        SimpleNamespace(name="LINEAR_TEAM", prompt="team?", required=True, secret=True, default=""),
    ]))

    import hermes_cli.config as config_mod
    real_save = config_mod.save_env_value
    calls = []

    def flaky_save(key, value):
        calls.append(key)
        if len(calls) == 2:
            raise RuntimeError("disk full")
        return real_save(key, value)

    with patch("tools.connectors.mcp._catalog_entry", return_value=entry), \
         patch("hermes_cli.mcp_catalog.card_install_config", return_value={"command": "new-linear"}), \
         patch("hermes_cli.mcp_config._probe_single_server", return_value=[("read", "desc")]), \
         patch("hermes_cli.config._publish_env_value", lambda *_a, **_k: None), \
         patch("hermes_cli.config.save_env_value", side_effect=flaky_save):
        with pytest.raises(RuntimeError, match="disk full"):
            mcp._CatalogBackend().install("linear", {"LINEAR_TOKEN": "new-token", "LINEAR_TEAM": "eng"})

    # write, failing write, then the rollback restoring the first key through the same path
    assert calls == ["LINEAR_TOKEN", "LINEAR_TEAM", "LINEAR_TOKEN"]
    from hermes_cli.config import load_config
    assert load_config().get("mcp_servers", {}).get("linear") == {"command": "old-linear"}
    env_text = (home / ".env").read_text()
    assert "LINEAR_TOKEN=old-token" in env_text and "LINEAR_TEAM" not in env_text


def test_e2e_carded_install_failure_restores_config_and_env(tmp_path, monkeypatch):
    """The carded production path end to end: manage_connections install -> the card supplies
    the secrets -> _start_install -> backend.install -> probe -> _save_mcp_server -> _save_env.
    A mid-env failure must report the failure AND restore both config.yaml and .env."""
    import tools.connectors.mcp as mcp

    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("LINEAR_TOKEN=old-token\n")
    (home / "config.yaml").write_text("mcp_servers:\n  linear:\n    command: old-linear\n")
    monkeypatch.setenv("HERMES_HOME", str(home))

    entry = SimpleNamespace(auth=SimpleNamespace(type="api_key", provider=None, env=[
        SimpleNamespace(name="LINEAR_TOKEN", prompt="token?", required=True, secret=True, default=""),
        SimpleNamespace(name="LINEAR_TEAM", prompt="team?", required=True, secret=True, default=""),
    ]))

    import hermes_cli.config as config_mod
    real_save = config_mod.save_env_value
    calls = []

    def flaky_save(key, value):
        calls.append(key)
        if len(calls) == 2:
            raise RuntimeError("disk full")
        return real_save(key, value)

    # A failed target stays resumable (failed is not a resolved state, so the card can offer
    # retry/skip). The callback sees only the opening payload, so a watcher thread plays the
    # renderer: it polls the live operation and answers "skipped" once the failure lands.
    approve = json.dumps({"targets": [{"name": "linear", "status": "approved",
                                       "env": {"LINEAR_TOKEN": "new-token", "LINEAR_TEAM": "eng"}}]})
    skip = json.dumps({"targets": [{"name": "linear", "status": "skipped"}]})
    seen_failed = []
    op_ids = []

    def answering(payload):
        op_ids.append(payload["op_id"])

        def respond():
            operation = live.get("s1", payload["op_id"])
            if operation is not None:
                apply_answer(operation, approve)

        threading.Timer(0.01, respond).start()
        return None

    def watch_for_failure():
        deadline = time.time() + 10
        while time.time() < deadline and not seen_failed:
            for op_id in op_ids:
                operation = live.get("s1", op_id)
                if operation is None:
                    continue
                target = operation.target("linear")
                if target is not None and target.state == TargetState.failed:
                    seen_failed.append(target.snapshot())
                    apply_answer(operation, skip)
                    return
            time.sleep(0.01)

    watcher = threading.Thread(target=watch_for_failure, daemon=True)
    watcher.start()

    with patch("tools.connectors.mcp._default_backend", return_value=mcp._CatalogBackend()), \
         patch("tools.connectors.mcp._catalog_entry", return_value=entry), \
         patch("hermes_cli.mcp_catalog.card_install_config", return_value={"command": "new-linear"}), \
         patch("hermes_cli.mcp_config._probe_single_server", return_value=[("read", "desc")]), \
         patch("hermes_cli.config._publish_env_value", lambda *_a, **_k: None), \
         patch("hermes_cli.config.save_env_value", side_effect=flaky_save):
        out = _mcp({"action": "install", "connectors": [_linear()]}, answering)
    watcher.join(timeout=5)

    (failed,) = seen_failed
    assert "disk full" in failed["detail"]
    assert out["targets"][0]["state"] == TargetState.skipped.value
    assert calls == ["LINEAR_TOKEN", "LINEAR_TEAM", "LINEAR_TOKEN"]
    from hermes_cli.config import load_config
    assert load_config().get("mcp_servers", {}).get("linear") == {"command": "old-linear"}
    env_text = (home / ".env").read_text()
    assert "LINEAR_TOKEN=old-token" in env_text and "LINEAR_TEAM" not in env_text


def test_probe_commit_on_commit_failure_rolls_back(tmp_path, monkeypatch):
    """probe_with_rollback undid only AttemptCanceled: a failing on_commit left the new config
    committed, the previous provider entry evicted and the pre-attempt tokens deleted."""
    from tools.connectors import mcp_oauth

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("mcp_servers:\n  linear:\n    command: old-linear\n")
    token_dir = home / "mcp-tokens"
    token_dir.mkdir()
    (token_dir / "linear.json").write_text('{"grant": "old"}')
    monkeypatch.setenv("HERMES_HOME", str(home))

    with patch("hermes_cli.mcp_config._probe_single_server", return_value=[("read", "desc")]), \
         patch("hermes_cli.mcp_config._oauth_tokens_present", return_value=True), \
         patch("hermes_cli.config._publish_env_value", lambda *_a, **_k: None):
        with pytest.raises(RuntimeError, match="env broke"):
            mcp_oauth.probe_with_rollback(
                "linear", {"command": "new-linear"}, str(home), None, False,
                on_commit=lambda: (_ for _ in ()).throw(RuntimeError("env broke")))

    from hermes_cli.config import load_config
    assert load_config().get("mcp_servers", {}).get("linear") == {"command": "old-linear"}
    assert (token_dir / "linear.json").read_text() == '{"grant": "old"}'
