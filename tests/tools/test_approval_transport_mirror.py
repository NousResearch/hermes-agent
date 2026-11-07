"""A phone transport and the built-in prompt share one bounded approval decision."""

import asyncio
import queue
import threading

import pytest

from hermes_cli import config
import hermes_cli.approval_transport as transport
from agent.memory_provider import spawn_context_thread
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from tools import approval, approval_context, approval_gateway_wait, approval_mirror, approval_prompt
from gateway.session_policy import LocalSessionPolicy, _ACTIVE_POLICY


SESSION = "mirror-fixture"
DATA = {"command": "rm -rf build", "description": "recursive delete",
        "pattern_key": "rm", "pattern_keys": ["rm"],
        "allow_session": True, "allow_permanent": True, "transport_eligible": True}


@pytest.fixture
def mirror(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    token = set_hermes_home_override(home)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch-profile"))
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="fixture", version="1.0.0", description="fixture",
                                     source="user", key="fixture"), manager)
    monkeypatch.setattr(approval_prompt, "get_plugin_manager", lambda: manager)
    hooks, invocations, threads, releases = [], [], [], []
    finished = threading.Event()
    real_invoke = transport.invoke_approval_transport

    def invoke(*args, **kwargs):
        invocations.append(kwargs)
        try:
            return real_invoke(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(transport, "invoke_approval_transport", invoke)
    monkeypatch.setattr(approval_context, "_fire_approval_hook", lambda name, **kw: hooks.append((name, kw)))

    def configure(mode="mirror", timeout=2):
        mode_line = "" if mode is None else f"    transport_mode: {mode}\n"
        (home / "config.yaml").write_text(
            f"approvals:\n  mode: manual\n  timeout: {timeout}\n"
            "security:\n  approval:\n    transport: phone\n" + mode_line)
        config._LOAD_CONFIG_CACHE.clear()

    def start(notify, data=None, guard=False, source="tui", gate=None, producer=None):
        outcome = {}

        def run():
            try:
                _ACTIVE_POLICY.set(LocalSessionPolicy(
                    source=source, platform=source, cwd=str(home), model=None, toolsets=(),
                    config_json="{}", request_json="{}", terminal_json="{}") if source else None)
                approval_context.set_current_session_key(SESSION)
                approval.register_gateway_notify(SESSION, notify)
                if producer:
                    outcome["value"] = producer()
                elif guard or gate:
                    outcome["value"] = approval._human_decision(
                        gate or approval._COMMAND_GATE, command=DATA["command"], description=DATA["description"],
                        pattern_key="rm", pattern_keys=["rm"], session_key=SESSION,
                        approval_callback=None, is_cli=False, is_gateway=True, is_ask=False)
                else:
                    outcome["value"] = approval_gateway_wait._await_gateway_decision(SESSION, notify, data or DATA)
            except Exception as exc:
                outcome["error"] = exc

        thread = spawn_context_thread(run, name="mirror-test-wait")
        threads.append(thread)
        thread.start()
        return thread, outcome

    configure()
    try:
        yield type("Mirror", (), dict(home=home, register=staticmethod(
            lambda present: ctx.register_approval_transport("phone", present)), configure=staticmethod(configure),
            start=staticmethod(start), hooks=hooks, invocations=invocations, finished=finished, releases=releases))()
    finally:
        for release in releases:
            release.set()
        approval.unregister_gateway_notify(SESSION)
        for thread in threads:
            thread.join(3)
        reset_hermes_home_override(token)
        config._LOAD_CONFIG_CACHE.clear()


def _done(thread, outcome):
    thread.join(3)
    assert not thread.is_alive()
    assert "error" not in outcome, outcome.get("error")
    return outcome["value"]


@pytest.mark.parametrize("async_present", [False, True])
def test_transport_once_shares_request_and_settles_builtin_once(mirror, async_present):
    seen, requests, settled = [], [], []

    def notify(data):
        seen.append(data)
        assert approval.register_gateway_settle(SESSION, data["request_id"], settled.append)

    def present(request):
        requests.append(request)
        assert seen
        assert get_hermes_home() == mirror.home
        return request.respond("once")

    async def present_async(request):
        return present(request)

    mirror.register(present_async if async_present else present)
    thread, outcome = mirror.start(notify)
    assert _done(thread, outcome) == {"resolved": True, "choice": "once", "reason": None}
    assert requests[0].request_id == seen[0]["request_id"]
    assert settled == ["resolved"]
    assert [name for name, _ in mirror.hooks] == ["pre_approval_request", "post_approval_response"]
    assert approval.resolve_gateway_approval(SESSION, "deny", request_id=requests[0].request_id) == 0


@pytest.mark.parametrize("captured", [2, 0])
def test_transport_waits_the_timeout_captured_for_the_request(mirror, captured):
    mirror.configure(timeout=captured)

    def notify(data):
        mirror.configure(timeout=7)  # edited while the prompt is being published

    mirror.register(lambda request: request.respond("deny"))
    thread, outcome = mirror.start(notify)
    _done(thread, outcome)
    assert mirror.finished.wait(3)
    assert mirror.invocations[0]["timeout_seconds"] == captured


def test_builtin_first_interrupts_transport_and_discards_late_answer(mirror):
    presented, returned, release = threading.Event(), threading.Event(), threading.Event()
    mirror.releases.append(release)
    requests, seen = [], []

    def present(request):
        requests.append(request)
        presented.set()
        release.wait(3)
        returned.set()
        return request.respond("always")

    mirror.register(present)
    thread, outcome = mirror.start(seen.append)
    assert presented.wait(2)
    rid = requests[0].request_id
    assert approval.resolve_gateway_approval(SESSION, "deny", request_id=rid) == 1
    assert _done(thread, outcome)["choice"] == "deny"
    assert mirror.invocations[0]["is_interrupted"]()
    release.set()
    assert returned.wait(2)
    assert mirror.finished.wait(2)
    assert approval.resolve_gateway_approval(SESSION, "always", request_id=rid) == 0
    assert approval.list_gateway_approvals(SESSION) == []
    assert [name for name, _ in mirror.hooks] == ["pre_approval_request", "post_approval_response"]


@pytest.mark.parametrize("failure", ["error", "timeout", "busy"])
def test_transport_failure_leaves_builtin_live(mirror, monkeypatch, failure):
    release = threading.Event()
    mirror.releases.append(release)
    seen, completed = [], threading.Event()
    real_invoke = transport.invoke_approval_transport

    def invoke(*args, **kwargs):
        if failure == "timeout":
            kwargs["timeout_seconds"] = 0.01
        result = real_invoke(*args, **kwargs)
        assert result.failure == failure
        completed.set()
        return result

    def present(request):
        if failure == "error":
            raise RuntimeError("offline")
        release.wait(3)
        return request.respond("deny")

    if failure == "busy":
        monkeypatch.setattr(approval_mirror, "_mirror_worker_slots", threading.BoundedSemaphore(0))
    monkeypatch.setattr(transport, "invoke_approval_transport", invoke)
    mirror.register(present)
    thread, outcome = mirror.start(seen.append)
    assert completed.wait(2)
    assert approval.list_gateway_approvals(SESSION)
    assert approval.resolve_gateway_approval(SESSION, "once", request_id=seen[0]["request_id"]) == 1
    assert _done(thread, outcome)["choice"] == "once"


@pytest.mark.parametrize("choice", ["session", "always"])
def test_smart_denied_transport_cannot_widen_scope(mirror, choice):
    requests, seen = [], []

    def present(request):
        requests.append(request)
        return request.respond(choice)

    mirror.register(present)
    thread, outcome = mirror.start(seen.append, {**DATA, "smart_denied": True,
                                               "allow_session": False, "allow_permanent": False})
    assert mirror.finished.wait(2)
    assert requests[0].allowed_choices == ("once", "deny")
    assert approval.list_gateway_approvals(SESSION)
    assert approval.resolve_gateway_approval(SESSION, "once", request_id=seen[0]["request_id"]) == 1
    assert _done(thread, outcome)["choice"] == "once"


@pytest.mark.parametrize("mode", ["replace", None, "MIRROR", "invalid"])
def test_default_or_invalid_mode_keeps_transport_replacement(mirror, mode):
    mirror.configure(mode)
    seen, requests = [], []
    mirror.register(lambda request: requests.append(request) or request.respond("once"))
    thread, outcome = mirror.start(seen.append, guard=True)
    assert _done(thread, outcome)["approved"] is True
    assert len(requests) == 1
    assert seen == []


def test_mirror_silence_keeps_existing_blocked_timeout_result(mirror):
    mirror.configure(timeout=0)
    seen = []
    mirror.register(lambda request: request.respond("once"))
    thread, outcome = mirror.start(seen.append, guard=True)
    result = _done(thread, outcome)
    assert seen
    assert result["approved"] is False
    assert result["outcome"] == "timeout"
    assert "BLOCKED" in result["message"]
    assert "timed out without user response" in result["message"]


def test_classic_cli_mirror_uses_builtin_only(mirror):
    requests, seen = [], []
    mirror.register(lambda request: requests.append(request) or request.respond("deny"))
    result = approval._human_decision(
        approval._COMMAND_GATE, command=DATA["command"], description=DATA["description"],
        pattern_key="rm", pattern_keys=["rm"], session_key=SESSION,
        approval_callback=lambda *args, **kw: seen.append(args) or "once",
        is_cli=True, is_gateway=False, is_ask=False)
    assert result["approved"] is True
    assert len(seen) == 1
    assert requests == []


def _observe_mirror(monkeypatch):
    """Observe the producer-thread admission before checking that no worker was spawned."""
    considered, workers = threading.Event(), []
    real_start, real_spawn = approval_mirror.start_mirror, approval_mirror.spawn_context_thread

    def start(*args):
        try:
            return real_start(*args)
        finally:
            considered.set()

    def spawn(*args, **kwargs):
        worker = real_spawn(*args, **kwargs)
        workers.append(worker)
        return worker

    monkeypatch.setattr(approval_mirror, "start_mirror", start)
    monkeypatch.setattr(approval_mirror, "spawn_context_thread", spawn)
    return considered, workers


@pytest.mark.parametrize("source", [None, "cli", "telegram", "discord", "webhook", "api_server", "cron", "kanban"])
def test_nonlocal_session_keeps_builtin_only(mirror, monkeypatch, source):
    considered, workers = _observe_mirror(monkeypatch)
    requests, seen = [], []
    mirror.register(lambda req: requests.append(req) or req.respond("once"))
    # A cannot be answered remotely, so it cannot leave a remotely-settled stale card
    # whose session-oldest button would consume B.
    for command in ("request A", "request B"):
        considered.clear()
        thread, outcome = mirror.start(seen.append, {**DATA, "command": command}, source=source)
        assert considered.wait(2)
        assert workers == []
        assert requests == []
        assert approval.list_gateway_approvals(SESSION)[0]["request_id"] == seen[-1]["request_id"]
        assert approval.resolve_gateway_approval(SESSION, "once") == 1
        assert _done(thread, outcome)["choice"] == "once"


@pytest.mark.parametrize("source", ["tui", "gui", "acp"])
@pytest.mark.parametrize("gate", [approval._COMMAND_GATE, approval._EXECUTE_CODE_GATE])
def test_local_transport_eligible_producers_mirror(mirror, source, gate):
    requests, seen = [], []
    mirror.register(lambda req: requests.append(req) or req.respond("once"))
    thread, outcome = mirror.start(seen.append, source=source, gate=gate)
    assert _done(thread, outcome)["approved"] is True
    assert requests[0].request_id == seen[0]["request_id"]


@pytest.mark.parametrize("producer", ["action", "protected_write", "elicitation", "editor"])
def test_nontransport_producers_keep_builtin_only(mirror, monkeypatch, producer):
    from acp_adapter.edit_approval import EditProposal
    from gateway.session_local_editor import request_editor_edit
    from tools.file_tools_write_guards import _request_protected_instruction_approval

    considered, workers = _observe_mirror(monkeypatch)
    monkeypatch.setattr(approval_context, "_is_gateway_approval_context", lambda: True)
    calls = {
        "protected_write": lambda: _request_protected_instruction_approval(["AGENTS.md"]),
        "elicitation": lambda: approval_prompt.request_elicitation_consent("Continue?", "MCP consent"),
        "editor": lambda: request_editor_edit(EditProposal("write_file", "file.txt", None, "text", {})),
    }
    mirror.register(lambda req: req.respond("once"))
    seen = []
    thread, outcome = mirror.start(seen.append, source="acp", gate=approval._ACTION_GATE,
                                   producer=calls.get(producer))
    assert considered.wait(2)
    assert workers == []
    assert approval.resolve_gateway_approval(SESSION, "deny", request_id=seen[0]["request_id"]) == 1
    result = _done(thread, outcome)
    if producer == "action":
        assert result["approved"] is False
    elif producer == "protected_write":
        assert "BLOCKED" in result
    else:
        assert result in (False, "decline")


def test_mirror_capacity_cannot_starve_replace(mirror):
    presented, release = queue.Queue(), threading.Event()
    mirror.releases.append(release)

    def present(request):
        presented.put(request)
        release.wait(10)
        return request.respond("once")

    mirror.register(present)
    for _ in range(transport._MAX_ACTIVE_TRANSPORT_WORKERS):
        thread, outcome = mirror.start(lambda data: None)
        request = presented.get(timeout=2)
        assert approval.resolve_gateway_approval(SESSION, "once", request_id=request.request_id) == 1
        assert _done(thread, outcome)["choice"] == "once"
    # All plugin callbacks still occupy their mirror slots after the built-in answers.
    result = transport.invoke_approval_transport(lambda req: req.respond("once"), request, timeout_seconds=1)
    assert result.failure is None
    assert result.choice == "once"


def test_sync_transport_can_run_its_own_event_loop(mirror):
    async def answer(request):
        assert get_hermes_home() == mirror.home
        return request.respond("once")

    mirror.register(lambda request: asyncio.run(answer(request)))
    thread, outcome = mirror.start(lambda data: None)
    assert _done(thread, outcome)["choice"] == "once"


def test_missing_scope_flags_offer_once_or_deny(mirror):
    requests, seen = [], []
    mirror.register(lambda req: requests.append(req) or req.respond("always"))
    data = {key: value for key, value in DATA.items() if key not in ("allow_session", "allow_permanent")}
    thread, outcome = mirror.start(seen.append, data)
    assert mirror.finished.wait(2)
    assert requests[0].allowed_choices == ("once", "deny")
    assert approval.resolve_gateway_approval(SESSION, "deny", request_id=seen[0]["request_id"]) == 1
    assert _done(thread, outcome)["choice"] == "deny"
