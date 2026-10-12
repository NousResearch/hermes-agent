"""TUI parity tests for the session-scoped ``/sessionflag`` command."""

from __future__ import annotations

import contextlib
import importlib
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import hermes_yaml as yaml

from hermes_state import SessionDB


POLICY_KEY = "test_session_override"


def _override(row):
    return SessionDB.session_policy_override(row, POLICY_KEY)


def _prepare_flag(raw_args):
    from hermes_cli.session_commands import SessionCommandRequest

    parts = raw_args.strip().lower().split()
    if len(parts) != 1 or parts[0] not in {"on", "off", "inherit", "status"}:
        raise ValueError("Usage: /sessionflag [on|off|inherit|status]")
    return SessionCommandRequest(parts[0] != "status", parts[0])


def _flag_command(request, session):
    from utils import is_truthy_value

    config = session.get_config()
    global_value = is_truthy_value((config.get("test_policy") or {}).get("enabled"), default=False)
    supported = True
    if request.write:
        supported = session.set_override(POLICY_KEY, {"on": True, "off": False, "inherit": None}[request.data])
    override = session.get_override(POLICY_KEY)
    enabled = global_value if override is None else override
    value = "ON" if enabled else "OFF"
    source = "global (inherited)" if override is None else "session"
    text = f"Session flag: {value}\nSource: {source}\nGlobal: {'ON' if global_value else 'OFF'}"
    if request.write:
        text = f"Session flag override saved: {value}\n{text}"
        if session.live and not supported:
            text += "\nCurrent context engine does not support switching session flag"
    return text


def _register_flag(manager):
    from hermes_cli.plugins import PluginContext, PluginManifest

    context = PluginContext(PluginManifest(name="sessionflag-probe", key="sessionflag-probe"), manager)
    context.register_session_policy(POLICY_KEY, config_path="test_policy.enabled",
                                   apply=lambda runtime, value: runtime.set_context_option("test_flag_enabled", value))
    context.register_session_command("sessionflag", _flag_command, prepare=_prepare_flag,
                                     args_hint="[on|off|inherit|status]",
                                     subcommands=("on", "off", "inherit", "status"),
                                     advertise_in_gateway_menu=False)


class RecordingEngine:
    def __init__(self, *, supported: bool = True) -> None:
        self.supported = supported
        self.enabled = None
        self.calls: list[bool] = []

    def set_session_option(self, name: str, enabled: bool) -> bool:
        assert name == "test_flag_enabled"
        if not self.supported:
            return False
        self.enabled = enabled
        self.calls.append(enabled)
        return True


class LiveAgent:
    def __init__(self, db: SessionDB, session_id: str, *, supported: bool = True) -> None:
        self.session_id = session_id
        self.model = "test-model"
        self.provider = "test-provider"
        self.context_compressor = RecordingEngine(supported=supported)
        self._session_init_model_config: dict = {}
        self._session_policy_overrides: dict = {}
        self._db = self._session_db = db
        self.ensure_calls = 0
        self.run_conversation_calls = 0

    def _ensure_db_session(self) -> None:
        self.ensure_calls += 1
        if self._db.get_session(self.session_id) is None:
            self._db.create_session(self.session_id, source="tui", model=self.model)

    def run_conversation(self, *_args, **_kwargs):
        self.run_conversation_calls += 1


class RecordingLease:
    def __init__(self, lease_id: str = "lease") -> None:
        self.lease_id = lease_id
        self.release_calls = 0

    def release(self) -> None:
        self.release_calls += 1


class RecordingWorker:
    instances: list[RecordingWorker] = []

    def __init__(self, *_args, **_kwargs) -> None:
        self.commands: list[str] = []
        self.closed = False
        self.__class__.instances.append(self)

    def run(self, command: str) -> str:
        self.commands.append(command)
        return "worker handled /sessionflag"

    def close(self) -> None:
        self.closed = True


class _PluginSessionControlAbort(BaseException):
    """Non-Exception failure used to prove lease cleanup covers BaseException."""


@pytest.fixture()
def gateway(tmp_path, monkeypatch):
    home = tmp_path / "launch-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import plugins

    manager = plugins.PluginManager()
    manager._discovered = True
    _register_flag(manager)
    monkeypatch.setattr(plugins, "_ensure_plugins_discovered", lambda: manager)
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"test_policy": {"enabled": False}}),
        encoding="utf-8",
    )

    server = importlib.import_module("tui_gateway.server")
    db = SessionDB(db_path=home / "state.db")
    old_methods = dict(server._methods)
    old_db = server._db
    old_db_error = server._db_error
    server._methods.clear()
    server._methods.update(old_methods)
    server._sessions.clear()
    server._db = db
    server._db_error = None
    RecordingWorker.instances.clear()
    try:
        yield server, db, home
    finally:
        for sid in list(server._sessions):
            session = server._sessions.pop(sid)
            worker = session.get("slash_worker")
            if worker is not None:
                worker.close()
        server._methods.clear()
        server._methods.update(old_methods)
        server._sessions.clear()
        server._db = old_db
        server._db_error = old_db_error
        db.close()


def _session(
    server,
    db,
    sid: str,
    key: str,
    agent,
    *,
    profile_home: Path | None = None,
    row_db: SessionDB | None = None,
):
    (row_db or db).create_session(key, source="tui", model="test-model")
    server._sessions[sid] = {
        "session_key": key,
        "agent": agent,
        "history": [],
        "history_lock": threading.Lock(),
        "running": False,
        "profile_home": str(profile_home) if profile_home else None,
        "slash_worker": None,
        "model": "test-model",
    }
    return server._sessions[sid]


def _call(server, method: str, **params) -> dict:
    return server._methods[method]("rid", params)


def _flag_call(server, method, sid, arg):
    params = {"session_id": sid}
    if method == "slash.exec":
        params["command"] = f"sessionflag {arg}"
    else:
        params.update(name="sessionflag", arg=arg)
    return _call(server, method, **params)


def _result(response: dict) -> dict:
    assert "result" in response, response
    return response["result"]


def test_catalog_exposes_plugin_session_subcommands_and_desktop_metadata(gateway):
    server, _db, _home = gateway
    result = _result(_call(server, "commands.catalog"))
    assert ["/sessionflag", "Plugin command"] in result["pairs"]
    assert result["sub"]["/sessionflag"] == ["on", "off", "inherit", "status"]
    assert result["commands"]["/sessionflag"] == {"argument_mode": "text", "desktop": None}


@pytest.mark.parametrize("method", ["slash.exec", "command.dispatch"])
def test_live_sessionflag_on_off_inherit_status_updates_exact_row_and_engine(gateway, method):
    server, db, _home = gateway
    agent = LiveAgent(db, "live-key")
    session = _session(server, db, "sid-live", "live-key", agent)

    on = _result(_flag_call(server, method, "sid-live", "on"))
    assert "override saved: ON" in on["output"]
    assert _override(db.get_session("live-key")) is True
    assert agent._session_policy_overrides[POLICY_KEY] is True
    assert agent.context_compressor.enabled is True
    assert agent.ensure_calls == 1

    status = _result(_flag_call(server, method, "sid-live", "status"))
    assert "Session flag: ON" in status["output"]
    assert "Source: session" in status["output"]
    assert agent.ensure_calls == 1

    off = _result(_flag_call(server, method, "sid-live", "off"))
    assert "override saved: OFF" in off["output"]
    assert _override(db.get_session("live-key")) is False
    assert agent._session_policy_overrides[POLICY_KEY] is False
    assert agent.context_compressor.enabled is False

    inherit = _result(_flag_call(server, method, "sid-live", "inherit"))
    assert "override saved: OFF" in inherit["output"]
    assert _override(db.get_session("live-key")) is None
    assert agent._session_policy_overrides.get(POLICY_KEY) is None
    assert agent.run_conversation_calls == 0
    assert session["history"] == []


def test_live_status_is_read_only_and_does_not_ensure_or_set_engine(gateway):
    server, db, _home = gateway
    key = "status-key"
    db.create_session(key, source="tui", model="test-model")
    agent = LiveAgent(db, key)
    _session(server, db, "sid-status", key, agent)
    calls_before = list(agent.context_compressor.calls)

    result = _result(_call(server, "slash.exec", session_id="sid-status", command="sessionflag status"))

    assert "Session flag: OFF" in result["output"]
    assert "Source: global (inherited)" in result["output"]
    assert agent.ensure_calls == 0
    assert agent.context_compressor.calls == calls_before
    assert _override(db.get_session(key)) is None


def test_live_invalid_sessionflag_command_has_no_mutation(gateway):
    server, db, _home = gateway
    agent = LiveAgent(db, "invalid-key")
    _session(server, db, "sid-invalid", "invalid-key", agent)

    response = _call(server, "slash.exec", session_id="sid-invalid", command="sessionflag maybe")

    assert "result" in response
    assert "on|off|inherit|status" in _result(response)["output"]
    assert agent.ensure_calls == 0
    assert agent.context_compressor.calls == []
    assert _override(db.get_session("invalid-key")) is None


def test_live_plugin_engine_saves_and_reports_unsupported(gateway):
    server, db, _home = gateway
    agent = LiveAgent(db, "plugin-key", supported=False)
    _session(server, db, "sid-plugin", "plugin-key", agent)

    result = _result(_call(server, "slash.exec", session_id="sid-plugin", command="sessionflag on"))

    assert "saved: ON" in result["output"]
    assert "does not support switching session flag" in result["output"]
    assert _override(db.get_session("plugin-key")) is True
    assert agent._session_policy_overrides[POLICY_KEY] is True
    assert agent.context_compressor.enabled is None


def test_two_live_sessions_are_isolated(gateway):
    server, db, _home = gateway
    agent_a = LiveAgent(db, "session-a")
    agent_b = LiveAgent(db, "session-b")
    _session(server, db, "sid-a", "session-a", agent_a)
    _session(server, db, "sid-b", "session-b", agent_b)

    _result(_call(server, "slash.exec", session_id="sid-a", command="sessionflag on"))
    _result(_call(server, "slash.exec", session_id="sid-b", command="sessionflag off"))

    assert _override(db.get_session("session-a")) is True
    assert _override(db.get_session("session-b")) is False
    assert agent_a.context_compressor.enabled is True
    assert agent_b.context_compressor.enabled is False
    assert agent_a.context_compressor.calls == [True]
    assert agent_b.context_compressor.calls == [False]


def test_busy_sessionflag_rejects_before_db_or_live_mutation(gateway):
    server, db, _home = gateway
    agent = LiveAgent(db, "busy-key")
    session = _session(server, db, "sid-busy", "busy-key", agent)
    session["running"] = True
    before = db.get_session("busy-key")

    response = _call(server, "slash.exec", session_id="sid-busy", command="sessionflag on")

    assert "error" in response
    assert response["error"]["code"] == 4009
    assert "busy" in response["error"]["message"].lower()
    assert agent.ensure_calls == 0
    assert agent.context_compressor.calls == []
    assert db.get_session("busy-key") == before


def test_no_live_agent_stays_in_backend_and_does_not_touch_other_live_agent(gateway, monkeypatch):
    server, db, _home = gateway
    other = LiveAgent(db, "other-key")
    _session(server, db, "sid-other", "other-key", other)
    cold = _session(server, db, "sid-cold", "cold-key", None)
    monkeypatch.setattr(server, "_SlashWorker", RecordingWorker)
    result = _result(_call(server, "slash.exec", session_id="sid-cold", command="/sessionflag on"))
    assert "override saved: ON" in result["output"]
    assert _override(db.get_session("cold-key")) is True
    assert cold["slash_worker"] is None
    assert not RecordingWorker.instances
    assert other.ensure_calls == 0
    assert other.context_compressor.calls == []
    assert _override(db.get_session("other-key")) is None


def test_profile_session_uses_profile_db_and_readonly_config(gateway, tmp_path):
    server, launch_db, launch_home = gateway
    profile_home = tmp_path / "profile-a"
    profile_home.mkdir()
    (profile_home / "config.yaml").write_text(
        yaml.safe_dump({"test_policy": {"enabled": True}}),
        encoding="utf-8",
    )
    profile_db = SessionDB(db_path=profile_home / "state.db")
    try:
        key = "profile-key"
        profile_db.create_session(key, source="tui", model="test-model")
        agent = LiveAgent(profile_db, key)
        _session(
            server,
            launch_db,
            "sid-profile",
            key,
            agent,
            profile_home=profile_home,
            row_db=profile_db,
        )

        result = _result(
            _call(server, "slash.exec", session_id="sid-profile", command="sessionflag status")
        )

        assert "Global: ON" in result["output"]
        assert "Source: global (inherited)" in result["output"]
        assert launch_db.get_session(key) is None
        assert _override(profile_db.get_session(key)) is None
        assert launch_home.joinpath("config.yaml").read_text(encoding="utf-8")
    finally:
        profile_db.close()


def test_backend_cold_write_hydrates_later_live_agent(gateway, monkeypatch):
    server, db, _home = gateway
    _session(server, db, "sid-cold-hydrate", "cold-hydrate", None)
    monkeypatch.setattr(server, "_SlashWorker", RecordingWorker)
    result = _result(_call(server, "slash.exec", session_id="sid-cold-hydrate", command="sessionflag on"))
    assert "override saved: ON" in result["output"]
    assert not RecordingWorker.instances
    agent = LiveAgent(db, "cold-hydrate")
    from agent.session_policies import hydrate_session_policies

    hydrate_session_policies(agent, db, "cold-hydrate")
    assert agent.context_compressor.enabled is True
    assert agent._session_policy_overrides.get(POLICY_KEY) is True


def test_slash_exec_live_sessionflag_never_calls_worker(gateway, monkeypatch):
    server, db, _home = gateway
    agent = LiveAgent(db, "direct-key")
    _session(server, db, "sid-direct", "direct-key", agent)

    def fail_worker(*_args, **_kwargs):
        pytest.fail("live /sessionflag must not start a slash worker")

    monkeypatch.setattr(server, "_SlashWorker", fail_worker)
    _result(_call(server, "slash.exec", session_id="sid-direct", command="sessionflag on"))
    assert agent.context_compressor.enabled is True


def test_live_sessionflag_rejects_mismatched_session_identity_without_mutation(gateway, monkeypatch):
    server, db, _home = gateway
    session_key = "session-row"
    agent_key = "agent-row"
    db.create_session(session_key, source="tui", model="test-model")
    db.create_session(agent_key, source="tui", model="test-model")
    agent = LiveAgent(db, agent_key)
    session = _session(server, db, "sid-mismatch", session_key, agent)
    session_row_before = db.get_session(session_key)
    agent_row_before = db.get_session(agent_key)
    engine_enabled_before = agent.context_compressor.enabled

    def fail_worker(*_args, **_kwargs):
        pytest.fail("mismatched live /sessionflag must not start a slash worker")

    monkeypatch.setattr(server, "_SlashWorker", fail_worker)
    response = _call(server, "slash.exec", session_id="sid-mismatch", command="/sessionflag on")

    output = _result(response)["output"]
    assert "session identity mismatch" in output.lower()
    assert db.get_session(session_key) == session_row_before
    assert db.get_session(agent_key) == agent_row_before
    assert _override(db.get_session(session_key)) is None
    assert _override(db.get_session(agent_key)) is None
    assert agent.ensure_calls == 0
    assert agent.context_compressor.enabled is engine_enabled_before
    assert agent.context_compressor.calls == []
    assert session["slash_worker"] is None


def _assert_history_lock_available(lock: threading.Lock) -> None:
    acquired = threading.Event()

    def acquire_and_release() -> None:
        with lock:
            acquired.set()

    thread = threading.Thread(target=acquire_and_release)
    thread.start()
    assert acquired.wait(2), "history_lock was held during blocking live work"
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_live_sessionflag_config_and_db_work_run_without_history_lock(gateway, monkeypatch):
    server, db, _home = gateway
    agent = LiveAgent(db, "outside-lock")
    session = _session(server, db, "sid-outside-lock", "outside-lock", agent)
    config_entered = threading.Event()
    config_release = threading.Event()
    setter_entered = threading.Event()
    setter_release = threading.Event()
    apply_entered = threading.Event()
    apply_release = threading.Event()
    original_setter = db.set_session_policy_override
    from agent import session_policies

    original_apply = session_policies.apply_session_policy

    def blocked_config():
        config_entered.set()
        assert config_release.wait(2)
        return {"test_policy": {"enabled": False}}

    def blocked_setter(session_id, policy_key, override):
        setter_entered.set()
        assert setter_release.wait(2)
        return original_setter(session_id, policy_key, override)

    def blocked_apply(*args, **kwargs):
        apply_entered.set()
        assert apply_release.wait(2)
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(server, "_session_plugin_config_loader", lambda _session: blocked_config)
    monkeypatch.setattr(db, "set_session_policy_override", blocked_setter)
    monkeypatch.setattr(session_policies, "apply_session_policy", blocked_apply)
    result = {}

    def run_sessionflag() -> None:
        result["response"] = _call(server, "slash.exec", session_id="sid-outside-lock", command="sessionflag on")

    thread = threading.Thread(target=run_sessionflag)
    thread.start()
    try:
        assert config_entered.wait(2)
        _assert_history_lock_available(session["history_lock"])
        config_release.set()
        assert setter_entered.wait(2)
        _assert_history_lock_available(session["history_lock"])
        setter_release.set()
        assert apply_entered.wait(2)
        _assert_history_lock_available(session["history_lock"])
    finally:
        config_release.set()
        setter_release.set()
        apply_release.set()
        thread.join(timeout=2)
    assert not thread.is_alive()
    assert "result" in result["response"]


@pytest.mark.parametrize("method", ["slash.exec", "command.dispatch"])
def test_live_sessionflag_lease_blocks_prompt_and_second_sessionflag_until_release(gateway, monkeypatch, method):
    server, db, _home = gateway
    agent = LiveAgent(db, "lease-busy")
    session = _session(server, db, "sid-lease-busy", "lease-busy", agent)
    setter_entered = threading.Event()
    setter_release = threading.Event()
    original_setter = db.set_session_policy_override

    def blocked_setter(session_id, policy_key, override):
        setter_entered.set()
        assert setter_release.wait(2)
        return original_setter(session_id, policy_key, override)

    monkeypatch.setattr(db, "set_session_policy_override", blocked_setter)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_args: None)
    first = {}
    second = {}
    prompt = {}
    second_done = threading.Event()
    prompt_done = threading.Event()

    def run_first() -> None:
        first["response"] = _flag_call(server, method, "sid-lease-busy", "on")

    def run_second() -> None:
        second["response"] = _call(server, "slash.exec", session_id="sid-lease-busy", command="sessionflag off")
        second_done.set()

    def run_prompt() -> None:
        prompt["response"] = _call(server, "prompt.submit", session_id="sid-lease-busy", text="must wait")
        prompt_done.set()

    first_thread = threading.Thread(target=run_first)
    second_thread = threading.Thread(target=run_second)
    prompt_thread = threading.Thread(target=run_prompt)
    first_thread.start()
    try:
        assert setter_entered.wait(2)
        second_thread.start()
        prompt_thread.start()
        assert second_done.wait(2), "second /sessionflag did not get a stable busy response"
        assert prompt_done.wait(2), "prompt.submit did not get a stable busy response"
    finally:
        setter_release.set()
        first_thread.join(timeout=2)
        second_thread.join(timeout=2)
        prompt_thread.join(timeout=2)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert not prompt_thread.is_alive()
    assert second["response"]["error"]["code"] == 4009
    assert prompt["response"]["error"]["code"] == 4009
    assert session["running"] is False
    assert session["history"] == []
    assert agent.run_conversation_calls == 0


def test_prompt_control_race_releases_only_new_active_session_lease(gateway, monkeypatch):
    server, db, _home = gateway
    agent = LiveAgent(db, "prompt-control-race")
    session = _session(server, db, "sid-prompt-control-race", "prompt-control-race", agent)

    lease = RecordingLease("prompt-control-race")
    claim_entered = threading.Event()
    claim_release = threading.Event()

    def blocked_claim(*_args, **_kwargs):
        claim_entered.set()
        assert claim_release.wait(5), "active-session claim was not released"
        return lease, None

    monkeypatch.setattr(server, "_claim_active_session_slot", blocked_claim)
    prompt_result: dict = {}

    def run_prompt() -> None:
        prompt_result["response"] = _call(server, "prompt.submit", session_id="sid-prompt-control-race", text="must not run")

    prompt_thread = threading.Thread(target=run_prompt)
    prompt_thread.start()
    assert claim_entered.wait(2), "prompt did not reach active-session claim"

    sessionflag_thread, sessionflag_release, sessionflag_result = _start_blocked_sessionflag(
        server,
        db,
        monkeypatch,
        sid="sid-prompt-control-race",
    )
    try:
        claim_release.set()
        prompt_thread.join(timeout=5)
        assert not prompt_thread.is_alive()
        response = prompt_result["response"]
        assert response["error"]["code"] == 4009
        assert session["running"] is False
        assert session["history"] == []
        assert agent.run_conversation_calls == 0
        assert session.get("active_session_lease") is None
        assert lease.release_calls == 1
    finally:
        _join_blocked_sessionflag(sessionflag_thread, sessionflag_release)
    assert "result" in sessionflag_result["response"]


def test_prompt_claim_releases_lease_when_session_is_closed_while_claiming(gateway, monkeypatch):
    server, db, _home = gateway
    agent = LiveAgent(db, "prompt-close-claim-race")
    sid = "sid-prompt-close-claim-race"
    session = _session(server, db, sid, "prompt-close-claim-race", agent)
    lease = RecordingLease("prompt-close-claim-race")
    claim_entered = threading.Event()
    claim_release = threading.Event()
    claim_result: dict = {}

    def blocked_claim(*_args, **_kwargs):
        claim_entered.set()
        assert claim_release.wait(5), "active-session claim was not released"
        return lease, None

    monkeypatch.setattr(server, "_claim_active_session_slot", blocked_claim)

    def run_claim() -> None:
        claim_result["value"] = server._claim_active_session_slot_for_prompt(sid, session)

    claim_thread = threading.Thread(target=run_claim)
    claim_thread.start()
    assert claim_entered.wait(2), "prompt did not reach active-session claim"

    popped = server._pop_session_by_id(sid)
    assert popped is session
    assert server._teardown_popped_session(popped) is True

    claim_release.set()
    claim_thread.join(timeout=5)
    assert not claim_thread.is_alive()
    assert claim_result["value"] == (None, None)
    assert sid not in server._sessions
    assert session.get("_closing") is True
    assert session.get("_finalized") is True
    assert session.get("active_session_lease") is None
    assert lease.release_calls == 1


def test_prompt_control_race_preserves_preexisting_active_session_lease(gateway, monkeypatch):
    server, db, _home = gateway
    agent = LiveAgent(db, "prompt-control-preexisting")
    session = _session(
        server,
        db,
        "sid-prompt-control-preexisting",
        "prompt-control-preexisting",
        agent,
    )
    lease = RecordingLease("preexisting")
    session["active_session_lease"] = lease
    config_entered = threading.Event()
    config_release = threading.Event()

    def blocked_config():
        config_entered.set()
        assert config_release.wait(5), "prompt preflight was not released"
        return {}

    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", blocked_config)
    prompt_result: dict = {}

    def run_prompt() -> None:
        prompt_result["response"] = _call(server, "prompt.submit", session_id="sid-prompt-control-preexisting", text="must not run")

    prompt_thread = threading.Thread(target=run_prompt)
    prompt_thread.start()
    assert config_entered.wait(2), "prompt did not reach the preflight barrier"
    sessionflag_thread, sessionflag_release, sessionflag_result = _start_blocked_sessionflag(
        server,
        db,
        monkeypatch,
        sid="sid-prompt-control-preexisting",
    )
    try:
        config_release.set()
        prompt_thread.join(timeout=5)
        assert not prompt_thread.is_alive()
        assert prompt_result["response"]["error"]["code"] == 4009
        assert session.get("active_session_lease") is lease
        assert lease.release_calls == 0
        assert session["running"] is False
        assert session["history"] == []
        assert agent.run_conversation_calls == 0
    finally:
        _join_blocked_sessionflag(sessionflag_thread, sessionflag_release)
    assert "result" in sessionflag_result["response"]


def test_prompt_exact_lease_cleanup_skips_replaced_session(gateway, monkeypatch):
    server, db, _home = gateway
    old_agent = LiveAgent(db, "prompt-aba-old")
    old_session = _session(server, db, "sid-prompt-aba", "prompt-aba-old", old_agent)
    old_session["profile_home"] = str(_home / "owning-profile")
    old_lease = RecordingLease("aba-old")
    old_session["active_session_lease"] = old_lease
    new_agent = LiveAgent(db, "prompt-aba-new")
    new_session = _session(server, db, "sid-prompt-aba-new", "prompt-aba-new", new_agent)
    new_lease = RecordingLease("aba-new")
    new_session["active_session_lease"] = new_lease

    registry_checked = threading.Event()
    registry_release = threading.Event()

    def blocked_child_check(_session_key: str, profile_home) -> bool:
        assert profile_home == old_session["profile_home"]
        registry_checked.set()
        assert registry_release.wait(5), "cleanup registry seam was not released"
        return False

    monkeypatch.setattr(server, "_child_run_active", blocked_child_check)
    cleanup_result: dict = {}

    def run_cleanup() -> None:
        cleanup_result["value"] = server._release_prompt_active_session_slot(
            "sid-prompt-aba",
            old_session,
            old_lease,
            lease_token=old_lease.lease_id,
        )

    cleanup_thread = threading.Thread(target=run_cleanup)
    cleanup_thread.start()
    assert registry_checked.wait(2), "cleanup did not reach the replacement seam"
    with server._sessions_lock:
        server._sessions["sid-prompt-aba"] = new_session
    registry_release.set()
    cleanup_thread.join(timeout=5)
    assert not cleanup_thread.is_alive()
    assert cleanup_result["value"] is False
    assert old_session.get("active_session_lease") is old_lease
    assert old_lease.release_calls == 0
    assert new_session.get("active_session_lease") is new_lease
    assert new_lease.release_calls == 0


def test_prompt_exact_lease_cleanup_skips_changed_lease(gateway):
    server, db, _home = gateway
    agent = LiveAgent(db, "prompt-lease-aba")
    session = _session(server, db, "sid-prompt-lease-aba", "prompt-lease-aba", agent)
    old_lease = RecordingLease("lease-old")
    new_lease = RecordingLease("lease-new")
    session["active_session_lease"] = new_lease

    result = server._release_prompt_active_session_slot(
        "sid-prompt-lease-aba",
        session,
        old_lease,
        lease_token=old_lease.lease_id,
    )

    assert result is False
    assert session.get("active_session_lease") is new_lease
    assert old_lease.release_calls == 0
    assert new_lease.release_calls == 0


def test_successful_prompt_transfers_new_lease_to_normal_turn_lifecycle(gateway, monkeypatch):
    server, db, _home = gateway
    agent = LiveAgent(db, "prompt-success-lease")
    session = _session(server, db, "sid-prompt-success-lease", "prompt-success-lease", agent)
    lease = RecordingLease("prompt-success")
    monkeypatch.setattr(server, "_claim_active_session_slot", lambda *a, **k: (lease, None))
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", dict)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *_a, **_k: None)

    response = _call(server, "prompt.submit", session_id="sid-prompt-success-lease", text="normal turn")

    assert response["result"]["status"] == "streaming"
    assert session["running"] is True
    assert session.get("active_session_lease") is lease
    assert lease.release_calls == 0
    server._release_active_session_slot(session)
    assert lease.release_calls == 1


def test_concurrent_prompts_do_not_release_winner_active_session_lease(gateway, monkeypatch):
    server, db, _home = gateway
    agent = LiveAgent(db, "prompt-two-winners")
    session = _session(server, db, "sid-prompt-two-winners", "prompt-two-winners", agent)
    lease = RecordingLease("prompt-two-winners")
    claim_entered = threading.Event()
    claim_release = threading.Event()
    preflight_barrier = threading.Barrier(2)
    claim_calls = 0
    claim_lock = threading.Lock()

    def blocked_claim(*_args, **_kwargs):
        nonlocal claim_calls
        with claim_lock:
            claim_calls += 1
            first = claim_calls == 1
        if first:
            claim_entered.set()
            assert claim_release.wait(5), "first prompt claim was not released"
            return lease, None
        raise AssertionError("second prompt tried to acquire a replacement lease")

    def synchronized_config():
        preflight_barrier.wait(timeout=5)
        return {}

    monkeypatch.setattr(server, "_claim_active_session_slot", blocked_claim)
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", synchronized_config)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *_a, **_k: None)
    monkeypatch.setattr(
        server,
        "_handle_busy_submit",
        lambda *_a, **_k: {"error": {"code": 4009, "message": "session busy"}},
    )
    responses: list[dict] = []

    def run_prompt() -> None:
        responses.append(
            _call(server, "prompt.submit", session_id="sid-prompt-two-winners", text="concurrent")
        )

    first_thread = threading.Thread(target=run_prompt)
    second_thread = threading.Thread(target=run_prompt)
    first_thread.start()
    assert claim_entered.wait(2), "first prompt did not reach active-session claim"
    second_thread.start()
    claim_release.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert len(responses) == 2
    assert any(response.get("result", {}).get("status") == "streaming" for response in responses)
    assert any(response.get("error", {}).get("code") == 4009 for response in responses)
    assert session["running"] is True
    assert session.get("active_session_lease") is lease
    assert lease.release_calls == 0
    server._release_active_session_slot(session)


def test_prompt_exact_lease_cleanup_failure_is_non_masking_and_unlocks(gateway, monkeypatch, caplog):
    server, db, _home = gateway
    agent = LiveAgent(db, "prompt-cleanup-failure")
    session = _session(server, db, "sid-prompt-cleanup-failure", "prompt-cleanup-failure", agent)
    lease = RecordingLease("cleanup-failure")
    session["active_session_lease"] = lease
    calls: list[tuple[tuple, dict]] = []

    class CleanupAbort(BaseException):
        pass

    def failing_canonical(*args, **kwargs):
        calls.append((args, kwargs))
        raise CleanupAbort("release seam failed")

    monkeypatch.setattr(server, "_release_active_session_slot", failing_canonical)
    result = server._release_prompt_active_session_slot(
        "sid-prompt-cleanup-failure",
        session,
        lease,
        lease_token=lease.lease_id,
    )

    assert result is False
    assert len(calls) == 1
    assert "sid-prompt-cleanup-failure" in caplog.text
    with session["history_lock"]:
        pass
    with server._sessions_lock:
        pass


def test_prompt_exact_lease_release_does_not_hold_coordination_locks(gateway):
    server, db, _home = gateway
    agent = LiveAgent(db, "prompt-release-seam")
    session = _session(server, db, "sid-prompt-release-seam", "prompt-release-seam", agent)
    release_entered = threading.Event()
    release_gate = threading.Event()

    class BlockingLease(RecordingLease):
        def release(self) -> None:
            release_entered.set()
            assert release_gate.wait(5), "lease release seam was not released"
            super().release()

    lease = BlockingLease("release-seam")
    session["active_session_lease"] = lease
    cleanup_result: dict = {}

    def run_cleanup() -> None:
        cleanup_result["value"] = server._release_prompt_active_session_slot(
            "sid-prompt-release-seam",
            session,
            lease,
            lease_token=lease.lease_id,
        )

    cleanup_thread = threading.Thread(target=run_cleanup)
    cleanup_thread.start()
    try:
        assert release_entered.wait(2), "cleanup did not reach the release seam"
        _assert_history_lock_available(session["history_lock"])
        _assert_history_lock_available(server._sessions_lock)
    finally:
        release_gate.set()
        cleanup_thread.join(timeout=5)
    assert not cleanup_thread.is_alive()
    assert cleanup_result["value"] is True
    assert lease.release_calls == 1


def test_manual_compression_routes_reject_before_any_mutation_during_sessionflag(
    gateway, monkeypatch
):
    server, db, _home = gateway
    key = "manual-compress-busy"
    agent = LiveAgent(db, key)
    session = _session(server, db, "sid-manual-compress-busy", key, agent)
    session["history"] = [{"role": "user", "content": "before"}]
    history_before = list(session["history"])
    row_before = db.get_session(key)
    key_before = session["session_key"]
    generation_before = session.get("session_generation", 0)
    compressor_calls: list[tuple] = []

    def fail_compress(*args, **kwargs):
        compressor_calls.append((args, kwargs))
        pytest.fail("manual compression must reject before invoking the compressor")

    monkeypatch.setattr(server, "_compress_session_history", fail_compress)
    thread, release, first = _start_blocked_sessionflag(
        server, db, monkeypatch, sid="sid-manual-compress-busy"
    )
    try:
        responses = [
            _call(server, "slash.exec", session_id="sid-manual-compress-busy", command="/compress"),
            _call(server, "slash.exec", session_id="sid-manual-compress-busy", command="/compact"),
            _call(
                server,
                "command.dispatch",
                session_id="sid-manual-compress-busy",
                name="compress",
                arg="",
            ),
            _call(
                server,
                "session.compress",
                session_id="sid-manual-compress-busy",
            ),
        ]
        assert all(response.get("error", {}).get("code") == 4009 for response in responses)
        assert all("busy" in response["error"]["message"].lower() for response in responses)
        assert compressor_calls == []
        assert session["session_key"] == key_before
        assert session.get("session_generation", 0) == generation_before
        assert session["history"] == history_before
        assert db.get_session(key) == row_before
    finally:
        _join_blocked_sessionflag(thread, release)
    assert "result" in first["response"]


def _assert_sessionflag_lease_cleared(
    session: dict, *, expected_running: bool = False
) -> None:
    assert session.get("_session_control_inflight") is None
    assert session.get("running") is expected_running


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt, _PluginSessionControlAbort])
def test_live_sessionflag_config_failure_clears_lease_and_allows_retry(gateway, failure_type):
    server, db, _home = gateway
    key = "config-failure"
    agent = LiveAgent(db, key)
    session = _session(server, db, "sid-config-failure", key, agent)
    row_before = db.get_session(key)
    failure = failure_type("config boom")

    def failing_config():
        raise failure

    with patch.object(server, "_session_plugin_config_loader", return_value=failing_config):
        if issubclass(failure_type, Exception):
            response = _call(server, "slash.exec", session_id="sid-config-failure", command="sessionflag on")
            assert "config boom" in _result(response)["output"]
        else:
            with pytest.raises(failure_type):
                _call(server, "slash.exec", session_id="sid-config-failure", command="sessionflag on")

    _assert_sessionflag_lease_cleared(session)
    assert db.get_session(key) == row_before
    assert agent.ensure_calls == 0
    assert agent.context_compressor.calls == []

    retry = _result(
        _call(server, "slash.exec", session_id="sid-config-failure", command="sessionflag on")
    )
    assert "override saved: ON" in retry["output"]
    assert _override(db.get_session(key)) is True
    assert agent.context_compressor.calls == [True]
    _assert_sessionflag_lease_cleared(session)


@pytest.mark.parametrize("boundary", ["db_context", "db_read", "ensure", "db_set", "live_apply"])
@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt, _PluginSessionControlAbort])
def test_live_sessionflag_failure_matrix_releases_lease_and_preserves_db_first_contract(
    gateway, monkeypatch, boundary, failure_type
):
    server, db, _home = gateway
    key = f"{boundary}-failure-{failure_type.__name__}"
    agent = LiveAgent(db, key)
    session = _session(server, db, f"sid-{key}", key, agent)
    row_before = db.get_session(key)
    failure = failure_type(f"{boundary} boom")

    def invoke_and_assert(*, propagates: bool = False):
        if propagates or not issubclass(failure_type, Exception):
            with pytest.raises(failure_type):
                _call(server, "slash.exec", session_id=f"sid-{key}", command="sessionflag on")
            return
        response = _call(server, "slash.exec", session_id=f"sid-{key}", command="sessionflag on")
        assert "result" in response, response
        expected = "does not support" if boundary == "live_apply" else boundary
        assert expected in _result(response)["output"]

    from hermes_cli.plugins import get_plugin_manager

    @contextlib.contextmanager
    def failing_session_db(*_args):
        raise failure
        yield None

    def failing_apply(*_args):
        raise failure

    entry = get_plugin_manager()._session_policies[POLICY_KEY]
    failure_patches = {
        "db_context": (patch.object(server, "_plugin_session_command_db", failing_session_db),),
        "db_read": (
            patch.object(agent, "_ensure_db_session", return_value=None),
            patch.object(db, "get_session", side_effect=failure),
        ),
        "ensure": (patch.object(agent, "_ensure_db_session", side_effect=failure),),
        "db_set": (patch.object(db, "set_session_policy_override", side_effect=failure),),
        "live_apply": (patch.dict(entry, apply=failing_apply),),
    }
    with contextlib.ExitStack() as stack:
        for failure_patch in failure_patches[boundary]:
            stack.enter_context(failure_patch)
        invoke_and_assert()

    _assert_sessionflag_lease_cleared(session)
    if boundary == "live_apply":
        assert _override(db.get_session(key)) is True
        assert agent.context_compressor.enabled is None
    else:
        assert db.get_session(key) == row_before
    assert agent.context_compressor.calls == []

    # Restore the failed seam and prove the same session can claim the lease
    # again.  ``off`` also checks that a DB-first failure did not leave a stale
    # durable override behind.
    retry = _result(
        _call(server, "slash.exec", session_id=f"sid-{key}", command="sessionflag off")
    )
    assert "override saved: OFF" in retry["output"]
    assert _override(db.get_session(key)) is False
    _assert_sessionflag_lease_cleared(session)


@pytest.mark.parametrize("failure_type", [RuntimeError, _PluginSessionControlAbort])
def test_cold_profile_session_db_open_failure_clears_lease_and_allows_retry(gateway, tmp_path, failure_type):
    server, db, _home = gateway
    key = f"profile-open-{failure_type.__name__}"
    profile_home = tmp_path / "profile-open-home"
    profile_home.mkdir()
    profile_db = SessionDB(db_path=profile_home / "state.db")
    profile_db.create_session(key, source="tui", model="test-model")
    session = _session(server, db, f"sid-{key}", key, None, profile_home=profile_home, row_db=profile_db)
    profile_db.close()
    row_before = db.get_session(key)
    failure = failure_type("profile SessionDB open boom")

    class FailingSessionDB:
        def __init__(self, **_kwargs):
            raise failure

    with patch("hermes_state.SessionDB", FailingSessionDB), patch.object(
        server, "_session_plugin_config_loader", return_value=lambda: {"test_policy": {}}
    ):
        if issubclass(failure_type, Exception):
            response = _call(server, "slash.exec", session_id=f"sid-{key}", command="sessionflag on")
            assert "canonical session persistence is not available" in _result(response)["output"]
        else:
            with pytest.raises(failure_type):
                _call(server, "slash.exec", session_id=f"sid-{key}", command="sessionflag on")
    _assert_sessionflag_lease_cleared(session)
    assert db.get_session(key) == row_before
    retry = _result(_call(server, "slash.exec", session_id=f"sid-{key}", command="sessionflag off"))
    assert "override saved: OFF" in retry["output"]
    check_db = SessionDB(db_path=profile_home / "state.db")
    try:
        assert _override(check_db.get_session(key)) is False
    finally:
        check_db.close()
    _assert_sessionflag_lease_cleared(session)


@pytest.mark.parametrize("busy_source", ["running", "child"])
def test_busy_sessionflag_rejects_before_lease_config_db_or_live_mutation(
    gateway, monkeypatch, busy_source
):
    server, db, _home = gateway
    key = f"busy-{busy_source}"
    agent = LiveAgent(db, key)
    session = _session(server, db, f"sid-{key}", key, agent)
    row_before = db.get_session(key)
    if busy_source == "running":
        session["running"] = True
    else:
        monkeypatch.setattr(server, "_child_run_active", lambda _key, _profile_home: True)

    def fail_config(_session):
        pytest.fail("busy /sessionflag must reject before loading config")

    monkeypatch.setattr(server, "_session_plugin_config_loader", fail_config)
    monkeypatch.setattr(
        db,
        "set_session_policy_override",
        lambda *_args, **_kwargs: pytest.fail("busy /sessionflag must not write the DB"),
    )

    response = _call(server, "slash.exec", session_id=f"sid-{key}", command="sessionflag on")

    assert response["error"]["code"] == 4009
    _assert_sessionflag_lease_cleared(
        session, expected_running=(busy_source == "running")
    )
    assert db.get_session(key) == row_before
    assert agent.ensure_calls == 0
    assert agent.context_compressor.calls == []


def _start_blocked_sessionflag(
    server, db, monkeypatch, *, sid: str, blocked_session_id: str | None = None
):
    entered = threading.Event()
    release = threading.Event()
    result: dict = {}
    original_setter = db.set_session_policy_override
    if blocked_session_id is None:
        blocked_session_id = str(
            server._sessions[sid].get("session_key") or sid
        )

    def blocked_setter(session_id, policy_key, override):
        if session_id == blocked_session_id:
            entered.set()
            assert release.wait(5), "blocked sessionflag was not released"
        return original_setter(session_id, policy_key, override)

    monkeypatch.setattr(db, "set_session_policy_override", blocked_setter)

    def run_sessionflag() -> None:
        result["response"] = _call(server, "slash.exec", session_id=sid, command="sessionflag on")

    thread = threading.Thread(target=run_sessionflag)
    thread.start()
    assert entered.wait(2), "sessionflag did not reach the blocking DB setter"
    return thread, release, result


def _join_blocked_sessionflag(thread: threading.Thread, release: threading.Event) -> None:
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_session_resume_reuse_rejects_during_sessionflag_and_reuses_after_release(gateway, monkeypatch):
    server, db, _home = gateway
    key = "resume-while-sessionflag"
    agent = LiveAgent(db, key)
    session = _session(server, db, "sid-resume-while-sessionflag", key, agent)
    thread, release, first = _start_blocked_sessionflag(
        server, db, monkeypatch, sid="sid-resume-while-sessionflag"
    )
    try:
        busy = _call(server, "session.resume", session_id=key)
        assert busy["error"]["code"] == 4009
        assert server._sessions.get("sid-resume-while-sessionflag") is session
        assert _override(db.get_session(key)) is None
    finally:
        _join_blocked_sessionflag(thread, release)

    assert "result" in first["response"]
    resumed = _call(server, "session.resume", session_id=key)
    assert "result" in resumed, resumed
    assert resumed["result"]["session_id"] == "sid-resume-while-sessionflag"
    assert server._sessions.get("sid-resume-while-sessionflag") is session
    assert _override(db.get_session(key)) is True


def test_session_reuse_releases_resume_lock_before_blocking_db_payload_read(gateway, monkeypatch):
    server, db, _home = gateway
    key = "resume-payload-lock"
    agent = LiveAgent(db, key)
    _session(server, db, "sid-resume-payload-lock", key, agent)
    db_read_entered = threading.Event()
    db_read_release = threading.Event()
    resume_result: dict = {}

    def blocked_messages(*_args, **_kwargs):
        db_read_entered.set()
        assert db_read_release.wait(5), "blocked resume payload read was not released"
        return []

    monkeypatch.setattr(db, "get_messages_as_conversation", blocked_messages)

    def resume() -> None:
        resume_result["response"] = _call(server, "session.resume", session_id=key)

    resume_thread = threading.Thread(target=resume)
    resume_thread.start()
    assert db_read_entered.wait(2)
    lock_acquired = threading.Event()

    def acquire_resume_lock() -> None:
        with server._session_resume_lock:
            lock_acquired.set()

    lock_thread = threading.Thread(target=acquire_resume_lock)
    lock_thread.start()
    try:
        assert lock_acquired.wait(2), "resume held the global lock during DB payload I/O"
    finally:
        db_read_release.set()
        resume_thread.join(timeout=5)
        lock_thread.join(timeout=5)
    assert not resume_thread.is_alive()
    assert not lock_thread.is_alive()
    assert "result" in resume_result["response"]


def test_session_close_rejects_during_sessionflag_and_succeeds_after_release(gateway, monkeypatch):
    server, db, _home = gateway
    key = "close-while-sessionflag"
    agent = LiveAgent(db, key)
    session = _session(server, db, "sid-close-while-sessionflag", key, agent)
    thread, release, first = _start_blocked_sessionflag(
        server, db, monkeypatch, sid="sid-close-while-sessionflag"
    )
    try:
        busy = _call(server, "session.close", session_id="sid-close-while-sessionflag")
        assert busy["error"]["code"] == 4009
        assert server._sessions.get("sid-close-while-sessionflag") is session
        assert _override(db.get_session(key)) is None
    finally:
        _join_blocked_sessionflag(thread, release)

    assert "result" in first["response"]
    closed = _call(server, "session.close", session_id="sid-close-while-sessionflag")
    assert closed["result"]["closed"] is True
    assert "sid-close-while-sessionflag" not in server._sessions
    assert _override(db.get_session(key)) is True


def test_compression_rotation_defers_during_sessionflag_and_reanchors_after_release(gateway, monkeypatch):
    server, db, _home = gateway
    old_key = "rotation-old"
    new_key = "rotation-new"
    agent = LiveAgent(db, old_key)
    session = _session(server, db, "sid-rotation", old_key, agent)
    db.create_session(new_key, source="tui", model="test-model")
    generation_before = session.get("session_generation", 0)
    monkeypatch.setattr(server, "_transfer_active_session_slot", lambda *_a, **_k: True)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_a, **_k: None)
    thread, release, first = _start_blocked_sessionflag(
        server, db, monkeypatch, sid="sid-rotation"
    )
    try:
        # Compression has already produced a continuation id, but the gateway
        # must defer the registry/session-key re-anchor while /sessionflag owns the
        # exact session object.  The production lease release below performs
        # the retry; this test deliberately never calls sync after release.
        agent.session_id = new_key
        status = server._sync_session_key_after_compress("sid-rotation", session)
        assert status == "deferred"
        assert session["session_key"] == old_key
        assert session.get("session_generation", 0) == generation_before
        assert _override(db.get_session(old_key)) is None
        assert _override(db.get_session(new_key)) is None
    finally:
        _join_blocked_sessionflag(thread, release)

    assert "result" in first["response"]
    assert "warning" in first["response"]["result"]["output"].lower()
    assert "old row" in first["response"]["result"]["output"].lower()
    assert session["session_key"] == new_key
    assert session["session_generation"] == generation_before + 1
    assert session.get("_session_control_inflight") is None
    assert session.get("_deferred_compression_rotation") is None
    # The defensive fail-closed path is explicit: /sessionflag changed only the old
    # row and the user must retry on the canonical continuation.
    assert _override(db.get_session(old_key)) is True
    assert _override(db.get_session(new_key)) is None


def test_deferred_rotations_coalesce_latest_target_and_canonical_fields(
    gateway, monkeypatch
):
    server, db, _home = gateway
    old_key = "coalesce-old"
    first_key = "coalesce-first"
    latest_key = "coalesce-latest"
    agent = LiveAgent(db, old_key)
    session = _session(server, db, "sid-coalesce", old_key, agent)
    db.create_session(first_key, source="tui", model="test-model")
    db.create_session(latest_key, source="tui", model="test-model")
    monkeypatch.setattr(server, "_transfer_active_session_slot", lambda *_a, **_k: True)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_a, **_k: None)
    generation_before = session.get("session_generation", 0)
    target, claim_error = server._claim_live_plugin_session_control("sid-coalesce", session)
    assert claim_error is None
    assert target is not None
    try:
        agent.session_id = first_key
        assert server._sync_session_key_after_compress("sid-coalesce", session) == "deferred"
        marker = session["_deferred_compression_rotation"]
        assert marker["agent_session_id"] == first_key
        agent.session_id = latest_key
        assert server._sync_session_key_after_compress("sid-coalesce", session) == "deferred"
        assert len(session["_deferred_compression_rotation"]) <= 7
        assert session["_deferred_compression_rotation"]["agent_session_id"] == latest_key
    finally:
        with patch("tools.approval.unregister_gateway_notify"), patch(
            "tools.approval.enable_session_yolo"
        ), patch("tools.approval.disable_session_yolo"), patch(
            "tools.approval.is_session_yolo_enabled", return_value=False
        ), patch("tools.approval.register_gateway_notify"):
            release_result = server._release_live_plugin_session_control(target)
    assert release_result
    assert session["session_key"] == latest_key
    assert session["session_generation"] == generation_before + 1
    assert session.get("_session_control_inflight") is None
    assert session.get("_deferred_compression_rotation") is None


def test_deferred_reanchor_failure_retains_marker_for_next_safe_trigger(
    gateway, monkeypatch
):
    server, db, _home = gateway
    old_key = "retry-old"
    new_key = "retry-new"
    agent = LiveAgent(db, old_key)
    session = _session(server, db, "sid-retry", old_key, agent)
    db.create_session(new_key, source="tui", model="test-model")
    monkeypatch.setattr(server, "_transfer_active_session_slot", lambda *_a, **_k: True)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_a, **_k: None)
    target, claim_error = server._claim_live_plugin_session_control("sid-retry", session)
    assert claim_error is None
    assert target is not None
    agent.session_id = new_key
    assert server._sync_session_key_after_compress("sid-retry", session) == "deferred"
    real_sync = server._sync_session_key_after_compress

    def fail_sync(*_args, **_kwargs):
        raise RuntimeError("injected re-anchor failure")

    monkeypatch.setattr(server, "_sync_session_key_after_compress", fail_sync)
    failed = server._release_live_plugin_session_control(target)
    assert not failed
    assert "retry" in failed.warning.lower()
    assert "injected re-anchor failure" in failed.warning
    assert session.get("_session_control_inflight") is None
    assert session.get("_deferred_compression_rotation") is not None
    assert session["session_key"] == old_key

    monkeypatch.setattr(server, "_sync_session_key_after_compress", real_sync)
    with patch("tools.approval.unregister_gateway_notify"), patch(
        "tools.approval.enable_session_yolo"
    ), patch("tools.approval.disable_session_yolo"), patch(
        "tools.approval.is_session_yolo_enabled", return_value=False
    ), patch("tools.approval.register_gateway_notify"):
        assert real_sync("sid-retry", session) == "applied"
    assert session["session_key"] == new_key
    assert session.get("_deferred_compression_rotation") is None


def test_release_close_barrier_does_not_touch_detached_target_or_new_lease(
    gateway, monkeypatch
):
    server, db, _home = gateway
    old_key = "close-barrier-old"
    new_key = "close-barrier-new"
    old_agent = LiveAgent(db, old_key)
    old_session = _session(server, db, "sid-close-barrier", old_key, old_agent)
    db.create_session(new_key, source="tui", model="test-model")
    target, claim_error = server._claim_live_plugin_session_control("sid-close-barrier", old_session)
    assert claim_error is None
    assert target is not None
    new_agent = LiveAgent(db, new_key)
    new_session = _session(server, db, "sid-close-barrier-new", new_key, new_agent)
    try:
        with server._sessions_lock:
            server._sessions.pop("sid-close-barrier", None)
            server._sessions["sid-close-barrier"] = new_session
        result = server._release_live_plugin_session_control(target)
        assert not result
        assert "changed" in result.warning.lower()
        assert old_session.get("_session_control_inflight") is None
        assert new_session.get("_session_control_inflight") is None
        assert server._sessions["sid-close-barrier"] is new_session
    finally:
        with server._sessions_lock:
            if server._sessions.get("sid-close-barrier") is new_session:
                server._sessions.pop("sid-close-barrier", None)


def test_deferred_reanchor_entry_guard_blocks_stale_mutation_after_registry_gap(
    gateway, monkeypatch
):
    server, db, _home = gateway
    old_key = "entry-gap-old"
    new_key = "entry-gap-new"
    old_agent = LiveAgent(db, old_key)
    old_session = _session(server, db, "sid-entry-gap", old_key, old_agent)
    db.create_session(new_key, source="tui", model="test-model")
    monkeypatch.setattr(server, "_transfer_active_session_slot", lambda *_a, **_k: True)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_a, **_k: None)

    target, claim_error = server._claim_live_plugin_session_control("sid-entry-gap", old_session)
    assert claim_error is None
    assert target is not None
    old_generation = int(old_session.get("session_generation", 0) or 0)
    old_row = db.get_session(old_key)
    new_row = db.get_session(new_key)
    old_agent.session_id = new_key
    assert server._sync_session_key_after_compress("sid-entry-gap", old_session) == "deferred"
    marker = old_session["_deferred_compression_rotation"]

    replacement_agent = LiveAgent(db, new_key)
    replacement_session = _session(
        server, db, "sid-entry-gap-replacement", new_key, replacement_agent
    )
    replacement_token = object()
    replacement_state = {"replacement": True}
    replacement_session["_session_control_inflight"] = replacement_token
    replacement_session["_plugin_session_control_state"] = replacement_state
    replacement_key = replacement_session["session_key"]
    replacement_generation = int(replacement_session.get("session_generation", 0) or 0)

    sync_entered = threading.Event()
    sync_release = threading.Event()
    real_sync = server._sync_session_key_after_compress

    def gated_sync(*args, **kwargs):
        sync_entered.set()
        assert sync_release.wait(5), "deferred sync entry was not released"
        return real_sync(*args, **kwargs)

    monkeypatch.setattr(server, "_sync_session_key_after_compress", gated_sync)
    mutation_calls: list[str] = []
    monkeypatch.setattr(
        server,
        "_transfer_active_session_slot",
        lambda *_a, **_k: mutation_calls.append("lease") or True,
    )
    monkeypatch.setattr(
        server,
        "_restart_slash_worker",
        lambda *_a, **_k: mutation_calls.append("worker"),
    )
    with patch("tools.approval.unregister_gateway_notify", lambda *_a: mutation_calls.append("unregister")), patch(
        "tools.approval.enable_session_yolo", lambda *_a: mutation_calls.append("enable")), patch(
        "tools.approval.disable_session_yolo", lambda *_a: mutation_calls.append("disable")), patch(
        "tools.approval.is_session_yolo_enabled", return_value=False), patch(
        "tools.approval.register_gateway_notify", lambda *_a, **_k: mutation_calls.append("register")
    ):
        release_result: dict = {}

        def release_target() -> None:
            release_result["value"] = server._release_live_plugin_session_control(target)

        release_thread = threading.Thread(target=release_target)
        release_thread.start()
        try:
            assert sync_entered.wait(2), "release did not reach the sync-entry barrier"
            with server._sessions_lock:
                server._sessions["sid-entry-gap"] = replacement_session
            sync_release.set()
            release_thread.join(timeout=5)
        finally:
            sync_release.set()
            release_thread.join(timeout=5)

    assert not release_thread.is_alive()
    result = release_result["value"]
    assert not result
    assert "changed" in result.warning.lower()
    assert old_session["session_key"] == old_key
    assert int(old_session.get("session_generation", 0) or 0) == old_generation
    assert old_session.get("_session_control_inflight") is None
    assert old_session.get("_deferred_compression_rotation") is marker
    assert mutation_calls == []
    assert db.get_session(old_key) == old_row
    assert db.get_session(new_key) == new_row
    assert server._sessions["sid-entry-gap"] is replacement_session
    assert replacement_session["session_key"] == replacement_key
    assert int(replacement_session.get("session_generation", 0) or 0) == replacement_generation
    assert replacement_session.get("_session_control_inflight") is replacement_token
    assert replacement_session.get("_plugin_session_control_state") is replacement_state


def test_deferred_reanchor_reservation_blocks_production_transitions_until_sync_finishes(
    gateway, monkeypatch
):
    server, db, _home = gateway
    old_key = "reservation-old"
    new_key = "reservation-new"
    agent = LiveAgent(db, old_key)
    session = _session(server, db, "sid-reservation", old_key, agent)
    db.create_session(new_key, source="tui", model="test-model")
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_a, **_k: None)
    target, claim_error = server._claim_live_plugin_session_control("sid-reservation", session)
    assert claim_error is None
    assert target is not None
    agent.session_id = new_key
    assert server._sync_session_key_after_compress("sid-reservation", session) == "deferred"

    transfer_entered = threading.Event()
    transfer_release = threading.Event()

    def blocked_transfer(*_args, **_kwargs):
        transfer_entered.set()
        assert transfer_release.wait(5), "re-anchor transfer seam was not released"
        return True

    monkeypatch.setattr(server, "_transfer_active_session_slot", blocked_transfer)
    release_result: dict = {}
    with patch("tools.approval.unregister_gateway_notify"), patch(
        "tools.approval.enable_session_yolo"
    ), patch("tools.approval.disable_session_yolo"), patch(
        "tools.approval.is_session_yolo_enabled", return_value=False
    ), patch("tools.approval.register_gateway_notify"):
        release_thread = threading.Thread(
            target=lambda: release_result.setdefault(
                "value", server._release_live_plugin_session_control(target)
            )
        )
        release_thread.start()
        try:
            assert transfer_entered.wait(2), "release did not retain control through transfer"
            assert server._pop_session_by_id("sid-reservation") is server._SESSION_CONTROL_BUSY
            assert server._live_session_payload(
                "sid-reservation", session, reject_session_control=True
            ) is None
            assert server._claim_or_reuse_live(
                "sid-reservation-reuse", new_key, {}, None
            ) is server._SESSION_CONTROL_BUSY
            assert server._claim_or_reuse_live(
                "sid-reservation-reuse-old", old_key, {}, None
            ) is server._SESSION_CONTROL_BUSY
            manual_token, manual_error = server._claim_manual_compression_control(session)
            assert manual_token is None
            assert manual_error is server._SESSION_CONTROL_BUSY
            assert server._try_claim_session_control(session) is None
            close_response = server._methods["session.close"](
                "rid", {"session_id": "sid-reservation"}
            )
            assert close_response["error"]["code"] == 4009
            with server._sessions_lock:
                assert server._sessions["sid-reservation"] is session
            transfer_release.set()
            release_thread.join(timeout=5)
        finally:
            transfer_release.set()
            release_thread.join(timeout=5)

    assert not release_thread.is_alive()
    assert release_result["value"]
    assert session["session_key"] == new_key
    assert session.get("_deferred_compression_rotation") is None
    assert session.get("_session_control_inflight") is None


def test_sync_session_key_after_compress_legacy_no_history_lock_compatibility(
    gateway, monkeypatch
):
    server, db, _home = gateway
    old_key = "legacy-lockless-old"
    new_key = "legacy-lockless-new"
    agent = LiveAgent(db, old_key)
    db.create_session(old_key, source="tui", model="test-model")
    db.create_session(new_key, source="tui", model="test-model")
    session = {
        "agent": agent,
        "session_key": old_key,
        "session_generation": 0,
        "pending_title": "old title",
        "profile_home": None,
        "slash_worker": None,
    }
    agent.session_id = new_key
    monkeypatch.setattr(server, "_transfer_active_session_slot", lambda *_a, **_k: True)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_a, **_k: None)
    with patch("tools.approval.unregister_gateway_notify"), patch(
        "tools.approval.enable_session_yolo"
    ), patch("tools.approval.disable_session_yolo"), patch(
        "tools.approval.is_session_yolo_enabled", return_value=False
    ), patch("tools.approval.register_gateway_notify"):
        status = server._sync_session_key_after_compress("sid-legacy-lockless", session)

    assert status == "applied"
    assert session["session_key"] == new_key
    assert session["session_generation"] == 1
    assert session["pending_title"] is None


def test_release_reanchor_leaves_history_and_registry_locks_available(
    gateway, monkeypatch
):
    server, db, _home = gateway
    old_key = "lock-race-old"
    new_key = "lock-race-new"
    agent = LiveAgent(db, old_key)
    session = _session(server, db, "sid-lock-race", old_key, agent)
    db.create_session(new_key, source="tui", model="test-model")
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_a, **_k: None)
    target, claim_error = server._claim_live_plugin_session_control("sid-lock-race", session)
    assert claim_error is None
    assert target is not None
    agent.session_id = new_key
    assert server._sync_session_key_after_compress("sid-lock-race", session) == "deferred"
    transfer_entered = threading.Event()
    transfer_release = threading.Event()

    def blocked_transfer(*_args, **_kwargs):
        transfer_entered.set()
        assert transfer_release.wait(2), "blocked re-anchor was not released"
        return True

    monkeypatch.setattr(server, "_transfer_active_session_slot", blocked_transfer)
    release_result: dict = {}

    def release_target() -> None:
        with patch("tools.approval.unregister_gateway_notify"), patch(
            "tools.approval.enable_session_yolo"
        ), patch("tools.approval.disable_session_yolo"), patch(
            "tools.approval.is_session_yolo_enabled", return_value=False
        ), patch("tools.approval.register_gateway_notify"):
            release_result["value"] = server._release_live_plugin_session_control(target)

    release_thread = threading.Thread(target=release_target)
    release_thread.start()
    assert transfer_entered.wait(2)
    history_acquired = threading.Event()
    registry_acquired = threading.Event()

    def acquire_history() -> None:
        with session["history_lock"]:
            history_acquired.set()

    def acquire_registry() -> None:
        with server._sessions_lock:
            registry_acquired.set()

    history_thread = threading.Thread(target=acquire_history)
    registry_thread = threading.Thread(target=acquire_registry)
    history_thread.start()
    registry_thread.start()
    try:
        assert history_acquired.wait(2), "history_lock remained held during re-anchor I/O"
        assert registry_acquired.wait(2), "_sessions_lock remained held during re-anchor I/O"
    finally:
        transfer_release.set()
        release_thread.join(timeout=5)
        history_thread.join(timeout=5)
        registry_thread.join(timeout=5)
    assert not release_thread.is_alive()
    assert not history_thread.is_alive()
    assert not registry_thread.is_alive()
    assert release_result["value"]


def test_old_sessionflag_finally_cannot_clear_replacement_session_lease(gateway, monkeypatch):
    server, db, _home = gateway
    old_key = "aba-old"
    new_key = "aba-new"
    old_agent = LiveAgent(db, old_key)
    old_session = _session(server, db, "sid-aba", old_key, old_agent)
    thread, release, first = _start_blocked_sessionflag(server, db, monkeypatch, sid="sid-aba")
    new_target = None
    try:
        new_agent = LiveAgent(db, new_key)
        db.create_session(new_key, source="tui", model="test-model")
        new_session = {
            "session_key": new_key,
            "agent": new_agent,
            "history": [],
            "history_lock": threading.Lock(),
            "history_version": 0,
            "session_generation": 1,
            "running": False,
            "profile_home": None,
            "slash_worker": None,
            "model": "test-model",
        }
        with server._sessions_lock:
            server._sessions["sid-aba"] = new_session
        new_target, claim_error = server._claim_live_plugin_session_control("sid-aba", new_session)
        assert claim_error is None
        assert new_target is not None
        new_token = new_session["_session_control_inflight"]
        assert new_token is new_target.token

        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert "result" in first["response"]
        output = first["response"]["result"]["output"]
        assert isinstance(output, str)
        assert "live session changed" in output
        assert old_session.get("_session_control_inflight") is None
        assert new_session.get("_session_control_inflight") is new_token
        assert _override(db.get_session(old_key)) is True
        assert _override(db.get_session(new_key)) is None
        assert old_agent.context_compressor.calls == [True]
        assert new_agent.context_compressor.calls == []
    finally:
        release.set()
        thread.join(timeout=5)
        if new_target is not None:
            server._release_live_plugin_session_control(new_target)


def test_blocked_sessionflag_on_one_session_does_not_block_other_session_or_status(
    gateway, monkeypatch
):
    server, db, _home = gateway
    agent_a = LiveAgent(db, "isolated-a")
    agent_b = LiveAgent(db, "isolated-b")
    _session(server, db, "sid-isolated-a", "isolated-a", agent_a)
    _session(server, db, "sid-isolated-b", "isolated-b", agent_b)
    thread, release, first = _start_blocked_sessionflag(
        server,
        db,
        monkeypatch,
        sid="sid-isolated-a",
        blocked_session_id="isolated-a",
    )
    other_result: dict = {}
    other_done = threading.Event()

    def run_other_sessionflag() -> None:
        other_result["response"] = _call(server, "slash.exec", session_id="sid-isolated-b", command="sessionflag on")
        other_done.set()

    other_thread = threading.Thread(target=run_other_sessionflag)
    other_thread.start()
    try:
        assert other_done.wait(2), "session B was blocked by session A's sessionflag I/O"
        assert "result" in other_result["response"]
        assert _override(db.get_session("isolated-b")) is True
        status = _call(server, "session.status", session_id="sid-isolated-b")
        assert "result" in status
    finally:
        _join_blocked_sessionflag(thread, release)
        other_thread.join(timeout=5)
    assert "result" in first["response"]
    assert not other_thread.is_alive()


@pytest.mark.parametrize("method", ["slash.exec", "command.dispatch"])
@pytest.mark.parametrize("arg", ["", "maybe", "on off"])
def test_invalid_session_command_prepares_before_lease_config_or_db(gateway, monkeypatch, method, arg):
    server, db, home = gateway
    agent = LiveAgent(db, "parse-first")
    session = _session(server, db, "sid-parse-first", agent.session_id, agent)
    before, config = db.get_session(agent.session_id), (home / "config.yaml").read_bytes()
    for name in ("_claim_live_plugin_session_control", "_load_cfg", "_session_plugin_config_loader", "_session_db"):
        monkeypatch.setattr(server, name, lambda *_a: pytest.fail("invalid command reached host state"))
    result = _result(_flag_call(server, method, "sid-parse-first", arg))
    assert "on|off|inherit|status" in result["output"]
    assert db.get_session(agent.session_id) == before and (home / "config.yaml").read_bytes() == config
    assert agent.ensure_calls == 0 and session.get("_session_control_inflight") is None


@pytest.mark.parametrize("method", ["slash.exec", "command.dispatch"])
def test_session_command_authority_failure_never_falls_through(gateway, monkeypatch, method):
    server, db, _home = gateway
    agent = LiveAgent(db, "denied")
    _session(server, db, "sid-denied", agent.session_id, agent)
    denied = server._err("rid", 4001, "transport rejected")
    monkeypatch.setattr(server, "_sess_nowait", lambda *_a: (None, denied))
    monkeypatch.setattr(server, "_SlashWorker", lambda *_a: pytest.fail("authority fell through to worker"))
    monkeypatch.setattr(server, "_dispatch_quick", lambda *_a: pytest.fail("authority fell through to quick command"))
    assert _flag_call(server, method, "sid-denied", "on") == denied
    assert agent.ensure_calls == 0 and _override(db.get_session(agent.session_id)) is None


@pytest.mark.parametrize("method", ["slash.exec", "command.dispatch"])
def test_cold_profile_status_does_not_create_db_and_write_uses_canonical_row(gateway, tmp_path, method):
    server, db, _home = gateway
    profile = tmp_path / "cold-profile"
    profile.mkdir()
    (profile / "config.yaml").write_text("test_policy:\n  enabled: true\n", encoding="utf-8")
    session = _session(server, db, "sid-profile-cold", "cold-canonical", None)
    session["profile_home"] = str(profile)
    with patch.object(server, "_ensure_session_db_row", side_effect=AssertionError("status ensured")):
        assert "Global: ON" in _result(_flag_call(server, method, "sid-profile-cold", "status"))["output"]
    assert not (profile / "state.db").exists()
    assert "override saved: OFF" in _result(_flag_call(server, method, "sid-profile-cold", "off"))["output"]
    with contextlib.closing(SessionDB(db_path=profile / "state.db", read_only=True)) as profile_db:
        row = profile_db.get_session("cold-canonical")
        assert row["source"] == "tui" and _override(row) is False
    with patch.object(server, "_session_db", side_effect=AssertionError("status opened a writer")):
        assert "Session flag: OFF" in _result(_flag_call(server, method, "sid-profile-cold", "status"))["output"]
    assert _override(db.get_session("cold-canonical")) is None and session["slash_worker"] is None


@pytest.mark.parametrize("method", ["slash.exec", "command.dispatch"])
def test_live_session_command_rejects_foreign_db_before_ensure(gateway, tmp_path, method):
    server, db, _home = gateway
    agent = LiveAgent(db, "foreign-db")
    session = _session(server, db, "sid-foreign-db", agent.session_id, agent)
    session["profile_home"] = str(tmp_path / "foreign-profile")
    output = _result(_flag_call(server, method, "sid-foreign-db", "on"))["output"]
    assert "database profile mismatch" in output
    assert agent.ensure_calls == 0 and agent.context_compressor.calls == []
    assert _override(db.get_session(agent.session_id)) is None
    _assert_sessionflag_lease_cleared(session)


@pytest.mark.parametrize("method", ["slash.exec", "command.dispatch"])
def test_cold_config_failure_does_not_create_state_db(gateway, tmp_path, method):
    server, db, _home = gateway
    profile = tmp_path / "config-failed-profile"
    profile.mkdir()
    session = _session(server, db, "sid-config-cold", "config-cold", None)
    session["profile_home"] = str(profile)
    with patch.object(server, "_session_plugin_config_loader", return_value=lambda: None):
        output = _result(_flag_call(server, method, "sid-config-cold", "on"))["output"]
    assert "could not read global configuration" in output
    assert not (profile / "state.db").exists()
    _assert_sessionflag_lease_cleared(session)


@pytest.mark.parametrize("new_row", [False, True])
def test_rotation_inherit_requires_exact_row_and_live_cache(gateway, new_row):
    server, db, _home = gateway
    agent = LiveAgent(db, "inherit-old")
    session = _session(server, db, "sid-inherit-rotate", agent.session_id, agent)
    target, err = server._claim_live_plugin_session_control("sid-inherit-rotate", session)
    assert err is None
    if new_row:
        db.create_session("inherit-new", source="tui")
        agent._session_policy_overrides[POLICY_KEY] = False
    try:
        warning = server._plugin_session_rotation_override_warning(
            target, {"agent_session_id": "inherit-new"}, {"writes": {POLICY_KEY: None}})
        assert ("Agent policy" if new_row else "old row") in warning
    finally:
        server._release_live_plugin_session_control(target)


@pytest.mark.parametrize("fails_after_write", [False, True])
def test_rotation_verifies_all_committed_keys_even_without_success_text(gateway, monkeypatch, fails_after_write):
    server, db, _home = gateway
    from hermes_cli.plugins import PluginContext, PluginManifest, get_plugin_manager
    manager = get_plugin_manager()
    context = PluginContext(PluginManifest(name="second-probe", key="second-probe"), manager)
    context.register_session_policy("second_override", config_path="test_policy.enabled", apply=lambda *_a: True)
    agent = LiveAgent(db, "multi-old")
    session = _session(server, db, "sid-multi", agent.session_id, agent)
    db.create_session("multi-new", source="tui")
    entered, release, result = threading.Event(), threading.Event(), {}

    def handler(_request, scoped):
        scoped.set_override(POLICY_KEY, True)
        scoped.set_override("second_override", False)
        entered.set()
        assert release.wait(5)
        if fails_after_write:
            raise RuntimeError("after committed writes")
        return "done"

    monkeypatch.setattr(manager._plugin_commands["sessionflag"]["handler"], "handler", handler)
    monkeypatch.setattr(server, "_transfer_active_session_slot", lambda *_a, **_k: True)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_a, **_k: None)
    thread = threading.Thread(target=lambda: result.update(response=_flag_call(server, "command.dispatch", "sid-multi", "on")))
    thread.start()
    try:
        assert entered.wait(2)
        db.set_session_policy_override("multi-new", POLICY_KEY, True)
        agent.session_id = "multi-new"
        assert server._sync_session_key_after_compress("sid-multi", session) == "deferred"
    finally:
        _join_blocked_sessionflag(thread, release)
    output = _result(result["response"])["output"]
    assert "old row" in output and ("after committed writes" if fails_after_write else "done") in output
    assert SessionDB.session_policy_override(db.get_session("multi-old"), "second_override") is False
    assert SessionDB.session_policy_override(db.get_session("multi-new"), "second_override") is None


def test_real_native_plugin_discovery_reaches_both_tui_dispatchers(tmp_path, monkeypatch):
    from hermes_cli import plugins
    home = tmp_path / "discovered-home"
    plugin = home / "plugins" / "native-probe"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: native-probe\nversion: 1.0.0\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(
        "from hermes_cli.session_commands import SessionCommandRequest\n"
        "def prepare(arg):\n"
        "    if arg not in ('on', 'status'): raise ValueError('Use on or status')\n"
        "    return SessionCommandRequest(arg == 'on', arg)\n"
        "def command(request, session):\n"
        "    if request.write: session.set_override('probe_override', True)\n"
        "    return f'{session.session_id}:{session.get_override(\"probe_override\")}'\n"
        "def register(ctx):\n"
        "    ctx.register_session_policy('probe_override', config_path='probe.enabled', apply=lambda *_a: True)\n"
        "    ctx.register_session_command('native-probe', command, prepare=prepare, subcommands=('on', 'status'))\n",
        encoding="utf-8",
    )
    (home / "config.yaml").write_text("plugins:\n  enabled: [native-probe]\n  isolation: in_process\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    plugins._reset_plugin_managers_for_tests()
    server = importlib.import_module("tui_gateway.server")
    db = SessionDB(db_path=home / "state.db")
    old_db = server._db
    server._db = db
    session = _session(server, db, "sid-native", "native-row", None, profile_home=home)
    try:
        response = server.handle_request({"id": "rid", "method": "command.dispatch", "params": {
            "session_id": "sid-native", "name": "native-probe", "arg": "on"}})
        assert _result(response)["output"] == "native-row:True"
        response = server.handle_request({"id": "rid", "method": "slash.exec", "params": {
            "session_id": "sid-native", "command": "native-probe status"}})
        assert _result(response)["output"] == "native-row:True" and session["slash_worker"] is None
    finally:
        server._sessions.pop("sid-native", None)
        server._db = old_db
        db.close()
        plugins._reset_plugin_managers_for_tests()
