"""Reasoning-effort session scoping in the TUI gateway (desktop backend).

Covers the "desktop reverts thinking to medium after one turn" report:

1. ``_session_info`` must report ``reasoning_effort: "none"`` when reasoning
   is disabled — reporting ``""`` (indistinguishable from "unset") made the
   desktop adopt the empty value after the first turn, wiping its sticky
   "thinking off" pick so every later chat reverted to the default effort.

2. ``config.set key=reasoning`` with a live session must be session-scoped:
   it must NOT rewrite the global ``agent.reasoning_effort`` in config.yaml
   (the desktop model menu applies a per-model preset on every selection,
   which was silently clobbering the user's configured value), and it must
   land on ``create_reasoning_override`` so lazily-built sessions (agent not
   constructed until the first prompt) don't drop the change.

3. ``_load_reasoning_config`` must honor a YAML boolean False
   (``reasoning_effort: false`` / ``off`` / ``no``) as thinking-disabled.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tui_gateway.server as server
from tui_gateway.server import _session_info


@pytest.fixture
def reasoning_factory(tmp_path, monkeypatch):
    """Real profile loader, factory and DB; no provider request is needed to build."""
    from hermes_state import SessionDB
    from hermes_yaml import safe_dump

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_IGNORE_RULES", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(safe_dump({
        "model": {"default": "review-model", "provider": "custom:fixture",
                  "base_url": "http://127.0.0.1:1/v1"},
        "agent": {"reasoning_effort": "medium",
                  "reasoning_overrides": {"review-model": "high"},
                  "adaptive_reasoning": {"enabled": True, "min_effort": "low"}},
        "custom_providers": [{"name": "fixture", "base_url": "http://127.0.0.1:1/v1",
                              "api_key": "test-only", "model": "review-model"}],
        "toolsets": {"desktop": []},
    }))
    db = SessionDB(db_path=tmp_path / "fixture.db")
    session = {
        "agent": None, "session_key": "reasoning-fixture", "source": "desktop",
        "model_override": {"model": "review-model", "provider": "custom:fixture",
                           "base_url": "http://127.0.0.1:1/v1",
                           "api_key": "test-only", "api_mode": "chat_completions"},
    }
    yield session, db
    db.close()


@pytest.mark.parametrize("level,explicit", [("high", True), ("medium", True)])
def test_explicit_reasoning_survives_deferred_and_stored_build(reasoning_factory, monkeypatch, level, explicit):
    import io
    import threading
    import json
    from tui_gateway.compute_host import ComputeHost
    from agent.adaptive_reasoning import adaptive_reasoning_turn

    session, db = reasoning_factory
    sid = session["session_key"]
    response = server._set_reasoning("pin", {"scope": "session"}, "reasoning", level, session)
    assert "error" not in response
    agent = server._make_agent(sid, sid, **server._deferred_build_agent_kwargs(session, db))
    assert agent.reasoning_user_override is explicit
    with adaptive_reasoning_turn(agent, "thanks"):
        assert agent.reasoning_config["effort"] == level
    # Persist and read the actual row, then restore through the same resume factory seam.
    db.create_session(sid, source="desktop", model=agent.model)
    session["agent"] = agent
    server._persist_live_session_runtime(session)
    row = db.get_session(sid)
    assert json.loads(row["model_config"])["reasoning_user_override"] is explicit
    restored = server._stored_session_runtime_overrides(row)
    rebuilt = server._make_agent(sid, sid, session_db=db, **restored)
    assert rebuilt.reasoning_user_override is explicit
    with adaptive_reasoning_turn(rebuilt, "thanks"):
        assert rebuilt.reasoning_config["effort"] == level

    # Cold resume -> compute transport -> real factory must retain the same pin.
    session.update(agent=None, resume_runtime_overrides=restored, history_lock=threading.Lock())
    session.pop("reasoning_user_override")
    session.pop("create_reasoning_override")
    assert server._overrides_have_routable_provider(restored)
    frame = server._compute_host_turn_frame("turn", sid, session, "thanks")
    assert frame["reasoning_user_override"] is True
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_start_session_services", lambda *a: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *a: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)
    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    try:
        hosted = host._build_server_session(server, frame, sid)
        assert hosted["agent"].reasoning_user_override is True
        replacement = server._rebuild_session_agent(sid, hosted)
        with adaptive_reasoning_turn(replacement, "thanks"):
            assert replacement.reasoning_config["effort"] == level
        # /new also discards the cold-resume snapshot so another deferred build
        # cannot resurrect the old pin after the live agent was reset.
        hosted["resume_runtime_overrides"] = restored.copy()
        server._reset_session_agent(sid, hosted)
        assert not hosted["agent"].reasoning_user_override
        assert "reasoning_user_override" not in hosted
        later = server._make_agent(sid, sid, **server._deferred_build_agent_kwargs(hosted, db))
        assert not later.reasoning_user_override
    finally:
        server._sessions.pop(sid, None)
        host.close()

    # A session pick after lazy resume beats the stored snapshot; --global clears both.
    server._set_reasoning("repin", {"scope": "session"}, "reasoning", "xhigh", session)
    fresh = server._make_agent(sid, sid, **server._deferred_build_agent_kwargs(session, db))
    assert fresh.reasoning_config["effort"] == "xhigh" and fresh.reasoning_user_override
    response = server._set_reasoning("global", {"scope": "global"}, "reasoning", "medium", session)
    assert "error" not in response
    fresh = server._make_agent(sid, sid, **server._deferred_build_agent_kwargs(session, db))
    assert not fresh.reasoning_user_override


@pytest.mark.parametrize("level,explicit", [("high", True), ("medium", False), (None, False)])
def test_desktop_factory_compares_composer_global_not_model_default(reasoning_factory, level, explicit):
    from hermes_constants import parse_reasoning_effort
    from agent.adaptive_reasoning import adaptive_reasoning_turn

    session, db = reasoning_factory
    if level:
        session["create_reasoning_override"] = parse_reasoning_effort(level)
    agent = server._make_agent("desktop", "desktop", **server._deferred_build_agent_kwargs(session, db))
    assert agent.reasoning_user_override is explicit
    if level:
        _, row_config = server._workdir_row_model_config(session)
        assert row_config["reasoning_user_override"] is explicit
    with adaptive_reasoning_turn(agent, "thanks"):
        assert agent.reasoning_config["effort"] == (level if explicit else "low")
    # False provenance must survive even though a model baseline differs from the global default.
    stored = server._runtime_model_config(agent)
    restored = server._stored_session_runtime_overrides({"model_config": stored, "model": agent.model})
    again = server._make_agent("desktop", "desktop", session_db=db, **restored)
    assert again.reasoning_user_override is explicit


@pytest.mark.parametrize("recorded,explicit", [(False, False), (True, True), (None, True)])
def test_rebuild_carries_reasoning_provenance_with_the_pin(reasoning_factory, monkeypatch, recorded, explicit):
    """A rebuild carries the session's effort pin; its recorded provenance must ride along, or a
    snapshot restored as not-a-user-pick (an old row on a compute host) is re-inferred as one."""
    from hermes_constants import parse_reasoning_effort
    from agent.adaptive_reasoning import adaptive_reasoning_turn

    session, db = reasoning_factory
    monkeypatch.setattr(server, "_get_db", lambda: db)
    session["create_reasoning_override"] = parse_reasoning_effort("xhigh")
    if recorded is not None:
        session["reasoning_user_override"] = recorded
    rebuilt = server._rebuild_session_agent(session["session_key"], session)
    assert rebuilt.reasoning_config["effort"] == "xhigh"
    assert rebuilt.reasoning_user_override is explicit
    with adaptive_reasoning_turn(rebuilt, "thanks"):
        assert rebuilt.reasoning_config["effort"] == ("xhigh" if explicit else "low")


@pytest.mark.parametrize("reasoning", [None, {}, {"enabled": False}, {"effort": "high"}])
@pytest.mark.parametrize("pinned", [False, True])
def test_runtime_reasoning_provenance_requires_a_current_config(reasoning, pinned):
    agent = _agent(reasoning)
    agent.reasoning_user_override = pinned
    stored = server._runtime_model_config(agent, {
        "reasoning_config": {"effort": "xhigh"}, "reasoning_user_override": True,
    })
    restored = server._stored_session_runtime_overrides({"model_config": stored})
    if reasoning is None:
        assert "reasoning_config" not in stored
        assert "reasoning_user_override" not in stored
        assert "reasoning_config_override" not in restored
        assert "reasoning_user_override" not in restored
    else:
        assert stored["reasoning_config"] == reasoning
        assert stored["reasoning_user_override"] is pinned
        assert restored["reasoning_config_override"] == reasoning
        assert restored["reasoning_user_override"] is pinned


def _agent(reasoning_config, **overrides):
    return SimpleNamespace(**{
        "reasoning_config": reasoning_config,
        "service_tier": None,
        "model": "glm-5",
        "provider": "zai",
        "session_id": "sess-key",
        **overrides,
    })


class TestSessionInfoReasoningEffort:
    """Disabled reasoning must be reported as 'none', never ''."""

    def test_disabled_reports_none(self) -> None:
        info = _session_info(_agent({"enabled": False}))
        assert info["reasoning_effort"] == "none"

    def test_enabled_reports_effort(self) -> None:
        info = _session_info(_agent({"enabled": True, "effort": "high"}))
        assert info["reasoning_effort"] == "high"

    def test_unset_reports_empty(self) -> None:
        info = _session_info(_agent(None))
        assert info["reasoning_effort"] == ""
        assert info["reasoning_effort_wire"] == ""

    def test_wire_level_is_what_the_route_actually_sends(self) -> None:
        """`ultra` is a Hermes-internal step (#61634): the route clamps it, and the Desktop must be able to
        say so ("ultra sends max on this route") instead of presenting Ultra as a distinct wire level."""
        info = _session_info(_agent({"enabled": True, "effort": "ultra"}))
        assert info["reasoning_effort"] == "ultra"
        assert info["reasoning_effort_wire"] == "max"
        # Verbatim levels report themselves, so clients only annotate a real clamp.
        assert _session_info(_agent({"enabled": True, "effort": "high"}))["reasoning_effort_wire"] == "high"
        assert _session_info(_agent({"enabled": False}))["reasoning_effort_wire"] == ""
        # On the Codex app-server ``ultra`` is codex's own harness mode, sent verbatim.
        app_server = _agent({"enabled": True, "effort": "ultra"}, provider="openai-codex",
                            model="gpt-5.6-sol", api_mode="codex_app_server")
        assert _session_info(app_server)["reasoning_effort_wire"] == "ultra"


class TestConfigSetReasoningSessionScope:
    """Session-targeted reasoning changes must not touch global config."""

    def _dispatch(self, params: dict) -> dict:
        handler = server._methods["config.set"]
        return handler("rid-1", params)

    def test_session_scoped_set_skips_global_write(self) -> None:
        agent = _agent(None)
        session = {"session_key": "k1", "agent": agent}
        with patch.dict(server._sessions, {"s1": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key, \
                patch.object(server, "_persist_live_session_runtime"), \
                patch.object(server, "_emit"):
            resp = self._dispatch(
                {"key": "reasoning", "session_id": "s1", "value": "none"}
            )
        assert resp["result"]["value"] == "none"
        assert agent.reasoning_config == {"enabled": False}
        write_key.assert_not_called()


    def test_no_session_persists_globally(self) -> None:
        with patch.object(server, "_write_config_key") as write_key:
            resp = self._dispatch({"key": "reasoning", "value": "low"})
        assert resp["result"]["value"] == "low"
        write_key.assert_called_once_with("agent.reasoning_effort", "low")

    def test_unknown_value_rejected(self) -> None:
        resp = self._dispatch({"key": "reasoning", "value": "bogus"})
        assert "error" in resp


class TestLoadReasoningConfigYamlBoolean:
    """YAML `reasoning_effort: false` means disabled, not default."""

    def test_boolean_false_disables(self) -> None:
        with patch.object(
            server, "_load_cfg", return_value={"agent": {"reasoning_effort": False}}
        ):
            assert server._load_reasoning_config() == {"enabled": False}

    def test_string_false_disables(self) -> None:
        with patch.object(
            server, "_load_cfg", return_value={"agent": {"reasoning_effort": "false"}}
        ):
            assert server._load_reasoning_config() == {"enabled": False}


class TestSessionNoneReachesDeepSeekWire:
    """Desktop ``config.set value=none`` must disable DeepSeek V4 thinking.

    ``{effort: "none"}`` without ``enabled: False`` is what ``_session_info``
    already reports as Off; the profile used to ignore it and send enabled.
    """

    def test_session_info_effort_none_without_enabled_reports_none(self) -> None:
        info = _session_info(_agent({"effort": "none"}))
        assert info["reasoning_effort"] == "none"

    def test_config_set_none_on_lazy_session_pins_disabled_override(self) -> None:
        session = {"session_key": "k-lazy", "agent": None}
        with patch.dict(server._sessions, {"s-lazy": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key:
            resp = server._methods["config.set"](
                "rid-1", {"key": "reasoning", "session_id": "s-lazy", "value": "none"}
            )
        assert resp["result"]["value"] == "none"
        assert session["create_reasoning_override"] == {"enabled": False}
        write_key.assert_not_called()
        kw = server._deferred_build_agent_kwargs(session, session_db=None)
        assert kw["reasoning_config_override"] == {"enabled": False}

    def test_effort_none_override_emits_thinking_disabled(self) -> None:
        import model_tools  # noqa: F401
        import providers
        from agent.transports.chat_completions import ChatCompletionsTransport

        profile = providers.get_provider_profile("deepseek")
        kwargs = ChatCompletionsTransport().build_kwargs(
            model="deepseek-v4.1-flash-expires-on-0910",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=profile,
            reasoning_config={"effort": "none"},
            base_url="https://api.deepseek.com/v1",
            provider_name="deepseek",
        )
        assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
        assert "reasoning_effort" not in kwargs


@pytest.mark.parametrize("launch_effort,worker_effort,picked,explicit", [
    ("medium", "high", "high", False),  # mirrors the WORKER default -> inherited, not a pick
    ("high", "medium", "high", True),   # distinct from the worker default -> explicit pick
])
def test_prebuild_row_infers_reasoning_intent_from_the_sessions_own_profile(
        tmp_path, monkeypatch, launch_effort, worker_effort, picked, explicit):
    """A secondary-profile session's first row (written before its agent exists) must judge the
    composer effort against THAT profile's default, not the launch profile's; resume honors the row."""
    import hashlib
    import json
    from hermes_constants import get_hermes_home_override, parse_reasoning_effort
    from hermes_state import SessionDB
    from hermes_yaml import safe_dump

    launch, worker = tmp_path / "launch", tmp_path / "profiles" / "worker"
    for home, effort in ((launch, launch_effort), (worker, worker_effort)):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(safe_dump({
            "model": {"default": f"{home.name}-model"},
            "agent": {"reasoning_effort": effort, "adaptive_reasoning": {"enabled": True}}}))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_hermes_home", launch)
    for name in ("_cfg_cache", "_cfg_sig", "_cfg_path"):
        monkeypatch.setattr(server, name, None)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (launch / "config.yaml", worker / "config.yaml")}
    launch_db = SessionDB(db_path=launch / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: launch_db)
    session = {"agent": None, "session_key": "worker-row", "source": "desktop", "profile_home": str(worker),
               "create_reasoning_override": parse_reasoning_effort(picked)}
    override_before = get_hermes_home_override()

    _, row_config = server._workdir_row_model_config(session)
    assert row_config["reasoning_user_override"] is explicit
    assert server._ensure_session_db_row(session) is True

    assert get_hermes_home_override() == override_before, "no profile scope leaks out of row creation"
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before} == before, "no config writes"
    assert launch_db.get_session("worker-row") is None, "the row never lands in the launch profile db"
    launch_db.close()
    worker_db = SessionDB(db_path=worker / "state.db")
    try:
        row = worker_db.get_session("worker-row")
        assert json.loads(row["model_config"])["reasoning_user_override"] is explicit
        restored = server._stored_session_runtime_overrides(row)
        assert restored["reasoning_user_override"] is explicit
        assert restored["reasoning_config_override"] == parse_reasoning_effort(picked)
    finally:
        worker_db.close()


def test_prebuild_row_recorded_intent_wins_over_profile_inference(tmp_path, monkeypatch):
    """An explicit recorded pick (config.set scope=session) is never re-inferred."""
    from hermes_constants import parse_reasoning_effort

    monkeypatch.setattr(server, "_explicit_reasoning_override",
                        lambda *a: (_ for _ in ()).throw(AssertionError("no inference")))
    monkeypatch.setattr(server, "_session_default_route", lambda session: ("m", ""))
    session = {"agent": None, "session_key": "k", "profile_home": str(tmp_path),
               "create_reasoning_override": parse_reasoning_effort("medium"), "reasoning_user_override": True}
    assert server._workdir_row_model_config(session)[1]["reasoning_user_override"] is True



@pytest.mark.parametrize("persist_global,one_turn,pinned", [
    (False, False, True), (True, False, False), (False, True, True)])
def test_tui_model_switch_reasoning_carries_pick_provenance(monkeypatch, persist_global, one_turn, pinned):
    """`/model X --reasoning L` in the TUI: a session/--once pick pins the effort against adaptive
    adjustment; --global is a new baseline; --once hands the prior provenance back."""
    from agent.adaptive_reasoning import adaptive_reasoning_turn

    monkeypatch.setattr(server, "_write_config_key", lambda *a: None)
    monkeypatch.setattr(server, "_persist_live_session_runtime", lambda *a: None)
    monkeypatch.setattr(server, "_emit_session_info", lambda *a: None)
    agent = SimpleNamespace(
        reasoning_config={"enabled": True, "effort": "medium"}, reasoning_user_override=False,
        adaptive_reasoning={"enabled": True, "max_effort": "xhigh", "min_effort": "low"},
        _adaptive_prev_effort=None, _adaptive_last_notified_effort=None, notice_callback=None,
        platform="tui", model="m", provider="p", base_url="", api_key="", api_mode="")
    session = {"agent": agent, "session_key": "k", "reasoning_user_override": True} if persist_global else {
        "agent": agent, "session_key": "k"}
    snapshot = server._snapshot_agent_model_runtime(agent) if one_turn else None
    server._apply_switch_reasoning("sid", session, agent, "medium", persist_global=persist_global, one_turn=one_turn)
    assert agent.reasoning_user_override is pinned
    if not one_turn:
        assert session.get("reasoning_user_override") is (True if pinned else None)
    with adaptive_reasoning_turn(agent, "Why does the gateway keep failing after I restart it? error: refused"):
        assert agent.reasoning_config["effort"] == ("medium" if pinned else "high")
    if one_turn:
        agent.switch_model = lambda **kw: None
        server._restore_agent_model_runtime(agent, snapshot)
        assert agent.reasoning_user_override is False


@pytest.mark.parametrize("setter", ["reasoning", "model"])
def test_global_pick_clears_lazy_resume_pin(monkeypatch, tmp_path, setter):
    """A --global effort (via /reasoning or /model ... --reasoning) is a new baseline: a lazily resumed
    session's stored pin must not resurrect the old effort/provenance at the deferred build."""
    from hermes_cli import model_switch
    from hermes_constants import parse_reasoning_effort

    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "model:\n  default: old-model\n  provider: openai\nagent:\n  reasoning_effort: medium\n")
    stale = {"model_override": {"model": "old-model", "provider": "openai"}, "provider_override": "openai",
             "reasoning_config_override": parse_reasoning_effort("low"), "reasoning_user_override": True}
    session = server._deferred_session_record("probe", cols=80, cwd=str(tmp_path), history=[], lease=None,
                                              resume_runtime_overrides=stale, model_override=stale["model_override"])
    monkeypatch.setitem(server._sessions, "probe", session)
    writes = []
    monkeypatch.setattr(server, "_write_config_key", lambda *a: writes.append(a))
    monkeypatch.setattr(server, "_current_model_runtime",
                        lambda *a: ("openai", "old-model", "https://example.invalid/v1", "test"))
    monkeypatch.setattr(server, "_expensive_model_confirm", lambda *a, **k: None)
    monkeypatch.setattr(model_switch, "persist_model_selection", lambda *a: None)
    monkeypatch.setattr(model_switch, "switch_model", lambda **k: model_switch.ModelSwitchResult(
        success=True, new_model="old-model", target_provider="openai", api_mode="chat_completions"))
    if setter == "model":
        params = {"session_id": "probe", "key": "model", "confirm_expensive_model": True,
                  "value": "old-model --provider openai --reasoning high --global"}
    else:
        params = {"session_id": "probe", "key": "reasoning", "value": "high", "scope": "global"}
    response = server._methods["config.set"]("rid", params)
    assert "error" not in response, response
    assert ("agent.reasoning_effort", "high") in writes
    built = server._deferred_build_agent_kwargs(session, None)
    assert built.get("reasoning_user_override", False) is False
    assert "reasoning_config_override" not in built


def test_config_set_reasoning_marks_user_override_for_adaptive(tmp_path, monkeypatch):
    """A session-scoped effort pick suppresses adaptive escalation on the live
    agent; a global write is a new baseline and re-enables it."""
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("agent:\n  reasoning_effort: medium\n", encoding="utf-8")
    agent = SimpleNamespace(reasoning_config=None, reasoning_user_override=False)
    monkeypatch.setitem(server._sessions, "sid", {
        "agent": agent, "session_key": "session-key", "history": [], "history_lock": threading.Lock(),
        "history_version": 0, "running": False, "attached_images": [], "image_counter": 0, "cols": 80,
        "slash_worker": None, "show_reasoning": False, "tool_progress_mode": "all"})

    server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"session_id": "sid", "key": "reasoning", "value": "low"},
        }
    )
    assert agent.reasoning_user_override is True

    server.handle_request(
        {
            "id": "2",
            "method": "config.set",
            "params": {
                "session_id": "sid",
                "key": "reasoning",
                "value": "high",
                "scope": "global",
            },
        }
    )
    assert agent.reasoning_user_override is False


def test_explicit_reasoning_override_ignores_profile_seeded_default():
    """The Desktop composer ships its seeded (profile-default) effort on
    session.create — an override merely mirroring the profile config is
    inherited state and must NOT suppress adaptive escalation. Only a
    distinct per-session pick counts as explicit."""
    medium = {"enabled": True, "effort": "medium"}
    high = {"enabled": True, "effort": "high"}

    # No override at all → inherited.
    assert server._explicit_reasoning_override(None, medium) is False
    # The defect scenario: seeded composer value equals the profile config.
    assert server._explicit_reasoning_override(dict(medium), medium) is False
    assert server._explicit_reasoning_override(dict(high), high) is False
    # An unset profile default resolves as the backend fallback (medium).
    assert server._explicit_reasoning_override(dict(medium), None) is False
    # Distinct picks are explicit overrides.
    assert server._explicit_reasoning_override(dict(high), medium) is True
    assert server._explicit_reasoning_override(dict(high), None) is True
    assert server._explicit_reasoning_override({"enabled": False}, medium) is True
    # Thinking-disabled profile: the seeded 'none' mirrors it; re-enabling is
    # explicit.
    assert server._explicit_reasoning_override({"enabled": False}, {"enabled": False}) is False
    assert server._explicit_reasoning_override(dict(medium), {"enabled": False}) is True
