"""Temporary fallback must survive persistence/rebuild without becoming a model pick.

Provider clients are inert; model selection, SQLite persistence, AIAgent snapshots and
turn admission use production code. No provider request or live profile is used.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tui_gateway import server

PRIMARY = "claude-opus-5-5"
FALLBACK = "gpt-6-astra"


@pytest.fixture(params=["default", "worker"])
def runtime_env(monkeypatch, tmp_path, request):
    from hermes_state import SessionDB
    from hermes_constants import get_hermes_home

    def no_network(*args, **kwargs):
        raise AssertionError("Network forbidden in fallback recovery regression")
    monkeypatch.setattr("socket.socket.connect", no_network)
    monkeypatch.setattr("socket.create_connection", no_network)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_IGNORE_RULES", "1")
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr(server, "_sessions", {})
    (tmp_path / "config.yaml").write_text(
        f"model:\n  default: {PRIMARY}\n  provider: anthropic\n"
        f"fallback_providers:\n  - provider: openai-codex\n    model: {FALLBACK}\n")
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_: [])
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda **_: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", MagicMock())
    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_client", MagicMock())
    monkeypatch.setattr("agent.context_compressor.get_model_context_length", lambda *a, **k: 200000)
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *a, **k: 200000)
    from agent.credential_pool import load_pool
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda *a, **k: None)
    clock = {"wall": 1000.0, "mono": 100.0, "auth_failed": False, "load_pool": load_pool}
    monkeypatch.setattr("time.time", lambda: clock["wall"])
    monkeypatch.setattr("time.monotonic", lambda: clock["mono"])
    resolved = []

    def resolve(**kwargs):
        from hermes_cli.auth import AuthError
        provider = kwargs.get("requested") or "anthropic"
        resolved.append((str(get_hermes_home()), provider))
        if (provider == "anthropic" and clock["auth_failed"]) or (provider == "openai-codex" and clock.get("fallback_auth_failed")):
            if on_failure := clock.get("on_auth_failure"):
                on_failure()
            raise AuthError("test primary unavailable")
        configured = server._load_cfg().get("model", {})
        configured = configured if configured.get("provider") == provider else {}
        return {"provider": provider, "api_key": "test-not-a-secret",
                "base_url": kwargs.get("explicit_base_url") or configured.get("base_url") or "https://example.invalid",
                "api_mode": configured.get("api_mode") or "chat_completions"}

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    launch_db = SessionDB(db_path=tmp_path / "state.db")
    home = None if request.param == "default" else tmp_path / "profiles" / "worker"
    if home:
        home.mkdir(parents=True)
        (home / "config.yaml").write_text((tmp_path / "config.yaml").read_text())
        db = SessionDB(db_path=home / "state.db")
        launch_db.create_session("key", source="desktop", model="launch-canary")
    else:
        db = launch_db
    monkeypatch.setattr(server, "_get_db", lambda: launch_db)
    yield clock, db, resolved, home
    if home:
        assert launch_db.get_session("key")["model"] == "launch-canary"
        db.close()
    launch_db.close()


def _build(home, db, sid, key, **kwargs):
    with server._profile_build_scope(home):
        return server._make_agent(sid, key, session_db=db, **kwargs)


def _admit(monkeypatch, session, *, real_config_sync=False):
    """Drive real admission through the fallback-recovery boundary, stopping before tools."""
    class Stop(Exception):
        pass

    monkeypatch.setattr(server, "_set_session_context", lambda *a, **k: [])
    monkeypatch.setattr(server, "_wire_callbacks", lambda _: None)
    if not real_config_sync:
        monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *a: None)
    for name in ("_apply_pending_model_switch", "_sync_agent_compression_with_config", "_sync_agent_fallback_with_config"):
        monkeypatch.setattr(server, name, lambda *a: None)
    monkeypatch.setattr(server, "_sync_bot_capabilities", lambda *a: (_ for _ in ()).throw(Stop()))
    st = server._TurnRun(agent=session["agent"], one_turn_restore=None,
                         terminal_callback=None, receipt_committed=False)
    try:
        with pytest.raises(Stop):
            server._prepare_turn_input("sid", session, st, "continue", [])
    finally:
        server._release_profile_runtime_scope_tokens(st.scopes)
        if st.scopes.approval is not None:
            from tools.approval_context import reset_current_session_key
            reset_current_session_key(st.scopes.approval)


def _inert_selection(monkeypatch, during_resolution=None):
    """Replace provider preflight only; keep config sync and the switch transaction real."""
    def resolve(**kwargs):
        cfg = server._load_cfg()["model"]
        result = SimpleNamespace(
            success=True, new_model=kwargs["raw_input"], target_provider=kwargs["explicit_provider"],
            api_key="test-not-a-secret", base_url=cfg.get("base_url") or "https://example.invalid",
            api_mode=cfg.get("api_mode") or "chat_completions", warning_message="",
            runtime_capabilities=None)
        if during_resolution:
            during_resolution()
        return result
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", resolve)
    for name in ("_restart_slash_worker", "_persist_live_session_system_prompt",
                 "_append_model_switch_marker", "_emit_session_info", "_emit"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)


def _request_fallback(monkeypatch, agent, home):
    from agent.error_classifier import FailoverReason
    client = MagicMock(base_url="https://fallback.invalid", api_key="test-not-a-secret")
    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", lambda *a, **k: (client, FALLBACK))
    with server._profile_build_scope(home):
        assert agent._try_activate_fallback(reason=FailoverReason.rate_limit)
    assert agent.model == FALLBACK
    assert agent._credential_pool is None  # no separate reset can mask a lost local deadline
    assert agent._rate_limited_until == 160.0


@pytest.mark.parametrize("boundary", ["resume", "rebuild", "resume-secondary-auth"])
@pytest.mark.parametrize("primary", [PRIMARY, "claude-sonnet-5-5"])
def test_request_fallback_recovers_across_boundaries(monkeypatch, runtime_env, boundary, primary):
    clock, db, resolved, home = runtime_env
    override = {"model": primary, "provider": "anthropic"}
    agent = _build(home, db, "sid", "key", model_override=override)
    db.create_session("key", source="desktop", model=primary)
    session = {"agent": agent, "session_key": "key", "model_override": override, "profile_home": home}
    # The core has already activated fallback after a failed request. Its snapshot
    # remains the actual configured/explicit primary, not the displayed fallback.
    agent.model, agent.provider = FALLBACK, "openai-codex"
    agent._fallback_activated = True
    agent._rate_limited_until = 200.0
    agent._cached_system_prompt = "stable fallback prefix"
    # Exercise production finally-path wiring, not just the serializer.
    monkeypatch.setattr(server, "_sessions_quiescent", lambda **_: False)
    st = server._TurnRun(agent=agent, one_turn_restore=None, terminal_callback=None, receipt_committed=False)
    with server._profile_build_scope(home):
        server._release_turn_scopes("sid", session, st)
    row = db.get_session("key")
    assert row["model"] == FALLBACK  # observability remains truthful
    assert "test-not-a-secret" not in row["model_config"]
    # First prove the lost-primary defect after the deadline, without depending on
    # any new helper API. This fails on unchanged Desktop/TUI production code.
    clock.update(wall=1101.0, mono=201.0)
    if boundary.startswith("resume"):
        rebuilt = _build(home, db, "resumed", "key", **server._stored_session_runtime_overrides(row))
    else:
        rebuilt = server._rebuild_session_agent("sid", session)
    assert (rebuilt.model, rebuilt.provider) == (primary, "anthropic")
    assert rebuilt._primary_runtime["model"] == primary

    # Reopen the same saved checkpoint DURING cooldown, then exercise actual
    # turn admission at the deadline. No re-resolution or cache churn while held.
    clock.update(wall=1050.0, mono=150.0)
    expected_fallback = FALLBACK
    if boundary == "resume-secondary-auth":
        clock["fallback_auth_failed"] = True
        expected_fallback = "secondary-fallback"
        monkeypatch.setattr(server, "_load_fallback_model", lambda: [
            {"model": FALLBACK, "provider": "openai-codex"},
            {"model": expected_fallback, "provider": "openrouter"}])
    resumed = _build(home, db, "resumed", "key", **server._stored_session_runtime_overrides(row))
    resumed._cached_system_prompt = "stable fallback prefix"
    reopened = {"agent": resumed, "session_key": "key", "profile_home": home}
    calls = len(resolved)
    _admit(monkeypatch, reopened)
    assert resumed.model == expected_fallback
    assert resumed._cached_system_prompt == "stable fallback prefix"
    assert len(resolved) == calls
    clock.update(wall=1101.0, mono=201.0)
    _admit(monkeypatch, reopened)
    assert (resumed.model, resumed.provider) == (primary, "anthropic")
    assert resumed._primary_runtime["model"] == primary
    assert resumed._fallback_activated is False
    server._persist_live_session_runtime(reopened)
    assert db.get_session("key")["model"] == primary
    assert "fallback_recovery" not in db.get_session("key")["model_config"]
    if home:
        assert {scope for scope, _ in resolved} == {str(home)}
    for item in (agent, rebuilt, resumed):
        item.close()


def test_auth_fallback_retries_primary_without_pinning_or_quota_loop(monkeypatch, runtime_env):
    clock, db, resolved, home = runtime_env
    clock["auth_failed"] = True
    agent = _build(home, db, "sid", "key")
    assert agent.model == FALLBACK
    session = {"agent": agent, "session_key": "key", "profile_home": home}
    calls = len(resolved)
    _admit(monkeypatch, session)
    assert len(resolved) == calls  # bounded retry, not every immediate turn
    clock.update(wall=1100.0, mono=200.0)
    _admit(monkeypatch, session)  # AuthError persists: delay rather than retry every turn.
    assert agent.model == FALLBACK
    calls = len(resolved)
    _admit(monkeypatch, session)
    assert len(resolved) == calls
    clock.update(wall=1200.0, mono=300.0, auth_failed=False)
    _admit(monkeypatch, session)
    assert (agent.model, agent.provider) == (PRIMARY, "anthropic")
    assert agent._primary_runtime["model"] == PRIMARY
    assert agent._fallback_chain == [{"provider": "openai-codex", "model": FALLBACK}]
    assert agent._pending_fallback_notice is None
    # Deliberately selecting the same Astra route has no recovery provenance and
    # must not be "healed" to the profile default, including after persistence.
    manual = _build(home, db, "manual", "manual-key",
                    model_override={"model": FALLBACK, "provider": "openai-codex"})
    db.create_session("manual-key", source="desktop", model=FALLBACK)
    manual_session = {"agent": manual, "session_key": "manual-key", "profile_home": home}
    server._persist_live_session_runtime(manual_session)
    row = db.get_session("manual-key")
    assert "fallback_recovery" not in row["model_config"]
    pinned = _build(home, db, "pinned", "manual-key", **server._stored_session_runtime_overrides(row))
    _admit(monkeypatch, {"agent": pinned, "session_key": "manual-key", "profile_home": home})
    assert (pinned.model, pinned.provider) == (FALLBACK, "openai-codex")
    for item in (agent, manual, pinned):
        item.close()


@pytest.mark.parametrize("boundary", ["resume", "rebuild", "resume-secondary-auth"])
def test_expired_local_deadline_respects_later_persisted_pool_reset(monkeypatch, runtime_env, boundary):
    import json
    from agent.credential_pool import PooledCredential, STATUS_EXHAUSTED

    clock, db, resolved, home = runtime_env
    agent = _build(home, db, "sid", "key")
    db.create_session("key", source="desktop", model=PRIMARY)
    agent.model, agent.provider = FALLBACK, "openai-codex"
    agent._fallback_activated = True
    agent._rate_limited_until = 120.0  # wall deadline 1020, before the pool's 1300
    session = {"agent": agent, "session_key": "key", "profile_home": home}
    with server._profile_build_scope(home):
        server._persist_live_session_runtime(session)
    row = db.get_session("key")
    owner = Path(home or server._hermes_home)
    credential = PooledCredential(
        provider="anthropic", id="isolated-primary", label="test", auth_type="api_key",
        priority=1, source="manual", access_token="test-pool-not-a-secret",
        last_status=STATUS_EXHAUSTED, last_status_at=1000.0,
        last_error_code=429, last_error_reset_at=1300.0)
    (owner / "auth.json").write_text(json.dumps({
        "version": 1, "credential_pool": {"anthropic": [credential.to_dict()]}}))
    monkeypatch.setattr("agent.credential_pool.load_pool", clock["load_pool"])
    clock.update(wall=1100.0, mono=201.0 if boundary == "rebuild" else 5.0)
    # Rebuild shares a monotonic epoch; a cold resume does not.
    if home:
        other = credential.to_dict()
        other["last_error_reset_at"] = 9000.0
        (Path(server._hermes_home) / "auth.json").write_text(json.dumps({
            "version": 1, "credential_pool": {"anthropic": [other]}}))
    resolved.clear()
    expected_fallback = FALLBACK
    if boundary == "resume-secondary-auth":
        clock["fallback_auth_failed"] = True
        expected_fallback = "secondary-fallback"
        monkeypatch.setattr(server, "_load_fallback_model", lambda: [
            {"model": FALLBACK, "provider": "openai-codex"},
            {"model": expected_fallback, "provider": "openrouter"}])
    rebuilt = None
    try:
        if boundary == "rebuild":
            rebuilt = server._rebuild_session_agent("sid", session)
        else:
            with server._profile_build_scope(home):
                overrides = server._stored_session_runtime_overrides(row)
            rebuilt = _build(home, db, "resumed", "key", **overrides)
        assert rebuilt.model == expected_fallback
        assert not any(provider == "anthropic" for _, provider in resolved)
        reopened = {"agent": rebuilt, "session_key": "key", "profile_home": home}
        rebuilt._cached_system_prompt = "held prefix"
        calls = len(resolved)
        _admit(monkeypatch, reopened)
        assert rebuilt.model == expected_fallback
        assert rebuilt._cached_system_prompt == "held prefix"
        assert len(resolved) == calls
        clock.update(wall=1301.0, mono=206.0)
        _admit(monkeypatch, reopened)
        assert (rebuilt.model, rebuilt.provider) == (PRIMARY, "anthropic")
        assert rebuilt._primary_runtime["model"] == PRIMARY
        server._persist_live_session_runtime(reopened)
        assert "fallback_recovery" not in db.get_session("key")["model_config"]
        assert {scope for scope, _ in resolved} == {str(owner)}
    finally:
        with server._profile_build_scope(home):
            agent.close()
            if rebuilt is not None:
                rebuilt.close()


@pytest.mark.parametrize("fallback_kind", ["request", "construction"])
@pytest.mark.parametrize("once", [False, True])
def test_explicit_switch_supersedes_recovery_unless_one_turn(monkeypatch, runtime_env, fallback_kind, once):
    clock, db, resolved, home = runtime_env
    clock["auth_failed"] = fallback_kind == "construction"
    agent = _build(home, db, "sid", "key")
    db.create_session("key", source="desktop", model=PRIMARY)
    if fallback_kind == "request":
        agent.model, agent.provider = FALLBACK, "openai-codex"
        agent._fallback_activated = True
        agent._rate_limited_until = 160.0
    session = {"agent": agent, "session_key": "key", "profile_home": home}
    for name in ("_restart_slash_worker", "_persist_live_session_system_prompt",
                 "_append_model_switch_marker", "_emit_session_info"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    snapshot = server._snapshot_agent_model_runtime(agent) if once else None
    result = SimpleNamespace(new_model="manual-model", target_provider="openrouter",
                             api_key="test-not-a-secret", base_url="https://example.invalid",
                             api_mode="chat_completions")
    try:
        with server._profile_build_scope(home):
            server._commit_agent_switch("sid", session, agent, result, FALLBACK, snapshot)
            assert "fallback_recovery" not in db.get_session("key")["model_config"]
            if once:
                server._restore_agent_model_runtime(agent, session.pop("one_turn_model_restore"))
                assert agent.model == FALLBACK  # do not bypass the local cooldown
                server._persist_live_session_runtime(session)
                assert "fallback_recovery" in db.get_session("key")["model_config"]
        calls = len(resolved)
        _admit(monkeypatch, session)
        assert len(resolved) == calls
        clock.update(wall=1100.0, mono=200.0, auth_failed=False)
        _admit(monkeypatch, session)
        assert agent.model == (PRIMARY if once else "manual-model")
        if once:
            assert {"provider": "openai-codex", "model": FALLBACK} in agent._fallback_chain
    finally:
        with server._profile_build_scope(home):
            agent.close()


@pytest.mark.parametrize("adopt", [True, False])
def test_config_adoption_retires_recovery_through_real_admission(monkeypatch, runtime_env, adopt):
    clock, db, resolved, home = runtime_env
    clock["auth_failed"] = True
    agent = _build(home, db, "sid", "key")
    db.create_session("key", source="desktop", model=PRIMARY)
    session = {"agent": agent, "session_key": "key", "profile_home": home,
               "config_model_seen": (PRIMARY, "anthropic")}
    owner = Path(home or server._hermes_home)
    try:
        with server._profile_build_scope(home):
            server._persist_live_session_runtime(session)
        if adopt:
            (owner / "config.yaml").write_text(f"model:\n  default: {FALLBACK}\n  provider: openai-codex\n")
        clock.update(wall=1100.0, mono=200.0, auth_failed=False)
        _admit(monkeypatch, session, real_config_sync=True)
        assert agent.model == (FALLBACK if adopt else PRIMARY)
        assert agent._tui_fallback_recovery is None
        if adopt:
            # Adoption itself persists, rather than relying on a later completed turn.
            row = db.get_session("key")
            assert row["model"] == FALLBACK
            assert "fallback_recovery" not in row["model_config"]
            calls = len(resolved)
            _admit(monkeypatch, session, real_config_sync=True)
            assert agent.model == FALLBACK and len(resolved) == calls
            with server._profile_build_scope(home):
                overrides = server._stored_session_runtime_overrides(row)
            reopened = _build(home, db, "reopened", "key", **overrides)
            try:
                assert reopened.model == FALLBACK and getattr(reopened, "_tui_fallback_recovery", None) is None
            finally:
                reopened.close()
    finally:
        agent.close()


@pytest.mark.parametrize("boundary", ["cold", "deferred", "eager"])
@pytest.mark.parametrize("intent", ["unchanged", "changed", "explicit", "legacy"])
def test_bot_chat_resume_preserves_only_current_profile_recovery(monkeypatch, runtime_env, boundary, intent):
    import json
    clock, db, resolved, home = runtime_env
    owner = Path(home or server._hermes_home)
    # Launch and owning worker routes deliberately disagree.
    if home:
        (Path(server._hermes_home) / "config.yaml").write_text("model:\n  default: launch-model\n  provider: openrouter\n")
    primary = "claude-sonnet-5-5" if intent == "explicit" else PRIMARY
    agent = _build(home, db, "original", "key", model_override={"model": primary, "provider": "anthropic"})
    metadata = {} if intent == "legacy" else {"follow_profile_config": True}
    if intent == "explicit":
        metadata["composer_override_profile"] = {"model": PRIMARY, "provider": "anthropic"}
    db.create_session("key", source="desktop", model=primary, model_config=metadata)
    db.set_session_title("key", "Bot Chat" if intent == "legacy" else "Canonical renamed chat")
    db.append_message("key", "user", "hi")
    db.append_message("key", "assistant", "hello")
    agent.model, agent.provider = FALLBACK, "openai-codex"
    agent._fallback_activated, agent._rate_limited_until = True, 200.0
    original = {"agent": agent, "session_key": "key", "profile_home": home}
    with server._profile_build_scope(home):
        server._persist_live_session_runtime(original)
    agent.close()
    if intent == "changed":
        (owner / "config.yaml").write_text(f"model:\n  default: {FALLBACK}\n  provider: openai-codex\n")
    monkeypatch.setattr(server, "_profile_home", lambda profile: home)
    for name in ("_enable_gateway_prompts", "_schedule_agent_build", "_schedule_session_cap_enforcement",
                 "_emit", "_wire_session_agent", "_start_session_services", "_schedule_mcp_late_refresh",
                 "_restart_slash_worker", "_persist_live_session_system_prompt", "_append_model_switch_marker",
                 "_emit_session_info"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *a, **k: None)
    monkeypatch.setattr(server, "_default_session_cwd", lambda *a, **k: str(owner))
    monkeypatch.setattr(server, "_session_info", lambda *a, **k: {})
    def hydration(sid, key, handle, *, close_db, **kwargs):
        # Synchronous transcript IO only; no background worker/handle is leaked.
        record = server._sessions[sid]
        record["history"] = handle.get_messages_as_conversation(key)
        record["resume_history_ready"].set()
        if close_db:
            server._release_db(handle)
    monkeypatch.setattr(server, "_schedule_resume_hydration", hydration)
    params = {"session_id": "key", "source": "desktop", "profile": "worker" if home else "default"}
    params.update({"defer_history": True} if boundary == "deferred" else {"eager_build": boundary == "eager"})
    resumed = None
    try:
        response = server.handle_request({"id": "resume", "method": "session.resume", "params": params})
        assert "error" not in response, response
        sid = response["result"]["session_id"]
        session = server._sessions[sid]
        if boundary != "eager":
            with server._profile_build_scope(home):
                kwargs = server._deferred_build_agent_kwargs(session, db)
                resumed = server._make_agent(sid, "key", **kwargs)
                server._attach_built_agent(sid, session, resumed)
        else:
            resumed = session["agent"]
        assert resumed.model == FALLBACK  # no pool reset to mask a lost local deadline
        assert bool(session.get("model_override")) == (intent == "explicit")
        if intent == "changed":
            assert getattr(resumed, "_tui_fallback_recovery", None) is None
        else:
            assert resumed._tui_fallback_recovery["retry_at"] == 1100.0
        calls = len(resolved)
        _admit(monkeypatch, session, real_config_sync=True)
        assert resumed.model == FALLBACK and len(resolved) == calls
        clock.update(wall=1101.0, mono=201.0)
        _admit(monkeypatch, session, real_config_sync=True)
        assert resumed.model == (FALLBACK if intent == "changed" else primary)
        with server._profile_build_scope(home):
            server._persist_live_session_runtime(session)
        assert "fallback_recovery" not in json.loads(db.get_session("key")["model_config"])
        if home:
            assert {scope for scope, _ in resolved} == {str(home)}
        if intent in {"unchanged", "legacy"}:
            # Keep provider preflight inert as well as construction. Config sync,
            # the switch transaction, AIAgent mutation and persistence stay real.
            monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kwargs: SimpleNamespace(
                success=True, new_model=kwargs["raw_input"], target_provider=kwargs["explicit_provider"],
                api_key="test-not-a-secret", base_url="https://example.invalid", api_mode="chat_completions",
                warning_message="", runtime_capabilities=None))
            # A subsequent real config edit must switch, not just update the seen marker.
            (owner / "config.yaml").write_text(f"model:\n  default: {FALLBACK}\n  provider: openai-codex\n")
            _admit(monkeypatch, session, real_config_sync=True)
            assert session["config_model_seen"] == (FALLBACK, "openai-codex")
            assert resumed.model == FALLBACK
            assert not session.get("model_override")
            assert "fallback_recovery" not in db.get_session("key")["model_config"]
    finally:
        if resumed is not None:
            resumed.close()
        server._sessions.clear()


@pytest.mark.parametrize("boundary", ["cold", "deferred", "eager"])
@pytest.mark.parametrize("fallback_kind", ["request", "construction", "adopted-request"])
@pytest.mark.parametrize("edit", ["unchanged", "defaults", "remove", "replace", "legacy", "malformed"])
def test_canonical_resume_preserves_historical_profile_intent(
        monkeypatch, runtime_env, boundary, fallback_kind, edit):
    """R3: production capture -> SQLite -> canonical resume, including edits after ACK."""
    import json
    import hermes_yaml as yaml

    clock, db, resolved, home = runtime_env
    owner = Path(home or server._hermes_home)
    cfg_path = owner / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    if edit != "defaults":
        cfg["model"].update(base_url="https://proxy.invalid/anthropic", api_mode="anthropic_messages")
    cfg["model"]["api_key"] = "profile-secret-canary"
    cfg_path.write_text(yaml.safe_dump(cfg))
    if home:
        (Path(server._hermes_home) / "config.yaml").write_text(
            "model:\n  default: launch-canary\n  provider: openrouter\n")
    clock["auth_failed"] = fallback_kind == "construction"
    if boundary == "cold" and edit == "replace":
        def during_resolution():
            edited = {**cfg, "model": {**cfg["model"], "base_url": "https://replacement.invalid",
                                      "api_mode": "chat_completions"}}
            cfg_path.write_text(yaml.safe_dump(edited))
        clock["on_auth_failure"] = during_resolution
    original = _build(home, db, "original", "key")  # no manually populated recovery
    primary = PRIMARY
    if fallback_kind == "adopted-request":
        primary = "claude-sonnet-5-5"
        cfg["model"]["default"] = primary
        cfg_path.write_text(yaml.safe_dump(cfg))
        _inert_selection(monkeypatch)
        session = {"agent": original, "session_key": "key", "profile_home": home,
                   "config_model_seen": (PRIMARY, "anthropic")}
        _admit(monkeypatch, session, real_config_sync=True)
        assert original.model == original._primary_runtime["model"] == primary
        assert not session.get("model_override")
        _request_fallback(monkeypatch, original, home)
    db.create_session("key", source="desktop", model=primary,
                      model_config={"follow_profile_config": True})
    db.set_session_title("key", "Canonical renamed chat")
    db.append_message("key", "user", "hi")
    db.append_message("key", "assistant", "hello")
    if fallback_kind == "request":
        original.model, original.provider = FALLBACK, "openai-codex"
        original.base_url, original.api_mode = "https://fallback.invalid", "chat_completions"
        original._fallback_activated, original._rate_limited_until = True, 160.0
    if boundary == "cold" and edit == "remove":
        # The serializer must not recapture the edited profile as historical intent.
        edited = {**cfg, "model": {k: v for k, v in cfg["model"].items() if k not in {"base_url", "api_mode"}}}
        cfg_path.write_text(yaml.safe_dump(edited))
    with server._profile_build_scope(home):
        server._persist_live_session_runtime({"agent": original, "session_key": "key", "profile_home": home})
    original.close()
    saved = json.loads(db.get_session("key")["model_config"])
    deadline = saved["fallback_recovery"]["retry_at"]
    intent = saved["fallback_recovery"]["profile_intent"]
    expected_intent = {"model": primary, "provider": "anthropic",
                       "base_url": cfg["model"].get("base_url", ""), "api_mode": cfg["model"].get("api_mode", "")}
    if fallback_kind == "construction" and edit != "defaults":
        assert saved["fallback_recovery"]["primary"]["base_url"] == cfg["model"]["base_url"]
        assert saved["fallback_recovery"]["primary"]["api_mode"] == cfg["model"]["api_mode"]
    assert deadline == 1060.0
    assert "profile-secret-canary" not in json.dumps(saved)
    if edit in {"legacy", "malformed"}:
        state = saved["fallback_recovery"]
        if edit == "legacy":
            state.pop("profile_intent", None)
        else:
            state["profile_intent"] = {"model": PRIMARY, "base_url": ["malformed"]}
        db.patch_session_model_config("key", {"fallback_recovery": state})
    # Remove/replace before reopening for cold/eager, after ACK for deferred.
    def change_config():
        if edit == "remove":
            cfg["model"].pop("base_url")
            cfg["model"].pop("api_mode")
        elif edit == "replace":
            cfg["model"].update(base_url="https://replacement.invalid", api_mode="chat_completions")
        cfg_path.write_text(yaml.safe_dump(cfg))
    if boundary != "deferred":
        change_config()
    clock.update(auth_failed=False, wall=1010.0, mono=10.0)
    resolved.clear()
    monkeypatch.setattr(server, "_profile_home", lambda profile: home)
    for name in ("_enable_gateway_prompts", "_schedule_agent_build", "_schedule_session_cap_enforcement",
                 "_emit", "_wire_session_agent", "_start_session_services", "_schedule_mcp_late_refresh",
                 "_restart_slash_worker", "_persist_live_session_system_prompt", "_append_model_switch_marker",
                 "_emit_session_info", "_maybe_schedule_auto_continue"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    monkeypatch.setattr(server, "_default_session_cwd", lambda *a, **k: str(owner))
    monkeypatch.setattr(server, "_session_info", lambda *a, **k: {})
    def hydration(sid, key, handle, *, close_db, **kwargs):
        record = server._sessions[sid]
        record["history"] = handle.get_messages_as_conversation(key)
        record["resume_history_ready"].set()
        if close_db:
            server._release_db(handle)
    monkeypatch.setattr(server, "_schedule_resume_hydration", hydration)
    params = {"session_id": "key", "source": "desktop", "profile": "worker" if home else "default"}
    params.update({"defer_history": True} if boundary == "deferred" else {"eager_build": boundary == "eager"})
    resumed = None
    try:
        response = server.handle_request({"id": "resume", "method": "session.resume", "params": params})
        assert "error" not in response, response
        sid = response["result"]["session_id"]
        session = server._sessions[sid]
        if boundary == "deferred":
            change_config()
        if boundary != "eager":
            with server._profile_build_scope(home):
                resumed = server._make_agent(sid, "key", **server._deferred_build_agent_kwargs(session, db))
                server._attach_built_agent(sid, session, resumed)
        else:
            resumed = session["agent"]
        unchanged = edit in {"unchanged", "defaults"}
        assert resumed.model == (FALLBACK if unchanged else primary)
        assert not session.get("model_override")  # temporary recovery is never a canonical pin
        assert intent == expected_intent
        if unchanged:
            assert resumed._tui_fallback_recovery["retry_at"] == deadline
            assert not any(provider == "anthropic" for _, provider in resolved)
        else:
            assert getattr(resumed, "_tui_fallback_recovery", None) is None
        clock.update(wall=deadline + 1, mono=80.0)
        _admit(monkeypatch, session, real_config_sync=True)
        assert resumed.model == primary
        assert resumed.base_url == (cfg["model"].get("base_url") or "https://example.invalid")
        assert resumed.api_mode == (cfg["model"].get("api_mode") or "chat_completions")
        with server._profile_build_scope(home):
            server._persist_live_session_runtime(session)
        assert "fallback_recovery" not in json.loads(db.get_session("key")["model_config"])
        assert not session.get("model_override")
        assert {scope for scope, _ in resolved} == {str(owner)}
    finally:
        if resumed is not None:
            resumed.close()
        server._sessions.clear()


@pytest.mark.parametrize("selection", [
    "resolve-failure", "swap-failure", "capture-before-resolution", "no-op",
    "request-adoption", "construction-adoption", "manual", "once-primary", "once-fallback",
    "request-swap-failure", "construction-swap-failure", "once-restore-failure",
])
def test_selection_intent_commits_and_restores_with_its_primary(monkeypatch, runtime_env, selection):
    """Only successful selections publish intent; one-turn detours restore its history."""
    import copy
    import hermes_yaml as yaml
    from tui_gateway.fallback_recovery import profile_intent, recovery_state

    clock, db, resolved, home = runtime_env
    owner = Path(home or server._hermes_home)
    cfg_path = owner / "config.yaml"
    original_cfg = yaml.safe_load(cfg_path.read_text())
    clock["auth_failed"] = selection.startswith("construction")
    agent = _build(home, db, "original", "key")
    db.create_session("key", source="desktop", model=PRIMARY,
                      model_config={"follow_profile_config": True})
    session = {"agent": agent, "session_key": "key", "profile_home": home,
               "config_model_seen": (PRIMARY, "anthropic")}
    old_intent = copy.deepcopy(agent._tui_profile_intent)
    _inert_selection(monkeypatch)
    cfg = copy.deepcopy(original_cfg)
    cfg["model"]["default"] = "claude-sonnet-5-5"
    if selection in {"request-adoption", "request-swap-failure", "once-fallback"}:
        _request_fallback(monkeypatch, agent, home)
    old_recovery = recovery_state(agent)
    old_primary = agent._primary_runtime
    old_model = agent.model
    if selection in {"request-adoption", "construction-adoption"}:
        cfg["model"] = {"default": FALLBACK, "provider": "openai-codex"}
    elif selection == "no-op":
        cfg = original_cfg
        session.pop("config_model_seen")
        agent._tui_profile_intent = None  # first sync must publish this successful no-op adoption
    cfg_path.write_text(yaml.safe_dump(cfg))

    def fail_resolution():
        # Even a concurrent edit must not replace the old provenance on failure.
        cfg_path.write_text(yaml.safe_dump(original_cfg))
        raise RuntimeError("inert provider refusal")

    if selection == "resolve-failure":
        _inert_selection(monkeypatch, fail_resolution)
    elif selection.endswith("swap-failure"):
        # Exercise the real switch rollback, not a mocked successful switch.
        def refuse_client(*args, **kwargs):
            raise RuntimeError("inert client failure")
        monkeypatch.setattr("agent.agent_runtime_helpers._build_switched_client", refuse_client)
    elif selection == "capture-before-resolution":
        _inert_selection(monkeypatch, lambda: cfg_path.write_text(yaml.safe_dump(original_cfg)))
    try:
        with server._profile_build_scope(home):
            if selection.startswith("once") or selection == "manual":
                flag = "--once" if selection.startswith("once") else "--session"
                server._apply_model_switch("sid", session, f"manual-model --provider anthropic {flag}",
                                           confirm_expensive_model=True)
                assert agent._tui_profile_intent is None  # a manual selection is not profile adoption
                if selection == "once-restore-failure":
                    monkeypatch.setattr(agent, "_restore_primary_runtime", lambda: False)
                    def refuse_restore(*args, **kwargs):
                        raise RuntimeError("inert restore failure")
                    monkeypatch.setattr("agent.agent_runtime_helpers._build_switched_client", refuse_restore)
                    with pytest.raises(RuntimeError, match="inert restore failure"):
                        server._restore_agent_model_runtime(agent, session.pop("one_turn_model_restore"))
                    assert agent._tui_profile_intent is None
                    assert agent.model == "manual-model"
                elif selection.startswith("once"):
                    server._restore_agent_model_runtime(agent, session.pop("one_turn_model_restore"))
                    assert agent._tui_profile_intent == old_intent
                    assert recovery_state(agent) == old_recovery
                    assert agent.model == old_model
                else:
                    assert session["model_override"]["model"] == "manual-model"
            else:
                server._sync_agent_model_with_config("sid", session)
                if selection.endswith("failure"):
                    assert agent._tui_profile_intent == old_intent
                    assert agent._primary_runtime is old_primary
                    assert agent.model == old_model
                    assert recovery_state(agent) == old_recovery
                else:
                    assert agent._tui_profile_intent == profile_intent(cfg)
                    assert agent._primary_runtime["model"] == cfg["model"]["default"]
                    assert agent.model == cfg["model"]["default"]
                    assert recovery_state(agent) is None
                    assert not session.get("model_override")
            server._persist_live_session_runtime(session)
        if selection == "once-primary":
            # A later request fallback must still carry the restored PRIMARY's intent.
            _request_fallback(monkeypatch, agent, home)
            assert recovery_state(agent)["profile_intent"] == old_intent
    finally:
        agent.close()
