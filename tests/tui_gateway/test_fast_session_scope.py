"""Fast-mode (service tier) session scoping in the TUI gateway (desktop backend).

Sibling of test_reasoning_session_scope.py — the ``reasoning`` key was made
session-scoped when a session is targeted, but ``fast`` kept writing the
global ``agent.service_tier`` to config.yaml on every call. The desktop's
per-model presets call ``config.set key=fast`` on every model selection, so
toggling fast in ONE session silently flipped the tier for every other
session, profile, CLI, and gateway build ("switch one session, switches
everywhere").

Contract under test:

1. ``config.set key=fast`` with a session must NOT write config.yaml; it pins
   ``create_service_tier_override`` ("priority" / "" for explicit normal) so
   lazily-built sessions and rebuilds keep the choice.
2. Without a session it persists globally, unchanged.
3. ``config.get key=fast`` must read a pre-build session's pin.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import tui_gateway.server as server

FAST_OVERRIDES = {"service_tier": "priority"}


def _agent(service_tier=None):
    return SimpleNamespace(
        reasoning_config=None,
        service_tier=service_tier,
        request_overrides={},
        model="gpt-6",
        provider="openai",
        session_id="sess-key",
    )


def _set(params: dict) -> dict:
    return server._methods["config.set"]("rid-1", params)


def _get(params: dict) -> dict:
    return server._methods["config.get"]("rid-1", params)


class TestConfigSetFastSessionScope:
    """Session-targeted fast changes must never touch global config."""

    def test_session_scoped_fast_skips_global_write(self) -> None:
        agent = _agent()
        session = {"session_key": "k1", "agent": agent}
        with patch.dict(server._sessions, {"s1": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key, \
                patch.object(server, "_persist_live_session_runtime"), \
                patch.object(server, "_emit"), \
                patch(
                    "hermes_cli.models.resolve_fast_mode_overrides",
                    return_value=FAST_OVERRIDES,
                ):
            resp = _set({"key": "fast", "session_id": "s1", "value": "fast"})
        assert resp["result"]["value"] == "fast"
        assert agent.service_tier == "priority"
        assert session["create_service_tier_override"] == "priority"
        write_key.assert_not_called()


    def test_lazy_session_pins_create_override(self) -> None:
        """A pre-build (agent=None) session must keep the change for the
        deferred agent build instead of dropping it."""
        session = {
            "session_key": "k3",
            "agent": None,
            "model_override": {"model": "gpt-6", "provider": "openai"},
        }
        with patch.dict(server._sessions, {"s3": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key, \
                patch(
                    "hermes_cli.models.resolve_fast_mode_overrides",
                    return_value=FAST_OVERRIDES,
                ):
            resp = _set({"key": "fast", "session_id": "s3", "value": "fast"})
        assert resp["result"]["value"] == "fast"
        assert session["create_service_tier_override"] == "priority"
        write_key.assert_not_called()


    def test_toggle_flips_prebuild_pin(self) -> None:
        """An empty value toggles from the session's pin, not the global."""
        session = {
            "session_key": "k5",
            "agent": None,
            "create_service_tier_override": "priority",
        }
        with patch.dict(server._sessions, {"s5": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key:
            resp = _set({"key": "fast", "session_id": "s5", "value": ""})
        assert resp["result"]["value"] == "normal"
        assert session["create_service_tier_override"] == ""
        write_key.assert_not_called()

    def test_no_session_persists_globally(self) -> None:
        with patch.object(server, "_write_config_key") as write_key:
            resp = _set({"key": "fast", "value": "normal"})
        assert resp["result"]["value"] == "normal"
        write_key.assert_called_once_with("agent.service_tier", "normal")

    def test_session_flex_pins_without_global_write(self) -> None:
        agent = _agent()
        agent.provider = "openrouter"
        agent.base_url = "https://openrouter.ai/api/v1"
        session = {"session_key": "k-flex", "agent": agent}
        with patch.dict(server._sessions, {"s-flex": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key, \
                patch.object(server, "_persist_live_session_runtime"), \
                patch.object(server, "_emit"):
            resp = _set({"key": "fast", "session_id": "s-flex", "value": "flex"})
        assert resp["result"]["value"] == "flex"
        assert agent.service_tier == "flex"
        assert agent._service_tier_session_pinned is True
        assert session["create_service_tier_override"] == "flex"
        write_key.assert_not_called()

    def test_openrouter_prebuild_fast_without_first_party_model(self) -> None:
        session = {
            "session_key": "k-or",
            "agent": None,
            "model_override": {
                "model": "meta-llama/llama-3.1-8b-instruct",
                "provider": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
            },
        }
        with patch.dict(server._sessions, {"s-or": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key:
            resp = _set({"key": "fast", "session_id": "s-or", "value": "fast"})
        assert resp["result"]["value"] == "fast"
        assert session["create_service_tier_override"] == "priority"
        write_key.assert_not_called()


class TestConfigGetFastSessionScope:
    def test_reads_prebuild_pin(self) -> None:
        session = {
            "session_key": "k6",
            "agent": None,
            "create_service_tier_override": "priority",
        }
        with patch.dict(server._sessions, {"s6": session}, clear=False):
            resp = _get({"key": "fast", "session_id": "s6"})
        assert resp["result"]["value"] == "fast"


    def test_falls_back_to_global(self) -> None:
        with patch.object(server, "_load_service_tier", return_value="priority"):
            resp = _get({"key": "fast"})
        assert resp["result"]["value"] == "fast"

    def test_unpinned_status_and_toggle_use_per_model_tier(self) -> None:
        """Unpinned per-model flex: status shows flex; toggle turns it off."""
        from agent import fast_mode

        agent = _agent(service_tier=None)
        agent.model = "openai/gpt-5"
        agent.provider = "openrouter"
        agent.base_url = "https://openrouter.ai/api/v1"
        agent._service_tier_session_pinned = False
        session = {"session_key": "k-pm", "agent": agent}
        cfg = {
            "agent": {
                "service_tier": "",
                "service_tier_overrides": {"openai/gpt-5": "flex"},
            }
        }
        with patch.dict(server._sessions, {"s-pm": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key, \
                patch.object(server, "_persist_live_session_runtime"), \
                patch.object(server, "_emit"), \
                patch("hermes_cli.config.load_config_readonly", return_value=cfg):
            assert _get({"key": "fast", "session_id": "s-pm"})["result"]["value"] == "flex"
            assert fast_mode.effective_request_overrides(agent)["service_tier"] == "flex"
            resp = _set({"key": "fast", "session_id": "s-pm", "value": "toggle"})
            assert resp["result"]["value"] == "normal"
            assert agent.service_tier is None
            assert agent._service_tier_session_pinned is True
            write_key.assert_not_called()


class TestSlashFastGlobalUnpins:
    """``/fast X --global`` persists and unpins so ``/model`` follows overlays."""

    def test_fast_flex_global_then_model_follows_overlay(self) -> None:
        agent = _agent()
        agent.model = "openai/gpt-5"
        agent.provider = "openrouter"
        agent.base_url = "https://openrouter.ai/api/v1"
        agent.service_tier = "priority"
        agent._service_tier_session_pinned = True
        session = {
            "session_key": "k-g",
            "agent": agent,
            "create_service_tier_override": "priority",
        }
        cfg = {
            "agent": {
                "service_tier": "flex",
                "service_tier_overrides": {
                    "openai/gpt-5": "flex",
                    "moonshotai/kimi-k2.6": "priority",
                },
            }
        }
        with patch.dict(server._sessions, {"s-g": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key, \
                patch.object(server, "_persist_live_session_runtime"), \
                patch.object(server, "_emit"), \
                patch.object(server, "_session_info", return_value={}), \
                patch("hermes_cli.config.load_config_readonly", return_value=cfg):
            warning = server._mirror_slash_side_effects(
                "s-g", session, "/fast flex --global",
            )
            assert warning == ""
            write_key.assert_called_once_with("agent.service_tier", "flex")
            assert agent._service_tier_session_pinned is False
            assert "create_service_tier_override" not in session

            from agent import fast_mode

            assert fast_mode.effective_request_overrides(agent)["service_tier"] == "flex"
            agent.model = "moonshotai/kimi-k2.6"
            assert fast_mode.effective_request_overrides(agent)["service_tier"] == "priority"


_MATCH = "openai/gpt-5"
_OPENROUTER = "https://openrouter.ai/api/v1"


def _unpinned_runtime_agent():
    return SimpleNamespace(
        model=_MATCH,
        provider="openrouter",
        base_url=_OPENROUTER,
        api_mode="chat_completions",
        reasoning_config=None,
        service_tier="priority",
        _service_tier_session_pinned=False,
    )


class TestColdResumePinProvenance:
    """Unpinned global/per-model tiers must not become a session pin on resume."""

    def test_unpinned_persist_omits_tier_override(self):
        from tui_gateway.server import _runtime_model_config, _stored_session_runtime_overrides

        persisted = _runtime_model_config(
            _unpinned_runtime_agent(), {"service_tier": "priority"},
        )
        assert "service_tier" not in persisted
        assert "service_tier_session_pinned" not in persisted
        overrides = _stored_session_runtime_overrides({
            "model": _MATCH,
            "model_config": json.dumps(persisted),
        })
        assert "service_tier_override" not in overrides

    def test_unpinned_cold_resume_uses_per_model_override(self, monkeypatch):
        from run_agent import AIAgent
        from agent.service_tier_escalation import escalation_is_active
        from tui_gateway.server import _runtime_model_config, _stored_session_runtime_overrides

        import hermes_cli.config as config_mod

        cfg = {
            "agent": {
                "service_tier": "priority",
                "service_tier_overrides": {_MATCH: "flex"},
                "service_tier_escalation": {"enabled": True, "ttft_threshold_seconds": 8.0, "consecutive_slow_requests": 1},
            }
        }
        monkeypatch.setattr(config_mod, "load_config_readonly", lambda: cfg)
        persisted = _runtime_model_config(_unpinned_runtime_agent())
        restored = _stored_session_runtime_overrides({
            "model": _MATCH,
            "model_config": json.dumps({**persisted, "model": _MATCH, "provider": "openrouter"}),
        })
        assert "service_tier_override" not in restored
        agent = AIAgent(
            api_key="k",
            base_url=_OPENROUTER,
            provider="openrouter",
            api_mode="chat_completions",
            model=_MATCH,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            save_trajectories=False,
            enabled_toolsets=["file"],
            service_tier=None,
        )
        try:
            agent._service_tier_session_pinned = restored.get("service_tier_override") is not None
            kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
            assert kwargs["service_tier"] == "flex"
            assert agent._service_tier_session_pinned is False
            assert escalation_is_active(agent) is True
        finally:
            agent.close()

    def test_explicit_fast_survives_resume_as_pin(self, monkeypatch):
        from run_agent import AIAgent
        from agent.service_tier_escalation import escalation_is_active
        from tui_gateway.server import _runtime_model_config, _stored_session_runtime_overrides

        import hermes_cli.config as config_mod

        cfg = {
            "agent": {
                "service_tier": "priority",
                "service_tier_overrides": {_MATCH: "flex"},
                "service_tier_escalation": {"enabled": True, "ttft_threshold_seconds": 8.0, "consecutive_slow_requests": 1},
            }
        }
        monkeypatch.setattr(config_mod, "load_config_readonly", lambda: cfg)
        live = _unpinned_runtime_agent()
        live._service_tier_session_pinned = True
        persisted = _runtime_model_config(live)
        assert persisted["service_tier"] == "priority"
        assert persisted["service_tier_session_pinned"] is True
        restored = _stored_session_runtime_overrides({
            "model": _MATCH,
            "model_config": json.dumps({**persisted, "model": _MATCH, "provider": "openrouter"}),
        })
        assert restored.get("service_tier_override") == "priority"
        agent = AIAgent(
            api_key="k",
            base_url=_OPENROUTER,
            provider="openrouter",
            api_mode="chat_completions",
            model=_MATCH,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            save_trajectories=False,
            enabled_toolsets=["file"],
            service_tier=restored["service_tier_override"],
        )
        try:
            agent._service_tier_session_pinned = restored.get("service_tier_override") is not None
            kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
            assert kwargs["service_tier"] == "priority"
            assert agent._service_tier_session_pinned is True
            assert escalation_is_active(agent) is False
        finally:
            agent.close()

    def test_legacy_stored_normal_without_flag_restores_as_pin(self):
        """Rows that stored ``normal`` before the pin flag still resume as an explicit pin."""
        from tui_gateway.server import _stored_session_runtime_overrides

        restored = _stored_session_runtime_overrides({
            "model": _MATCH,
            "model_config": json.dumps({"service_tier": "normal"}),
        })
        assert restored.get("service_tier_override") == ""


def test_tui_background_from_pinned_session_child_does_not_inherit_bake(monkeypatch):
    """TUI background inherits the session pin; a delegated child does not inherit the bake."""
    from run_agent import AIAgent
    from agent.fast_mode import set_framework_baked_tier_keys
    from tools import delegate_tool as dt
    import tools.delegate_tool_config as dtc
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "load_config_readonly",
        lambda: {"agent": {"service_tier": "", "service_tier_overrides": {}}},
    )
    monkeypatch.setattr(dt, "_load_config", lambda: {})
    monkeypatch.setattr(dtc, "_load_config", lambda: {})

    parent = AIAgent(
        api_key="k",
        base_url=_OPENROUTER,
        provider="openrouter",
        api_mode="chat_completions",
        model=_MATCH,
        platform="tui",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        enabled_toolsets=["file"],
        request_overrides={"service_tier": "priority"},
        service_tier="priority",
        session_id="p-tui-fast",
    )
    parent._service_tier_session_pinned = True
    set_framework_baked_tier_keys(parent, {"service_tier": "priority"})
    parent._fallback_chain = []
    bg = None
    child = None
    try:
        with patch.object(server, "_get_db", return_value=None), \
                patch.object(server, "_load_cfg", return_value={"agent": {}}), \
                patch.object(server, "_load_enabled_toolsets", return_value=["file"]), \
                patch.object(server, "_load_service_tier", return_value=None), \
                patch.object(server, "_resolve_model", return_value=_MATCH), \
                patch.object(server, "_load_reasoning_config", return_value=None):
            snap = server._background_tier_snapshot(parent)
            kwargs = server._background_agent_kwargs(parent, "bg-tui-fast", snap)
        kwargs.update(
            skip_context_files=True,
            skip_memory=True,
            save_trajectories=False,
            session_db=None,
            enabled_toolsets=["file"],
        )
        bg = AIAgent(**kwargs)
        server._apply_background_tier_provenance(bg, parent, snap)
        assert bg._service_tier_session_pinned is True
        assert "service_tier" in (getattr(bg, "_framework_baked_tier_keys", None) or ())
        assert bg._build_api_kwargs([{"role": "user", "content": "hi"}])["service_tier"] == "priority"
        creds = dt._resolve_delegation_credentials({"model": "", "provider": ""}, bg)
        child = dt._build_child_agent(
            task_index=0,
            goal="goal",
            context=None,
            toolsets=["file"],
            model=None,
            max_iterations=4,
            task_count=1,
            parent_agent=bg,
            override_request_overrides=creds.get("request_overrides"),
        )
        assert child._service_tier_session_pinned is False
        assert "service_tier" not in (child.request_overrides or {})
        child_kwargs = child._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert "service_tier" not in child_kwargs
        assert "speed" not in child_kwargs
    finally:
        if child is not None:
            child.close()
        if bg is not None:
            bg.close()
        parent.close()


def test_tui_background_unpinned_does_not_spurious_mark():
    """Unpinned TUI parent: background provenance must not invent a bake mark."""
    from agent.fast_mode import TIER_WIRE_KEYS

    parent = SimpleNamespace(
        base_url=_OPENROUTER,
        api_key="k",
        provider="openrouter",
        api_mode="chat_completions",
        acp_command=None,
        acp_args=[],
        ephemeral_system_prompt=None,
        providers_allowed=None,
        providers_ignored=None,
        providers_order=None,
        provider_sort=None,
        provider_data_collection=None,
        openrouter_min_coding_score=None,
        model=_MATCH,
        enabled_toolsets=["file"],
        provider_require_parameters=False,
        reasoning_config=None,
        service_tier=None,
        request_overrides={"service_tier": "flex", "extra_body": {"keep": 1}},
        _service_tier_session_pinned=False,
        _fallback_chain=[],
    )
    with patch.object(server, "_get_db", return_value=None), \
            patch.object(server, "_load_cfg", return_value={"agent": {}}), \
            patch.object(server, "_load_enabled_toolsets", return_value=["file"]), \
            patch.object(server, "_load_service_tier", return_value=None), \
            patch.object(server, "_resolve_model", return_value=_MATCH), \
            patch.object(server, "_load_reasoning_config", return_value=None):
        snap = server._background_tier_snapshot(parent)
        kwargs = server._background_agent_kwargs(parent, "bg-tui-unpinned", snap)
    bg = SimpleNamespace(
        request_overrides=dict(kwargs.get("request_overrides") or {}),
        _framework_baked_tier_keys=frozenset({"service_tier"}),
        _service_tier_session_pinned=True,
    )
    server._apply_background_tier_provenance(bg, parent, snap)
    assert not bg._framework_baked_tier_keys
    assert bg._service_tier_session_pinned is False
    assert bg.request_overrides.get("service_tier") == "flex"
    assert bg.request_overrides.get("extra_body") == {"keep": 1}
    assert all(
        key in bg.request_overrides or key not in (parent.request_overrides or {})
        for key in TIER_WIRE_KEYS
    )


def test_tui_background_pinned_normal_ignores_global_priority(monkeypatch):
    """Pinned ``/fast normal`` must not fall back to a global priority tier."""
    from run_agent import AIAgent
    from agent.fast_mode import set_framework_baked_tier_keys
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "load_config_readonly",
        lambda: {"agent": {"service_tier": "priority", "service_tier_overrides": {}}},
    )
    parent = AIAgent(
        api_key="k",
        base_url=_OPENROUTER,
        provider="openrouter",
        api_mode="chat_completions",
        model=_MATCH,
        platform="tui",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        enabled_toolsets=["file"],
        request_overrides={},
        service_tier=None,
        session_id="p-tui-normal",
    )
    parent._service_tier_session_pinned = True
    set_framework_baked_tier_keys(parent, None)
    parent._fallback_chain = []
    bg = None
    try:
        with patch.object(server, "_get_db", return_value=None), \
                patch.object(server, "_load_cfg", return_value={"agent": {"service_tier": "priority"}}), \
                patch.object(server, "_load_enabled_toolsets", return_value=["file"]), \
                patch.object(server, "_load_service_tier", return_value="priority"), \
                patch.object(server, "_resolve_model", return_value=_MATCH), \
                patch.object(server, "_load_reasoning_config", return_value=None):
            snap = server._background_tier_snapshot(parent)
            kwargs = server._background_agent_kwargs(parent, "bg-tui-normal", snap)
        assert kwargs["service_tier"] is None
        assert snap["pinned"] is True
        kwargs.update(
            skip_context_files=True,
            skip_memory=True,
            save_trajectories=False,
            session_db=None,
            enabled_toolsets=["file"],
        )
        bg = AIAgent(**kwargs)
        server._apply_background_tier_provenance(bg, parent, snap)
        assert bg._service_tier_session_pinned is True
        assert bg.service_tier is None
        assert not (getattr(bg, "_framework_baked_tier_keys", None) or ())
        wire = bg._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert "service_tier" not in wire
        assert "speed" not in wire
    finally:
        if bg is not None:
            bg.close()
        parent.close()


def test_tui_background_snapshot_overrides_ignore_later_fast_toggle(monkeypatch):
    """Snapshot overrides+marker stay pre-toggle if /fast flips before kwargs."""
    from run_agent import AIAgent
    from agent.fast_mode import set_framework_baked_tier_keys
    from tools import delegate_tool as dt
    import tools.delegate_tool_config as dtc
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "load_config_readonly",
        lambda: {"agent": {"service_tier": "", "service_tier_overrides": {}}},
    )
    monkeypatch.setattr(dt, "_load_config", lambda: {})
    monkeypatch.setattr(dtc, "_load_config", lambda: {})

    parent = AIAgent(
        api_key="k",
        base_url=_OPENROUTER,
        provider="openrouter",
        api_mode="chat_completions",
        model=_MATCH,
        platform="tui",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        enabled_toolsets=["file"],
        request_overrides={"extra_body": {"keep": 1}},
        service_tier=None,
        session_id="p-tui-snap-race",
    )
    parent._service_tier_session_pinned = True
    set_framework_baked_tier_keys(parent, None)
    parent._fallback_chain = []
    bg = None
    child = None
    try:
        with patch.object(server, "_get_db", return_value=None), \
                patch.object(server, "_load_cfg", return_value={"agent": {}}), \
                patch.object(server, "_load_enabled_toolsets", return_value=["file"]), \
                patch.object(server, "_load_service_tier", return_value="priority"), \
                patch.object(server, "_resolve_model", return_value=_MATCH), \
                patch.object(server, "_load_reasoning_config", return_value=None):
            snap = server._background_tier_snapshot(parent)
            parent.service_tier = "priority"
            parent.request_overrides = {"service_tier": "priority", "extra_body": {"keep": 1}}
            set_framework_baked_tier_keys(parent, {"service_tier": "priority"})
            kwargs = server._background_agent_kwargs(parent, "bg-tui-snap-race", snap)
        assert snap["pinned"] is True
        assert snap["service_tier"] is None
        assert "service_tier" not in (snap.get("request_overrides") or {})
        assert kwargs["service_tier"] is None
        assert "service_tier" not in (kwargs.get("request_overrides") or {})
        assert kwargs["request_overrides"].get("extra_body") == {"keep": 1}
        kwargs.update(
            skip_context_files=True,
            skip_memory=True,
            save_trajectories=False,
            session_db=None,
            enabled_toolsets=["file"],
        )
        bg = AIAgent(**kwargs)
        server._apply_background_tier_provenance(bg, parent, snap)
        assert bg._service_tier_session_pinned is True
        assert not (getattr(bg, "_framework_baked_tier_keys", None) or ())
        assert "service_tier" not in (bg.request_overrides or {})
        creds = dt._resolve_delegation_credentials({"model": "", "provider": ""}, bg)
        child = dt._build_child_agent(
            task_index=0,
            goal="goal",
            context=None,
            toolsets=["file"],
            model=None,
            max_iterations=4,
            task_count=1,
            parent_agent=bg,
            override_request_overrides=creds.get("request_overrides"),
        )
        assert child._service_tier_session_pinned is False
        assert "service_tier" not in (child.request_overrides or {})
        child_kwargs = child._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert "service_tier" not in child_kwargs
        assert "speed" not in child_kwargs
    finally:
        if child is not None:
            child.close()
        if bg is not None:
            bg.close()
        parent.close()
