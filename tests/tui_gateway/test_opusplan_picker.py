"""Offline picker-mode tests: inventory, session switch, persisted resume and normal routes."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import inventory
from hermes_cli.opusplan import picker_model_ids
from tui_gateway import server

PAIR = {"lab": {"base_url": "http://lab.invalid/v1", "opusplan": {"plan": "planner", "exec": "worker"}}}
CFG = {"model": {"default": "ordinary", "provider": "lab"}, "providers": PAIR}


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: CFG)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: CFG)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: CFG)
    monkeypatch.setattr("providers.get_provider_profile", lambda p: None)
    for name in ("_restart_slash_worker", "_persist_live_session_runtime", "_persist_live_session_system_prompt",
                 "_append_model_switch_marker", "_emit_session_info"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)


def test_configured_pair_and_unsupported_provider(offline):
    assert picker_model_ids(["planner", "worker"], "lab", user_providers=PAIR) == ["opusplan", "planner", "worker"]
    assert picker_model_ids(["ordinary"], "unsupported", user_providers=PAIR) == ["ordinary"]
    assert picker_model_ids(["opusplan", "ordinary"], "auto", user_providers=PAIR) == ["ordinary"]


def test_inventory_options_has_selectable_model_row(offline, monkeypatch):
    rows = [{"slug": "lab", "models": ["planner", "worker"], "total_models": 2},
            {"slug": "unsupported", "models": ["ordinary"], "total_models": 1}]
    monkeypatch.setattr("hermes_cli.model_switch.list_authenticated_providers", lambda **k: copy.deepcopy(rows))
    for name in ("_local_runtime_row", "_moa_provider_row"):
        monkeypatch.setattr(inventory, name, lambda *a: None)
    for name in ("_apply_picker_hints", "_apply_pricing", "_apply_capabilities", "_apply_featured",
                 "_apply_custom_aliases", "_apply_limits", "_apply_usage", "_prewarm_pricing_async"):
        monkeypatch.setattr(inventory, name, lambda *a, **k: None)
    ctx = inventory.ConfigContext("lab", "ordinary", "http://lab.invalid/v1", PAIR, [])
    payload = inventory.build_model_options_payload(ctx)
    assert payload["providers"][0]["models"][0] == "opusplan"
    assert payload["providers"][0]["total_models"] == 3
    assert payload["providers"][1]["models"] == ["ordinary"]
    assert payload["model"] == "ordinary"
    # Non-picker consumers (recommended-default selection, raw catalogs) must not
    # silently adopt an orchestration preset just because it is the first row.
    ordinary = inventory.build_models_payload(ctx)
    assert ordinary["providers"][0]["models"] == ["planner", "worker"]


class Agent:
    def __init__(self):
        self.model, self.provider = "ordinary", "lab"
        self.base_url, self.api_key, self.api_mode = "", "", "chat_completions"
        self.opusplan_active = False

    def switch_model(self, **kw):
        self.model, self.provider = kw["new_model"], kw["new_provider"]


def test_native_switch_pins_mode_and_switch_away_disables(offline, monkeypatch):
    def switch(**kw):
        active = kw["raw_input"] == "opusplan"
        return SimpleNamespace(success=True, new_model="planner" if active else "ordinary", opusplan=active,
                               target_provider="lab", base_url="", api_key="", api_mode="chat_completions",
                               runtime_capabilities=None, warning_message="", model_info=None)
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", switch)
    persist = Mock()
    monkeypatch.setattr("hermes_cli.model_switch.persist_model_selection", persist)
    session = {"agent": Agent()}
    out = server._apply_model_switch("sid", session, "opusplan --session", confirm_expensive_model=True)
    assert out["value"] == "opusplan"
    assert session["agent"].model == "planner"
    assert session["agent"].opusplan_active is True
    assert session["model_override"]["opusplan"] is True
    server._apply_model_switch("sid", session, "ordinary --session", confirm_expensive_model=True)
    assert session["agent"].opusplan_active is False
    assert session["model_override"]["opusplan"] is False
    persist.assert_not_called()


@pytest.mark.parametrize("active", [True, False])
def test_runtime_mode_survives_persist_resume_and_resolution(offline, monkeypatch, active):
    agent = Agent()
    agent.opusplan_active = active
    agent.model = "planner"  # selecting the concrete planner is not selecting the preset
    cfg = server._runtime_model_config(agent)
    assert cfg["opusplan"] is active
    monkeypatch.setattr(server, "_is_routable_provider", lambda p: True)
    row = {"model": agent.model, "model_config": json.dumps(cfg)}
    override = server._stored_session_runtime_overrides(row)["model_override"]
    assert override["opusplan"] is active
    seen = []
    def resolve(kw):
        seen.append(kw)
        return SimpleNamespace(runtime={"provider": "lab"}, used_fallback=False)
    monkeypatch.setattr(server, "_resolve_runtime_with_fallback", resolve)
    model, runtime = server._resolve_agent_model_runtime(override, None)
    assert model == "planner" and runtime["_opusplan_active"] is active
    assert seen[0]["target_model"] == "planner"
    assert "api_key" not in cfg


def test_desktop_create_keyword_resolves_before_provider(offline, monkeypatch):
    seen = []
    def resolve(kw):
        seen.append(kw)
        return SimpleNamespace(runtime={"provider": "lab"}, used_fallback=False)
    monkeypatch.setattr(server, "_resolve_runtime_with_fallback", resolve)
    model, runtime = server._resolve_agent_model_runtime({"model": "opusplan", "provider": "lab"}, None)
    assert model == "planner"
    assert runtime["_opusplan_active"] is True
    assert seen[0]["target_model"] == "planner"


def test_once_restore_restores_mode(offline):
    agent = Agent()
    agent.opusplan_active = True
    snapshot = server._snapshot_agent_model_runtime(agent)
    agent.opusplan_active = False
    server._restore_agent_model_runtime(agent, snapshot)
    assert agent.opusplan_active is True


@pytest.mark.asyncio
async def test_http_options_inventory_includes_preset(offline, monkeypatch):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    # Stub catalogs/decorators only; exercise the real shared payload and HTTP handler.
    test_inventory_options_has_selectable_model_row(offline, monkeypatch)
    ctx = inventory.ConfigContext("lab", "ordinary", "http://lab.invalid/v1", PAIR, [])
    monkeypatch.setattr(inventory, "load_picker_context", lambda: ctx)
    monkeypatch.setattr(inventory, "_append_unconfigured_rows", lambda *a, **k: [])
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    app = web.Application()
    app.router.add_get("/api/model/options", adapter._handle_model_options)
    async with TestClient(TestServer(app)) as client:
        response = await client.get("/api/model/options?include_unconfigured=false")
        assert response.status == 200
        payload = await response.json()
    lab = next(row for row in payload["providers"] if row["slug"] == "lab")
    assert lab["models"][0] == "opusplan" and lab["total_models"] == 3


@pytest.mark.asyncio
async def test_desktop_rest_options_inventory_includes_preset(offline, monkeypatch):
    from contextlib import nullcontext
    from hermes_cli.web_routers import models
    test_inventory_options_has_selectable_model_row(offline, monkeypatch)
    ctx = inventory.ConfigContext("lab", "ordinary", "http://lab.invalid/v1", PAIR, [])
    monkeypatch.setattr(inventory, "load_picker_context", lambda: ctx)
    monkeypatch.setattr(models, "_dashboard_code_skew_guard", lambda: "")
    monkeypatch.setattr(models, "_config_profile_scope", lambda p: nullcontext())
    payload = await models.get_model_options(profile=None, refresh=False, include_unconfigured=False, explicit_only=False)
    lab = next(row for row in payload["providers"] if row["slug"] == "lab")
    assert lab["models"][0] == "opusplan"


def test_native_options_reports_selected_preset(offline, monkeypatch):
    ctx = inventory.ConfigContext("lab", "ordinary", "", PAIR, [])
    monkeypatch.setattr(inventory, "load_picker_context", lambda: ctx)
    agent = Agent()
    agent.model, agent.opusplan_active = "planner", True
    assert server._model_picker_context(agent).current_model == "opusplan"
    agent.opusplan_active = False
    assert server._model_picker_context(agent).current_model == "planner"
