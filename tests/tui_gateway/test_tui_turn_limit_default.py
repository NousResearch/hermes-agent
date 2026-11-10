"""With ``agent.max_turns`` unset the TUI (and desktop, which runs through it) resolves the same
unlimited turn budget as load_config, the CLI, the gateway and cron, and says so in /config."""

from unittest.mock import patch

from hermes_cli.config import TURN_LIMIT_UNLIMITED
from tui_gateway import server


def _stub_make_agent(monkeypatch, cfg):
    monkeypatch.delenv("HERMES_TUI_MAX_TURNS", raising=False)
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    monkeypatch.setattr(server, "_resolve_startup_runtime", lambda: ("test-model", None))
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, target_model=None: {
            "provider": None, "base_url": None, "api_key": None, "api_mode": None,
            "command": None, "args": None, "credential_pool": None})
    monkeypatch.setattr(server, "_load_tool_progress_mode", lambda: "off")
    monkeypatch.setattr(server, "_load_reasoning_config", lambda model="": None)
    monkeypatch.setattr(server, "_load_service_tier", lambda: None)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_agent_cbs", lambda sid: {})


def test_an_unset_max_turns_gives_the_tui_agent_an_unlimited_budget(monkeypatch):
    _stub_make_agent(monkeypatch, {})
    with patch("run_agent.AIAgent") as agent:
        server._make_agent("sid", "key")
    assert agent.call_args.kwargs["max_iterations"] == TURN_LIMIT_UNLIMITED


def test_the_config_panel_shows_unlimited_or_the_user_value(monkeypatch):
    monkeypatch.delenv("HERMES_TUI_MAX_TURNS", raising=False)

    def max_turns_row(cfg):
        monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
        response = server._methods["config.show"]("rid", {})
        assert "error" not in response, response
        sections = {s["title"]: dict(s["rows"]) for s in response["result"]["sections"]}
        return sections["Agent"]["Max Turns"]

    assert max_turns_row({}) == "unlimited"
    assert max_turns_row({"agent": {"max_turns": 12}}) == "12"
