"""Actual response-header capture -> credit latches -> CLI/TUI presentation."""
import json
import time
from types import SimpleNamespace

import pytest
from run_agent import AIAgent
from tui_gateway import server
from hermes_cli.cli_stream_mixin import CLIStreamMixin
from tests.agent.test_credits_tracker import HEALTHY_HEADERS, DEPLETED_HEADERS


class CLI(CLIStreamMixin):
    def _invalidate(self):
        pass


@pytest.mark.parametrize("setting", [None, False, True])
@pytest.mark.parametrize("surface", ["cli", "tui"])
def test_actual_credit_capture_depletion_and_recovery(tmp_path, monkeypatch, setting, surface):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    cfg = {} if setting is None else {"display": {"suppress_warning_notifications": setting}}
    (tmp_path / "config.yaml").write_text(json.dumps(cfg))
    agent = object.__new__(AIAgent)
    agent.provider = "nous"
    agent.model = "paid-model"
    agent.base_url = ""
    agent._credits_state = agent._credits_session_start_micros = None
    agent._credits_notices_enabled_cache = True
    agent._notification_config = cfg
    observed, clears, frames = [], [], []
    cli = CLI()
    cli.agent = agent
    monkeypatch.setattr(server, "write_json", lambda frame: frames.append(frame) or True)
    monkeypatch.setitem(server._sessions, "credit", {"agent": agent, "profile_home": str(tmp_path)})
    callback = cli._on_notice if surface == "cli" else server._agent_cbs("credit")["notice_callback"]
    def observe(notice):
        observed.append(notice)
        callback(notice)
    agent.notice_callback = observe
    agent.notice_clear_callback = clears.append
    for headers in (HEALTHY_HEADERS, DEPLETED_HEADERS, DEPLETED_HEADERS, HEALTHY_HEADERS):
        current = dict(headers)
        current['x-nous-credits-as-of-ms'] = str(int(time.time() * 1000))
        agent._capture_credits(SimpleNamespace(headers=current))
    assert sum(n.key == "credits.depleted" for n in observed) == 1
    assert "credits.depleted" in clears
    assert agent.get_credits_state().paid_access is True
    assert agent.get_credits_state().remaining_micros == int(HEALTHY_HEADERS['x-nous-credits-remaining-micros'])
    projected = getattr(cli, "_pending_credit_notices", []) if surface == "cli" else [
        f for f in frames if f.get("params", {}).get("type") == "notification.show"]
    levels = [row[0] for row in projected] if surface == "cli" else [
        frame["params"]["payload"]["level"] for frame in projected]
    assert ("error" in levels) is not (setting is True)
    # Every credit-service notice is an automatic diagnostic, recovery included: a "restored"
    # line after a hidden depletion notice would be orphan noise (same rule as the gateway).
    assert ("success" in levels) is not (setting is True)
    assert sum(n.key == "credits.restored" for n in observed) == 1
