"""``/model X --once`` restores the context window the session learned on its primary.

The one-turn snapshot copies ``_primary_runtime``; a window corrected since construction
(turn_overflow adopts a provider-reported limit with ``update_model``) must come back with the
restore, not the construction-time guess. The TUI and the CLI each take their own snapshot.
"""

from types import SimpleNamespace

import pytest

from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin
from run_agent import AIAgent
from tui_gateway import server

# Loopback discard port: every probe is refused at once, nothing leaves the machine.
PRIMARY_URL = "http://127.0.0.1:9/primary/v1"
ONCE_URL = "http://127.0.0.1:9/once/v1"


def _tui_round_trip(agent, detour):
    snapshot = server._snapshot_agent_model_runtime(agent)
    detour()
    server._restore_agent_model_runtime(agent, snapshot)


def _cli_round_trip(agent, detour):
    cli = SimpleNamespace(
        agent=agent, model=agent.model, provider=agent.provider, requested_provider=agent.provider,
        api_key=agent.api_key, _explicit_api_key=agent.api_key, base_url=agent.base_url,
        _explicit_base_url=agent.base_url, api_mode=agent.api_mode, reasoning_config=None,
    )
    snapshot = CLIModelSwitchMixin._snapshot_model_runtime(cli)
    detour()
    CLIModelSwitchMixin._restore_model_runtime_snapshot(cli, snapshot)


@pytest.mark.parametrize("round_trip", [_tui_round_trip, _cli_round_trip], ids=["tui", "cli"])
def test_once_restore_keeps_the_primary_window_learned_mid_session(round_trip):
    agent = AIAgent(
        model="primary-model", provider="custom", base_url=PRIMARY_URL, api_key="primary-key",
        quiet_mode=True, enabled_toolsets=[], skip_context_files=True, skip_memory=True,
    )
    compressor = agent.context_compressor
    learned = 65_536
    assert compressor.context_length != learned
    compressor.update_model(
        model=agent.model, context_length=learned, base_url=agent.base_url,
        api_key=agent.api_key, provider=agent.provider, api_mode=agent.api_mode,
    )

    round_trip(agent, lambda: agent.switch_model(
        new_model="once-model", new_provider="custom", api_key="once-key",
        base_url=ONCE_URL, api_mode="chat_completions"))

    assert (agent.model, compressor.context_length) == ("primary-model", learned)
