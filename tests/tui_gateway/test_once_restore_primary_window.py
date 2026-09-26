"""``/model X --once`` restores the context window the session learned on its primary.

The one-turn snapshot copies ``_primary_runtime``; a window corrected since construction
(turn_overflow adopts a provider-reported limit with ``update_model``) must come back with the
restore, not the construction-time guess.
"""

from run_agent import AIAgent
from tui_gateway import server

# Loopback discard port: every probe is refused at once, nothing leaves the machine.
PRIMARY_URL = "http://127.0.0.1:9/primary/v1"
ONCE_URL = "http://127.0.0.1:9/once/v1"


def test_once_restore_keeps_the_primary_window_learned_mid_session():
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

    snapshot = server._snapshot_agent_model_runtime(agent)
    agent.switch_model(new_model="once-model", new_provider="custom", api_key="once-key",
                       base_url=ONCE_URL, api_mode="chat_completions")
    server._restore_agent_model_runtime(agent, snapshot)

    assert (agent.model, compressor.context_length) == ("primary-model", learned)
