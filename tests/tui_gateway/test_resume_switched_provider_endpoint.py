"""A chat switched to another provider must resume on THAT provider's endpoint.

The messaging gateway's /model wrote only ``model``/``provider`` over a row the Desktop had persisted
with the Nous Portal route, so resuming a chat switched to openai-codex sent the Codex slug to the
Portal's chat/completions ("Model 'gpt-6-luna-900k' isn't available on ChatGPT or Codex").
"""

import json

import pytest

from hermes_cli.cli_model_switch_mixin import stored_session_route
from hermes_state import SessionDB
from tui_gateway.server import _stored_session_runtime_overrides

NOUS_ROUTE = {"base_url": "https://inference-api.nousresearch.com/v1", "api_mode": "chat_completions"}


@pytest.mark.parametrize("row_origin", ["gateway_model_switch", "row_written_by_older_build"])
def test_resume_drops_previous_providers_endpoint(tmp_path, row_origin):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="telegram", model="openai/gpt-6-luna")
    if row_origin == "gateway_model_switch":
        db.update_session_meta("s1", json.dumps({"model": "openai/gpt-6-luna", "provider": "nous", **NOUS_ROUTE}),
                               "openai/gpt-6-luna")
        db.update_session_model("s1", "gpt-6-luna-900k", provider="openai-codex")
    else:
        db.update_session_meta("s1", json.dumps({"model": "gpt-6-luna-900k", "provider": "openai-codex", **NOUS_ROUTE}),
                               "gpt-6-luna-900k")
    row = db.get_session("s1")

    desktop = _stored_session_runtime_overrides(row)["model_override"]
    assert (desktop["provider"], desktop["base_url"], desktop["api_mode"]) == ("openai-codex", None, None)
    assert stored_session_route(row, current_model="openai/gpt-6-luna", current_provider="nous") == (
        "gpt-6-luna-900k", "openai-codex", None, None, True)


def test_resume_after_fallback_served_first_call_keeps_the_picked_pair(tmp_path):
    """The first accounted call records the serving (fallback) route in the billing columns; the chat's
    picked model+provider must still resume together, not as the fallback model on the picked provider."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="desktop", model="claude-opus-4")
    db.update_session_meta("s1", json.dumps({"model": "claude-opus-4", "provider": "anthropic"}), "claude-opus-4")
    db.update_token_counts("s1", input_tokens=10, output_tokens=5, model="deepseek/deepseek-v4-flash",
                           billing_provider="openrouter", api_call_count=1)
    row = db.get_session("s1")
    # The billing record keeps the serving route.
    assert (row["model"], row["billing_provider"]) == ("deepseek/deepseek-v4-flash", "openrouter")

    restored = _stored_session_runtime_overrides(row)["model_override"]
    assert (restored["model"], restored["provider"]) == ("claude-opus-4", "anthropic")
