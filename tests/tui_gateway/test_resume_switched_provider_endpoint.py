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


def test_resume_honors_gateway_runtime_over_first_call_billing_provider():
    """The messaging gateway persists the route it last ran only in ``model_config.gateway_runtime``;
    ``billing_provider`` is frozen at the first accounted call. Desktop resume must read the nested
    route like CLI ``--resume`` does, not recombine the current model with the stale bucket (#125942)."""
    row = {"model": "gpt-6-sol", "billing_provider": "anthropic", "model_config": json.dumps(
        {"gateway_runtime": {"provider": "openai-codex", "base_url": None, "api_mode": "codex_responses"}})}
    desktop = _stored_session_runtime_overrides(row)["model_override"]
    assert (desktop["provider"], desktop["base_url"], desktop["api_mode"]) == ("openai-codex", None, "codex_responses")
    assert stored_session_route(row, current_model="gpt-6-sol", current_provider="anthropic")[1:4] == (
        "openai-codex", None, "codex_responses")


def test_resume_prefers_gateway_runtime_over_stale_top_level_keys():
    """A gateway turn after a Desktop pick rewrites only ``gateway_runtime``, so the nested route is the
    newest one — same precedence as ``SessionDB.session_gateway_runtime``."""
    row = {"model": "gpt-6-sol", "model_config": json.dumps(
        {"provider": "anthropic", **NOUS_ROUTE, "gateway_runtime": {"provider": "openai-codex", "api_mode": "codex_responses"}})}
    desktop = _stored_session_runtime_overrides(row)["model_override"]
    assert (desktop["provider"], desktop["base_url"], desktop["api_mode"]) == ("openai-codex", None, "codex_responses")


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


@pytest.mark.parametrize("picked", [
    {"model": "claude-opus-4", "provider": "anthropic"},
    {"model": "claude-opus-4"},
], ids=["picked-pair", "picked-model-only"])
def test_resume_never_pairs_billing_model_with_picked_provider(picked):
    """A chat whose first call was served by a fallback has the fallback route in the billing columns
    (``first_accounted_route``) and the user's pick in ``model_config``. Resume must restore the pick as a
    pair, never the fallback model on the picked provider (or the picked model on the fallback provider)."""
    ov = _stored_session_runtime_overrides({
        "model": "deepseek/deepseek-v4-flash", "billing_provider": "openrouter", "model_config": picked,
    })
    restored = ov["model_override"]
    assert restored["model"] == picked["model"]
    assert restored["provider"] == picked.get("provider")
    assert ov.get("provider_override") == picked.get("provider")

def test_resume_after_gateway_turn_keeps_the_gateway_pair_over_an_older_desktop_pick():
    """A gateway turn after a Desktop pick writes the model column with ``gateway_runtime`` but leaves the
    Desktop's top-level pick; resume takes the gateway's pair, never the old pick on the gateway's provider."""
    row = {"model": "gpt-6-sol", "billing_provider": "anthropic", "model_config": json.dumps({
        "model": "claude-opus-4", "provider": "anthropic",
        "gateway_runtime": {"provider": "openai-codex", "api_mode": "codex_responses"}})}
    desktop = _stored_session_runtime_overrides(row)["model_override"]
    assert (desktop["model"], desktop["provider"]) == ("gpt-6-sol", "openai-codex")


def test_resume_keeps_a_confirmed_browser_model_lock_over_an_older_desktop_pick(tmp_path):
    """A confirmed Browser/API lock writes the model column but not the Desktop pick; resume keeps the
    locked model rather than silently reverting the chat to the pick the lock replaced."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="desktop", model="claude-opus-4")
    db.update_session_meta("s1", json.dumps({"model": "claude-opus-4", "provider": "anthropic"}), "claude-opus-4")
    db.update_session_runtime_lock("s1", model="claude-sonnet-5", provider="anthropic", confirmed=True)
    restored = _stored_session_runtime_overrides(db.get_session("s1"))["model_override"]
    assert (restored["model"], restored["provider"]) == ("claude-sonnet-5", "anthropic")
