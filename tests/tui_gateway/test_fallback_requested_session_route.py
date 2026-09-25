"""Regression for #121506: a served fallback must not become the resumed route."""

import json
from types import SimpleNamespace

from hermes_state import SessionDB
from tui_gateway.server import _persist_live_session_runtime, _stored_session_runtime_overrides


def _fallback_agent(db):
    return SimpleNamespace(
        _session_db=db,
        _fallback_activated=True,
        _primary_runtime={
            "model": "primary-model", "provider": "anthropic",
            "base_url": "https://primary.example/v1", "api_mode": "anthropic_messages",
            "reasoning_config": {"effort": "high"}, "service_tier": None,
        },
        model="fallback-model", provider="nous",
        base_url="https://fallback.example/v1", api_mode="chat_completions",
        reasoning_config={"effort": "low"}, service_tier=None,
    )


def test_desktop_persist_and_resume_keep_requested_route(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("fallback-session", source="desktop", model="primary-model")
        agent = _fallback_agent(db)
        _persist_live_session_runtime({"agent": agent, "session_key": "fallback-session"})
        row = db.get_session("fallback-session")
        config = json.loads(row["model_config"])
        assert row["model"] == config["model"] == "primary-model"
        assert config["provider"] == "anthropic"
        assert config["base_url"] == "https://primary.example/v1"
        assert config["api_mode"] == "anthropic_messages"
        assert config["reasoning_config"] == {"effort": "high"}
        restored = _stored_session_runtime_overrides(row)
        assert restored["model_override"]["model"] == "primary-model"
        assert restored["provider_override"] == "anthropic"
        agent._fallback_activated = False
        _persist_live_session_runtime({"agent": agent, "session_key": "fallback-session"})
        assert db.get_session("fallback-session")["model"] == "fallback-model"


def test_cli_lazy_row_uses_primary_snapshot(tmp_path):
    from run_agent import AIAgent
    from hermes_cli.cli_model_switch_mixin import stored_session_route

    with SessionDB(tmp_path / "state.db") as db:
        agent = _fallback_agent(db)
        agent._session_init_model_config = {"reasoning_config": {"effort": "high"}}
        agent.session_id = "cli-fallback"
        agent._session_db_created = False
        agent._persist_disabled = False
        agent.platform = "cli"
        agent._cached_system_prompt = "System prompt"
        agent._parent_session_id = None
        agent._ensure_db_session = lambda: AIAgent._ensure_db_session(agent)
        agent._session_row_model_config = lambda: AIAgent._session_row_model_config(agent)
        agent._ensure_db_session()
        row = db.get_session(agent.session_id)
        assert row["model"] == "primary-model"
        config = json.loads(row["model_config"])
        assert {key: config[key] for key in ("model", "provider", "base_url", "api_mode")} == {
            key: agent._primary_runtime[key] for key in ("model", "provider", "base_url", "api_mode")
        }
        db.update_token_counts(
            agent.session_id, model="fallback-model", billing_provider="nous",
            billing_base_url="https://fallback.example/v1", input_tokens=17,
            output_tokens=3, api_call_count=1,
        )
        row = db.get_session(agent.session_id)
        assert row["model"] == "primary-model"
        assert row["billing_provider"] == "nous"
        assert stored_session_route(row, current_model="fallback-model", current_provider="nous")[:2] == (
            "primary-model", "anthropic")
        with db._lock:
            usage = db._conn.execute(
                "SELECT model, billing_provider, input_tokens FROM session_model_usage WHERE session_id = ?",
                (agent.session_id,),
            ).fetchone()
        assert tuple(usage) == ("fallback-model", "nous", 17)
        assert json.loads(row["model_config"])["reasoning_config"] == {"effort": "high"}
