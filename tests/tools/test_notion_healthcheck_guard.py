"""Regression tests for the narrow Notion healthcheck guard."""

from __future__ import annotations

from agent.runtime_evidence import clear_runtime_evidence, current_runtime_evidence, seed_turn_runtime_evidence
from agent import notion_healthcheck as notion_guard


def test_missing_notion_route_tool_blocks_notion_dependent_action(monkeypatch):
    clear_runtime_evidence()
    monkeypatch.setattr(notion_guard, "get_env_value", lambda key: None)
    monkeypatch.setattr(notion_guard.os, "getenv", lambda key, default=None: None)

    block = notion_guard.require_fresh_notion_healthcheck(
        action_label="memory",
        user_task="Run a Notion healthcheck.",
    )

    assert block is not None
    assert "missing_config" in block or "missing" in block.lower()
    assert "NOTION_API_KEY" in block
    clear_runtime_evidence()


def test_unauthenticated_config_failure_blocks(monkeypatch):
    clear_runtime_evidence()
    monkeypatch.setattr(
        notion_guard,
        "get_env_value",
        lambda key: "notion-token" if key == "NOTION_API_KEY" else None,
    )

    def _route_unavailable(url: str, token: str):
        raise notion_guard.NotionAuthError("Blocked Notion healthcheck: auth/config failed.")

    result = notion_guard.probe_notion_healthcheck(fetcher=_route_unavailable, now=100.0)
    assert result["ok"] is False
    assert result["reason"] == "auth_config"
    assert "auth" in result["message"].lower() or "config" in result["message"].lower()
    block = notion_guard.require_fresh_notion_healthcheck(
        action_label="memory",
        user_task="Use Notion memory.",
        now=101.0,
    )
    assert block is not None
    assert "auth" in block.lower() or "config" in block.lower()
    clear_runtime_evidence()


def test_empty_or_invalid_health_result_blocks(monkeypatch):
    clear_runtime_evidence()
    monkeypatch.setattr(notion_guard, "get_env_value", lambda key: "notion-token")

    result = notion_guard.probe_notion_healthcheck(
        fetcher=lambda url, token: {"status_code": 200, "body": {}},
        now=200.0,
    )
    assert result["ok"] is False
    assert result["reason"] == "invalid_result"
    assert "invalid" in result["message"].lower() or "empty" in result["message"].lower()

    block = notion_guard.require_fresh_notion_healthcheck(
        action_label="memory",
        user_task="Use Notion knowledge.",
        now=201.0,
    )
    assert block is not None
    assert "invalid" in block.lower() or "empty" in block.lower()
    clear_runtime_evidence()


def test_stale_health_result_blocks(monkeypatch):
    clear_runtime_evidence()
    monkeypatch.setattr(
        notion_guard,
        "get_env_value",
        lambda key: "notion-token" if key == "NOTION_API_KEY" else None,
    )
    seed_turn_runtime_evidence(latest_user_request="Use Notion memory.")
    update = notion_guard.probe_notion_healthcheck(
        fetcher=lambda url, token: {
            "status_code": 200,
            "body": {
                "object": "user",
                "id": "user-123",
                "type": "bot",
                "name": "Hermes Notion",
                "bot": {"owner_id": "bot-owner", "workspace_name": "Acme"},
            },
        },
        now=1_000.0,
    )
    assert update["ok"] is True

    evidence = current_runtime_evidence()
    assert evidence is not None
    evidence["notion_healthcheck_state"]["checked_at"] = 100.0

    block = notion_guard.require_fresh_notion_healthcheck(
        action_label="memory",
        user_task="Use Notion memory.",
        now=1_000.0 + 600.0,
        evidence=evidence,
    )
    assert block is not None
    assert "stale" in block.lower() or "fresh" in block.lower()
    clear_runtime_evidence()


def test_valid_fresh_health_result_allows(monkeypatch):
    clear_runtime_evidence()
    seed_turn_runtime_evidence(latest_user_request="Use Notion memory.")
    monkeypatch.setattr(
        notion_guard,
        "get_env_value",
        lambda key: "notion-token" if key == "NOTION_API_KEY" else None,
    )

    result = notion_guard.probe_notion_healthcheck(
        fetcher=lambda url, token: {
            "status_code": 200,
            "body": {
                "object": "user",
                "id": "user-123",
                "type": "bot",
                "name": "Hermes Notion",
                "bot": {"owner_id": "bot-owner", "workspace_name": "Acme"},
            },
        },
        now=1_500.0,
    )
    assert result["ok"] is True
    assert result["workspace"] == "Acme"
    assert result["bot_name"] == "Hermes Notion"
    assert "notion-token" not in result["route_url"]

    block = notion_guard.require_fresh_notion_healthcheck(
        action_label="memory",
        user_task="Use Notion memory.",
        now=1_500.0 + 1.0,
    )
    assert block is None
    clear_runtime_evidence()


def test_unrelated_normal_commands_still_pass():
    clear_runtime_evidence()
    block = notion_guard.require_fresh_notion_healthcheck(
        action_label="terminal",
        user_task="Continue PR work.",
        now=2_000.0,
    )
    assert block is None
    clear_runtime_evidence()
