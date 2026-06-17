from datetime import datetime, timezone
from types import SimpleNamespace

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from agent.usage_guard import (
    UsageGuardConfig,
    UsageGuardLevel,
    active_valid_tool_names,
    evaluate_usage_guard,
    load_usage_guard_config,
    should_require_fallback_confirmation,
)


def _snapshot(*windows):
    return AccountUsageSnapshot(
        provider="openai-codex",
        source="usage_api",
        fetched_at=datetime.now(timezone.utc),
        windows=tuple(windows),
    )


def test_load_usage_guard_config_accepts_requested_schema():
    cfg = load_usage_guard_config(
        {
            "usage_guard": {
                "enabled": True,
                "provider": "openai-codex",
                "warn_at_percent": 75,
                "wind_down_at_percent": 90,
                "block_new_long_tasks_at_percent": 85,
                "fallback_requires_user_confirmation": True,
                "safe_mode_toolsets": ["file", "terminal", "messaging"],
            }
        }
    )

    assert cfg.enabled is True
    assert cfg.provider == "openai-codex"
    assert cfg.warn_at_percent == 75.0
    assert cfg.block_new_long_tasks_at_percent == 85.0
    assert cfg.wind_down_at_percent == 90.0
    assert cfg.fallback_requires_user_confirmation is True
    assert cfg.safe_mode_toolsets == ("file", "terminal", "messaging")


def test_evaluate_usage_guard_uses_highest_account_window_for_wind_down():
    cfg = UsageGuardConfig(
        enabled=True,
        provider="openai-codex",
        warn_at_percent=75,
        block_new_long_tasks_at_percent=85,
        wind_down_at_percent=90,
        fallback_requires_user_confirmation=True,
        safe_mode_toolsets=("file", "terminal", "messaging"),
    )
    decision = evaluate_usage_guard(
        cfg,
        _snapshot(
            AccountUsageWindow(label="Session", used_percent=72),
            AccountUsageWindow(label="Weekly", used_percent=91),
        ),
    )

    assert decision.level is UsageGuardLevel.WIND_DOWN
    assert decision.used_percent == 91.0
    assert decision.window_label == "Weekly"
    assert "safe wind-down" in decision.context.lower()
    assert "file, terminal, messaging" in decision.context


def test_active_valid_tool_names_filters_to_safe_mode_toolsets_without_schema_mutation():
    agent = SimpleNamespace(
        tools=[
            {"function": {"name": "terminal"}},
            {"function": {"name": "patch"}},
            {"function": {"name": "send_message"}},
            {"function": {"name": "browser_navigate"}},
        ],
        valid_tool_names={"terminal", "patch", "send_message", "browser_navigate"},
        _usage_guard_safe_tool_names={"terminal", "patch", "send_message"},
    )

    assert active_valid_tool_names(agent) == {"terminal", "patch", "send_message"}
    assert [tool["function"]["name"] for tool in agent.tools] == [
        "terminal",
        "patch",
        "send_message",
        "browser_navigate",
    ]


def test_fallback_confirmation_required_blocks_automatic_primary_fallback():
    agent = SimpleNamespace(
        provider="openai-codex",
        _primary_runtime={"provider": "openai-codex"},
        _fallback_chain=[{"provider": "anthropic", "model": "claude-sonnet"}],
        _usage_guard_config=UsageGuardConfig(
            enabled=True,
            provider="openai-codex",
            fallback_requires_user_confirmation=True,
        ),
    )

    should_block, message = should_require_fallback_confirmation(agent)

    assert should_block is True
    assert "explicit user confirmation" in message
    assert "openai-codex" in message
