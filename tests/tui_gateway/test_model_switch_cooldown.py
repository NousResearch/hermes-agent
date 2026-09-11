from __future__ import annotations

import copy
from types import SimpleNamespace


def test_one_turn_restore_clears_primary_manager_cooldown(monkeypatch):
    from agent.cooldown_manager import (
        CooldownManager, build_cooldown_key, get_cooldown_manager, set_cooldown_manager,
    )
    from tui_gateway import model_switch

    monkeypatch.setattr(model_switch, "copy", copy, raising=False)
    original = get_cooldown_manager()
    manager = CooldownManager(storage_path=False)
    key = build_cooldown_key("openrouter", "sk-old", "rate_limit")
    manager.mark_failure(key, "rate_limit", cooldown_seconds=60)
    set_cooldown_manager(manager)
    agent = SimpleNamespace(_primary_runtime=None, _fallback_activated=False)
    agent._restore_primary_runtime = lambda: not manager.is_cooling(key)
    try:
        model_switch._restore_agent_model_runtime(
            agent,
            {"primary_runtime": {"provider": "openrouter", "api_key": "sk-old"}},
        )
    finally:
        set_cooldown_manager(original)

    assert not manager.is_cooling(key)
