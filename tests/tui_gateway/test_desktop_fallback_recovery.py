"""Desktop fallback-chain and configured-primary recovery regressions for #119195."""

import time
from pathlib import Path

import pytest

from agent.credential_pool import PooledCredential, load_pool
from hermes_cli.config import save_config
from tui_gateway import server


_PRIMARY_MODEL = "claude-sonnet-4-6"
_BACKUP_ALIAS = "deepseek-chat"
_BACKUP_MODEL = "deepseek-flash"
_LATER_MODEL = "openai/gpt-4o-mini"


def _bench(pool, credential_id, seconds):
    pool.mark_exhausted_and_rotate(
        credential_id=credential_id, status_code=429,
        error_context={"reset_at": time.time() + seconds},
    )


@pytest.fixture
def desktop_chain(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(server, "_hermes_home", home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_IGNORE_RULES", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek-fallback")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-openrouter-fallback")
    save_config({
        "model": {"provider": "anthropic", "default": _PRIMARY_MODEL},
        "fallback_providers": [
            {"provider": "deepseek", "model": _BACKUP_ALIAS},
            {"provider": "openrouter", "model": _LATER_MODEL},
        ],
    })
    primary = load_pool("anthropic")
    entry = primary.add_entry(PooledCredential(
        provider="anthropic", id="primary", label="Primary", auth_type="api_key",
        priority=0, source="manual", access_token="sk-ant-api03-test-primary",
    ))
    _bench(primary, entry.id, 3600)
    return entry.id


@pytest.mark.parametrize(("cooldown", "expected"), [(540, _BACKUP_ALIAS), (660, _LATER_MODEL)])
def test_startup_fallback_respects_the_retry_horizon(desktop_chain, cooldown, expected):
    backup = load_pool("deepseek")
    entry = backup.select(model=_BACKUP_MODEL)
    assert entry is not None
    _bench(backup, entry.id, cooldown)

    resolution = server._resolve_runtime_with_fallback({
        "requested": "anthropic", "target_model": _PRIMARY_MODEL,
    })

    assert resolution.selected_model == expected


@pytest.fixture
def desktop_agent(desktop_chain, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_launch_persistence", lambda: None)
    agent = server._make_agent(
        "fallback-test", "fallback-session", platform_override="desktop",
        cwd_override=str(tmp_path),
    )
    try:
        assert (agent.provider, agent.model) == ("deepseek", _BACKUP_MODEL)
        yield agent
    finally:
        agent.close()


def test_transient_fallback_stays_turn_scoped_until_the_configured_primary_recovers(
    desktop_agent, desktop_chain, monkeypatch,
):
    agent = desktop_agent
    notices = []
    monkeypatch.setattr(agent, "_emit_diagnostic_status", notices.append)
    agent._emit_pending_fallback_notice()
    notices.clear()

    for _ in range(2):
        assert agent._try_activate_fallback()
        assert (agent.provider, agent.model) == ("openrouter", _LATER_MODEL)
        assert agent._restore_primary_runtime()
        assert (agent.provider, agent.model) == ("deepseek", _BACKUP_MODEL)
        assert not any("Primary model restored" in str(notice) for notice in notices)

    load_pool("anthropic").reset_status(desktop_chain)
    assert agent._restore_primary_runtime()
    assert (agent.provider, agent.model) == ("anthropic", _PRIMARY_MODEL)
    assert any("Primary model restored" in str(notice) for notice in notices)


def test_live_chain_removal_leaves_the_remaining_fallback_reachable(desktop_agent):
    save_config({
        "model": {"provider": "anthropic", "default": _PRIMARY_MODEL},
        "fallback_providers": [{"provider": "openrouter", "model": _LATER_MODEL}],
    })
    server._sync_agent_fallback_with_config("fallback-test", {"agent": desktop_agent})

    assert not desktop_agent._restore_primary_runtime()
    assert desktop_agent._try_activate_fallback()
    assert (desktop_agent.provider, desktop_agent.model) == ("openrouter", _LATER_MODEL)


@pytest.mark.parametrize("one_turn", [True, False])
def test_desktop_model_pick_only_supersedes_recovery_if_it_persists(
    desktop_agent, desktop_chain, monkeypatch, one_turn,
):
    agent = desktop_agent
    session = {"agent": agent}
    for name in (
        "_restart_slash_worker", "_persist_live_session_runtime", "_persist_live_session_system_prompt",
        "_append_model_switch_marker", "_emit_session_info",
    ):
        monkeypatch.setattr(server, name, lambda *_args, **_kwargs: None)
    mode = "--once" if one_turn else "--session"

    server._apply_model_switch(
        "fallback-test", session, f"{_LATER_MODEL} --provider openrouter {mode}",
        confirm_expensive_model=True,
    )
    load_pool("anthropic").reset_status(desktop_chain)
    assert not agent._restore_primary_runtime()
    assert (agent.provider, agent.model) == ("openrouter", _LATER_MODEL)

    if one_turn:
        server._restore_agent_model_runtime(agent, session.pop("one_turn_model_restore"))
        assert (agent.provider, agent.model) == ("deepseek", _BACKUP_MODEL)
        assert agent._restore_primary_runtime()
        assert (agent.provider, agent.model) == ("anthropic", _PRIMARY_MODEL)
