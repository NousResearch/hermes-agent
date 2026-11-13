"""A card's ``model`` is a palier (``model.aliases``) until the override is STORED as its route.

Regression for t_c578eb82: ``model: std`` was stored verbatim and the worker died on
``Model "std" is not supported on this endpoint`` (HTTP 400) four attempts running. A palier now
stores the model id AND the provider it must be sent to, because ``commandcode`` and
``commandcode-anthropic`` are two registered providers (``chat_completions`` vs
``anthropic_messages``) — not one provider with a routing prefix.
"""
from __future__ import annotations

import json

import pytest

ALIASES = {
    "eco": "xiaomi/mimo-v2.6-flash",
    "std": "commandcode-anthropic/claude-haiku-5-5",
}


def _config(aliases=None, provider="commandcode"):
    return {"model": {"provider": provider, "aliases": ALIASES if aliases is None else aliases}}


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

def test_palier_expands_to_the_route_its_worker_launches():
    from hermes_cli.kanban_model import resolve_model_override

    assert resolve_model_override("std", config=_config()) == (
        "claude-haiku-5-5", "commandcode-anthropic")
    assert resolve_model_override(" STD ", config=_config()) == (
        "claude-haiku-5-5", "commandcode-anthropic")


def test_a_prefixed_palier_keeps_its_own_provider():
    """``model.aliases`` strings split ``provider/model`` — the same reading ``/model eco`` has."""
    from hermes_cli.kanban_model import resolve_model_override

    assert resolve_model_override("eco", config=_config()) == ("mimo-v2.6-flash", "xiaomi")


def test_wire_ids_pass_through_untouched():
    from hermes_cli.kanban_model import resolve_model_override

    for wire_id in ("gpt-6-luna", "claude-haiku-5-5"):
        assert resolve_model_override(wire_id, config=_config()) == (wire_id, None)


def test_a_configured_provider_prefix_is_split_off_the_model():
    from hermes_cli.kanban_model import resolve_model_override

    cfg = {**_config(), "providers": {"commandcode": {"api": "https://api.commandcode.ai/provider/v1"}}}
    assert resolve_model_override("commandcode/xiaomi/mimo-v2.6-pro", config=cfg) == (
        "xiaomi/mimo-v2.6-pro", "commandcode")


def test_unknown_palier_fails_loudly_and_names_the_declared_paliers():
    from hermes_cli.kanban_model import resolve_model_override

    for typo in ("frontier", "stdd", "leger"):
        with pytest.raises(ValueError) as exc:
            resolve_model_override(typo, config=_config())
        assert f"unknown palier {typo!r}" in str(exc.value)
        assert "eco, std" in str(exc.value)


def test_provider_declared_by_an_entry_is_used_and_a_contradiction_is_an_error():
    from hermes_cli.kanban_model import resolve_model_override

    cfg = {"model": {"aliases": {"far": {"model": "far-model", "provider": "openrouter"}}}}
    assert resolve_model_override("far", config=cfg) == ("far-model", "openrouter")
    with pytest.raises(ValueError, match="targets provider 'openrouter'"):
        resolve_model_override("far", "nous", config=cfg)


def test_model_aliases_entries_are_resolved_too():
    """The table is the core's merged one: ``model_aliases:`` entries resolve, and win."""
    from hermes_cli.kanban_model import resolve_model_override

    cfg = {
        "model_aliases": {"theta": {"model": "theta-1", "provider": "nous"}},
        "model": {"provider": "commandcode", "aliases": {"theta": "other/x-2"}},
    }
    assert resolve_model_override("theta", config=cfg) == ("theta-1", "nous")


def test_a_palier_pinning_its_own_endpoint_is_refused():
    """A card carries only the pair: an alias endpoint would silently be replaced."""
    from hermes_cli.kanban_model import resolve_model_override

    cfg = {"model_aliases": {"loc": {"model": "qwen3", "provider": "custom",
                                     "base_url": "https://gpu.lan/v1"}}}
    with pytest.raises(ValueError, match="pins its own base_url"):
        resolve_model_override("loc", config=cfg)


def test_provider_without_model_is_rejected():
    from hermes_cli.kanban_model import resolve_model_override

    with pytest.raises(ValueError, match="requires 'model'"):
        resolve_model_override(None, "commandcode", config=_config())
    assert resolve_model_override(None, config=_config()) == (None, None)


# ---------------------------------------------------------------------------
# kanban_create stores the route
# ---------------------------------------------------------------------------

@pytest.fixture
def board_home(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n"
        "  provider: commandcode\n"
        "  aliases:\n"
        "    eco: xiaomi/mimo-v2.6-flash\n"
        "    std: commandcode-anthropic/claude-haiku-5-5\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def test_create_stores_the_palier_as_model_and_provider(board_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_create({"title": "pinned", "assignee": "peer", "model": "std"}))
    assert out["ok"] is True
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, out["task_id"])
    assert task is not None
    assert task.model_override == "claude-haiku-5-5"
    assert task.provider_override == "commandcode-anthropic"


def test_the_stored_pair_builds_the_anthropic_route(board_home, tmp_path):
    """The point of the pair: the worker's policy lands on the /messages provider, not chat."""
    from gateway.session_policy import build_policy
    from providers import get_provider_profile

    policy = build_policy(
        {"source": "cli", "cwd": str(tmp_path), "model": "claude-haiku-5-5",
         "provider": "commandcode-anthropic"},
        {"model": {"provider": "commandcode", "base_url": "https://api.commandcode.ai/provider/v1",
                   "api_mode": "chat_completions"},
         "providers": {"commandcode": {"api": "https://api.commandcode.ai/provider/v1"}}})
    assert policy.model == "claude-haiku-5-5"
    assert policy.config()["model"]["provider"] == "commandcode-anthropic"
    profile = get_provider_profile("commandcode-anthropic")
    assert profile is not None and profile.api_mode == "anthropic_messages"


def test_create_rejects_an_unknown_palier_before_any_card_exists(board_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_create({"title": "typo", "assignee": "peer", "model": "stdd"}))
    assert "error" in out and "unknown palier 'stdd'" in out["error"]
    with kbc.connect_closing() as conn:
        assert kb.list_tasks(conn) == []
