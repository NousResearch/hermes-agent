"""A card's ``model`` is a palier (``model.aliases``) until the override is STORED expanded.

Regression for t_c578eb82: ``model: std`` was stored verbatim and the worker died on
``Model "std" is not supported on this endpoint`` (HTTP 400) four attempts running.
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

def test_palier_expands_to_its_wire_id():
    from hermes_cli.kanban_model import resolve_model_override

    assert resolve_model_override("std", config=_config()) == (
        "commandcode-anthropic/claude-haiku-5-5", None)
    assert resolve_model_override(" STD ", config=_config())[0] == (
        "commandcode-anthropic/claude-haiku-5-5")


def test_wire_ids_pass_through_untouched():
    from hermes_cli.kanban_model import resolve_model_override

    for wire_id in ("gpt-6-luna", "claude-haiku-5-5", "commandcode/xiaomi/mimo-v2.6-pro"):
        assert resolve_model_override(wire_id, config=_config()) == (wire_id, None)
    assert resolve_model_override("std", "commandcode", config=_config()) == (
        "commandcode-anthropic/claude-haiku-5-5", "commandcode")


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


def test_provider_without_model_is_rejected():
    from hermes_cli.kanban_model import resolve_model_override

    with pytest.raises(ValueError, match="requires 'model'"):
        resolve_model_override(None, "commandcode", config=_config())
    assert resolve_model_override(None, config=_config()) == (None, None)


# ---------------------------------------------------------------------------
# kanban_create stores the expanded id
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


def test_create_stores_the_palier_as_its_wire_id(board_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_create({"title": "pinned", "assignee": "peer", "model": "std"}))
    assert out["ok"] is True
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, out["task_id"])
    assert task is not None
    assert task.model_override == "commandcode-anthropic/claude-haiku-5-5"
    assert task.provider_override is None


def test_create_rejects_an_unknown_palier_before_any_card_exists(board_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_create({"title": "typo", "assignee": "peer", "model": "stdd"}))
    assert "error" in out and "unknown palier 'stdd'" in out["error"]
    with kbc.connect_closing() as conn:
        assert kb.list_tasks(conn) == []
