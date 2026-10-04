"""``profiles.list`` must never emit a non-finite float out of ``profile.yaml`` ui_meta.

YAML 1.1 resolves an unquoted Bot Mode chat id like ``20260101_120000_1e0400`` to ``inf``
(``e0400`` reads as an exponent). JSON has no non-finite spelling, so the pre-fix listing
serialized it as the bare token ``Infinity`` and a strict client dropped the whole profile list
(#132800). The JSON boundary maps such scalars to ``null``: the roster still loads, only the
unsalvageable value is lost, and the frame stays strict-JSON clean end to end.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import tui_gateway.server as srv
from tui_gateway import profile_roster_cache as cache


@pytest.fixture(autouse=True)
def _clear_memo():
    cache.invalidate()
    yield
    cache.invalidate()


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # ``list_profiles`` also scans ~/.local/bin for wrapper aliases; on a dev box that dir
    # holds real binaries (node, …) the home_io_guard correctly refuses to open. The alias
    # scan is orthogonal to ui_meta, so serve an empty map instead.
    import hermes_cli.profiles as profiles_mod
    monkeypatch.setattr(profiles_mod, "build_alias_map", lambda: {})
    (tmp_path / "config.yaml").write_text("model:\n  provider: openai\n", encoding="utf-8")
    bob = tmp_path / "profiles" / "bob"
    bob.mkdir(parents=True)
    (bob / "config.yaml").write_text("model:\n  provider: openai\n", encoding="utf-8")
    return tmp_path


def _write_chat_line(home: Path, chat_line: str) -> None:
    (home / "profiles" / "bob" / "profile.yaml").write_text(
        f"display_name: Bob\nui_meta:\n  hermes-bots:\n    title: Bob\n    {chat_line}\n"
        "_ui_meta_revisions:\n  hermes-bots: 1\n", encoding="utf-8")


def _envelope() -> dict:
    return srv._methods["profiles.list"](1, {"include_sessions": False})


@pytest.mark.parametrize("chat_line", [
    "chat: 20260101_120000_1e0400",  # the issue's id: YAML 1.1 exponent -> inf
    "chat: -.inf",                   # YAML 1.1 spells negative infinity
    "chat: .nan",                    # YAML 1.1 spells NaN
])
def test_a_non_finite_ui_meta_scalar_becomes_null(home, chat_line):
    """The roster still loads and sibling keys survive; only the unsalvageable scalar is lost."""
    _write_chat_line(home, chat_line)
    row = next(p for p in _envelope()["result"]["profiles"] if p["name"] == "bob")
    assert row["ui_meta"]["hermes-bots"]["title"] == "Bob"
    assert row["ui_meta"]["hermes-bots"]["chat"] is None


def test_the_listing_envelope_stays_strict_json(home):
    """Wire contract: the whole reply serializes with ``allow_nan=False`` — pre-fix this raised
    ``ValueError`` because the envelope carried ``inf`` where a bare ``Infinity`` would go."""
    _write_chat_line(home, "chat: 20260101_120000_1e0400")
    assert json.dumps(_envelope(), allow_nan=False)
