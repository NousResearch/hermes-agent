"""``profiles.list`` re-parses a profile's ``profile.yaml`` for ui_meta only when it has changed.

The roster polls this every 5s per connection, and the listing already parsed that same file once
(``read_profile_meta``, for description/display_name) before parsing it again here for ui_meta.

The memo must be invisible: an edit — including one made through the ui_meta CAS writer, which
reads the raw document uncached and writes it back — has to show on the very next listing. And
``has_avatar`` is deliberately NOT cached, so an avatar dropped in without touching profile.yaml
still appears.
"""
from __future__ import annotations

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
    (tmp_path / "config.yaml").write_text("model:\n  provider: openai\n", encoding="utf-8")
    bob = tmp_path / "profiles" / "bob"
    bob.mkdir(parents=True)
    (bob / "config.yaml").write_text("model:\n  provider: openai\n", encoding="utf-8")
    (bob / "profile.yaml").write_text(
        "display_name: Bob\nui_meta:\n  hermes-bots:\n    title: Bob\n"
        "_ui_meta_revisions:\n  hermes-bots: 1\n", encoding="utf-8")
    return tmp_path

def _row(name="bob", **params):
    envelope = srv._methods["profiles.list"](1, {"include_sessions": False, **params})
    return next(p for p in envelope["result"]["profiles"] if p["name"] == name)

def test_the_cas_writer_round_trips_through_the_listing(home):
    """The real write path: profiles.configure reads the raw document, mutates and writes it back."""
    before = _row()["ui_meta_revisions"]["hermes-bots"]

    envelope = srv._methods["profiles.configure"](2, {
        "name": "bob",
        "ui_meta": {"hermes-bots": {"title": "Bobby"}},
        "ui_meta_expected_revisions": {"hermes-bots": before},
    })
    assert envelope["result"]["applied"]["ui_meta"] is True

    row = _row()
    assert row["ui_meta"]["hermes-bots"]["title"] == "Bobby"
    assert row["ui_meta_revisions"]["hermes-bots"] == before + 1

def test_an_avatar_added_without_touching_profile_yaml_is_still_seen(home):
    """``has_avatar`` stays live — that is why it is not part of the cached value."""
    assert _row()["has_avatar"] is False

    assets = home / "profiles" / "bob" / "assets"
    assets.mkdir(parents=True)
    (assets / "avatar.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    assert _row()["has_avatar"] is True
