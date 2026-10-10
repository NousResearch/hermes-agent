"""``profiles.list`` carries an ``avatar_rev`` that moves whenever the stored avatar file changes.

Desktop caches each bot's avatar data URL indefinitely and only re-downloads it when this token
differs from the one it fetched at, so replacing ``assets/avatar.*`` in place (same name, new
bytes) must change the token, and ``profiles.get_asset`` must report the same token it serves.
"""
from __future__ import annotations

import os

import pytest

import tui_gateway.server as srv
from tui_gateway import profile_roster_cache as cache


@pytest.fixture(autouse=True)
def _clean_memo():
    cache.invalidate()
    yield
    cache.invalidate()


@pytest.fixture
def tutor(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    d = tmp_path / "profiles" / "tutor"
    (d / "assets").mkdir(parents=True)
    (d / "config.yaml").write_text("model:\n  provider: openai\n", encoding="utf-8")
    return d


def _row(name="tutor"):
    rows = srv._methods["profiles.list"](1, {})["result"]["profiles"]
    return next(r for r in rows if r["name"] == name)


def _asset(name="tutor"):
    return srv._methods["profiles.get_asset"](1, {"name": name, "asset": "avatar"})["result"]


def test_no_avatar_has_no_rev(tutor):
    row = _row()
    assert row["has_avatar"] is False
    assert row["avatar_rev"] is None


def test_rev_moves_when_file_replaced_in_place(tutor):
    path = tutor / "assets" / "avatar.webp"
    path.write_bytes(b"RIFF....WEBPold")
    os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    first = _row()["avatar_rev"]
    assert first and first.startswith("webp:")
    assert _asset()["rev"] == first

    path.write_bytes(b"RIFF....WEBPnew-art")
    os.utime(path, ns=(2_000_000_000, 2_000_000_000))
    second = _row()["avatar_rev"]
    assert second != first
    assert _asset()["rev"] == second


def test_rev_tracks_the_file_get_asset_serves(tutor):
    (tutor / "assets" / "avatar.png").write_bytes(b"\x89PNG....")
    rev = _row()["avatar_rev"]
    assert rev.startswith("png:")
    assert _asset()["rev"] == rev
