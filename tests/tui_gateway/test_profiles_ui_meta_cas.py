"""Gateway-owned revisions for concurrent profile UI metadata updates.

Clients use ``profiles.list`` to read a shared key and its revision, merge a
local mutation, then pass that revision back to ``profiles.configure``.  The
gateway must let exactly one concurrent writer advance the key and make every
stale writer retry instead of silently replacing somebody else's state.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

import tui_gateway.server as srv


@pytest.fixture
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def _configure(ui_meta, expected=None):
    params = {"name": "default", "ui_meta": ui_meta}
    if expected is not None:
        params["ui_meta_expected_revisions"] = expected
    return srv._methods["profiles.configure"]("configure", params)["result"]["applied"]


def _default_profile():
    rows = srv._methods["profiles.list"](
        "list", {"include_sessions": False}
    )["result"]["profiles"]
    return next(row for row in rows if row["name"] == "default")


def test_ui_meta_revision_advances_and_stale_compare_and_swap_is_rejected(home):
    first = _configure({"shared-room": {"messages": ["one"]}})
    assert first["ui_meta"] is True
    assert first["ui_meta_revisions"] == {"shared-room": 1}

    second = _configure(
        {"shared-room": {"messages": ["one", "two"]}},
        {"shared-room": 1},
    )
    assert second["ui_meta"] is True
    assert second["ui_meta_revisions"] == {"shared-room": 2}

    stale = _configure(
        {"shared-room": {"messages": ["stale replacement"]}},
        {"shared-room": 1},
    )
    assert stale["ui_meta"] is False
    assert stale["ui_meta_conflicts"] == {
        "shared-room": {"expected": 1, "actual": 2}
    }

    row = _default_profile()
    assert row["ui_meta"]["shared-room"] == {"messages": ["one", "two"]}
    assert row["ui_meta_revisions"]["shared-room"] == 2


def test_ui_meta_revision_survives_key_deletion(home):
    _configure({"shared-room": {"messages": ["one"]}})
    deleted = _configure({"shared-room": None}, {"shared-room": 1})

    assert deleted["ui_meta"] is True
    assert deleted["ui_meta_revisions"] == {"shared-room": 2}
    row = _default_profile()
    assert "shared-room" not in row.get("ui_meta", {})
    assert row["ui_meta_revisions"]["shared-room"] == 2

    stale_recreate = _configure(
        {"shared-room": {"messages": ["resurrected"]}},
        {"shared-room": 1},
    )
    assert stale_recreate["ui_meta"] is False
    assert stale_recreate["ui_meta_conflicts"]["shared-room"]["actual"] == 2


def test_profiles_list_normalizes_yaml_timestamps_in_ui_meta(home):
    """#92506: an unquoted ISO timestamp under ui_meta is parsed by YAML as datetime; the
    profiles.list row must still be JSON-serializable (the value becomes its ISO string)."""
    (home / "profile.yaml").write_text(
        "ui_meta:\n  hermes-bots:\n    created: 2026-08-22T00:00:00Z\n",
        encoding="utf-8",
    )

    row = _default_profile()

    assert row["ui_meta"]["hermes-bots"]["created"] == "2026-08-22T00:00:00+00:00"
    json.dumps(row)


def test_two_concurrent_writers_cannot_both_replace_the_same_revision(home):
    def write(label):
        return _configure(
            {"shared-room": {"messages": [label]}},
            {"shared-room": 0},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write, ("alpha", "beta")))

    assert sum(result["ui_meta"] is True for result in results) == 1
    assert sum(result["ui_meta"] is False for result in results) == 1
    loser = next(result for result in results if result["ui_meta"] is False)
    assert loser["ui_meta_conflicts"]["shared-room"]["actual"] == 1
    assert _default_profile()["ui_meta_revisions"]["shared-room"] == 1


_SEEDED_PROFILE_YAML = (
    "role: setup\n"
    "display_name: Coder\n"
    "previous_names:\n- dev\n"
    "description: original\n"
    "ui_meta:\n  hermes-bots:\n    title: Coder\n"
)


def test_unparseable_profile_yaml_is_refused_not_replaced(home):
    """A ui_meta edit must not rewrite a profile.yaml it could not parse: the read-modify-write
    used to read it as ``{}`` and replace the file with only ui_meta, dropping role/display_name/
    previous_names/description while reporting the section applied."""
    path = home / "profile.yaml"
    path.write_text(_SEEDED_PROFILE_YAML + "extra: [unterminated\n", encoding="utf-8")
    before = path.read_bytes()

    applied = _configure({"hermes-bots": {"title": "Renamed"}})

    assert applied["ui_meta"] is False
    assert path.read_bytes() == before


def test_read_error_on_profile_yaml_is_refused_not_replaced(home, monkeypatch):
    """A transient read error (EMFILE/EIO) on an intact profile.yaml must fail the ui_meta
    section, not wipe every field it did not touch. A description in the same request must not
    land on a truncated file either."""
    from pathlib import Path

    path = home / "profile.yaml"
    path.write_text(_SEEDED_PROFILE_YAML, encoding="utf-8")
    before = path.read_bytes()
    real_read_text = Path.read_text

    def flaky_read_text(self, *args, **kwargs):
        if self.name == "profile.yaml":
            raise OSError(24, "Too many open files")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)
    applied = srv._methods["profiles.configure"]("configure", {
        "name": "default", "ui_meta": {"hermes-bots": {"title": "Renamed"}},
        "description": "edited",
    })["result"]["applied"]
    monkeypatch.setattr(Path, "read_text", real_read_text)

    assert applied["ui_meta"] is False
    assert path.read_bytes() == before
