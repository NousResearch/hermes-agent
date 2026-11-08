# ABOUTME: Verify group chat history persists outside profile metadata through the real RPCs.
# ABOUTME: Cover migration, revisions, profile isolation, and failed writes without losing logs.

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import subprocess
import sys

import pytest

import hermes_yaml as yaml
import tui_gateway.server as srv
from tui_gateway import profile_roster_cache


KEY = "hermes-bots-groups"
STATE = Path("state/group-chat.yaml")


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for path in (root, root / "profiles" / "helper"):
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.yaml").write_text("model: {}\n", encoding="utf-8")
        (path / "profile.yaml").write_text("display_name: Example\n", encoding="utf-8")
    profile_roster_cache.invalidate()
    return root


def snapshot(text):
    return {"version": 3, "rooms": {"id:room": {"log": [{"text": text, "at": 1}]}}}


def configure(value, revision, name="default"):
    return srv._methods["profiles.configure"]("write", {
        "name": name, "ui_meta": {KEY: value},
        "ui_meta_expected_revisions": {KEY: revision},
    })["result"]["applied"]


def row(name="default"):
    result = srv._methods["profiles.list"]("read", {"include_sessions": False})
    return next(p for p in result["result"]["profiles"] if p["name"] == name)


@pytest.mark.parametrize("name", ["default", "helper"])
def test_history_round_trips_without_changing_profile_yaml(home, name):
    target = home if name == "default" else home / "profiles" / name
    before = (target / "profile.yaml").read_bytes()
    assert KEY not in row(name).get("ui_meta", {})  # prime the roster cache

    assert configure(snapshot("first"), 0, name)["ui_meta"] is True
    assert (target / "profile.yaml").read_bytes() == before
    assert yaml.safe_load((target / STATE).read_text())["snapshot"] == snapshot("first")
    assert row(name)["ui_meta"][KEY] == snapshot("first")
    assert configure(snapshot("second"), 1, name)["ui_meta"] is True
    assert row(name)["ui_meta"][KEY] == snapshot("second")
    assert row(name)["ui_meta_revisions"][KEY] == 2
    assert (target / "profile.yaml").read_bytes() == before
    other = "helper" if name == "default" else "default"
    assert KEY not in row(other).get("ui_meta", {})


def test_legacy_history_migrates_on_metadata_write_without_losing_bot_settings(home):
    legacy = {"display_name": "Example", "ui_meta": {
        "hermes-bots": {"title": "Example", "pinned": True}, KEY: snapshot("existing")},
        "_ui_meta_revisions": {KEY: 7, "hermes-bots": 2}}
    (home / "profile.yaml").write_text(yaml.safe_dump(legacy), encoding="utf-8")
    assert row()["ui_meta"][KEY] == snapshot("existing")

    applied = srv._methods["profiles.configure"]("write", {
        "name": "default", "ui_meta": {"other-plugin": {"color": "blue"}},
    })["result"]["applied"]

    assert applied["ui_meta"] is True
    meta = yaml.safe_load((home / "profile.yaml").read_text())
    assert KEY not in meta["ui_meta"]
    assert KEY not in meta["_ui_meta_revisions"]
    assert meta["ui_meta"]["hermes-bots"] == legacy["ui_meta"]["hermes-bots"]
    assert row()["ui_meta"][KEY] == snapshot("existing")
    assert row()["ui_meta_revisions"][KEY] == 7


def test_deleted_history_keeps_revision_and_rejects_stale_recreation(home):
    assert configure(snapshot("first"), 0)["ui_meta"] is True
    assert configure(None, 1)["ui_meta"] is True
    assert KEY not in row().get("ui_meta", {})
    assert row()["ui_meta_revisions"][KEY] == 2
    stale = configure(snapshot("stale"), 1)
    assert stale["ui_meta"] is False
    assert stale["ui_meta_conflicts"][KEY]["actual"] == 2
    assert yaml.safe_load((home / STATE).read_text()) == {"revision": 2, "snapshot": None}


def test_concurrent_group_writers_preserve_compare_and_swap(home):
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda text: configure(snapshot(text), 0), ["one", "two"]))
    assert sum(o["ui_meta"] is True for o in outcomes) == 1
    assert sum(o["ui_meta"] is False for o in outcomes) == 1
    assert row()["ui_meta_revisions"][KEY] == 1
    assert KEY not in yaml.safe_load((home / "profile.yaml").read_text()).get("ui_meta", {})


def test_failed_state_write_preserves_legacy_history(home, monkeypatch):
    import utils
    original = yaml.safe_dump({"ui_meta": {KEY: snapshot("existing")}, "_ui_meta_revisions": {KEY: 4}})
    (home / "profile.yaml").write_text(original, encoding="utf-8")
    real_write = utils.atomic_yaml_write

    def fail_state(path, data, **kwargs):
        if Path(path) == home / STATE:
            raise OSError("disk full")
        return real_write(path, data, **kwargs)

    monkeypatch.setattr(utils, "atomic_yaml_write", fail_state)
    assert configure(snapshot("replacement"), 4)["ui_meta"] is False
    assert (home / "profile.yaml").read_text() == original
    assert row()["ui_meta"][KEY] == snapshot("existing")


def test_corrupt_state_is_not_overwritten(home):
    (home / STATE).parent.mkdir()
    (home / STATE).write_text("[invalid", encoding="utf-8")
    before = (home / "profile.yaml").read_bytes()
    assert configure(snapshot("replacement"), 0)["ui_meta"] is False
    assert (home / STATE).read_text() == "[invalid"
    assert (home / "profile.yaml").read_bytes() == before


def test_interrupted_cleanup_keeps_new_history_and_can_resume(home, monkeypatch):
    import utils
    from tui_gateway.group_chat_storage import migrate_group_chat_state

    original = {"ui_meta": {KEY: snapshot("old")}, "_ui_meta_revisions": {KEY: 3}}
    (home / "profile.yaml").write_text(yaml.safe_dump(original), encoding="utf-8")
    real_write = utils.atomic_yaml_write

    def fail_profile(path, data, **kwargs):
        if Path(path) == home / "profile.yaml":
            raise OSError("profile temporarily read-only")
        return real_write(path, data, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(utils, "atomic_yaml_write", fail_profile)
        assert configure(snapshot("new"), 3)["ui_meta"] is True

    assert row()["ui_meta"][KEY] == snapshot("new")
    assert row()["ui_meta_revisions"][KEY] == 4
    migrate_group_chat_state(home)
    assert KEY not in yaml.safe_load((home / "profile.yaml").read_text()).get("ui_meta", {})
    before = (home / STATE).read_bytes()
    migrate_group_chat_state(home)
    assert (home / STATE).read_bytes() == before
    assert row()["ui_meta"][KEY] == snapshot("new")


def test_equal_revision_tombstone_wins_over_leftover_legacy_history(home):
    (home / "profile.yaml").write_text(yaml.safe_dump({
        "ui_meta": {KEY: snapshot("old")}, "_ui_meta_revisions": {KEY: 2},
    }), encoding="utf-8")
    (home / STATE).parent.mkdir()
    (home / STATE).write_text(yaml.safe_dump({"revision": 2, "snapshot": None}), encoding="utf-8")
    assert KEY not in row().get("ui_meta", {})
    assert row()["ui_meta_revisions"][KEY] == 2


def test_separate_backend_processes_cannot_replace_the_same_revision(home):
    script = """
import json
import sys
import tui_gateway.server as srv
result = srv._methods['profiles.configure']('write', {
    'name': 'default', 'ui_meta': {'hermes-bots-groups': {'rooms': {}, 'label': sys.argv[1]}},
    'ui_meta_expected_revisions': {'hermes-bots-groups': 0},
})
print(json.dumps(result['result']['applied']['ui_meta']), file=sys.__stdout__)
"""

    def write(label):
        result = subprocess.run([sys.executable, "-c", script, label], capture_output=True,
                                text=True, check=True, timeout=30,
                                env=os.environ | {"HOME": str(home.parent), "HERMES_HOME": str(home)})
        return result.stdout.strip().splitlines()[-1]

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(write, ["one", "two"]))
    assert sorted(outcomes) == ["false", "true"]
    assert row()["ui_meta_revisions"][KEY] == 1


@pytest.mark.parametrize("name", ["default", "helper"])
@pytest.mark.parametrize("contents", ["[invalid", "{}\n", "revision: 2\n"])
def test_corrupt_group_state_does_not_hide_healthy_profiles(home, name, contents):
    target = home if name == "default" else home / "profiles" / name
    (target / "profile.yaml").write_text(yaml.safe_dump({
        "display_name": "Example", "ui_meta": {"hermes-bots": {"color": "blue"}},
    }), encoding="utf-8")
    (target / STATE).parent.mkdir(exist_ok=True)
    (target / STATE).write_text(contents, encoding="utf-8")

    damaged = row(name)
    assert damaged["ui_meta"]["hermes-bots"] == {"color": "blue"}
    assert row("helper" if name == "default" else "default")["display_name"] == "Example"
    assert configure(snapshot("replacement"), 0, name)["ui_meta"] is False
    assert (target / STATE).read_text() == contents


def test_mixed_update_is_committed_and_recoverable_when_profile_write_fails(home, monkeypatch):
    import utils
    from tui_gateway.group_chat_storage import migrate_group_chat_state

    initial = {"display_name": "Example", "ui_meta": {
        KEY: snapshot("old"), "example-settings": {"color": "blue"}},
        "_ui_meta_revisions": {KEY: 3, "example-settings": 1}}
    (home / "profile.yaml").write_text(yaml.safe_dump(initial), encoding="utf-8")
    request = {"name": "default", "ui_meta": {
        KEY: snapshot("new"), "example-settings": {"color": "green"}},
        "ui_meta_expected_revisions": {KEY: 3, "example-settings": 1}}
    real_write = utils.atomic_yaml_write

    def fail_profile(path, data, **kwargs):
        if Path(path) == home / "profile.yaml":
            raise OSError("synthetic metadata failure")
        return real_write(path, data, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(utils, "atomic_yaml_write", fail_profile)
        applied = srv._methods["profiles.configure"]("write", request)["result"]["applied"]
        assert applied["ui_meta"] is True
        visible = row()
        assert visible["ui_meta"]["example-settings"] == {"color": "green"}
        assert visible["ui_meta"][KEY] == snapshot("new")
        assert visible["ui_meta_revisions"] == {KEY: 4, "example-settings": 2}
        assert yaml.safe_load((home / "profile.yaml").read_text()) == initial

    # A separate metadata editor must not lose unrelated fields during replay.
    edited = initial | {"description": "An independent edit"}
    edited["ui_meta"]["unrelated"] = {"pinned": True}
    (home / "profile.yaml").write_text(yaml.safe_dump(edited), encoding="utf-8")
    migrate_group_chat_state(home)
    stored = yaml.safe_load((home / "profile.yaml").read_text())
    assert stored["description"] == "An independent edit"
    assert stored["ui_meta"]["unrelated"] == {"pinned": True}
    assert stored["ui_meta"]["example-settings"] == {"color": "green"}
    assert KEY not in stored["ui_meta"]
    assert stored["_ui_meta_revisions"]["example-settings"] == 2
    assert row()["ui_meta_revisions"][KEY] == 4
    assert "pending_ui_meta" not in yaml.safe_load((home / STATE).read_text())
    stale = srv._methods["profiles.configure"]("retry", request)["result"]["applied"]
    assert stale["ui_meta"] is False
    assert stale["ui_meta_conflicts"][KEY]["actual"] == 4


def test_roster_does_not_mix_stale_metadata_with_a_completed_group_update(home, monkeypatch):
    import threading

    (home / "profile.yaml").write_text(yaml.safe_dump({
        "ui_meta": {KEY: snapshot("old"), "example-settings": {"color": "blue"}},
        "_ui_meta_revisions": {KEY: 3, "example-settings": 1},
    }), encoding="utf-8")
    metadata_read = threading.Event()
    writer_done = threading.Event()
    read_metadata = srv._read_profile_yaml

    def gated_read(profile_dir):
        metadata = read_metadata(profile_dir)
        if threading.current_thread().name == "roster" and Path(profile_dir) == home:
            metadata_read.set()
            writer_done.wait(2)
        return metadata

    monkeypatch.setattr(srv, "_read_profile_yaml", gated_read)
    observed = {}

    def read():
        observed.update(row())

    reader = threading.Thread(target=read, name="roster")
    reader.start()
    assert metadata_read.wait(5)
    try:
        srv._methods["profiles.configure"]("write", {
            "name": "default", "ui_meta": {
                KEY: snapshot("new"), "example-settings": {"color": "green"}},
            "ui_meta_expected_revisions": {KEY: 3, "example-settings": 1},
        })
    finally:
        writer_done.set()
        reader.join(timeout=5)
    assert not reader.is_alive()
    visible = observed["ui_meta"]
    assert (visible[KEY], visible["example-settings"]) in [
        (snapshot("old"), {"color": "blue"}),
        (snapshot("new"), {"color": "green"}),
    ]


def test_transient_roster_read_failure_is_not_cached_as_missing_history(home, monkeypatch):
    from tui_gateway import group_chat_storage

    assert configure(snapshot("first"), 0)["ui_meta"] is True
    profile_roster_cache.invalidate()

    def unavailable(profile_dir):
        raise OSError("synthetic transient state read failure")

    with monkeypatch.context() as fault:
        fault.setattr(group_chat_storage, "_read_state", unavailable)
        assert KEY not in row().get("ui_meta", {})
    assert row()["ui_meta"][KEY] == snapshot("first")
