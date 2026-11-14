"""One malformed record in a hub state file must not take every other record down with it."""
import json

from tools.skills_hub import HubLockFile, TapsManager


def test_lock_keeps_good_entries_beside_a_malformed_one(tmp_path):
    path = tmp_path / "lock.json"
    path.write_text(json.dumps({"version": 1, "installed": {
        "bad": "oops", "num": 3, "ok": {"install_path": "ok", "identifier": "a/b/ok"}}}))
    lock = HubLockFile(path)
    assert lock.list_installed() == [{"name": "ok", "install_path": "ok", "identifier": "a/b/ok"}]
    assert lock.get_installed("bad") is None
    assert set(lock.by_identifier()) == {"a/b/ok"}
    # A wrong-shaped file reads as empty instead of raising in every consumer.
    path.write_text('{"version": 1, "installed": []}')
    assert HubLockFile(path).list_installed() == []


def test_taps_skip_a_malformed_tap_and_still_accept_new_ones(tmp_path):
    path = tmp_path / "taps.json"
    path.write_text(json.dumps({"taps": [1, {"path": "x/"}, {"repo": "a/b", "path": "s/"}]}))
    taps = TapsManager(path)
    assert taps.load() == [{"repo": "a/b", "path": "s/"}]
    assert taps.add("c/d") is True
    assert [t["repo"] for t in taps.load()] == ["a/b", "c/d"]
    assert taps.remove("a/b") is True
    assert [t["repo"] for t in taps.load()] == ["c/d"]
