"""Regression tests: corrupt/wrong-shape hub state files must read as empty.

`_JsonStateFile._read` promises EMPTY "on every miss/corrupt read", but it
returned any successfully-parsed JSON — `[]`, `null`, scalars — so a corrupt
lock.json/taps.json crashed every consumer downstream (TypeError/KeyError/
AttributeError).  Shape is now validated: top level must be an object, declared
container keys must keep their EMPTY types, missing container keys are filled,
malformed items are dropped, and undecodable bytes degrade like #68053.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.skills_hub import HubLockFile, TapsManager


def _file(tmp_path: Path, content) -> Path:
    p = tmp_path / "state.json"
    p.write_bytes(content if isinstance(content, bytes) else content.encode())
    return p


@pytest.mark.parametrize("content", ['[]', '"x"', '5', 'null', 'true', '{bad'])
def test_lock_wrong_shape_reads_empty(tmp_path, content):
    lock = HubLockFile(_file(tmp_path, content))
    assert lock.load() == {"version": 1, "installed": {}}
    assert lock.list_installed() == []
    assert lock.get_installed("anything") is None


@pytest.mark.parametrize("content", ['[]', '"x"', '5', 'null', '{bad'])
def test_taps_wrong_shape_reads_empty(tmp_path, content):
    taps = TapsManager(_file(tmp_path, content))
    assert taps.load() == []
    with pytest.raises(ValueError, match="corrupt"):  # corrupt state is never clobbered
        taps.add("owner/repo")


def test_lock_wrong_container_type_reads_empty(tmp_path):
    lock = HubLockFile(_file(tmp_path, '{"installed": [], "version": 1}'))
    assert lock.load() == {"version": 1, "installed": {}}


def test_lock_missing_installed_key_is_filled(tmp_path):
    """A dict missing `installed` gets the EMPTY container, not a KeyError."""
    lock = HubLockFile(_file(tmp_path, '{"version": 2, "extra": 1}'))
    data = lock.load()
    assert data["installed"] == {}
    assert data["version"] == 2 and data["extra"] == 1  # other keys preserved


def test_lock_nondict_entries_filtered(tmp_path):
    lock = HubLockFile(_file(tmp_path,
        '{"installed": {"bad": "str", "ok": {"install_path": "x/ok"}, "num": 3}}'))
    assert lock.list_installed() == [{"name": "ok", "install_path": "x/ok"}]
    assert lock.get_installed("bad") is None
    assert lock.get_installed("ok") == {"install_path": "x/ok"}


def test_lock_valid_file_unchanged(tmp_path):
    entry = {"install_path": "cat/s", "source": "github"}
    lock = HubLockFile(_file(tmp_path, json.dumps({"version": 1, "installed": {"s": entry}})))
    assert lock.list_installed() == [{"name": "s", **entry}]


def test_taps_malformed_items_dropped(tmp_path):
    taps = TapsManager(_file(tmp_path, json.dumps(
        {"taps": [1, "x", {"no_repo": 1}, {"repo": "a/b", "path": "s/"}]})))
    assert taps.load() == [{"repo": "a/b", "path": "s/"}]


def test_taps_wrong_container_type_reads_empty(tmp_path):
    assert TapsManager(_file(tmp_path, '{"taps": "abc"}')).load() == []
    assert TapsManager(_file(tmp_path, '{"taps": {"r": 1}}')).load() == []


def test_lock_non_utf8_degrades_not_crash(tmp_path):
    """Mirror of #68053: a Windows-1252 byte must not kill the whole file."""
    raw = b'{"version": 1, "installed": {"s": {"note": "caf\x97"}}}'
    lock = HubLockFile(_file(tmp_path, raw))
    assert lock.get_installed("s") is not None


def test_empty_shape_not_mutated(tmp_path):
    """The EMPTY deep-copy must not be shared/mutated by returned data."""
    lock = HubLockFile(_file(tmp_path, "[]"))
    lock.load()["installed"]["x"] = 1
    lock2 = HubLockFile(_file(tmp_path, "[]"))
    assert lock2.load()["installed"] == {}
    assert HubLockFile.EMPTY == {"version": 1, "installed": {}}


def test_record_install_works_after_shape_fill(tmp_path):
    """record_install on a file missing `installed` writes instead of KeyError."""
    lock = HubLockFile(_file(tmp_path, '{"version": 2}'))
    lock.record_install("sk", "github", "o/r", "community", "clean", "h", "a/sk", [])
    assert lock.get_installed("sk")["install_path"] == "a/sk"
    assert lock.load()["version"] == 2


# --- Writes fail closed on corrupt state (reads still degrade) ----------------


@pytest.mark.parametrize("content", ['[]', '{bad', '{"installed": []}'])
def test_write_refuses_to_clobber_corrupt_lock(tmp_path, content):
    """Reads degrade to EMPTY, but a corrupt lock must never be silently
    overwritten — its provenance is the only record of what was installed."""
    p = _file(tmp_path, content)
    lock = HubLockFile(p)
    assert lock.list_installed() == []           # read path works
    with pytest.raises(ValueError, match="corrupt"):
        lock.record_install("sk", "github", "o/r", "community", "clean", "h", "a/sk", [])
    assert p.read_text() == (content if isinstance(content, str) else content.decode())  # untouched


def test_taps_add_refuses_on_corrupt_file(tmp_path):
    p = _file(tmp_path, '{"taps": "broken"}')
    taps = TapsManager(p)
    assert taps.load() == []
    with pytest.raises(ValueError, match="corrupt"):
        taps.add("a/b")
    assert p.read_text() == '{"taps": "broken"}'


def test_write_to_missing_or_healthy_file_works(tmp_path):
    p = tmp_path / "fresh.json"                    # missing: fine
    TapsManager(p).add("a/b")
    assert json.loads(p.read_text())["taps"] == [{"repo": "a/b", "path": "skills/"}]
    p2 = _file(tmp_path, '{"taps": [{"repo": "a/b"}]}')  # healthy: fine
    assert TapsManager(p2).add("c/d") is True
    assert len(json.loads(p2.read_text())["taps"]) == 2


def test_write_allowed_when_only_scalars_differ(tmp_path):
    """A wrong-typed scalar (version) is not corruption — writes still go through."""
    p = _file(tmp_path, '{"version": "two", "installed": {}}')
    lock = HubLockFile(p)
    lock.record_install("sk", "github", "o/r", "community", "clean", "h", "a/sk", [])
    assert lock.get_installed("sk") is not None


def test_taps_nonstr_path_or_bucket_dropped(tmp_path):
    taps = TapsManager(_file(tmp_path, json.dumps({"taps": [
        {"repo": "a/b", "path": 5}, {"repo": "c/d", "bucket": None},
        {"repo": "e/f", "path": "s/"}]})))
    assert taps.load() == [{"repo": "e/f", "path": "s/"}]


# --- Shared index cache: a corrupt entry must be a miss, not a crash ---------


import contextlib


@contextlib.contextmanager
def _injected(**attrs):
    """Set test-injected resolver attrs and DELETE them on teardown — the module's
    __getattr__ means monkeypatch.setattr would capture a resolved path and restore
    it as a permanent real attribute."""
    import tools.skills_hub as hub_mod
    for name, value in attrs.items():
        setattr(hub_mod, name, value)
    try:
        yield
    finally:
        for name in attrs:
            delattr(hub_mod, name)


def _cache_dir(tmp_path) -> Path:
    return tmp_path


def test_cached_metas_wrong_shape_misses(tmp_path):
    from tools.skills_hub_models import _cached_metas
    d = tmp_path
    with _injected(INDEX_CACHE_DIR=d):
        for content in ('{"not": "a list"}', '[1, "x"]', '5'):
            (d / "k.json").write_text(content)
            assert _cached_metas("k") is None


def test_memo_json_wrong_shape_recomputes(tmp_path):
    from tools.skills_hub_models import _memo_json
    with _injected(INDEX_CACHE_DIR=tmp_path):
        (tmp_path / "lobehub_index.json").write_text('"just a string"')
        assert _memo_json("lobehub_index", lambda: {"agents": []},
                          valid=lambda c: isinstance(c, (dict, list))) == {"agents": []}


def test_stale_index_cache_wrong_shape_returns_none(tmp_path):
    from tools.skills_hub_search import _load_stale_index_cache
    with _injected(INDEX_CACHE_DIR=tmp_path):
        (tmp_path / "hermes-index.json").write_text('[1, 2]')
        assert _load_stale_index_cache() is None


def test_index_cache_non_utf8_is_a_miss(tmp_path):
    from tools.skills_hub_models import _cached_metas
    with _injected(INDEX_CACHE_DIR=tmp_path):
        (tmp_path / "k.json").write_bytes(b'{"x": \x97}')  # Windows-1252 byte outside a string
        assert _cached_metas("k") is None


# --- Snapshot import: user-supplied file shapes ------------------------------


def _import(tmp_path, content, capsys):
    from hermes_cli.skills_hub import do_snapshot_import
    with _injected(TAPS_FILE=tmp_path / "taps.json"):
        do_snapshot_import(str(_file(tmp_path, content)))
    return capsys.readouterr().out


def test_snapshot_import_non_dict(tmp_path, capsys):
    out = _import(tmp_path, '["not", "a", "snapshot"]', capsys)
    assert "Invalid snapshot shape" in out


def test_snapshot_import_malformed_items_skipped(tmp_path, capsys):
    out = _import(tmp_path, json.dumps({
        "taps": [1, {"repo": "a/b"}], "skills": ["junk", {"no_id": 1}]}),
        capsys)
    assert "Restored 1 tap(s)" in out
    assert "Skipping malformed entry" in out
    assert "Skipping entry with no identifier" in out


def test_snapshot_import_non_utf8_no_crash(tmp_path, capsys):
    out = _import(tmp_path, b'{"skills": [\x97]}', capsys)
    assert "Skipping malformed entry" in out or "Invalid JSON" in out or "import" in out.lower()


# --- e2e: `hermes skills list` with a corrupt lock.json -----------------------


def test_do_list_survives_corrupt_lock(tmp_path, capsys):
    from hermes_cli.skills_hub import do_list
    skills_dir = tmp_path / "skills"
    hub_dir = skills_dir / ".hub"
    hub_dir.mkdir(parents=True)
    (hub_dir / "lock.json").write_text('[]')
    with _injected(SKILLS_DIR=skills_dir):  # resolver seam — all hub paths follow
        do_list()  # the finding's repro command — must print a table, not crash
    assert "Installed Skills" in capsys.readouterr().out
