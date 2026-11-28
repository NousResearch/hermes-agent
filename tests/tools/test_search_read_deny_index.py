"""search_files resolves the credential read denylist once per search, not once per result.

Regression for #159084: the result filter rebuilt the denylist (a ``resolve()`` per profile
home and store name) for every result path, so a search slowed down with the number of profiles.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import agent.file_safety as fs
from tools.file_operations import SearchResult
from tools.file_tools import _filter_read_blocked_search_results

DENIED = (
    "auth.json", ".env", "auth/google_oauth.json", "mcp-tokens", "mcp-tokens/github.json",
    "vault/vault.key", "skills/.hub/index.json",
)


@pytest.fixture()
def profile_homes(tmp_path, monkeypatch):
    """Session runs in profile ``active``; ``other`` is a sibling. Returns (root, active, sibling)."""
    root = tmp_path / "hermes_root"
    active, sibling = root / "profiles" / "active", root / "profiles" / "other"
    active.mkdir(parents=True)
    sibling.mkdir()
    monkeypatch.setattr(fs, "_hermes_home_path", lambda: active)
    monkeypatch.setattr(fs, "_hermes_root_path", lambda: root)
    return root, active, sibling


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        path.write_text("x", encoding="utf-8")
    return str(path)


def _count_calls(monkeypatch, name: str) -> list[int]:
    calls = [0]
    original = getattr(fs, name)

    def counted(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fs, name, counted)
    return calls


def test_index_gives_the_same_verdict_as_a_fresh_denylist(profile_homes, tmp_path):
    root, active, sibling = profile_homes
    (active / "mcp-tokens").mkdir()
    paths = [_touch(home / rel) for home in (root, active, sibling) for rel in DENIED if rel != "mcp-tokens"]
    paths += [str(active / "mcp-tokens"), _touch(tmp_path / "project" / ".env"), "/proc/self/environ",
              _touch(tmp_path / "project" / "main.py"), _touch(sibling / "notes.md"), str(tmp_path / "missing.txt")]
    index = fs.read_deny_index()
    for path in paths:
        assert fs.get_read_block_error(path, index=index) == fs.get_read_block_error(path), path
    assert fs.get_read_block_error(paths[0], index=index)
    assert fs.get_read_block_error(str(tmp_path / "project" / "main.py"), index=index) is None


def test_filter_drops_credential_hits_from_every_result_shape(profile_homes, tmp_path):
    root, active, sibling = profile_homes
    kept = _touch(tmp_path / "project" / "main.py")
    denied = [_touch(home / rel) for home in (root, active, sibling) for rel in ("auth.json", ".env", "vault/vault.key")]
    result = SearchResult(files=[kept, *denied], counts={kept: 2, **dict.fromkeys(denied, 1)})
    assert _filter_read_blocked_search_results(result, "deny-index") == 2 * len(denied)
    assert result.files == [kept]
    assert result.counts == {kept: 2}


def test_denylist_is_built_once_however_many_results(profile_homes, tmp_path, monkeypatch):
    builds = _count_calls(monkeypatch, "_credential_store_dirs")
    per_size = []
    for size in (2, 40):
        builds[0] = 0
        files = [_touch(tmp_path / "project" / f"f{i}.py") for i in range(size)]
        _filter_read_blocked_search_results(SearchResult(files=files), "deny-index")
        per_size.append(builds[0])
    assert per_size[0] == per_size[1] == 1


def test_profile_created_between_searches_is_denied_in_the_next(profile_homes):
    root, _active, _sibling = profile_homes
    late = root / "profiles" / "late" / "auth.json"

    def search() -> list[str]:
        result = SearchResult(files=[str(late)])
        _filter_read_blocked_search_results(result, "deny-index")
        return result.files

    search()
    _touch(late)
    assert search() == []
