"""Tests for ``skill_ledger.entry_diff`` — blob-reconstructed unified diffs.

The ledger stores full before/after file contents content-addressed; ``entry_diff``
turns a row back into a human-readable answer to "what did this mutation change?".
Degradation paths (missing blob, truncation, identical hashes) must never raise.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def ledger_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so blob writes and path shortening stay inside tmp_path."""
    from tools import skill_ledger

    home = tmp_path / "home"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setattr(skill_ledger, "get_hermes_home", lambda: home)
    return home


def _blob(env, text: str) -> str:
    from tools import skill_ledger

    return skill_ledger._store_blob(text.encode("utf-8"))


def _path(env, name="SKILL.md") -> str:
    return str(env / "skills" / "my-skill" / name)


def _entry(env, old=None, new=None):
    from tools import skill_ledger

    before = [{"path": _path(env), "sha256": _blob(env, old)}] if old is not None else []
    after = [{"path": _path(env), "sha256": _blob(env, new)}] if new is not None else []
    return {"action": "patch", "skill": "my-skill", "evidence": {},
            "before": before, "after": after}


def test_modified_file_yields_unified_diff(ledger_env):
    from tools.skill_ledger import entry_diff

    diff = entry_diff(_entry(ledger_env, "line one\nline two\n", "line one\nline TWO\n"))
    assert any(l.startswith("--- a") for l in diff)
    assert any(l.startswith("+++ b") for l in diff)
    assert "-line two" in diff
    assert "+line TWO" in diff
    # paths are shortened relative to HERMES_HOME
    assert any("/skills/my-skill/SKILL.md" in l for l in diff if l.startswith("--- a"))


def test_created_file_diffs_against_empty(ledger_env):
    from tools.skill_ledger import entry_diff

    diff = entry_diff(_entry(ledger_env, None, "fresh content\n"))
    body = [l for l in diff if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
    assert body == ["+fresh content"]


def test_deleted_file_diffs_to_empty(ledger_env):
    from tools.skill_ledger import entry_diff

    diff = entry_diff(_entry(ledger_env, "gone soon\n", None))
    body = [l for l in diff if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
    assert body == ["-gone soon"]


def test_identical_hashes_are_skipped(ledger_env):
    from tools.skill_ledger import entry_diff

    same = _blob(ledger_env, "unchanged\n")
    entry = {"before": [{"path": _path(ledger_env), "sha256": same}],
             "after": [{"path": _path(ledger_env), "sha256": same}]}
    assert entry_diff(entry) == []


def test_missing_blob_degrades_to_hash_note(ledger_env):
    from tools.skill_ledger import entry_diff

    entry = {"before": [{"path": _path(ledger_env), "sha256": "f" * 64}],
             "after": [{"path": _path(ledger_env), "sha256": _blob(ledger_env, "x")}],
             "evidence": {}}
    diff = entry_diff(entry)
    assert len(diff) == 1
    assert "blob missing" in diff[0] and "f" * 64 in diff[0]


def test_long_diff_is_truncated_with_marker(ledger_env):
    from tools.skill_ledger import entry_diff

    old = "".join(f"old {i}\n" for i in range(500))
    new = "".join(f"new {i}\n" for i in range(500))
    diff = entry_diff(_entry(ledger_env, old, new), max_lines=10)
    assert len(diff) == 11
    assert "truncated" in diff[-1]


def test_non_utf8_blob_decodes_with_replacement(ledger_env):
    from tools import skill_ledger
    from tools.skill_ledger import entry_diff

    sha = skill_ledger._store_blob(b"\xff\xfe\x00binary-ish")
    entry = {"before": [], "after": [{"path": _path(ledger_env), "sha256": sha}], "evidence": {}}
    diff = entry_diff(entry)  # must not raise
    assert any(l.startswith("+") for l in diff)


def test_empty_entry_is_empty(ledger_env):
    from tools.skill_ledger import entry_diff

    assert entry_diff({"before": [], "after": [], "evidence": {}}) == []
