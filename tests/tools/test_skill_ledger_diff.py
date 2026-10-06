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


def test_sparse_invalid_utf8_still_renders_as_text(ledger_env):
    from tools import skill_ledger
    from tools.skill_ledger import entry_diff

    # Sparse bad bytes (low replacement ratio) decode as text; dense corruption or NULs
    # are what the binary-content note is for.
    sha = skill_ledger._store_blob("almost utf-8 \xff with one bad byte\n".encode("latin-1"))
    entry = {"before": [], "after": [{"path": _path(ledger_env), "sha256": sha}], "evidence": {}}
    diff = entry_diff(entry)  # must not raise
    assert any(l.startswith("+almost utf-8") for l in diff)


def test_empty_entry_is_empty(ledger_env):
    from tools.skill_ledger import entry_diff

    assert entry_diff({"before": [], "after": [], "evidence": {}}) == []


# ─── malformed manifests (hand-edited ledger rows) must never raise ─────────


@pytest.mark.parametrize("bad", [
    {"before": 7, "after": None},
    {"before": object()},
])
def test_noniterable_manifests_yield_note_not_traceback(ledger_env, bad):
    from tools.skill_ledger import entry_diff

    out = entry_diff(dict(bad, evidence={}))
    assert any("diff unavailable" in l for l in out)


@pytest.mark.parametrize("weird", [
    {"before": "oops", "after": []},
    {"before": ["/not/a/dict"], "after": []},
    {"after": {"path": "x"}},
])
def test_iterable_but_wrong_shape_manifests_degrade_quietly(ledger_env, weird):
    """Str/list-of-non-dict/dict manifests filter to empty — no traceback, no note."""
    from tools.skill_ledger import entry_diff

    assert entry_diff(dict(weird, evidence={})) == []


def test_created_and_deleted_empty_files_get_explicit_note(ledger_env):
    from tools.skill_ledger import entry_diff

    empty = _blob(ledger_env, "")
    rel = "/skills/my-skill/SKILL.md"  # home-stripped display path
    created = {"before": [], "evidence": {},
               "after": [{"path": _path(ledger_env), "sha256": empty}]}
    assert entry_diff(created) == [f"{rel}: created empty file"]
    deleted = {"after": [], "evidence": {},
               "before": [{"path": _path(ledger_env), "sha256": empty}]}
    assert entry_diff(deleted) == [f"{rel}: deleted empty file"]


def test_binary_content_gets_hash_note(ledger_env):
    from tools import skill_ledger
    from tools.skill_ledger import entry_diff

    sha = skill_ledger._store_blob(b"\x00\x01\x02\x00binary-with-nuls")
    entry = {"before": [], "evidence": {},
             "after": [{"path": _path(ledger_env), "sha256": sha}]}
    diff = entry_diff(entry)
    assert len(diff) == 1
    assert "binary content" in diff[0] and sha in diff[0]


def test_both_blobs_missing_list_both_shas(ledger_env):
    from tools.skill_ledger import entry_diff

    entry = {"evidence": {},
             "before": [{"path": _path(ledger_env), "sha256": "a" * 64}],
             "after": [{"path": _path(ledger_env), "sha256": "b" * 64}]}
    diff = entry_diff(entry)
    assert len(diff) == 1
    assert f"before {'a' * 64}" in diff[0] and f"after {'b' * 64}" in diff[0]


def test_global_budget_stops_with_remaining_count(ledger_env):
    from tools.skill_ledger import entry_diff

    p1, p2 = _path(ledger_env, "a.md"), _path(ledger_env, "b.md")
    entry = {
        "evidence": {},
        "before": [{"path": p1, "sha256": _blob(ledger_env, "old1\n")},
                   {"path": p2, "sha256": _blob(ledger_env, "old2\n")}],
        "after": [{"path": p1, "sha256": _blob(ledger_env, "new1\n")},
                  {"path": p2, "sha256": _blob(ledger_env, "new2\n")}],
    }
    diff = entry_diff(entry, max_total_lines=2)
    assert "+new1" in "\n".join(diff)
    assert any("more changed path(s) not shown" in l for l in diff)
    assert "+new2" not in "\n".join(diff)
