"""A checkpoint remembers which prompt it was taken for (tools/checkpoint_manager.py).

Contracts pinned here:
* a checkpoint taken after ``new_turn(session_id, prompt_uid)`` lists back with that link, and its
  subject (the ``reason``) is exactly what it was before;
* a checkpoint taken with no link, and one written before the link existed, list and restore exactly
  as they always did, with no extra keys;
* the link survives the snapshot-count trim, which rebuilds every kept commit;
* the lookup "what did this session checkpoint from prompt X on" is scoped to one session, takes the
  earliest checkpoint of the picked prompt or any later one, and counts files against the live tree.
"""

import subprocess

import pytest

from tools.checkpoint_manager import (
    CheckpointManager,
    _parse_link,
    _link_trailers,
    _project_hash,
    _ref_name,
    _run_git,
    _store_path,
    format_checkpoint_list,
)

LEGACY_KEYS = {"hash", "short_hash", "timestamp", "reason", "files_changed", "insertions", "deletions"}


@pytest.fixture()
def work_dir(tmp_path):
    d = tmp_path / "project"
    d.mkdir()
    for name in ("a.txt", "b.txt", "c.txt"):
        (d / name).write_text(f"{name} v0\n")
    return d


@pytest.fixture()
def mgr(checkpoint_base, monkeypatch):
    monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
    return CheckpointManager(enabled=True, max_snapshots=50)


@pytest.fixture()
def checkpoint_base(tmp_path):
    return tmp_path / "checkpoints"


def _edit(work_dir, name, text):
    (work_dir / name).write_text(text)


def _turn(mgr, work_dir, session, prompt, name, text, reason="before write_file"):
    """One agent iteration: new_turn, checkpoint the tree, then the tool edits a file."""
    mgr.new_turn(session_id=session, prompt_uid=prompt)
    assert mgr.ensure_checkpoint(str(work_dir), reason) is True
    _edit(work_dir, name, text)


def _message(work_dir, commit_hash):
    ok, out, _ = _run_git(["log", "-1", "--format=%B", commit_hash], _store_path(), str(work_dir))
    assert ok
    return out


# ---------------------------------------------------------------------------
# Linking
# ---------------------------------------------------------------------------

def test_checkpoint_records_session_and_prompt_and_keeps_its_subject(mgr, work_dir):
    _turn(mgr, work_dir, "sess-1", "uid-1", "a.txt", "a.txt v1\n")

    (entry,) = mgr.list_checkpoints(str(work_dir))

    assert (entry["session_id"], entry["prompt_uid"]) == ("sess-1", "uid-1")
    assert entry["reason"] == "before write_file"  # the link rides below the subject, never in it


def test_one_prompt_can_own_several_checkpoints(mgr, work_dir):
    # new_turn runs once per tool-loop iteration, so a prompt that edits in two iterations
    # produces two checkpoints, both linked to it.
    _turn(mgr, work_dir, "sess-1", "uid-1", "a.txt", "a.txt v1\n")
    _turn(mgr, work_dir, "sess-1", "uid-1", "b.txt", "b.txt v1\n")

    links = {(cp["session_id"], cp["prompt_uid"]) for cp in mgr.list_checkpoints(str(work_dir))}

    assert links == {("sess-1", "uid-1")}
    assert len(mgr.list_checkpoints(str(work_dir))) == 2


def test_no_link_is_recorded_when_the_caller_does_not_know_the_prompt(mgr, work_dir):
    mgr.new_turn()
    assert mgr.ensure_checkpoint(str(work_dir), "before write_file") is True

    (entry,) = mgr.list_checkpoints(str(work_dir))

    assert set(entry) == LEGACY_KEYS
    assert _message(work_dir, entry["hash"]) == "before write_file"  # byte-identical to a pre-link commit


def test_new_turn_without_arguments_clears_the_previous_link(mgr, work_dir):
    _turn(mgr, work_dir, "sess-1", "uid-1", "a.txt", "a.txt v1\n")
    mgr.new_turn()
    mgr.ensure_checkpoint(str(work_dir), "before terminal: rm x")

    newest, older = mgr.list_checkpoints(str(work_dir))

    assert "prompt_uid" not in newest and older["prompt_uid"] == "uid-1"


@pytest.mark.parametrize("session_id, prompt_uid", [
    ("sess 1", "uid-1"),                      # whitespace
    ("sess-1\nHermes-Prompt: forged", "u"),   # trailer injection
    ("sess-1", ""),                           # missing half
    ("", "uid-1"),
])
def test_an_unsafe_or_incomplete_link_is_not_written(mgr, work_dir, session_id, prompt_uid):
    mgr.new_turn(session_id=session_id, prompt_uid=prompt_uid)
    mgr.ensure_checkpoint(str(work_dir), "before write_file")

    (entry,) = mgr.list_checkpoints(str(work_dir))

    assert set(entry) == LEGACY_KEYS
    assert _message(work_dir, entry["hash"]) == "before write_file"


def test_a_pre_rollback_snapshot_is_not_attributed_to_the_last_prompt(mgr, work_dir):
    _turn(mgr, work_dir, "sess-1", "uid-1", "a.txt", "a.txt v1\n")
    target = mgr.list_checkpoints(str(work_dir))[0]["hash"]

    assert mgr.restore(str(work_dir), target)["success"] is True

    newest = mgr.list_checkpoints(str(work_dir))[0]
    assert newest["hash"] != target and "prompt_uid" not in newest


def test_a_linked_checkpoint_restores_like_any_other(mgr, work_dir):
    _turn(mgr, work_dir, "sess-1", "uid-1", "a.txt", "a.txt v1\n")
    target = mgr.list_checkpoints(str(work_dir))[0]["hash"]

    result = mgr.restore(str(work_dir), target)

    assert result["success"] is True
    assert (work_dir / "a.txt").read_text() == "a.txt v0\n"
    assert result["reason"] == "before write_file"


def test_a_commit_written_before_the_link_existed_lists_and_restores_as_today(mgr, work_dir):
    """Build the commit exactly as the pre-link ``_take`` did, trailer-free, and read it back."""
    mgr.new_turn()
    assert mgr.ensure_checkpoint(str(work_dir), "before write_file")
    _edit(work_dir, "a.txt", "a.txt v1\n")
    store, ref = _store_path(), _ref_name(_project_hash(str(work_dir.resolve())))
    ok, tip, _ = _run_git(["rev-parse", ref], store, str(work_dir))
    ok, tree, _ = _run_git(["rev-parse", f"{tip}^{{tree}}"], store, str(work_dir))
    ok, legacy, _ = _run_git(["commit-tree", tree, "-p", tip, "-m", "before patch", "--no-gpg-sign"], store, str(work_dir))
    assert ok and _run_git(["update-ref", ref, legacy], store, str(work_dir))[0]

    newest = mgr.list_checkpoints(str(work_dir))[0]

    assert newest["hash"] == legacy and set(newest) == LEGACY_KEYS and newest["reason"] == "before patch"
    assert mgr.restore(str(work_dir), legacy)["success"] is True


def test_parsing_needs_both_trailers_and_ignores_prose():
    body = "Hermes-Session: s1\nunrelated line\n"
    assert _parse_link(body) is None
    assert _parse_link(_link_trailers(("s1", "u1"))) == ("s1", "u1")
    assert _parse_link("Hermes-Session: s1\nHermes-Prompt: bad value with spaces") is None


# ---------------------------------------------------------------------------
# The link survives the trim (the pruner rebuilds every kept commit)
# ---------------------------------------------------------------------------

def test_the_link_survives_the_snapshot_count_trim(checkpoint_base, work_dir, monkeypatch):
    monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
    mgr = CheckpointManager(enabled=True, max_snapshots=3)
    for i in range(1, 7):
        _turn(mgr, work_dir, "sess-1", f"uid-{i}", "a.txt", f"a.txt v{i}\n")

    kept = mgr.list_checkpoints(str(work_dir))

    assert [cp["prompt_uid"] for cp in kept] == ["uid-6", "uid-5", "uid-4"]
    assert all(cp["reason"] == "before write_file" for cp in kept)
    assert _message(work_dir, kept[0]["hash"]).splitlines() == [
        "before write_file", "", "Hermes-Session: sess-1", "Hermes-Prompt: uid-6"]


# ---------------------------------------------------------------------------
# The lookup
# ---------------------------------------------------------------------------

def _three_prompts(mgr, work_dir):
    """p1 edits a; p2 edits b then c in two iterations; checkpoints are pre-edit snapshots."""
    _turn(mgr, work_dir, "sess-1", "p1", "a.txt", "a.txt v1\n")
    _turn(mgr, work_dir, "sess-1", "p2", "b.txt", "b.txt v1\n")
    _turn(mgr, work_dir, "sess-1", "p2", "c.txt", "c.txt v1\n")


def test_lookup_returns_the_picked_prompt_and_later_ones_with_files_changed_since(mgr, work_dir):
    _three_prompts(mgr, work_dir)

    from_p1 = mgr.checkpoints_for_prompts("sess-1", ["p1", "p2"])
    from_p2 = mgr.checkpoints_for_prompts("sess-1", ["p2"])

    # Newest first. Restoring cp(p1) would touch a, b and c; cp(p2, first) b and c; cp(p2, second) c.
    assert [(cp["prompt_uid"], cp["files_changed_since"]) for cp in from_p1] == [("p2", 1), ("p2", 2), ("p1", 3)]
    assert [(cp["prompt_uid"], cp["files_changed_since"]) for cp in from_p2] == [("p2", 1), ("p2", 2)]
    assert {cp["workdir"] for cp in from_p1} == {str(work_dir.resolve())}


def test_lookup_counts_new_and_deleted_files(mgr, work_dir):
    mgr.new_turn(session_id="sess-1", prompt_uid="p1")
    mgr.ensure_checkpoint(str(work_dir), "before terminal: edit")
    (work_dir / "new.txt").write_text("n\n")
    (work_dir / "b.txt").unlink()
    _edit(work_dir, "a.txt", "changed\n")

    (cp,) = mgr.checkpoints_for_prompts("sess-1", ["p1"])

    assert cp["files_changed_since"] == 3  # created + deleted + modified


def test_lookup_skips_a_picked_prompt_that_edited_nothing(mgr, work_dir):
    _turn(mgr, work_dir, "sess-1", "p1", "a.txt", "a.txt v1\n")
    # p2 answered without touching a file, so it owns no checkpoint; p3 did.
    _turn(mgr, work_dir, "sess-1", "p3", "b.txt", "b.txt v1\n")

    cps = mgr.checkpoints_for_prompts("sess-1", ["p2", "p3"])

    assert [cp["prompt_uid"] for cp in cps] == ["p3"]


def test_two_sessions_in_one_directory_do_not_cross_link(mgr, work_dir):
    _turn(mgr, work_dir, "sess-A", "uA", "a.txt", "from A\n")
    _turn(mgr, work_dir, "sess-B", "uB", "b.txt", "from B\n")

    assert [cp["prompt_uid"] for cp in mgr.checkpoints_for_prompts("sess-A", ["uA"])] == ["uA"]
    assert [cp["prompt_uid"] for cp in mgr.checkpoints_for_prompts("sess-B", ["uB"])] == ["uB"]
    assert mgr.checkpoints_for_prompts("sess-A", ["uB"]) == []   # B's prompt is not A's
    assert mgr.checkpoints_for_prompts("sess-C", ["uA", "uB"]) == []


def test_lookup_spans_working_directories(mgr, work_dir, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "x.txt").write_text("x0\n")
    _turn(mgr, work_dir, "sess-1", "p1", "a.txt", "a.txt v1\n")
    mgr.new_turn(session_id="sess-1", prompt_uid="p1")
    mgr.ensure_checkpoint(str(other), "before terminal: edit")
    (other / "x.txt").write_text("x1\n")

    cps = mgr.checkpoints_for_prompts("sess-1", ["p1"])

    assert {cp["workdir"] for cp in cps} == {str(work_dir.resolve()), str(other.resolve())}


def test_lookup_changes_nothing(mgr, work_dir):
    _three_prompts(mgr, work_dir)
    before = [cp["hash"] for cp in mgr.list_checkpoints(str(work_dir))]
    tree = {p.name: p.read_text() for p in work_dir.iterdir()}

    mgr.checkpoints_for_prompts("sess-1", ["p1", "p2"])

    assert [cp["hash"] for cp in mgr.list_checkpoints(str(work_dir))] == before
    assert {p.name: p.read_text() for p in work_dir.iterdir()} == tree
    # The scratch index is back on the project ref, so the next checkpoint still sees only real change.
    mgr.new_turn(session_id="sess-1", prompt_uid="p3")
    assert mgr.ensure_checkpoint(str(work_dir), "after lookup") is True


def test_lookup_degrades_to_empty(mgr, work_dir, checkpoint_base):
    assert mgr.checkpoints_for_prompts("sess-1", ["p1"]) == []      # no store yet
    _three_prompts(mgr, work_dir)
    assert mgr.checkpoints_for_prompts("", ["p1"]) == []
    assert mgr.checkpoints_for_prompts("sess-1", []) == []
    assert mgr.checkpoints_for_prompts("sess-1", ["nope"]) == []


# ---------------------------------------------------------------------------
# /rollback listing
# ---------------------------------------------------------------------------

def test_listing_quotes_the_prompt_only_for_linked_checkpoints(mgr, work_dir):
    _turn(mgr, work_dir, "sess-1", "p1", "a.txt", "a.txt v1\n")
    mgr.new_turn()
    mgr.ensure_checkpoint(str(work_dir), "before terminal: rm x")
    entries = mgr.list_checkpoints(str(work_dir))
    linked = next(cp for cp in entries if cp.get("prompt_uid") == "p1")
    linked["prompt_preview"] = "fix the login bug"   # what the gateway sends alongside the checkpoint

    text = format_checkpoint_list(entries, str(work_dir))
    lines = text.splitlines()

    quoted = [i for i, line in enumerate(lines) if "prompt:" in line]
    assert len(quoted) == 1 and "fix the login bug" in lines[quoted[0]]
    assert "before write_file" in lines[quoted[0] - 1]          # directly under its own checkpoint
    assert text.count("prompt:") == 1


def test_listing_without_links_is_unchanged(mgr, work_dir):
    mgr.new_turn()
    mgr.ensure_checkpoint(str(work_dir), "before write_file")
    entries = mgr.list_checkpoints(str(work_dir))

    text = format_checkpoint_list(entries, str(work_dir))
    lines = text.splitlines()

    assert "prompt:" not in text
    # header + blank + one line per checkpoint + blank + three /rollback hints: nothing else.
    assert len(lines) == 6 + len(entries)
    assert lines[2].strip().endswith("before write_file")


def test_listing_survives_an_unreadable_prompt(mgr, work_dir, monkeypatch):
    _turn(mgr, work_dir, "sess-1", "p1", "a.txt", "a.txt v1\n")
    # No state.db in the isolated HERMES_HOME: the checkpoint still lists, just without the quote.
    text = format_checkpoint_list(mgr.list_checkpoints(str(work_dir)), str(work_dir))

    assert "before write_file" in text and "prompt:" not in text


# ---------------------------------------------------------------------------
# Prompt lookups (tools/checkpoint_prompts.py): read-only questions put to state.db
# ---------------------------------------------------------------------------

@pytest.fixture()
def state_db(tmp_path):
    """The columns the lookups read; ids and uids are what the real schema stores."""
    import sqlite3

    path = tmp_path / "state.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, "
                 "message_uid TEXT, active INTEGER DEFAULT 1)")
    rows = [
        (1, "s1", "user", "first prompt", "u1", 1),
        (2, "s1", "assistant", "reply", "a1", 1),
        (3, "s1", "user", "second prompt " + "x" * 200, "u2", 1),
        (4, "s1", "user", '[{"type": "text", "text": "multi\\nmodal"}, {"type": "image_url"}]', "u3", 1),
        (5, "s1", "user", "rewound away", "u4", 0),
        (6, "s2", "user", "other session", "u9", 1),
        (7, "s1", "user", "old copy", "u2", 0),   # compaction left an archived copy of u2
    ]
    conn.executemany("INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return path


def test_prompt_uids_run_from_the_picked_row_to_the_newest_active_prompt(state_db):
    from tools.checkpoint_prompts import prompt_uids_from

    assert prompt_uids_from("s1", 1, state_db) == ["u1", "u2", "u3"]
    assert prompt_uids_from("s1", 3, state_db) == ["u2", "u3"]


@pytest.mark.parametrize("row_id", [2, 5, 6, 999, 0, -1])
def test_prompt_uids_refuse_a_row_that_is_not_an_active_user_message_of_the_session(state_db, row_id):
    from tools.checkpoint_prompts import prompt_uids_from

    # 2 assistant row, 5 rewound row, 6 another session's row, the rest absent.
    assert prompt_uids_from("s1", row_id, state_db) == []


def test_prompt_uids_survive_garbage_input(state_db, tmp_path):
    from tools.checkpoint_prompts import prompt_uids_from

    assert prompt_uids_from("s1", "not-a-number", state_db) == []   # type: ignore[arg-type]
    assert prompt_uids_from("s1", 1, tmp_path / "missing.db") == []


def test_previews_quote_one_short_line_and_prefer_the_live_copy(state_db):
    from tools.checkpoint_prompts import PREVIEW_CHARS, prompt_previews

    got = prompt_previews([("s1", "u1"), ("s1", "u2"), ("s1", "u3"), ("s2", "u9"), ("s1", "gone")], state_db)

    assert got[("s1", "u1")] == "first prompt"
    assert got[("s1", "u2")].startswith("second prompt ") and len(got[("s1", "u2")]) == PREVIEW_CHARS
    assert got[("s1", "u2")].endswith("…")
    assert got[("s1", "u3")] == "multi modal"          # JSON content flattened to its text, one line
    assert got[("s2", "u9")] == "other session"
    assert ("s1", "gone") not in got                     # a deleted message just has no quote


def test_previews_never_cross_sessions(state_db):
    from tools.checkpoint_prompts import prompt_previews

    assert prompt_previews([("s2", "u1")], state_db) == {}


def test_previews_are_empty_without_a_database(tmp_path):
    from tools.checkpoint_prompts import prompt_previews

    assert prompt_previews([("s1", "u1")], tmp_path / "missing.db") == {}
    assert prompt_previews([], tmp_path / "missing.db") == {}


def test_the_lookups_open_the_database_read_only(state_db):
    import sqlite3

    from tools.checkpoint_prompts import _connect

    conn = _connect(state_db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM messages")
    finally:
        conn.close()
