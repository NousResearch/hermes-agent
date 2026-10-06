"""Behavior tests for the indexed (locate) filename engine (#127861).

The engine is a positive cache: it answers only when a fresh database exists, and
every other outcome — including "no match" — leaves the walk in charge.
"""

import os
import shlex
import time

import pytest

from tools import file_search_index
from tools.file_operations import ShellFileOperations

DAY = 24 * 3600
INDEX = ("plocate", "mlocate")
LIMIT = 50
FETCH = LIMIT + 1
WINDOW = file_search_index.fetch_window(FETCH)


def argv_of(command):
    """The argv the shell would see, for one bounded command string."""
    words = command.removeprefix("set -o pipefail; ").split(" 2>/dev/null | head -n ")[0]
    return shlex.split(words)


class RecordingEnvironment:
    """Records commands and answers per engine, never touching a real filesystem."""

    is_local = False
    cwd = "/repo"

    def __init__(self, index_output="", index_code=0, walk_output="", walk_code=0):
        self.commands = []
        self.index_output = index_output
        self.index_code = index_code
        self.walk_output = walk_output
        self.walk_code = walk_code

    def execute(self, command, **kwargs):
        self.commands.append(command)
        if "--version" in command:
            return {"output": "ripgrep 14.1.1\n", "returncode": 0}
        if command.startswith("test -e "):
            return {"output": "exists\n", "returncode": 0}
        if "plocate" in command or "mlocate" in command:
            return {"output": self.index_output, "returncode": self.index_code}
        if "--files" in command:
            return {"output": self.walk_output, "returncode": self.walk_code}
        return {"output": "", "returncode": 1}

    @property
    def index_commands(self):
        return [command for command in self.commands
                if "plocate" in command or "mlocate" in command]

    @property
    def walk_commands(self):
        return [command for command in self.commands if "--files" in command]


@pytest.fixture()
def database(monkeypatch, tmp_path):
    """A fresh locate database; plocate is the fake PATH's first engine."""
    path = tmp_path / "plocate.db"
    path.write_bytes(b"index")
    monkeypatch.setenv("HERMES_FILE_SEARCH_LOCATE_DB", str(path))
    monkeypatch.delenv("HERMES_FILE_SEARCH_ENGINE", raising=False)
    monkeypatch.delenv("HERMES_FILE_SEARCH_INDEX_MAX_AGE", raising=False)
    return path


@pytest.fixture()
def tree(tmp_path):
    """A real search root: the existence probe before dispatch is the native lane,
    and the filter stats the paths an index would hand back."""
    root = tmp_path / "tree"
    for relative in ("found.py", "keep.py", "walked.py", "linked.py", "f0.py", "f1.py", "f2.py",
                     "sub/other.py", "sub/keep2.py", "sub/.dotfile", ".hidden/secret.py",
                     "node_modules/dep.py", "sub/target/debug.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    (root / "linked.py").unlink()
    (root / "linked.py").symlink_to(root / "found.py")
    (tmp_path / "treebar").mkdir()
    (tmp_path / "treebar" / "trap.py").write_text("", encoding="utf-8")
    return root


def build_ops(monkeypatch, env, *engines):
    """A local backend (index eligible) with rg and *engines* on the fake PATH."""
    ops = ShellFileOperations(env)
    monkeypatch.setattr(ops, "_lsp_local_only", lambda: True)
    monkeypatch.setattr(ops, "_native_read_enabled", lambda: False)
    monkeypatch.setattr(ops, "_has_command", lambda name: name == "rg")
    monkeypatch.setattr(file_search_index, "which_exists", lambda name: name in engines)
    return ops


def test_fresh_index_answers_without_walking(monkeypatch, database, tree):
    env = RecordingEnvironment(index_output=f"{tree}/found.py\n{tree}/sub/other.py\n")
    ops = build_ops(monkeypatch, env, *INDEX)

    result = ops.search("*.py", path=str(tree), target="files", limit=LIMIT)

    assert result.files == [f"{tree}/found.py", f"{tree}/sub/other.py"]
    assert result.total_count == 2
    assert result.warning == file_search_index.SNAPSHOT_WARNING
    assert env.walk_commands == []
    assert argv_of(env.index_commands[0]) == [
        "plocate", "-d", str(database), "-e", "-l", str(WINDOW), "--", f"{tree}/*.py"]


def test_index_answer_is_filtered_to_what_the_walk_hides(monkeypatch, database, tree, tmp_path):
    env = RecordingEnvironment(index_output="\n".join([
        f"{tree}/keep.py",
        f"{tree}/.hidden/secret.py",
        f"{tree}/sub/.dotfile",
        f"{tree}/sub/keep2.py",
        f"{tree}/node_modules/dep.py",     # not hidden: the walk reports it too
        f"{tree}/linked.py",               # a symlink: the walk does not follow it
        f"{tree}/sub",                     # a directory is not a file
        f"{tmp_path}/outside.py",
        f"{tree}bar/trap.py",
    ]) + "\n")
    ops = build_ops(monkeypatch, env, *INDEX)

    result = ops.search("*.py", path=str(tree), target="files", limit=LIMIT)

    assert result.files == [f"{tree}/keep.py", f"{tree}/sub/keep2.py", f"{tree}/node_modules/dep.py"]
    assert env.walk_commands == []


def test_capped_window_is_a_lower_bound(monkeypatch, database, tree):
    monkeypatch.setattr(file_search_index, "fetch_window", lambda fetch_limit: 3)
    env = RecordingEnvironment(index_output="".join(
        f"{tree}/f{index}.py\n" for index in range(3)))
    ops = build_ops(monkeypatch, env, *INDEX)

    result = ops.search("*.py", path=str(tree), target="files", limit=2)

    assert result.files == [f"{tree}/f0.py", f"{tree}/f1.py"]
    assert result.total_count == 3
    assert result.to_dict()["total_count_is_lower_bound"] is True


def test_short_page_from_a_full_window_falls_through_to_the_walk(monkeypatch, database, tree):
    """A window full of filtered-out paths is a path-order slice, not a sample."""
    monkeypatch.setattr(file_search_index, "fetch_window", lambda fetch_limit: 4)
    env = RecordingEnvironment(index_output="\n".join([
        f"{tree}/.hidden/secret.py", f"{tree}/sub/.dotfile",
        f"{tree}/linked.py", f"{tree}/keep.py",
    ]) + "\n", walk_output=f"{tree}/walked.py\n")
    ops = build_ops(monkeypatch, env, *INDEX)

    result = ops.search("*.py", path=str(tree), target="files", limit=LIMIT)

    assert result.files == [f"{tree}/walked.py"]
    assert len(env.walk_commands) == 1

    # The same window with room to spare answers from the index.
    env.index_output = f"{tree}/.hidden/secret.py\n{tree}/keep.py\n"
    result = ops.search("*.py", path=str(tree), target="files", limit=LIMIT)
    assert result.files == [f"{tree}/keep.py"]


def test_older_engine_is_used_when_it_is_the_only_one(monkeypatch, database, tree):
    env = RecordingEnvironment(index_output=f"{tree}/found.py\n")
    ops = build_ops(monkeypatch, env, "mlocate")

    result = ops.search("*.py", path=str(tree), target="files", limit=LIMIT)

    assert result.files == [f"{tree}/found.py"]
    assert argv_of(env.index_commands[0])[:3] == ["mlocate", "-d", str(database)]
    assert "-e" not in argv_of(env.index_commands[0])  # mlocate has no existence filter


def test_empty_index_answer_falls_through_to_the_walk(monkeypatch, database, tree):
    """A snapshot's blind spot: the file was created after the last updatedb."""
    env = RecordingEnvironment(index_output="", index_code=1, walk_output=f"{tree}/walked.py\n")
    ops = build_ops(monkeypatch, env, *INDEX)

    result = ops.search("*.py", path=str(tree), target="files", limit=LIMIT)

    assert result.files == [f"{tree}/walked.py"]
    assert len(env.index_commands) == 1 and len(env.walk_commands) == 1


def test_index_hits_outside_the_roots_fall_through_to_the_walk(monkeypatch, database, tree, tmp_path):
    env = RecordingEnvironment(index_output=f"{tmp_path}/outside.py\n",
                               walk_output=f"{tree}/walked.py\n")
    ops = build_ops(monkeypatch, env, *INDEX)

    result = ops.search("*.py", path=str(tree), target="files", limit=LIMIT)

    assert result.files == [f"{tree}/walked.py"]
    assert len(env.walk_commands) == 1


def test_stale_database_falls_through_to_the_walk(monkeypatch, database, tree):
    old = time.time() - 8 * DAY
    os.utime(database, (old, old))
    env = RecordingEnvironment(index_output=f"{tree}/found.py\n", walk_output=f"{tree}/walked.py\n")
    ops = build_ops(monkeypatch, env, *INDEX)

    result = ops.search("*.py", path=str(tree), target="files", limit=LIMIT)

    assert result.files == [f"{tree}/walked.py"]
    assert env.index_commands == []


def test_missing_database_falls_through_to_the_walk(monkeypatch, tree, tmp_path):
    monkeypatch.setenv("HERMES_FILE_SEARCH_LOCATE_DB", str(tmp_path / "absent.db"))
    env = RecordingEnvironment(index_output=f"{tree}/found.py\n", walk_output=f"{tree}/walked.py\n")
    ops = build_ops(monkeypatch, env, *INDEX)

    result = ops.search("*.py", path=str(tree), target="files", limit=LIMIT)

    assert result.files == [f"{tree}/walked.py"]
    assert env.index_commands == []


def test_failing_query_falls_through_to_the_walk(monkeypatch, database, tree):
    env = RecordingEnvironment(index_output="plocate: io error\n", index_code=2,
                               walk_output=f"{tree}/walked.py\n")
    ops = build_ops(monkeypatch, env, *INDEX)

    result = ops.search("*.py", path=str(tree), target="files", limit=LIMIT)

    assert result.files == [f"{tree}/walked.py"]
    assert len(env.walk_commands) == 1


def test_relative_root_and_remote_backend_never_ask_the_index(monkeypatch, database, tree):
    env = RecordingEnvironment(index_output=f"{tree}/found.py\n")
    ops = build_ops(monkeypatch, env, *INDEX)

    assert ops._search_files_index("*.py", [".", str(tree)], LIMIT, 0) is None

    monkeypatch.setattr(ops, "_lsp_local_only", lambda: False)
    assert ops._search_files_index("*.py", [str(tree)], LIMIT, 0) is None
    assert env.index_commands == []


def test_worktree_roots_keep_the_walk(monkeypatch, database, tmp_path):
    """Bounded, ignore-filtered and cheap there: the walk is already the right engine."""
    (tmp_path / ".git").mkdir()
    root = tmp_path / "src"
    (root / "found.py").parent.mkdir(parents=True, exist_ok=True)
    (root / "found.py").write_text("", encoding="utf-8")
    env = RecordingEnvironment(index_output=f"{root}/found.py\n",
                               walk_output=f"{root}/walked.py\n")
    ops = build_ops(monkeypatch, env, *INDEX)

    result = ops.search("*.py", path=str(root), target="files", limit=LIMIT)
    assert result.files == [f"{root}/walked.py"]

    (root / ".git").mkdir()
    assert ops._search_files_index("*.py", [str(root)], LIMIT, 0) is None
    assert env.index_commands == []


def test_modified_order_keeps_the_exact_rg_order(monkeypatch, database, tree):
    def _never(*args, **kwargs):
        raise AssertionError("an unordered index cannot serve order='modified'")

    monkeypatch.setattr(file_search_index, "indexed_argv", _never)
    env = RecordingEnvironment(index_output=f"{tree}/found.py\n", walk_output=f"{tree}/walked.py\n")
    ops = build_ops(monkeypatch, env, *INDEX)

    result = ops.search("*.py", path=str(tree), target="files", order="modified", limit=LIMIT)

    assert result.files == [f"{tree}/walked.py"]
    assert "--sortr=modified" in env.walk_commands[0]


def test_indexed_argv_scopes_every_root_into_the_query(monkeypatch, database, tmp_path):
    now = time.time()
    query = file_search_index.indexed_argv("*.py", ["/s", "/s/", "/other"], 7,
                                           lambda name: name in INDEX, now)
    assert query == ["plocate", "-d", str(database), "-e", "-l", "7", "--",
                     "/s/*.py", "/other/*.py"]

    # A relative root cannot be expressed as a locate pattern.
    assert file_search_index.indexed_argv("*.py", ["."], 7, lambda name: True, now) is None

    # mlocate has no existence filter, so no -e for it.
    second = tmp_path / "mlocate.db"
    second.write_bytes(b"index")
    monkeypatch.setenv("HERMES_FILE_SEARCH_LOCATE_DB", str(second))
    assert file_search_index.indexed_argv("foo", ["/s"], 3, lambda name: name == "mlocate", now) == [
        "mlocate", "-d", str(second), "-l", "3", "--", "/s/*foo"]


@pytest.mark.parametrize("pattern", ["**/deep", "{a,b}.py", "!keep", "back\\slash"])
def test_globs_locate_cannot_express_keep_to_the_walk(monkeypatch, database, pattern):
    assert file_search_index.indexed_argv(pattern, ["/s"], 5, lambda name: True, time.time()) is None


def test_index_is_optional_and_switchable(monkeypatch, database):
    now = time.time()
    assert file_search_index.indexed_argv("*.py", ["/s"], 5, lambda name: False, now) is None
    monkeypatch.setenv("HERMES_FILE_SEARCH_ENGINE", "RG")
    assert file_search_index.indexed_argv("*.py", ["/s"], 5, lambda name: True, now) is None
    monkeypatch.delenv("HERMES_FILE_SEARCH_ENGINE")
    monkeypatch.setenv("HERMES_FILE_SEARCH_INDEX_MAX_AGE", "60")
    assert file_search_index.indexed_argv("*.py", ["/s"], 5, lambda name: True, now + 90) is None
    assert file_search_index.indexed_argv("*.py", ["/s"], 5, lambda name: True, now + 30) is not None
    assert file_search_index.max_age_seconds() == 60


def test_default_engine_probe_is_the_host_path(monkeypatch, database):
    monkeypatch.setattr(file_search_index, "which_exists", lambda name: name == "locate")
    assert file_search_index.indexed_argv("*.py", ["/s"], 5, now=time.time())[0] == "locate"


def test_default_max_age_is_a_week():
    assert file_search_index.DEFAULT_MAX_AGE_SECONDS == 7 * DAY


def test_fetch_window_leaves_room_for_the_filter():
    assert file_search_index.fetch_window(1) == 200
    assert file_search_index.fetch_window(FETCH) == FETCH * 10


def test_worktree_probe_climbs_to_an_ancestor_worktree_root(tmp_path):
    assert file_search_index.inside_worktree(str(tmp_path)) is False
    (tmp_path / ".git").write_text("gitdir: /elsewhere\n", encoding="utf-8")
    assert file_search_index.inside_worktree(str(tmp_path / "a" / "b")) is True


def test_relative_match_ignores_the_root_s_own_name():
    """rg applies -g to the path relative to the root, so a pattern matching the root
    name must not match every file inside it."""
    assert file_search_index.relative_match("/s/tortoise/README.md", "/s/tortoise", "*tortoise*") is False
    assert file_search_index.relative_match("/s/tortoise/README.md", "/s/tortoise", "*README*") is True
    assert file_search_index.relative_match("/s/tortoise/sub/tortoise.txt", "/s/tortoise", "*tortoise*") is True
    assert file_search_index.relative_match("/s/tortoise/README.md", "/s/tortoise/", "*README*") is True


def test_keep_indexed_path_matches_the_walk_s_visibility(tree, tmp_path):
    keep = file_search_index.keep_indexed_path
    root, glob = str(tree), "*"

    assert keep(f"{tree}/keep.py", [root], glob) is True
    assert keep(f"{tree}/sub/keep2.py", [root], glob) is True
    assert keep(f"{tree}/node_modules/dep.py", [root], glob) is True   # not hidden
    assert keep(f"{tree}/.hidden/secret.py", [root], glob) is False
    assert keep(f"{tree}/sub/.dotfile", [root], glob) is False
    assert keep(f"{tree}/linked.py", [root], glob) is False            # symlink
    assert keep(f"{tree}/sub", [root], glob) is False                  # directory
    assert keep(f"{tmp_path}/outside.py", [root], glob) is False        # sibling root
    assert keep(f"{tree}bar/trap.py", [root], glob) is False            # name-prefix sibling
    assert keep("relative/keep.py", [root], glob) is False              # not absolute
    assert keep(f"{tree}/keep.py", [root, str(tmp_path)], glob) is True

    # A dot-named ROOT is the caller's own choice, but not its dot-named children.
    hidden = tmp_path / ".cfg"
    (hidden / "sub").mkdir(parents=True)
    (hidden / "app.py").write_text("", encoding="utf-8")
    (hidden / "sub" / "deep.py").write_text("", encoding="utf-8")
    assert keep(f"{hidden}/app.py", [str(hidden)], glob) is True
    assert keep(f"{hidden}/sub/deep.py", [str(hidden)], "*deep*") is True
    assert keep(f"{hidden}/app.py", [str(tmp_path)], "*app*") is False

    # A symlinked directory level is the operand-following case (#116270): from
    # above, the walk does not enter it, so the index must not report from it either.
    (tmp_path / "linktree").symlink_to(tree)
    assert keep(f"{tmp_path}/linktree/keep.py", [str(tmp_path / "linktree")], glob) is False
