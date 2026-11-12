"""Two V4A operations spelling one file differently validate as one file (no partial write)."""

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_agent.tools.patch_parser import apply_v4a_operations, parse_v4a_patch


class _AliasingFileOps:
    """A non-host backend: relative paths resolve against ``cwd`` like a remote shell would."""

    def __init__(self, files):
        self.files, self.env = dict(files), SimpleNamespace(cwd="/work")

    def _key(self, path):
        import posixpath
        return posixpath.normpath(posixpath.join(self.env.cwd, path.strip()))

    def read_file_raw(self, path):
        if self._key(path) in self.files:
            return SimpleNamespace(content=self.files[self._key(path)], error=None)
        return SimpleNamespace(content=None, error=f"File not found: {path}", not_found=True)

    def write_file(self, path, content, pre_content=None):
        self.files[self._key(path)] = content
        return SimpleNamespace(error=None)

    def delete_file(self, path):
        self.files.pop(self._key(path), None)
        return SimpleNamespace(error=None)


def _update(path, old, new):
    return f"*** Update File: {path}\n@@\n-{old}\n+{new}\n"


@pytest.mark.parametrize("alias", ["./dup.txt", "/work/dup.txt", ".//dup.txt"])
def test_alias_of_an_edited_file_validates_against_the_first_edit(alias):
    ops, err = parse_v4a_patch("*** Begin Patch\n" + _update("dup.txt", "before", "first")
                               + _update(alias, "before", "second") + "*** End Patch\n")
    assert err is None
    fo = _AliasingFileOps({"/work/dup.txt": "before\n"})
    result = apply_v4a_operations(ops, fo)
    assert result.success is False
    assert "no files were modified" in result.error
    assert fo.files == {"/work/dup.txt": "before\n"}


def test_alias_spelled_sequential_edits_and_delete_then_add_still_apply():
    patch = ("*** Begin Patch\n" + _update("seq.txt", "a", "A") + _update("./seq.txt", "b", "B")
             + "*** Delete File: re.txt\n*** Add File: ./re.txt\n+fresh\n*** End Patch\n")
    ops, err = parse_v4a_patch(patch)
    assert err is None
    fo = _AliasingFileOps({"/work/seq.txt": "a\nb\n", "/work/re.txt": "old\n"})
    result = apply_v4a_operations(ops, fo)
    assert result.success is True, result.error
    assert fo.files == {"/work/seq.txt": "A\nB\n", "/work/re.txt": "fresh"}


@pytest.mark.platforms("posix")
def test_dotdot_through_a_symlinked_dir_is_a_different_file(tmp_path):
    """``sub/link/../f.txt`` is ``real/f.txt`` when ``link -> real/deep``: two files, both edited."""
    from hermes_agent.tools.environments.local import LocalEnvironment
    from hermes_agent.tools.file_operations import ShellFileOperations
    for rel in ("sub/f.txt", "real/f.txt"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("foo\n")
    (tmp_path / "real/deep").mkdir()
    (tmp_path / "sub/link").symlink_to(tmp_path / "real/deep")
    env = LocalEnvironment(cwd=str(tmp_path), timeout=30, persistent=False)
    env.uses_host_filesystem = False  # the lexical overlay path, as on a remote backend
    try:
        result = ShellFileOperations(env, cwd=str(tmp_path)).patch_v4a(
            "*** Begin Patch\n" + _update("sub/f.txt", "foo", "X")
            + _update("sub/link/../f.txt", "foo", "Y") + "*** End Patch\n")
    finally:
        env.cleanup()
    assert result.success is True, result.error
    assert ((tmp_path / "sub/f.txt").read_text(), (tmp_path / "real/f.txt").read_text()) == ("X\n", "Y\n")


@contextlib.contextmanager
def _non_host_file_ops(cwd, **env_attrs):
    """Real ShellFileOperations on a LocalEnvironment taking the remote backends' lexical path."""
    from hermes_agent.tools.environments.local import LocalEnvironment
    from hermes_agent.tools.file_operations import ShellFileOperations
    env = LocalEnvironment(cwd=str(cwd), timeout=30, persistent=False)
    env.uses_host_filesystem = False
    for name, value in env_attrs.items():
        setattr(env, name, value)
    try:
        yield ShellFileOperations(env, cwd=str(cwd))
    finally:
        env.cleanup()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("dir_spelling", ["a.txt/", "a.txt/.", "a.txt/.//"])
def test_a_trailing_slash_needs_a_directory_so_never_aliases_the_file(tmp_path, dir_spelling):
    (tmp_path / "a.txt").write_text("before\n")
    with _non_host_file_ops(tmp_path) as fo:
        result = fo.patch_v4a("*** Begin Patch\n" + _update("a.txt", "before", "first")
                              + _update(dir_spelling, "first", "second") + "*** End Patch\n")
    assert result.success is False
    assert "no files were modified" in result.error
    assert (tmp_path / "a.txt").read_text() == "before\n"


@pytest.mark.platforms("posix")
def test_host_path_under_a_workspace_mount_aliases_the_container_path(tmp_path):
    """Docker's workspace mount: ``_expand_path`` rewrites ``<host_cwd>/a.txt`` to ``<mount>/a.txt``."""
    ws, host = tmp_path / "ws", tmp_path / "hostproj"
    ws.mkdir()
    host.mkdir()
    (ws / "a.txt").write_text("before\n")
    with _non_host_file_ops(ws, host_cwd=str(host), host_cwd_mount=str(ws)) as fo:
        result = fo.patch_v4a("*** Begin Patch\n" + _update(f"{host}/a.txt", "before", "first")
                              + _update("a.txt", "before", "second") + "*** End Patch\n")
    assert result.success is False
    assert "no files were modified" in result.error
    assert (ws / "a.txt").read_text() == "before\n"


@pytest.mark.platforms("posix")
def test_tilde_is_the_backend_home_not_a_dir_under_cwd(tmp_path):
    """``~/x.txt`` is the backend's $HOME/x.txt; ``./~/x.txt`` is a directory named ``~`` under cwd: two files."""
    work = tmp_path / "work"
    (work / "~").mkdir(parents=True)
    (work / "~" / "x.txt").write_text("foo\n")
    with _non_host_file_ops(work) as fo:
        home = Path(fo.env.execute("echo $HOME")["output"].strip())  # whatever HOME the backend runs with
        target = home / f"hermes-alias-probe-{tmp_path.name}.txt"
        target.write_text("foo\n")
        try:
            result = fo.patch_v4a("*** Begin Patch\n" + _update(f"~/{target.name}", "foo", "X")
                                  + _update("./~/x.txt", "foo", "Y") + "*** End Patch\n")
            assert result.success is True, result.error
            assert (target.read_text(), (work / "~/x.txt").read_text()) == ("X\n", "Y\n")
        finally:
            target.unlink(missing_ok=True)