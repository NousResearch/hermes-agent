"""Two V4A operations spelling one file differently validate as one file (no partial write)."""

from types import SimpleNamespace

import pytest

from tools.patch_parser import apply_v4a_operations, parse_v4a_patch


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


@pytest.mark.parametrize("alias", ["./dup.txt", "/work/dup.txt", "sub/../dup.txt"])
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
