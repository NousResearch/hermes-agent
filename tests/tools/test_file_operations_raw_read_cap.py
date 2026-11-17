"""read_file_raw's size cap and the V4A operations that must still work past it."""

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations


@pytest.mark.parametrize("stale_size", [False, True])
def test_raw_read_cap_with_real_file(tmp_path, monkeypatch, stale_size):
    import tools.file_operations as operations

    monkeypatch.setattr(operations, "_RAW_READ_MAX_BYTES", 64)
    target = tmp_path / "large.log"
    target.write_text("a" * 1024, encoding="utf-8")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))
    if stale_size:
        monkeypatch.setattr(ops, "_probe_regular_file", lambda path: (1, "ok"))
    result = ops.read_file_raw(str(target))
    assert result.error and "too large" in result.error
    assert not result.content


def test_raw_read_cap_holds_on_the_shell_transport(tmp_path, monkeypatch):
    """A backend without a native read (remote, sandbox) takes the base64 transport; a size probe
    that went stale before the read must not let that whole payload through either."""
    import tools.file_operations as operations

    monkeypatch.setattr(operations, "_RAW_READ_MAX_BYTES", 64)
    target = tmp_path / "large.log"
    target.write_text("a" * 65536, encoding="utf-8")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))
    monkeypatch.setattr(ops, "_native_read_enabled", lambda: False)
    monkeypatch.setattr(ops, "_probe_regular_file", lambda path: (1, "ok"))
    replies = []
    real_exec = ops._exec

    def recording_exec(command, *args, **kwargs):
        replies.append(real_exec(command, *args, **kwargs))
        return replies[-1]

    monkeypatch.setattr(ops, "_exec", recording_exec)
    result = ops.read_file_raw(str(target))
    assert result.error and "too large" in result.error
    assert not result.content
    assert max(len(r.stdout) for r in replies) < 4096  # the transfer stopped near the cap, not at 64 KiB


@pytest.mark.parametrize("transport", ["base64", "od"])
def test_capped_shell_read_still_returns_files_under_the_cap(tmp_path, monkeypatch, transport):
    target = tmp_path / "small.bin"
    target.write_bytes(bytes(range(64)))
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))
    monkeypatch.setattr(ops, "_native_read_enabled", lambda: False)
    if transport == "od":
        real_fenced = ops._fenced_read
        monkeypatch.setattr(ops, "_fenced_read", lambda body, *more: real_fenced(
            body.replace("| base64", "| __hermes_no_such_base64"), *more))
    assert ops._read_exact_bytes(str(target), max_bytes=64) == (bytes(range(64)), None)
    data, failed = ops._read_exact_bytes(str(target), max_bytes=63)
    assert data is None and "too large" in failed.stdout


@pytest.mark.parametrize("stale_size", [False, True])
def test_v4a_delete_removes_an_oversized_file(tmp_path, monkeypatch, stale_size):
    """Delete needs no content, so a file past the raw-read cap is still deletable."""
    import tools.file_operations as operations
    from tools.patch_parser import apply_v4a_operations, parse_v4a_patch

    monkeypatch.setattr(operations, "_RAW_READ_MAX_BYTES", 64)
    target = tmp_path / "large.log"
    target.write_text("a" * 1024, encoding="utf-8")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))
    if stale_size:  # the cap is only found by the read itself
        monkeypatch.setattr(ops, "_probe_regular_file", lambda path: (1, "ok"))
    operations_, err = parse_v4a_patch(f"*** Begin Patch\n*** Delete File: {target}\n*** End Patch\n")
    assert err is None
    result = apply_v4a_operations(operations_, ops)
    assert result.success, result.error
    assert not target.exists()


def test_v4a_add_does_not_overwrite_an_oversized_file(tmp_path, monkeypatch):
    """The cap error is a read failure, not "missing": an Add over a file past the cap is refused."""
    import tools.file_operations as operations
    from tools.patch_parser import apply_v4a_operations, parse_v4a_patch

    monkeypatch.setattr(operations, "_RAW_READ_MAX_BYTES", 64)
    target = tmp_path / "large.log"
    target.write_text("a" * 1024, encoding="utf-8")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))
    patch = f"*** Begin Patch\n*** Add File: {target}\n+new\n*** End Patch\n"
    operations_, err = parse_v4a_patch(patch)
    assert err is None
    result = apply_v4a_operations(operations_, ops)
    assert not result.success and "too large" in result.error
    assert target.read_text(encoding="utf-8") == "a" * 1024


def test_v4a_second_delete_of_an_oversized_file_is_refused(tmp_path, monkeypatch):
    """The cap exemption follows the patch's own overlay: a file this patch already deleted is gone."""
    import tools.file_operations as operations
    from tools.patch_parser import apply_v4a_operations, parse_v4a_patch

    monkeypatch.setattr(operations, "_RAW_READ_MAX_BYTES", 64)
    target = tmp_path / "large.log"
    target.write_text("a" * 1024, encoding="utf-8")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))
    patch = f"*** Begin Patch\n*** Delete File: {target}\n*** Delete File: {target}\n*** End Patch\n"
    operations_, err = parse_v4a_patch(patch)
    assert err is None
    result = apply_v4a_operations(operations_, ops)
    assert not result.success
    assert target.exists()


def test_v4a_delete_reads_an_oversized_file_once(tmp_path, monkeypatch):
    """Validation keeps the first read's cap verdict instead of reading the file a second time."""
    import tools.file_operations as operations
    from tools.patch_parser import _validate_operations, parse_v4a_patch

    monkeypatch.setattr(operations, "_RAW_READ_MAX_BYTES", 64)
    target = tmp_path / "large.log"
    target.write_text("a" * 1024, encoding="utf-8")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))
    monkeypatch.setattr(ops, "_probe_regular_file", lambda path: (1, "ok"))
    reads = []
    real_read = ops.read_file_raw
    monkeypatch.setattr(ops, "read_file_raw", lambda path: reads.append(path) or real_read(path))
    operations_, err = parse_v4a_patch(f"*** Begin Patch\n*** Delete File: {target}\n*** End Patch\n")
    assert err is None
    assert _validate_operations(operations_, ops) == []
    assert len(reads) == 1
