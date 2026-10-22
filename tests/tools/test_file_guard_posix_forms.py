"""POSIX-rooted file guards match the POSIX spelling of a path on every host."""

import pytest

from tools.file_tools import _is_blocked_device_path
from tools.file_tools_write_guards import _check_sensitive_path


@pytest.mark.parametrize("path", ["/dev/zero", "/dev/stdin", "/proc/self/fd/0"])
def test_read_guard_refuses_posix_device_paths(path):
    assert _is_blocked_device_path(path) is True


@pytest.mark.parametrize("path", ["/etc/hosts", "/boot/grub/grub.cfg"])
def test_write_guard_refuses_posix_sensitive_paths(path):
    assert _check_sensitive_path(path) is not None


def test_ordinary_paths_pass_both_guards(tmp_path):
    """Positive control: a normal workspace file is neither a device nor sensitive."""
    target = str(tmp_path / "notes.txt")
    assert _is_blocked_device_path(target) is False
    assert _check_sensitive_path(target) is None


@pytest.mark.platforms("windows")
def test_windows_native_path_keeps_its_drive_anchor():
    from tools.file_tools_paths import _posix_match_forms

    bs = chr(92)
    native = bs.join(["C:", "Users", "op", "file.txt"])
    assert _posix_match_forms(native) == (native, "C:/Users/op/file.txt")
    assert _posix_match_forms("/etc/hosts") == (bs + "etc" + bs + "hosts", "/etc/hosts")
