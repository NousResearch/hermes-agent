"""The Bot Mode DM directory: a private, user-owned folder that housekeeping never creates.

Regression for the Windows lock-out: ``Path.mkdir(mode=0o700)`` on CPython 3.11.10+ / 3.12.4+
gives the folder a protected SYSTEM / Administrators / OWNER RIGHTS DACL, so when an elevated
process created ``%TEMP%\\hermes-dm`` its owner became ``BUILTIN\\Administrators`` and the same
user, unelevated, could no longer read it; ``path.chmod`` then raised WinError 5 on every send,
and the hourly ``cleanup_bot_dm_cache`` sweep created the folder on its own.
"""

import os
import subprocess
from pathlib import Path

import pytest

from tools import bot_mode_dm


@pytest.fixture()
def temp_root(tmp_path, monkeypatch):
    monkeypatch.setattr(bot_mode_dm.tempfile, "gettempdir", lambda: str(tmp_path))
    return tmp_path


def _sddl(path: Path) -> str:
    """Owner + DACL of *path* as SDDL, read straight from Win32."""
    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32.GetNamedSecurityInfoW.argtypes = [wintypes.LPCWSTR, ctypes.c_int, wintypes.DWORD] + [
        ctypes.POINTER(wintypes.LPVOID)] * 5
    advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW.argtypes = [
        wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(wintypes.LPWSTR), wintypes.LPVOID]
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    info = 0x1 | 0x4  # OWNER | DACL
    descriptor = wintypes.LPVOID()
    code = advapi32.GetNamedSecurityInfoW(str(path), 1, info, None, None, None, None, ctypes.byref(descriptor))
    assert code == 0, code
    text = wintypes.LPWSTR()
    try:
        assert advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW(descriptor, 1, info,
                                                                             ctypes.byref(text), None)
        return text.value
    finally:
        kernel32.LocalFree(ctypes.cast(text, wintypes.HLOCAL))
        kernel32.LocalFree(descriptor)


@pytest.mark.platforms("windows")
def test_windows_dm_dir_is_owned_by_the_user_sid_with_a_protected_dacl(temp_root):
    from tools import bot_mode_dm_winacl as winacl

    me = winacl.current_user_sid()
    dm_dir = bot_mode_dm._dm_dir()

    sddl = _sddl(dm_dir)
    assert sddl.startswith(f"O:{me}"), sddl
    dacl = sddl.split("D:", 1)[1]
    assert dacl.startswith("P"), f"DACL inherits from %TEMP%: {sddl}"
    # Grants go to the token USER SID (valid elevated or not), never OWNER RIGHTS / Administrators.
    assert f"(A;OICI;FA;;;{me})" in dacl
    assert "(A;OICI;FA;;;SY)" in dacl
    assert "OW" not in dacl and ";BA)" not in dacl
    # Repeat calls (every send) must not chmod — that is what raised WinError 5.
    assert bot_mode_dm._dm_dir() == dm_dir
    payload = Path(bot_mode_dm._write_dm_file("hello"))
    assert payload.parent == dm_dir
    assert payload.read_text(encoding="utf-8-sig") == "hello"


@pytest.mark.platforms("windows")
def test_windows_dm_dir_owned_by_another_account_is_rejected_actionably(temp_root, monkeypatch):
    from tools import bot_mode_dm_winacl as winacl

    dm_dir = bot_mode_dm._dm_dir()
    # An elevated process's folder is owned by BUILTIN\Administrators.
    monkeypatch.setattr(winacl, "owner_sid", lambda path: "S-1-5-32-544")

    with pytest.raises(PermissionError) as excinfo:
        bot_mode_dm._dm_dir()

    message = str(excinfo.value)
    assert str(dm_dir) in message
    assert "owned by another account (S-1-5-32-544)" in message
    assert "rd /s /q" in message and "unelevated" in message


@pytest.mark.platforms("windows")
def test_windows_dm_dir_the_user_cannot_list_is_rejected_actionably(temp_root):
    from tools import bot_mode_dm_winacl as winacl

    me = winacl.current_user_sid()
    dm_dir = bot_mode_dm._dm_dir_path()
    dm_dir.mkdir()
    # Owned by the user but granting only SYSTEM: the owner keeps READ_CONTROL/WRITE_DAC, nothing else.
    subprocess.run(["icacls", str(dm_dir), "/inheritance:r", "/grant:r", "*S-1-5-18:(OI)(CI)F"],
                   check=True, capture_output=True)
    try:
        with pytest.raises(PermissionError, match="cannot be listed by the current user") as excinfo:
            bot_mode_dm._dm_dir()
        assert str(dm_dir) in str(excinfo.value)
        assert bot_mode_dm.cleanup_bot_dm_cache() == 0, "housekeeping must swallow, not raise"
    finally:
        subprocess.run(["icacls", str(dm_dir), "/grant", f"*{me}:(OI)(CI)F"], check=True, capture_output=True)


def test_cleanup_never_creates_the_dm_dir(temp_root):
    dm_dir = bot_mode_dm._dm_dir_path()

    assert bot_mode_dm.cleanup_bot_dm_cache() == 0

    assert not os.path.lexists(dm_dir)


def test_cleanup_still_sweeps_an_existing_dm_dir(temp_root):
    stale = Path(bot_mode_dm._write_dm_file("stale"))
    os.utime(stale, (1, 1))

    assert bot_mode_dm.cleanup_bot_dm_cache() == 1
    assert not stale.exists()


@pytest.mark.platforms("posix")
def test_posix_dm_dir_keeps_mode_0700_and_uid_scope(temp_root):
    dm_dir = bot_mode_dm._dm_dir()

    assert dm_dir.name == f"{bot_mode_dm._DM_DIR_NAME}-{os.getuid()}"  # windows-footgun: ok — POSIX-only test
    assert dm_dir.stat().st_mode & 0o777 == 0o700
    assert dm_dir.stat().st_uid == os.getuid()  # windows-footgun: ok — POSIX-only test


@pytest.mark.platforms("posix")
def test_posix_cleanup_does_not_sweep_through_a_planted_symlink(temp_root):
    victim = temp_root / "victim"
    victim.mkdir()
    precious = victim / "notes.txt"
    precious.write_text("keep", encoding="utf-8")
    os.utime(precious, (1, 1))
    bot_mode_dm._dm_dir_path().symlink_to(victim, target_is_directory=True)

    bot_mode_dm.cleanup_bot_dm_cache()

    assert precious.exists()
