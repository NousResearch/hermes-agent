"""Windows security descriptors for the Bot Mode DM payload directory (stdlib ``ctypes`` only).

``Path.mkdir(mode=0o700)`` on Windows (CPython 3.11.10+ / 3.12.4+, gh-118486) gives a directory
a protected DACL of SYSTEM, Administrators and OWNER RIGHTS. OWNER RIGHTS follows the folder's
owner, and an elevated process makes ``BUILTIN\\Administrators`` the owner, so the same user
running unelevated is locked out of a folder Hermes itself created. Here the folder is created
with an explicit owner and ACL tied to the token's user SID, so elevated and unelevated
processes of one user both keep access and nothing is inherited from ``%TEMP%``.

Import is safe on every host; the functions call Win32 and run only on Windows.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import Path

_TOKEN_QUERY = 0x0008
_TOKEN_USER_CLASS = 1
_SE_FILE_OBJECT = 1
_OWNER_SECURITY_INFORMATION = 0x00000001
_SDDL_REVISION_1 = 1
_ERROR_ALREADY_EXISTS = 183


class _SecurityAttributes(ctypes.Structure):
    _fields_ = [("nLength", wintypes.DWORD), ("lpSecurityDescriptor", wintypes.LPVOID),
                ("bInheritHandle", wintypes.BOOL)]


def _advapi32():
    lib = ctypes.WinDLL("advapi32", use_last_error=True)
    lib.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    lib.OpenProcessToken.restype = wintypes.BOOL
    lib.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
                                        ctypes.POINTER(wintypes.DWORD)]
    lib.GetTokenInformation.restype = wintypes.BOOL
    lib.ConvertSidToStringSidW.argtypes = [wintypes.LPVOID, ctypes.POINTER(wintypes.LPWSTR)]
    lib.ConvertSidToStringSidW.restype = wintypes.BOOL
    lib.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(wintypes.LPVOID), ctypes.POINTER(wintypes.ULONG)]
    lib.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
    lib.GetNamedSecurityInfoW.argtypes = [
        wintypes.LPCWSTR, ctypes.c_int, wintypes.DWORD, ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID), ctypes.POINTER(wintypes.LPVOID), ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID)]
    lib.GetNamedSecurityInfoW.restype = wintypes.DWORD
    return lib


def _kernel32():
    lib = ctypes.WinDLL("kernel32", use_last_error=True)
    # HANDLE-typed returns: ctypes would otherwise truncate the pseudo-handle to c_int.
    lib.GetCurrentProcess.restype = wintypes.HANDLE
    lib.CloseHandle.argtypes = [wintypes.HANDLE]
    lib.CreateDirectoryW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(_SecurityAttributes)]
    lib.CreateDirectoryW.restype = wintypes.BOOL
    lib.LocalFree.argtypes = [wintypes.HLOCAL]
    lib.LocalFree.restype = wintypes.HLOCAL
    return lib


def _raise_last(what: str, path: Path | None = None, code: int | None = None) -> None:
    code = ctypes.get_last_error() if code is None else code
    raise ctypes.WinError(code, f"{what} failed" + (f" for {path}" if path is not None else ""))


def _sid_to_string(advapi32, kernel32, sid) -> str:
    text = wintypes.LPWSTR()
    if not advapi32.ConvertSidToStringSidW(sid, ctypes.byref(text)):
        _raise_last("ConvertSidToStringSidW")
    try:
        return text.value or ""
    finally:
        kernel32.LocalFree(ctypes.cast(text, wintypes.HLOCAL))


def current_user_sid() -> str:
    """String SID of the process token's user (the same for elevated and unelevated tokens)."""
    advapi32, kernel32 = _advapi32(), _kernel32()
    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), _TOKEN_QUERY, ctypes.byref(token)):
        _raise_last("OpenProcessToken")
    try:
        needed = wintypes.DWORD()
        advapi32.GetTokenInformation(token, _TOKEN_USER_CLASS, None, 0, ctypes.byref(needed))
        buffer = ctypes.create_string_buffer(needed.value)
        if not advapi32.GetTokenInformation(token, _TOKEN_USER_CLASS, buffer, needed, ctypes.byref(needed)):
            _raise_last("GetTokenInformation")
        # TOKEN_USER begins with SID_AND_ATTRIBUTES, whose first field is the PSID.
        sid = ctypes.cast(buffer, ctypes.POINTER(wintypes.LPVOID)).contents
        return _sid_to_string(advapi32, kernel32, sid)
    finally:
        kernel32.CloseHandle(token)


def private_directory_sddl(user_sid: str) -> str:
    """Owner = the user; protected DACL (nothing inherited) granting full control to SYSTEM and the user."""
    return f"O:{user_sid}D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;{user_sid})"


def create_private_directory(path: Path, user_sid: str) -> bool:
    """Create *path* with :func:`private_directory_sddl`; returns False when it already exists."""
    advapi32, kernel32 = _advapi32(), _kernel32()
    descriptor = wintypes.LPVOID()
    if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            private_directory_sddl(user_sid), _SDDL_REVISION_1, ctypes.byref(descriptor), None):
        _raise_last("ConvertStringSecurityDescriptorToSecurityDescriptorW")
    try:
        attributes = _SecurityAttributes(ctypes.sizeof(_SecurityAttributes), descriptor, False)
        created = bool(kernel32.CreateDirectoryW(str(path), ctypes.byref(attributes)))
        code = 0 if created else ctypes.get_last_error()
    finally:
        kernel32.LocalFree(descriptor)
    if code and code != _ERROR_ALREADY_EXISTS:
        _raise_last("CreateDirectoryW", path, code)
    return created


def owner_sid(path: Path) -> str:
    """String SID of *path*'s owner. Raises PermissionError when the caller may not read it."""
    advapi32, kernel32 = _advapi32(), _kernel32()
    owner, descriptor = wintypes.LPVOID(), wintypes.LPVOID()
    code = advapi32.GetNamedSecurityInfoW(str(path), _SE_FILE_OBJECT, _OWNER_SECURITY_INFORMATION,
                                          ctypes.byref(owner), None, None, None, ctypes.byref(descriptor))
    if code:
        _raise_last("GetNamedSecurityInfoW", path, code)
    try:
        return _sid_to_string(advapi32, kernel32, owner)
    finally:
        kernel32.LocalFree(descriptor)
