"""Cross-profile subtree fences: shared ancestors, exclusive resources, one stable OS-user namespace."""

from contextlib import contextmanager, ExitStack, suppress
import errno
import hashlib
import os
from pathlib import Path
import threading
import time

_held = threading.local()


def _windows_lock(fd, *, shared, release=False):  # pragma: no cover - native Windows CI
    import ctypes
    from ctypes import wintypes as wt
    import msvcrt

    class Overlapped(ctypes.Structure):
        _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                    ("Offset", wt.DWORD), ("OffsetHigh", wt.DWORD), ("hEvent", wt.HANDLE)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    handle, overlap = msvcrt.get_osfhandle(fd.fileno()), Overlapped()
    if release:
        call = kernel.UnlockFileEx
        call.argtypes = [wt.HANDLE, wt.DWORD, wt.DWORD, wt.DWORD, ctypes.POINTER(Overlapped)]
        args = handle, 0, 1, 0, ctypes.byref(overlap)
    else:
        call = kernel.LockFileEx
        call.argtypes = [wt.HANDLE, wt.DWORD, wt.DWORD, wt.DWORD, wt.DWORD, ctypes.POINTER(Overlapped)]
        args = handle, 1 | (0 if shared else 2), 0, 1, 0, ctypes.byref(overlap)
    call.restype = wt.BOOL
    if not call(*args):
        error = ctypes.get_last_error()
        if not release and error == 33:  # ERROR_LOCK_VIOLATION
            raise BlockingIOError(errno.EAGAIN, "Skill resource is busy")
        raise ctypes.WinError(error)


def _lock(fd, shared, *, release=False):
    if os.name == "nt":
        return _windows_lock(fd, shared=shared, release=release)
    import fcntl
    operation = fcntl.LOCK_UN if release else (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB
    fcntl.flock(fd, operation)


@contextmanager
def _resource_lock(path, shared, deadline):
    held = getattr(_held, "resources", None)
    if held is None:
        held = _held.resources = {}
    if path in held:
        if held[path] and not shared:
            raise OSError("Nested skill transaction requested an unsafe lock upgrade")
        yield
        return
    # Do not create catalog roots to obtain ownership; keep lock inodes outside skill trees.
    from hermes_constants import get_real_home
    # Subprocess HOME can be profile-local; the real-home contract survives that remap.
    namespace = Path(get_real_home()).resolve() / ".cache" / "hermes" / "skill-resource-locks"
    namespace.mkdir(parents=True, exist_ok=True, mode=0o700)
    name = hashlib.sha256(os.fsencode(os.path.normcase(str(path)))).hexdigest()
    with (namespace / name).open("a+b") as fd:
        while True:
            try:
                _lock(fd, shared)
                break
            except BlockingIOError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Another profile holds a shared skill resource.") from exc
                time.sleep(min(remaining, 0.05))
        held[path] = shared
        try:
            yield
        finally:
            held.pop(path)
            with suppress(OSError):
                _lock(fd, shared, release=True)  # Descriptor close releases ownership even after an unlock failure.


def requests_for(resources):
    """S on proper ancestors conflicts with X on an overlapping ancestor subtree."""
    requests = {}
    for path in resources:
        path = Path(path).resolve()
        for parent in path.parents:
            requests.setdefault(parent, True)
        requests[path] = False
    return requests


@contextmanager
def acquire_resources(resources, deadline):
    with ExitStack() as stack:
        for path, shared in sorted(requests_for(resources).items(), key=lambda item: os.path.normcase(str(item[0]))):
            stack.enter_context(_resource_lock(path, shared, deadline))
        yield


def covered(resources):
    """A nested operation must not enlarge ownership after any outer mutation."""
    held = getattr(_held, "resources", {})
    return all(any(not shared and (path == owned or path.is_relative_to(owned))
                   for owned, shared in held.items()) for path in resources)


def resources_held():
    return bool(getattr(_held, "resources", {}))
