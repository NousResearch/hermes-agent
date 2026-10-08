"""OS write boundary for basenames a profile has registered.

The command scanner cannot see a name built at runtime. This module denies the
write in the kernel of the process that actually opens the final path. It uses
the standard library only, so a container can exec the same source. A missing
boundary refuses the command; it does not run the command raw.
"""

from __future__ import annotations

import base64
import ctypes
import errno
import inspect
import json
import os
import re
import shlex
import shutil
import signal
import stat
import sys


_SAFE_NAME = re.compile(r"[A-Za-z0-9._-]+")
_LINUX_ENVS = {"docker", "singularity"}
_STD_FD = {"stdin": "0", "stdout": "1", "stderr": "2"}
_TRUSTED_PYTHON_DIRS = ("/usr/local/bin", "/usr/bin", "/bin")
_SCAN_LIMIT = 20000

AT_FDCWD = -100
O_ACCMODE = 0o3
O_CREAT = 0o100
O_TRUNC = 0o1000
O_APPEND = 0o2000
O_PATH = 0o10000000
WRITE_MASK = O_ACCMODE | O_CREAT | O_TRUNC | O_APPEND
RESOLVE_IN_ROOT = 0x10
AUDIT_ARCH_AARCH64 = 0xC00000B7
SECCOMP_SET_MODE_FILTER = 1
SECCOMP_RET_ALLOW = 0x7FFF0000
SECCOMP_RET_TRACE = 0x7FF00000
SECCOMP_RET_ERRNO = 0x00050000
PR_SET_NO_NEW_PRIVS = 38
SYS_SECCOMP = 277
SYS_PRCTL = 167
SYS_PTRACE = 117
PTRACE_TRACEME = 0
PTRACE_CONT = 7
PTRACE_SETOPTIONS = 0x4200
PTRACE_GETREGSET = 0x4204
PTRACE_SETREGSET = 0x4205
PTRACE_O_TRACESYSGOOD = 1
PTRACE_O_TRACEFORK = 2
PTRACE_O_TRACEVFORK = 4
PTRACE_O_TRACECLONE = 8
PTRACE_O_TRACEEXEC = 16
PTRACE_O_TRACESECCOMP = 0x80
PTRACE_O_EXITKILL = 0x100000
PTRACE_EVENT_FORK = 1
PTRACE_EVENT_VFORK = 2
PTRACE_EVENT_CLONE = 3
PTRACE_EVENT_SECCOMP = 7
NT_PRSTATUS = 1
NT_ARM_SYSTEM_CALL = 0x404
# linux/arch/arm64 uses the asm-generic numbers. openat through truncate were
# measured on the Big container's linuxkit kernel; mkdirat is the generic slot
# immediately before unlinkat.
AARCH64 = {
    "mkdirat": 34,
    "unlinkat": 35,
    "symlinkat": 36,
    "linkat": 37,
    "renameat": 38,
    "truncate": 45,
    "openat": 56,
    "mount": 40,
    "umount2": 39,
    "pivot_root": 41,
    "unshare": 97,
    "open_by_handle_at": 265,
    "setns": 268,
    "renameat2": 276,
    "io_uring_setup": 425,
    "io_uring_enter": 426,
    "io_uring_register": 427,
    "open_tree": 428,
    "move_mount": 429,
    "fsopen": 430,
    "fsconfig": 431,
    "fsmount": 432,
    "fspick": 433,
    "openat2": 437,
}

libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long


class SockFilter(ctypes.Structure):
    _fields_ = (
        ("code", ctypes.c_uint16),
        ("jt", ctypes.c_uint8),
        ("jf", ctypes.c_uint8),
        ("k", ctypes.c_uint32),
    )


class SockFprog(ctypes.Structure):
    _fields_ = (("len", ctypes.c_ushort), ("filter", ctypes.POINTER(SockFilter)))


class IOVec(ctypes.Structure):
    _fields_ = (("base", ctypes.c_void_p), ("len", ctypes.c_size_t))


def build_filter():
    """Classic BPF: trace the mutating path syscalls, allow ordinary reads.

    io_uring and open_by_handle_at never carry a path in the syscall arguments,
    so they are refused outright instead of being traced.
    """
    load, jump, equal, ret, word, absolute, constant = 0x00, 0x05, 0x10, 0x06, 0x00, 0x20, 0x00
    denied = (
        "open_by_handle_at", "io_uring_setup", "io_uring_enter", "io_uring_register",
        # A new mount can publish another path to the same inode. The traced
        # command has no mount capability in this container; refuse the attempt
        # instead of inspecting a namespace the supervisor does not share.
        "mount", "umount2", "pivot_root", "unshare", "setns",
        "open_tree", "move_mount", "fsopen", "fsconfig", "fsmount", "fspick",
    )
    always = (
        "mkdirat", "unlinkat", "renameat", "renameat2",
        "symlinkat", "linkat", "truncate", "openat2",
    )
    ops = [
        (load | word | absolute, 0, 0, 4),
        (jump | equal | constant, 1, 0, AUDIT_ARCH_AARCH64),
        (ret | constant, 0, 0, SECCOMP_RET_ERRNO | errno.EPERM),
        (load | word | absolute, 0, 0, 0),
    ]
    for name in denied:
        ops.append((jump | equal | constant, 0, 1, AARCH64[name]))
        ops.append((ret | constant, 0, 0, SECCOMP_RET_ERRNO | errno.EPERM))
    for name in always:
        ops.append((jump | equal | constant, 0, 1, AARCH64[name]))
        ops.append((ret | constant, 0, 0, SECCOMP_RET_TRACE))
    ops.append((jump | equal | constant, 0, 1, AARCH64["openat"]))
    ops.append((jump | constant, 0, 0, 0))
    allow_at = len(ops)
    ops.append((ret | constant, 0, 0, SECCOMP_RET_ALLOW))
    flag_at = len(ops)
    ja = ops[allow_at - 1]
    ops[allow_at - 1] = (ja[0], 0, 0, flag_at - allow_at)
    ops.extend((
        (load | word | absolute, 0, 0, 32),
        (0x54, 0, 0, WRITE_MASK),
        (jump | equal | constant, 0, 1, 0),
        (ret | constant, 0, 0, SECCOMP_RET_ALLOW),
        (ret | constant, 0, 0, SECCOMP_RET_TRACE),
    ))
    filters = (SockFilter * len(ops))()
    for index, (code, jt, jf, value) in enumerate(ops):
        filters[index].code = code
        filters[index].jt = jt
        filters[index].jf = jf
        filters[index].k = value
    return filters


FILTERS = build_filter()


def ptrace(request, pid, addr=0, data=0):
    rc = libc.syscall(SYS_PTRACE, request, pid, addr, data)
    if rc != 0:
        raise OSError(ctypes.get_errno(), f"ptrace {request} pid {pid}")
    return rc


def read_cstr(pid, addr):
    if not addr:
        return b""
    fd = os.open(f"/proc/{pid}/mem", os.O_RDONLY)
    try:
        try:
            blob = os.pread(fd, 4096, addr)
        except (OSError, OverflowError) as exc:
            raise OSError(errno.EIO, f"unreadable path address {addr:#x}") from exc
    finally:
        os.close(fd)
    return blob.split(b"\0", 1)[0]


def _join_rest(cur, parts):
    acc = cur or "/"
    for extra in parts:
        if extra in ("", "."):
            continue
        if extra == "..":
            acc = "/" if acc == "/" else os.path.dirname(acc)
        else:
            acc = "/" + extra if acc == "/" else acc + "/" + extra
    return acc


def _fd_component(cur, part, parts, index, pid):
    """Return ``(number, rest)`` when the next components name a process fd."""
    if cur == "/dev" and part in _STD_FD:
        return _STD_FD[part], parts[index + 1:]
    rest_at = None
    task = f"/proc/{pid}/task/"
    if (cur == f"/proc/{pid}" or cur == "/dev") and part == "fd":
        rest_at = index + 1
    elif cur.startswith(task) and cur.count("/") == 4 and part == "fd":
        rest_at = index + 1
    if rest_at is None or rest_at >= len(parts) or not parts[rest_at].isdigit():
        return None
    return parts[rest_at], parts[rest_at + 1:]


def _restart_from_fd(pid, number, rest, depth):
    link = f"/proc/{int(pid)}/fd/{number}"
    try:
        target = os.readlink(link)
    except OSError as exc:
        raise OSError(errno.EIO, f"unreadable {link}") from exc
    deleted = " (deleted)"
    if target.endswith(deleted):
        target = target[:-len(deleted)]
    if not target:
        raise OSError(errno.EIO, "empty fd target")
    if not target.startswith("/"):
        if rest:
            raise OSError(errno.EIO, "non-file fd")
        return link
    combined = target if not rest else target.rstrip("/") + "/" + "/".join(rest)
    return _follow_child_path(combined, pid, depth + 1)


def _follow_child_path(path, pid, depth=0):
    """Resolve *path* as the traced process would, not as the supervisor.

    ``/proc/self`` in the supervisor is the supervisor. Component walk rewrites
    that prefix, ``root``, ``cwd``, and ``fd`` links to the traced pid before
    following any symlink. ``..`` is applied after those links, so a lexical
    ``normpath`` cannot point the check at a different file than the kernel.
    """
    if depth > 16:
        raise OSError(errno.EIO, "path did not resolve")
    pid = int(pid)
    if not path.startswith("/"):
        try:
            base = os.readlink(f"/proc/{pid}/cwd")
        except OSError as exc:
            raise OSError(errno.EIO, "unreadable cwd") from exc
        path = base.rstrip("/") + "/" + path
    cur = "/"
    parts = [part for part in path.split("/") if part not in ("", ".")]
    index = 0
    while index < len(parts):
        part = parts[index]
        if part == "..":
            cur = "/" if cur == "/" else os.path.dirname(cur)
            index += 1
            continue
        if cur == "/proc" and part in ("self", "thread-self"):
            cur = f"/proc/{pid}"
            index += 1
            continue
        if cur == f"/proc/{pid}" and part == "root":
            rest = parts[index + 1:]
            restarted = "/" + "/".join(rest) if rest else "/"
            return _follow_child_path(restarted, pid, depth + 1)
        if cur == f"/proc/{pid}" and part == "cwd":
            try:
                base = os.readlink(f"/proc/{pid}/cwd")
            except OSError as exc:
                raise OSError(errno.EIO, "unreadable cwd") from exc
            rest = parts[index + 1:]
            restarted = base if not rest else base.rstrip("/") + "/" + "/".join(rest)
            return _follow_child_path(restarted, pid, depth + 1)
        found = _fd_component(cur, part, parts, index, pid)
        if found is not None:
            number, rest = found
            return _restart_from_fd(pid, number, rest, depth)
        nxt = "/" + part if cur == "/" else cur + "/" + part
        try:
            info = os.lstat(nxt)
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                return _join_rest(cur, parts[index:])
            raise OSError(errno.EIO, f"unreadable {nxt}") from exc
        if stat.S_ISLNK(info.st_mode):
            try:
                target = os.readlink(nxt)
            except OSError as exc:
                raise OSError(errno.EIO, f"unreadable symlink {nxt}") from exc
            if target.startswith("/"):
                combined = target
            else:
                combined = (cur if cur != "/" else "") + "/" + target
            rest = parts[index + 1:]
            if rest:
                combined = combined.rstrip("/") + "/" + "/".join(rest)
            return _follow_child_path(combined, pid, depth + 1)
        cur = nxt
        index += 1
    return cur


def _remember(names, paths, text):
    if not text:
        return
    base = os.path.basename(text.rstrip("/"))
    if base:
        names.add(base)
    if text not in paths:
        paths.append(text)


def _read_open_how(pid, addr, size):
    """Return ``(flags, resolve)`` from the traced process.

    The kernel accepts any ``open_how`` size from 24 bytes through a page, and
    ignores trailing zeros. Only a short size is EINVAL. A larger size is still
    checked; skipping it would let the write through.
    """
    size = int(size)
    if size < 24 or not addr:
        raise OSError(errno.EINVAL, "open_how size")
    fd = os.open(f"/proc/{int(pid)}/mem", os.O_RDONLY)
    try:
        try:
            blob = os.pread(fd, 24, addr)
        except (OSError, OverflowError) as exc:
            raise OSError(errno.EIO, "unreadable open_how") from exc
    finally:
        os.close(fd)
    if len(blob) < 24:
        raise OSError(errno.EIO, "short open_how")
    flags = int.from_bytes(blob[0:8], "little")
    resolve = int.from_bytes(blob[16:24], "little")
    return flags, resolve


def _directory_of(pid, dirfd):
    """Directory an ``openat2(RESOLVE_IN_ROOT)`` call uses as its root."""
    if int(dirfd) < 0 and int(dirfd) != AT_FDCWD:
        raise OSError(errno.EIO, "bad dirfd")
    link = f"/proc/{int(pid)}/cwd" if int(dirfd) == AT_FDCWD or int(dirfd) < 0 else f"/proc/{int(pid)}/fd/{int(dirfd)}"
    try:
        target = os.readlink(link)
    except OSError as exc:
        raise OSError(errno.EIO, f"unreadable {link}") from exc
    deleted = " (deleted)"
    if target.endswith(deleted):
        target = target[:-len(deleted)]
    if not target.startswith("/"):
        raise OSError(errno.EIO, "dirfd is not a directory")
    return target.rstrip("/") or "/"


def _stay_in_root(root, cur):
    if cur == root:
        return cur
    parent = os.path.dirname(cur) or "/"
    if root == "/" or parent == root or parent.startswith(root + "/"):
        return parent
    return root


def _follow_in_root(root, rel, pid, depth=0):
    """Resolve *rel* with ``dirfd`` as the root, the way ``RESOLVE_IN_ROOT`` does.

    A leading slash does not mean the real filesystem root. Absolute symlink
    targets stay under *root* too. The real ``/proc/self`` magic is used only
    when the walk actually reaches it.
    """
    if depth > 16:
        raise OSError(errno.EIO, "path did not resolve")
    root = root.rstrip("/") or "/"
    cur = root
    parts = [part for part in rel.split("/") if part not in ("", ".")]
    index = 0
    while index < len(parts):
        part = parts[index]
        if part == "..":
            cur = _stay_in_root(root, cur)
            index += 1
            continue
        if cur == "/proc" and part in ("self", "thread-self"):
            return _follow_child_path("/proc/" + "/".join(parts[index:]), pid, depth + 1)
        nxt = "/" + part if cur == "/" else cur + "/" + part
        try:
            info = os.lstat(nxt)
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                acc = cur
                for extra in parts[index:]:
                    if extra in ("", "."):
                        continue
                    if extra == "..":
                        acc = _stay_in_root(root, acc)
                    else:
                        acc = "/" + extra if acc == "/" else acc + "/" + extra
                return acc
            raise OSError(errno.EIO, f"unreadable {nxt}") from exc
        if stat.S_ISLNK(info.st_mode):
            try:
                target = os.readlink(nxt)
            except OSError as exc:
                raise OSError(errno.EIO, f"unreadable symlink {nxt}") from exc
            rest = parts[index + 1:]
            if target.startswith("/"):
                combined = target.lstrip("/")
            else:
                prefix = "" if cur == root else cur[len(root):].lstrip("/")
                combined = target if not prefix else prefix + "/" + target
            if rest:
                combined = combined.rstrip("/") + "/" + "/".join(rest)
            return _follow_in_root(root, combined, pid, depth + 1)
        cur = nxt
        index += 1
    return cur


def candidate_paths(pid, dirfd, addr, resolve=0):
    """Basenames and absolute paths the syscall is about to use.

    The raw bytes and the path the traced process will open are both included.
    ``/proc/self`` is the traced process, including after ``root``, ``cwd``,
    ``task``, or a symlink. ``openat2`` with ``RESOLVE_IN_ROOT`` keeps an
    absolute path under ``dirfd``. An unreadable link is denied. An empty path
    refers to *dirfd* itself (``AT_EMPTY_PATH``).
    """
    text = read_cstr(pid, addr).decode("utf-8", "surrogateescape")
    names = set()
    paths = []
    if not text:
        if dirfd == AT_FDCWD or int(dirfd) < 0:
            return set(), []
        full = f"/proc/{int(pid)}/fd/{int(dirfd)}"
    else:
        _remember(names, [], text)
        if text.startswith("/"):
            full = text
        elif dirfd == AT_FDCWD:
            try:
                base = os.readlink(f"/proc/{pid}/cwd")
            except OSError as exc:
                raise OSError(errno.EIO, "unreadable cwd") from exc
            full = base.rstrip("/") + "/" + text
        else:
            try:
                base = os.readlink(f"/proc/{pid}/fd/{int(dirfd)}")
            except OSError as exc:
                raise OSError(errno.EIO, f"unreadable dirfd {dirfd}") from exc
            deleted = " (deleted)"
            if base.endswith(deleted):
                base = base[:-len(deleted)]
            if not base.startswith("/"):
                raise OSError(errno.EIO, f"unreadable dirfd {dirfd}")
            full = base if text == "." else base.rstrip("/") + "/" + text
    if resolve & RESOLVE_IN_ROOT:
        root = _directory_of(pid, dirfd)
        resolved = _follow_in_root(root, text.lstrip("/"), pid)
    else:
        resolved = _follow_child_path(full, pid)
    _remember(names, paths, resolved)
    return {name for name in names if name}, paths


def path_names(pid, dirfd, addr):
    names, _paths = candidate_paths(pid, dirfd, addr)
    return names


def tree_has_protected(root, names, limit=20000):
    """True when a real directory holds a protected basename.

    A symlink is not walked: renaming the link does not move its target.
    A tree that cannot be read, or that is larger than *limit*, is treated as
    protected so a rename cannot carry an unseen authority file away.
    """
    try:
        info = os.lstat(root)
    except OSError as exc:
        return exc.errno != errno.ENOENT
    if not stat.S_ISDIR(info.st_mode):
        return False
    seen = 0
    for _dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        if set(dirnames) & names or set(filenames) & names:
            return True
        seen += len(dirnames) + len(filenames)
        if seen > limit:
            return True
    return False


def shares_protected_inode(path, names):
    """True when *path* is another name for a protected file, or that cannot be disproved.

    A new path (ENOENT) and a regular file with one link are left alone. A second
    link in another directory, or a parent that cannot be listed, is refused.
    """
    try:
        info = os.lstat(path)
    except OSError as exc:
        return exc.errno != errno.ENOENT
    if not stat.S_ISREG(info.st_mode) or info.st_nlink <= 1:
        return False
    parent = os.path.dirname(path) or "/"
    matched = 0
    try:
        with os.scandir(parent) as entries:
            for entry in entries:
                try:
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    return True
                if st.st_dev != info.st_dev or st.st_ino != info.st_ino:
                    continue
                matched += 1
                if entry.name in names:
                    return True
    except OSError:
        return True
    return matched < info.st_nlink


def protected_write(pid, regs, names):
    number = int(regs[8])

    def i32(index):
        return ctypes.c_int64(int(regs[index])).value

    paths = []
    rename_source = None
    open_resolve = 0
    if number == AARCH64["openat"]:
        flags = int(regs[2])
        if (flags & O_PATH) or not (flags & WRITE_MASK):
            return False
        paths = [(i32(0), int(regs[1]))]
    elif number == AARCH64["openat2"]:
        try:
            flags, open_resolve = _read_open_how(pid, int(regs[2]), int(regs[3]))
        except OSError as exc:
            return exc.errno != errno.EINVAL
        if (flags & O_PATH) or not (flags & WRITE_MASK):
            return False
        paths = [(i32(0), int(regs[1]))]
    elif number in (AARCH64["unlinkat"], AARCH64["mkdirat"]):
        paths = [(i32(0), int(regs[1]))]
    elif number == AARCH64["truncate"]:
        paths = [(AT_FDCWD, int(regs[0]))]
    elif number in (AARCH64["renameat"], AARCH64["renameat2"], AARCH64["linkat"]):
        paths = [(i32(0), int(regs[1])), (i32(2), int(regs[3]))]
        if number != AARCH64["linkat"]:
            rename_source = paths[0]
    elif number == AARCH64["symlinkat"]:
        paths = [(i32(1), int(regs[2])), (AT_FDCWD, int(regs[0]))]
    else:
        return False
    for dirfd, addr in paths:
        try:
            found, full = candidate_paths(pid, dirfd, addr, open_resolve)
            if found & names:
                return True
            for one in full:
                if shares_protected_inode(one, names):
                    return True
            if rename_source == (dirfd, addr):
                for root in full:
                    if tree_has_protected(root, names):
                        return True
        except OSError:
            return True
    return False


def get_regset(pid):
    regs = (ctypes.c_uint64 * 34)()
    iov = IOVec(ctypes.addressof(regs), ctypes.sizeof(regs))
    rc = libc.syscall(SYS_PTRACE, PTRACE_GETREGSET, pid, NT_PRSTATUS, ctypes.byref(iov))
    if rc != 0:
        raise OSError(ctypes.get_errno(), "getregset")
    return regs


def set_regset(pid, regs):
    iov = IOVec(ctypes.addressof(regs), ctypes.sizeof(regs))
    rc = libc.syscall(SYS_PTRACE, PTRACE_SETREGSET, pid, NT_PRSTATUS, ctypes.byref(iov))
    if rc != 0:
        raise OSError(ctypes.get_errno(), "setregset")


def deny_syscall(pid, regs):
    """Skip the syscall and return EACCES. Both halves are required on aarch64."""
    regs[0] = ctypes.c_uint64(-errno.EACCES).value
    set_regset(pid, regs)
    blocked = ctypes.c_int(-1)
    block_iov = IOVec(ctypes.addressof(blocked), ctypes.sizeof(blocked))
    rc = libc.syscall(
        SYS_PTRACE, PTRACE_SETREGSET, pid, NT_ARM_SYSTEM_CALL, ctypes.byref(block_iov),
    )
    if rc != 0:
        raise OSError(ctypes.get_errno(), "skip syscall")


def install_filter():
    rc = libc.syscall(SYS_PRCTL, PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
    if rc != 0:
        raise OSError(ctypes.get_errno(), "no_new_privs")
    prog = SockFprog(len(FILTERS), ctypes.cast(FILTERS, ctypes.POINTER(SockFilter)))
    rc = libc.syscall(SYS_SECCOMP, SECCOMP_SET_MODE_FILTER, 0, ctypes.byref(prog))
    if rc != 0:
        raise OSError(ctypes.get_errno(), "seccomp")


def _kill_child(pid):
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        os.waitpid(pid, 0)
    except OSError:
        pass


def supervise(names, command=None, argv=None):
    """Trace one command. Exit 126 without exec when the filter cannot be installed."""
    machine = os.uname().machine
    if machine not in {"aarch64", "arm64"}:
        sys.stderr.write(f"authority guard: unsupported architecture {machine}\n")
        return 126
    if not names or (command is None and not argv):
        sys.stderr.write("authority guard: nothing to supervise\n")
        return 126
    root = os.fork()
    if root == 0:
        try:
            ptrace(PTRACE_TRACEME, 0)
            os.kill(os.getpid(), signal.SIGSTOP)
            install_filter()
            if argv:
                os.execvp(argv[0], argv)
            os.execv("/bin/bash", ["bash", "-c", command])
        except OSError as exc:
            sys.stderr.write(f"authority guard failed: {exc}\n")
            os._exit(126)
    options = (
        PTRACE_O_TRACESYSGOOD | PTRACE_O_TRACEFORK | PTRACE_O_TRACEVFORK
        | PTRACE_O_TRACECLONE | PTRACE_O_TRACEEXEC | PTRACE_O_TRACESECCOMP
        | PTRACE_O_EXITKILL
    )
    try:
        os.waitpid(root, 0)
        ptrace(PTRACE_SETOPTIONS, root, 0, options)
    except OSError as exc:
        _kill_child(root)
        sys.stderr.write(f"authority guard failed: {exc}\n")
        return 126
    ptrace(PTRACE_CONT, root, 0, 0)
    alive = {root}
    root_status = None
    while alive:
        try:
            pid, status = os.waitpid(-1, 0)
        except InterruptedError:
            continue
        except ChildProcessError:
            break
        if os.WIFEXITED(status) or os.WIFSIGNALED(status):
            alive.discard(pid)
            if pid == root:
                root_status = status
            continue
        if not os.WIFSTOPPED(status):
            continue
        event = (status >> 16) & 0xFF
        if event == PTRACE_EVENT_SECCOMP:
            try:
                regs = get_regset(pid)
                if protected_write(pid, regs, names):
                    deny_syscall(pid, regs)
            except OSError as exc:
                sys.stderr.write(f"authority guard deny failed: {exc}\n")
                _kill_child(pid)
                alive.discard(pid)
                if pid == root:
                    root_status = None
                continue
        elif event in (PTRACE_EVENT_FORK, PTRACE_EVENT_VFORK, PTRACE_EVENT_CLONE):
            alive.add(pid)
        try:
            ptrace(PTRACE_CONT, pid, 0, 0)
        except OSError:
            alive.discard(pid)
    if root_status is None:
        return 126
    if os.WIFEXITED(root_status):
        return os.WEXITSTATUS(root_status)
    return 128 + os.WTERMSIG(root_status)


def supervise_main(payload_b64):
    try:
        payload = json.loads(base64.b64decode(payload_b64))
        names = {str(name) for name in (payload.get("names") or []) if str(name)}
        argv = payload.get("argv")
        if isinstance(argv, list) and argv and all(isinstance(item, str) and item for item in argv):
            code = supervise(names, argv=argv)
        elif isinstance(payload.get("command"), str):
            code = supervise(names, command=payload["command"])
        else:
            code = 126
    except Exception as exc:
        sys.stderr.write(f"authority guard failed: {exc}\n")
        code = 126
    raise SystemExit(code)


def _module_source() -> bytes:
    return open(__file__, "rb").read()


def _stub() -> str:
    return (
        "import base64,sys\n"
        "ns={}\n"
        "ns['__name__']='authority_os_guard'\n"
        "exec(base64.b64decode(sys.argv[1]), ns)\n"
        "ns['supervise_main'](sys.argv[2])\n"
    )


def _exec_trusted_python(code: str, args: list[str]) -> str:
    """Run *code* with an absolute trusted interpreter and ``-I``.

    ``PATH`` is not consulted. A user-site ``usercustomize`` therefore cannot
    start in the supervisor or the official-writer check.
    """
    listed = " ".join(
        shlex.quote(os.path.join(root, name))
        for root in _TRUSTED_PYTHON_DIRS
        for name in ("python3", "python")
    )
    tail = " ".join(shlex.quote(arg) for arg in args)
    return (
        "for _hermes_py in "
        + listed
        + '; do [ -x "$_hermes_py" ] && exec "$_hermes_py" -I -c '
        + shlex.quote(code)
        + ((" " + tail) if tail else "")
        + "; done; "
        "echo 'authority writer interpreter was not trusted' >&2; exit 126"
    )


def _linux_python_command(payload: dict) -> str:
    """Shell command whose outer shell never parses the supervised command."""
    source = base64.b64encode(_module_source()).decode("ascii")
    encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    return _exec_trusted_python(_stub(), [source, encoded])


def linux_shell_command(command: str, names) -> str:
    return _linux_python_command({"command": command, "names": sorted(names)})


def linux_argv_command(argv, names) -> str:
    return _linux_python_command({"argv": list(argv), "names": sorted(names)})


def darwin_profile(names) -> str | None:
    safe = [name for name in sorted(names) if _SAFE_NAME.fullmatch(name)]
    if len(safe) != len(list(names)):
        return None
    if not safe:
        return None
    body = "|".join(re.escape(name) for name in safe)
    return (
        "(version 1)\n"
        "(allow default)\n"
        f'(deny file-write* (regex "^/.*/({body})$"))\n'
    )


def _usable_names(names):
    raw = [str(name) for name in names]
    safe = [name for name in sorted(set(raw)) if _SAFE_NAME.fullmatch(name)]
    if len(safe) != len(set(raw)) or not safe:
        return None
    return safe


def exec_darwin_boundary():
    """Scan the current directory, then exec under a seatbelt profile.

    A static basename rule misses a pre-existing hard link and a rename of a
    directory that contains a protected file. Unreadable subtrees stay write-denied;
    ordinary file counts do not revoke unrelated write permission.
    """
    import json
    import os
    import re
    import stat
    import sys

    spec = json.loads(sys.argv[1])
    names = {str(name) for name in (spec.get("names") or []) if str(name)}
    sandbox = spec.get("sandbox")
    if not names or not isinstance(sandbox, str) or not os.path.isfile(sandbox):
        sys.stderr.write("authority guard: sandbox is unavailable\n")
        raise SystemExit(126)
    cwd = os.path.realpath(os.getcwd())
    roots = [cwd]
    for candidate in ("/tmp", "/private/tmp", os.environ.get("TMPDIR") or ""):
        if not candidate:
            continue
        try:
            real_root = os.path.realpath(candidate)
        except OSError:
            sys.stderr.write("authority guard: directory scan failed closed\n")
            raise SystemExit(126)
        if os.path.isdir(real_root):
            roots.append(real_root)
    roots = sorted(set(roots))
    covered = []
    for root in roots:
        if any(root == earlier or root.startswith(earlier + os.sep) for earlier in covered):
            continue
        covered.append(root)
    records = []
    unreadable = set()
    def scan_error(exc):
        # Deny the unscanned subtree, not every unrelated ordinary write.
        if not getattr(exc, "filename", None):
            raise exc
        unreadable.add(os.path.realpath(exc.filename))
    try:
        for root in covered:
            for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=scan_error):
                parent = os.path.realpath(dirpath)
                for filename in filenames:
                    full = os.path.join(dirpath, filename)
                    try:
                        info = os.lstat(full)
                    except FileNotFoundError:
                        continue  # A concurrently removed temp entry has no inode to protect.
                    except OSError:
                        unreadable.add(os.path.realpath(full))
                        continue
                    is_reg = stat.S_ISREG(info.st_mode)
                    # Only protected names and multiply-linked regular files
                    # affect the deny rules. Ordinary single-link files need no record.
                    if filename in names or (is_reg and info.st_nlink > 1):
                        records.append((
                            info.st_dev, info.st_ino, filename,
                            os.path.realpath(full) if is_reg else "", is_reg, parent,
                            info.st_nlink if is_reg else 0,
                        ))
                for dirname in dirnames:
                    if dirname in names:
                        records.append((0, 0, dirname, "", False, parent, 0))
    except OSError as exc:
        sys.stderr.write("authority guard: directory scan failed closed: %s\n" % exc)
        raise SystemExit(126)
    link_counts = {}
    link_totals = {}
    for dev, ino, _name, _real, is_reg, _parent, nlink in records:
        if not is_reg:
            continue
        key = (dev, ino)
        link_counts[key] = link_counts.get(key, 0) + 1
        link_totals[key] = nlink
    incomplete = set()
    for key, nlink in link_totals.items():
        if link_counts[key] < nlink:
            incomplete.add(key)
    protected_inodes = {
        (dev, ino) for dev, ino, name, _real, is_reg, _parent, _nlink in records
        if is_reg and name in names
    }
    write_deny = set()
    unlink_dirs = set()

    def protect_parents(parent):
        current = parent
        while current:
            unlink_dirs.add(current)
            if current == cwd or current == "/":
                break
            nxt = os.path.dirname(current)
            if nxt == current:
                break
            current = nxt

    for dev, ino, name, real, is_reg, parent, _nlink in records:
        if name in names:
            protect_parents(parent)
            if is_reg and real:
                write_deny.add(real)
        elif is_reg and real and ((dev, ino) in protected_inodes or (dev, ino) in incomplete):
            write_deny.add(real)
    allow_roots = []
    for root in covered:
        if '"' in root or "\\" in root or "\n" in root:
            sys.stderr.write("authority guard: directory path cannot be represented\n")
            raise SystemExit(126)
        allow_roots.append(root)
    body = "|".join(re.escape(name) for name in sorted(names))
    # A later deny wins over an earlier allow. Roots that were scanned may be
    # written; every other path stays denied. Basename and hard-link rules are
    # last so they still apply inside an allowed root.
    lines = [
        "(version 1)",
        "(allow default)",
        "(deny file-write*)",
    ]
    for root in allow_roots:
        lines.append('(allow file-write* (subpath "%s"))' % root)
    lines.append('(deny file-write* (regex "^/.*/(%s)$"))' % body)
    for path in sorted(unreadable):
        if '"' in path or "\\" in path or "\n" in path:
            sys.stderr.write("authority guard: unreadable path cannot be represented\n")
            raise SystemExit(126)
        lines.append('(deny file-write* (subpath "%s"))' % path)
    for real in sorted(write_deny):
        if '"' in real or "\\" in real or "\n" in real:
            sys.stderr.write("authority guard: hardlink path cannot be represented\n")
            raise SystemExit(126)
        lines.append('(deny file-write* (regex "^%s$"))' % re.escape(real))
    for directory in sorted(unlink_dirs):
        if '"' in directory or "\\" in directory or "\n" in directory:
            sys.stderr.write("authority guard: directory path cannot be represented\n")
            raise SystemExit(126)
        lines.append('(deny file-write-unlink (regex "^%s$"))' % re.escape(directory))
    profile = "\n".join(lines) + "\n"
    if "command" in spec:
        child = [sandbox, "-p", profile, "/bin/bash", "--noprofile", "--norc", "-c", spec["command"]]
    else:
        tail = list(sys.argv[2:])
        if not tail:
            sys.stderr.write("authority guard: nothing to run\n")
            raise SystemExit(126)
        child = [sandbox, "-p", profile, "--"] + tail
    try:
        os.execv(sandbox, child)
    except OSError as exc:
        sys.stderr.write("authority guard: sandbox is unavailable: %s\n" % exc)
        raise SystemExit(126)


def _darwin_launcher_code() -> str:
    return inspect.getsource(exec_darwin_boundary) + "\nexec_darwin_boundary()\n"


def _darwin_payload(names, binary, command=None) -> str | None:
    safe = _usable_names(names)
    if safe is None:
        return None
    payload = {"names": safe, "sandbox": binary}
    if command is not None:
        payload["command"] = command
    return json.dumps(payload, ensure_ascii=True)


def darwin_shell_command(command: str, names) -> str | None:
    binary = shutil.which("sandbox-exec")
    payload = _darwin_payload(names, binary, command) if binary else None
    if payload is None:
        return None
    return (
        shlex.quote(sys.executable)
        + " -I -c "
        + shlex.quote(_darwin_launcher_code())
        + " "
        + shlex.quote(payload)
    )


def darwin_spawn_prefix(names) -> list[str] | None:
    binary = shutil.which("sandbox-exec")
    payload = _darwin_payload(names, binary) if binary else None
    if payload is None:
        return None
    return [sys.executable, "-I", "-c", _darwin_launcher_code(), payload]


_WRITER_BOOT = (
    "import os, runpy, sys\n"
    "script = os.path.realpath(sys.argv[1])\n"
    "sys.path.insert(0, os.path.dirname(script))\n"
    "sys.argv = [script] + sys.argv[2:]\n"
    "runpy.run_path(script, run_name='__main__')\n"
)


def immutable_exec_command(argv) -> str:
    """Exec a trusted isolated interpreter only after the writer refuses a write open.

    ``PATH`` and ``argv[0]`` are not used for the final exec. A shell script
    named python3, a writable interpreter, and a user-site startup file cannot
    inherit the exemption. The caller's original text is not reused.
    """
    code = (
        "import errno, json, os, stat, sys\n"
        "spec = json.loads(sys.argv[1])\n"
        "script = spec.get('script')\n"
        "argv = spec.get('argv')\n"
        "boot = spec.get('boot')\n"
        "roots = spec.get('roots') or []\n"
        "if (not isinstance(script, str) or not script.startswith('/') or not isinstance(argv, list)\n"
        "        or len(argv) < 2 or argv[1] != script or not all(isinstance(item, str) for item in argv)\n"
        "        or not isinstance(boot, str) or not boot or not isinstance(roots, list)):\n"
        "    sys.exit(126)\n"
        "base = os.path.basename(argv[0])\n"
        "if base not in ('python', 'python3'):\n"
        "    sys.stderr.write('authority writer interpreter was not trusted\\n')\n"
        "    sys.exit(126)\n"
        "trusted = []\n"
        "for root in roots:\n"
        "    if not isinstance(root, str) or not root.startswith('/'):\n"
        "        continue\n"
        "    cand = os.path.join(root, base)\n"
        "    if not os.path.isfile(cand):\n"
        "        continue\n"
        "    real = os.path.realpath(cand)\n"
        "    if real not in trusted:\n"
        "        trusted.append(real)\n"
        "if not trusted:\n"
        "    sys.stderr.write('authority writer interpreter was not trusted\\n')\n"
        "    sys.exit(126)\n"
        "requested = argv[0]\n"
        "if os.path.isabs(requested):\n"
        "    try:\n"
        "        requested_real = os.path.realpath(requested)\n"
        "    except OSError:\n"
        "        sys.exit(126)\n"
        "    if requested_real not in trusted:\n"
        "        sys.stderr.write('authority writer interpreter was not trusted\\n')\n"
        "        sys.exit(126)\n"
        "    chosen = requested_real\n"
        "else:\n"
        "    chosen = trusted[0]\n"
        "try:\n"
        "    info = os.stat(chosen)\n"
        "except OSError:\n"
        "    sys.exit(126)\n"
        "if not stat.S_ISREG(info.st_mode):\n"
        "    sys.exit(126)\n"
        "if os.access(chosen, os.W_OK):\n"
        "    sys.stderr.write('authority writer interpreter is writable; refusing exemption\\n')\n"
        "    sys.exit(126)\n"
        "try:\n"
        "    handle = open(chosen, 'rb')\n"
        "except OSError:\n"
        "    sys.exit(126)\n"
        "else:\n"
        "    try:\n"
        "        magic = handle.read(2)\n"
        "    finally:\n"
        "        handle.close()\n"
        "if magic == b'#!':\n"
        "    sys.stderr.write('authority writer interpreter was not trusted\\n')\n"
        "    sys.exit(126)\n"
        "try:\n"
        "    fd = os.open(script, os.O_WRONLY | os.O_APPEND)\n"
        "except OSError as exc:\n"
        "    if exc.errno not in (errno.EROFS, errno.EACCES, errno.EPERM):\n"
        "        sys.stderr.write('authority writer check failed: %s\\n' % exc)\n"
        "        sys.exit(126)\n"
        "else:\n"
        "    os.close(fd)\n"
        "    sys.stderr.write('authority writer is writable; refusing exemption\\n')\n"
        "    sys.exit(126)\n"
        "try:\n"
        "    os.execv(chosen, [chosen, '-I', '-c', boot, script] + argv[2:])\n"
        "except OSError as exc:\n"
        "    sys.stderr.write('authority writer check failed: %s\\n' % exc)\n"
        "    sys.exit(126)\n"
    )
    payload = json.dumps({
        "script": argv[1],
        "argv": list(argv),
        "boot": _WRITER_BOOT,
        "roots": list(_TRUSTED_PYTHON_DIRS),
    }, ensure_ascii=True)
    return _exec_trusted_python(code, [payload])


def wrap_shell(command: str, names, *, env_type: str) -> str | None:
    """Return a command that enforces *names*, or None when this environment cannot."""
    if not names:
        return command
    if env_type == "local":
        if sys.platform == "darwin":
            return darwin_shell_command(command, names)
        if sys.platform.startswith("linux"):
            return linux_shell_command(command, names)
        return None
    if env_type in _LINUX_ENVS:
        return linux_shell_command(command, names)
    return None


def posix_spawn_argv(child_argv, names) -> list[str] | None:
    """Argv that starts *child_argv* under the same boundary. None fails closed."""
    if not names:
        return list(child_argv)
    if sys.platform == "darwin":
        prefix = darwin_spawn_prefix(names)
        if prefix is None:
            return None
        return prefix + list(child_argv)
    if sys.platform.startswith("linux"):
        source = base64.b64encode(_module_source()).decode("ascii")
        payload = base64.b64encode(json.dumps({
            "argv": list(child_argv),
            "names": sorted(names),
        }).encode("utf-8")).decode("ascii")
        return [sys.executable, "-I", "-c", _stub(), source, payload]
    return None
