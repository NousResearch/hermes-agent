#!/usr/bin/env python3
"""Exec a command inside a Landlock ruleset that hides Hermes secret stores.

    python -I -S landlock_exec.py --mode require|auto [--no-access PATH]... [--read-only PATH]... -- ARGV...

Stdlib only and import-free of Hermes: it runs as a fresh ``-I -S`` interpreter between the
spawning process and the agent's command, so nothing the agent controls (PYTHONPATH, cwd modules,
site-packages) is loaded before the sandbox is in place. The ruleset is inherited by every
descendant and cannot be lifted, so an agent re-running this helper cannot widen it.

Landlock is allow-list only; deny semantics are built from it:
- ``READ_DIR`` is granted on ``/``: every directory stays listable (names are not secrets).
- Directories on the path from ``/`` to a protected item ("chain" directories) get NO hierarchical
  rights. Their entries are walked one by one: protected entries get nothing (no-access) or read
  rights (read-only); every other entry gets the full handled set (directories) or file rights.
  Chain directories therefore cannot gain, lose or rename entries, which also blocks replacing a
  protected file, or smuggling it out by rename or hard link (``REFER`` is never granted there).
- Rules bind to inodes, so symlinks pointing into the protected set are resolved and refused.
- Mount points on a device that holds a protected item are walked too, so a second bind mount of
  ``HERMES_HOME`` cannot alias the protected files.

``require`` fails closed (exit 126) when Landlock is unavailable or a protected file has a hard
link elsewhere; ``auto`` runs the command unprotected in that case.
"""

import ctypes
import errno
import os
import re
import stat
import struct
import sys

# Landlock syscalls share these numbers on every architecture Hermes supports (x86_64, aarch64).
_SYS_CREATE_RULESET, _SYS_ADD_RULE, _SYS_RESTRICT_SELF = 444, 445, 446
_CREATE_RULESET_VERSION = 1
_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38

EXECUTE, WRITE_FILE, READ_FILE, READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
REFER, TRUNCATE, IOCTL_DEV = 1 << 13, 1 << 14, 1 << 15
_ABI1_ALL = (1 << 13) - 1  # EXECUTE .. MAKE_SYM

_EXIT_REFUSED = 126


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


def _path_beneath_attr(access: int, fd: int):
    """``struct landlock_path_beneath_attr`` is packed: u64 allowed_access, s32 parent_fd (12 bytes)."""
    return ctypes.create_string_buffer(struct.pack("=Qi", access, fd), 12)


def _libc():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    return libc


def _syscall(libc, number, *args):
    result = libc.syscall(ctypes.c_long(number), *args)
    if result < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return result


def abi_version(libc=None) -> int:
    """Landlock ABI version, 0 when unsupported (non-Linux, kernel < 5.13, disabled, seccomp)."""
    if not sys.platform.startswith("linux"):
        return 0
    try:
        return _syscall(libc or _libc(), _SYS_CREATE_RULESET, None, ctypes.c_size_t(0),
                        ctypes.c_uint32(_CREATE_RULESET_VERSION))
    except OSError:
        return 0


def handled_access(abi: int) -> int:
    return _ABI1_ALL | (REFER if abi >= 2 else 0) | (TRUNCATE if abi >= 3 else 0) | (IOCTL_DEV if abi >= 5 else 0)


def file_access(abi: int) -> int:
    return EXECUTE | WRITE_FILE | READ_FILE | (TRUNCATE if abi >= 3 else 0) | (IOCTL_DEV if abi >= 5 else 0)


def _inode(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_dev, st.st_ino


def _ancestors(path):
    parent = os.path.dirname(os.path.realpath(path))
    while True:
        yield parent
        if parent == os.sep:
            return
        parent = os.path.dirname(parent)


def _mount_points():
    try:
        with open("/proc/self/mountinfo", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return []
    unescape = re.compile(r"\\([0-7]{3})")
    return [unescape.sub(lambda m: chr(int(m.group(1), 8)), line.split()[4]) for line in lines if len(line.split()) > 4]


def _holds_secret(path) -> bool:
    """A protected path holds data: a file, or a directory with any non-directory entry beneath it.
    Hermes scaffolds empty secret dirs (``pairing/``) when it initialises a home; those hold nothing.
    A subtree that cannot be inspected counts as a secret (fail closed)."""
    if not os.path.isdir(path):
        return _inode(path) is not None
    errors = []
    for root, dirs, files in os.walk(path, onerror=errors.append):
        if files or any(os.path.islink(os.path.join(root, d)) for d in dirs):
            return True
    return bool(errors)


def _protection_sets(no_access, read_only):
    """``(deny, ro, walk)`` inode sets. ``walk`` (directories to freeze) is built ONLY from the
    ancestors of protected paths that hold a secret: a home holding no secrets is never frozen, so the
    agent terminal keeps normal file access there, while a home with real secrets has its
    secret-bearing directories frozen so those files cannot be read, replaced, renamed or deleted."""
    deny = {k for k in map(_inode, no_access) if k}
    ro = {k for k in map(_inode, read_only) if k}
    existing = [p for p in (*no_access, *read_only) if _holds_secret(p)]
    walk = {k for p in existing for k in map(_inode, _ancestors(p)) if k}
    return deny, ro, walk


def plan_rules(no_access, read_only, abi):
    """``[(path, allowed_access)]`` implementing the deny semantics described in the module doc."""
    deny, ro, walk = _protection_sets(no_access, read_only)
    protected_devs = {k[0] for k in deny | ro} | {k[0] for k in walk}
    for mount in _mount_points():
        key = _inode(mount)
        if key and key[0] in protected_devs and key not in walk:
            walk.update(k for k in map(_inode, _ancestors(mount)) if k)
    full, files = handled_access(abi), file_access(abi)
    ro_dir, ro_file = READ_DIR | READ_FILE | EXECUTE, READ_FILE | EXECUTE
    rules, stack, visited = [(os.sep, READ_DIR)], [os.sep], set()
    while stack:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            try:
                st = os.stat(entry.path)  # follow symlinks: rules bind to the target inode
            except OSError:
                continue
            key, is_dir = (st.st_dev, st.st_ino), stat.S_ISDIR(st.st_mode)
            if key in deny:
                continue
            if key in walk:
                if is_dir and not entry.is_symlink() and key not in visited:
                    visited.add(key)
                    stack.append(entry.path)
                continue
            if key in ro:
                rules.append((entry.path, ro_dir if is_dir else ro_file))
            else:
                rules.append((entry.path, full if is_dir else files))
    return rules


def _hard_linked(no_access):
    for path in no_access:
        try:
            st = os.stat(path)
        except OSError:
            continue
        if not stat.S_ISDIR(st.st_mode) and st.st_nlink > 1:
            yield path, st.st_nlink


def restrict_self(rules, abi, libc=None) -> None:
    libc = libc or _libc()
    attr = _RulesetAttr(handled_access(abi))
    ruleset = _syscall(libc, _SYS_CREATE_RULESET, ctypes.byref(attr), ctypes.c_size_t(ctypes.sizeof(attr)),
                       ctypes.c_uint32(0))
    try:
        for path, access in rules:
            try:
                fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            except OSError:
                continue
            try:
                rule = _path_beneath_attr(access, fd)
                _syscall(libc, _SYS_ADD_RULE, ctypes.c_int(ruleset), ctypes.c_int(_RULE_PATH_BENEATH),
                         rule, ctypes.c_uint32(0))
            except OSError as exc:
                # The entry changed type between stat and open: leave it denied.
                if exc.errno != errno.EINVAL:
                    raise
            finally:
                os.close(fd)
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            err = ctypes.get_errno()
            raise OSError(err, "prctl(PR_SET_NO_NEW_PRIVS): " + os.strerror(err))
        _syscall(libc, _SYS_RESTRICT_SELF, ctypes.c_int(ruleset), ctypes.c_uint32(0))
    finally:
        os.close(ruleset)


def _parse(argv):
    if "--" not in argv:
        raise SystemExit("usage: landlock_exec.py --mode require|auto [--no-access P]... [--read-only P]... -- ARGV...")
    split = argv.index("--")
    opts, command = argv[:split], argv[split + 1:]
    mode, no_access, read_only = "require", [], []
    it = iter(opts)
    for flag in it:
        value = next(it, None)
        if value is None:
            raise SystemExit(f"landlock_exec: {flag} needs a value")
        if flag == "--mode":
            mode = value
        elif flag == "--no-access":
            no_access.append(value)
        elif flag == "--read-only":
            read_only.append(value)
        else:
            raise SystemExit(f"landlock_exec: unknown option {flag}")
    if not command:
        raise SystemExit("landlock_exec: no command")
    return mode, no_access, read_only, command


def _refuse(reason: str) -> None:
    sys.stderr.write(
        f"hermes: refusing to run this command: {reason}. Without it the agent's shell could read "
        "Hermes secrets. Set security.terminal_secret_isolation: auto (run unprotected) or off in "
        "config.yaml to override.\n")
    sys.stderr.flush()
    os._exit(_EXIT_REFUSED)


def main(argv) -> None:
    mode, no_access, read_only, command = _parse(argv)
    libc = _libc() if sys.platform.startswith("linux") else None
    abi = abi_version(libc)
    if abi < 1:
        if mode == "require":
            _refuse("Landlock is unavailable (needs Linux >= 5.13 with landlock enabled and allowed by seccomp)")
    else:
        linked = list(_hard_linked(no_access))
        if linked and mode == "require":
            path, count = linked[0]
            _refuse(f"{path} has {count} hard links; an alias outside the protected directory would bypass isolation")
        try:
            restrict_self(plan_rules(no_access, read_only, abi), abi, libc)
        except OSError as exc:
            if mode == "require":
                _refuse(f"Landlock setup failed ({exc})")
    try:
        os.execvp(command[0], command)
    except OSError as exc:
        sys.stderr.write(f"hermes: cannot execute {command[0]}: {exc.strerror}\n")
        os._exit(127)


if __name__ == "__main__":
    main(sys.argv[1:])
