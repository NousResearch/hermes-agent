"""Standalone stdlib reader sent to the terminal backend; Hermes need not be installed there."""
import base64
import json
import os
import stat
import sys
from pathlib import PurePosixPath


def _opened_path(fd):
    # These are facts about the execution target, which may differ from the gateway host.
    if sys.platform == "linux":
        return os.readlink(f"/proc/self/fd/{fd}")
    if sys.platform == "darwin":
        import fcntl
        return os.fsdecode(fcntl.fcntl(fd, 50, b"\0" * 1024).split(b"\0", 1)[0])  # F_GETPATH
    raise ValueError("This backend cannot provide verified remote file reads.")


def _check_path(path, policy):
    target = PurePosixPath(path)
    if not target.is_absolute() or path.endswith(" (deleted)"):
        raise ValueError("The remote file changed while it was being shared. Try again.")
    if any(target == PurePosixPath(root) or PurePosixPath(root) in target.parents for root in policy["prefixes"]):
        raise ValueError("That remote file is private and cannot be shared.")
    parts = tuple(part.lower() for part in target.parts)
    trees = policy["private_trees"]
    if ".hermes" in parts:
        inside = parts[parts.index(".hermes") + 1:]
        if any(tuple(tree) == inside[i:i + len(tree)] for tree in trees for i in range(len(inside))):
            raise ValueError("That remote Hermes file is private and cannot be shared.")
    if parts[-1] in policy["private_names"] or policy["private_storage"] in parts or any(
            tuple(tree) == parts[i:i + len(tree)] for tree in policy["anywhere"] for i in range(len(parts))):
        raise ValueError("That remote file is private and cannot be shared.")


def read_verified(path, limit, policy):
    """Validate the opened file's path and metadata, then read only that descriptor."""
    _check_path(path, policy)
    resolved = os.path.realpath(path)
    _check_path(resolved, policy)
    flags = os.O_RDONLY | os.O_NOFOLLOW
    parts = PurePosixPath(resolved).parts
    directory = os.open("/", flags | os.O_DIRECTORY)
    try:
        for part in parts[1:-1]:
            child = os.open(part, flags | os.O_DIRECTORY, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(parts[-1], flags | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    try:
        opened = _opened_path(fd)
        _check_path(opened, policy)
        before = os.fstat(fd)
        if before.st_nlink != 1 or not stat.S_ISREG(before.st_mode):
            raise ValueError("Share a regular remote file without hard links.")
        if not 0 < before.st_size <= limit:
            raise ValueError(f"A shared file must be a non-empty file of at most {limit // 1_000_000} MB.")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            data = handle.read(limit + 1)
        after = os.fstat(fd)
        final_path = _opened_path(fd)
        _check_path(final_path, policy)
        if opened != final_path or len(data) != before.st_size or any(
                getattr(before, field) != getattr(after, field) for field in (
                    "st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")):
            raise ValueError("The remote file changed while it was being shared. Try again.")
        return data
    finally:
        os.close(fd)


if __name__ == "__main__":
    request = json.loads(sys.argv[1])
    try:
        result = {"data": base64.b64encode(read_verified(
            request["path"], request["limit"], request["policy"])).decode("ascii")}
    except (OSError, ValueError, AttributeError) as exc:
        result = {"error": f"Could not complete a verified remote file read: {exc}"}
    print(request["marker"] + json.dumps(result) + request["marker"])
