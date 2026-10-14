"""Descriptor-relative operations for an explicitly selected local terminal workspace.

The root is an operator-controlled runtime directory. No component below it follows a
symlink. Shell execution retains the terminal's existing authority; this is not a shell sandbox.
"""
import base64
import os
import shutil
import stat
import uuid
from contextlib import contextmanager
from pathlib import PurePosixPath


class WorkspacePathError(ValueError):
    pass


class WorkspaceFiles:
    def __init__(self, root: str):
        self.root = os.path.realpath(root)
        if self.root == "/":
            raise WorkspacePathError("A filesystem root cannot be a workspace")

    def parts(self, path: str) -> tuple[str, ...]:
        if "\x00" in path:
            raise WorkspacePathError("NUL in path")
        value = PurePosixPath(path)
        if value.is_absolute():
            if not value.is_relative_to(self.root):
                raise WorkspacePathError("Path is outside the workspace")
            value = value.relative_to(self.root)
        if ".." in value.parts:
            raise WorkspacePathError("Parent traversal is not permitted")
        return value.parts

    @contextmanager
    def parent(self, path: str, *, create=False):
        parts = self.parts(path)
        if not parts:
            raise WorkspacePathError("The workspace root cannot be a file target")
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in parts[:-1]:
                if create:
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=fd)
                    except FileExistsError:
                        pass  # The following no-follow open validates the existing component.
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            yield fd, parts[-1]
        finally:
            os.close(fd)

    def resolve(self, path: str) -> str:
        parts = self.parts(path)
        # Resolve is lexical so callers can name a destination before creating it.
        # Every actual file operation independently enforces descriptor containment.
        return str(PurePosixPath(self.root).joinpath(*parts))

    def cwd(self, path: str) -> str:
        value = self.resolve(path)
        # Verify every directory component without following symlinks.
        with self.parent(str(PurePosixPath(value) / ".workspace-cwd-probe")):
            return value

    def read(self, path: str, maximum: int) -> dict:
        try:
            with self.parent(path) as (parent, name):
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
                with os.fdopen(fd, "rb") as stream:
                    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                        raise WorkspacePathError("Only regular files can be read")
                    data = stream.read(maximum + 1)
        except FileNotFoundError:
            return {"kind": "missing"}
        if len(data) > maximum:
            raise WorkspacePathError("File exceeds max_bytes")
        return {"kind": "file", "data_base64": base64.b64encode(data).decode("ascii"), "size_bytes": len(data)}

    def write(self, path: str, data: bytes) -> None:
        with self.parent(path, create=True) as (parent, name):
            # Replacing a symlink never follows it. Replacing a directory fails.
            temporary = f".hermes-write-{uuid.uuid4().hex}"
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
                os.fsync(parent)
            finally:
                # Only our unique temporary entry is cleaned up after a failed rename.
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass

    def remove(self, path: str, *, force: bool, recursive: bool) -> None:
        try:
            with self.parent(path) as (parent, name):
                info = os.stat(name, dir_fd=parent, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    if recursive:
                        if not shutil.rmtree.avoids_symlink_attacks:
                            raise WorkspacePathError("Safe recursive removal is unavailable")
                        shutil.rmtree(name, dir_fd=parent)
                    else:
                        os.rmdir(name, dir_fd=parent)
                else:
                    os.unlink(name, dir_fd=parent)
        except FileNotFoundError:
            if not force:
                raise
