"""Host-independent path syntax and containment helpers.

Workstation evidence and policy inputs can describe paths belonging to a
different operating system than the process evaluating them.  ``Path`` and
``os.path`` intentionally follow the host OS, so using them to classify an
evidence path can turn an absolute Windows path into a relative POSIX path.
This module classifies the original spelling first and only performs lexical
containment when both paths use the same, known absolute syntax.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import ntpath
import posixpath
import re
from pathlib import PureWindowsPath


class PathSyntax(str, Enum):
    WINDOWS_ABSOLUTE = "windows_absolute"
    POSIX_ABSOLUTE = "posix_absolute"
    RELATIVE = "relative"


@dataclass(frozen=True, slots=True)
class ClassifiedPath:
    """A path classified without consulting the host filesystem semantics."""

    raw: str
    syntax: PathSyntax
    normalized: str

    @property
    def is_absolute(self) -> bool:
        return self.syntax is not PathSyntax.RELATIVE


_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_WINDOWS_DRIVE_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def classify_path(value: str) -> ClassifiedPath:
    """Classify a path string as Windows absolute, POSIX absolute, or relative.

    The test for ``//`` deliberately precedes the POSIX test: in the
    Workstation contract it is the portable spelling of a Windows UNC path.
    Drive-relative paths such as ``C:notes.txt`` remain relative and are not
    treated as safe absolute paths.
    """

    raw = str(value).strip()
    if not raw or _CONTROL_CHARS.search(raw):
        return ClassifiedPath(raw, PathSyntax.RELATIVE, raw)

    if _WINDOWS_DRIVE_ABSOLUTE.match(raw) or raw.startswith((r"\\", "//")):
        windows = ntpath.normpath(raw.replace("/", "\\"))
        # Keep case in the diagnostic value while callers compare normalized
        # Windows components case-insensitively.
        return ClassifiedPath(raw, PathSyntax.WINDOWS_ABSOLUTE, windows)

    if raw.startswith("/"):
        return ClassifiedPath(raw, PathSyntax.POSIX_ABSOLUTE, posixpath.normpath(raw))

    # This catches less common absolute spellings accepted by the Windows
    # grammar without making a host-dependent ``Path`` call.
    windows_path = PureWindowsPath(raw)
    if windows_path.is_absolute():
        return ClassifiedPath(raw, PathSyntax.WINDOWS_ABSOLUTE, ntpath.normpath(raw))

    return ClassifiedPath(raw, PathSyntax.RELATIVE, raw)


def path_is_within(parent: str, child: str) -> bool | None:
    """Return lexical containment, or ``None`` for incompatible path syntax.

    ``True`` includes equality.  ``None`` is intentionally distinct from
    ``False`` so callers can fail closed for policy checks while release
    qualification can avoid comparing a Windows clean-install path to a POSIX
    candidate checkout on the runner.
    """

    parent_path = classify_path(parent)
    child_path = classify_path(child)
    if not parent_path.is_absolute or not child_path.is_absolute:
        return None
    if parent_path.syntax is not child_path.syntax:
        return None

    parent_drive, parent_parts = _components(parent_path)
    child_drive, child_parts = _components(child_path)
    if parent_drive != child_drive:
        return None
    return child_parts[: len(parent_parts)] == parent_parts


def _components(path: ClassifiedPath) -> tuple[str, tuple[str, ...]]:
    if path.syntax is PathSyntax.WINDOWS_ABSOLUTE:
        normalized = ntpath.normpath(path.raw.replace("/", "\\"))
        drive, tail = ntpath.splitdrive(normalized)
        parts = tuple(part.casefold() for part in tail.split("\\") if part not in {"", "."})
        return drive.casefold(), parts

    normalized = posixpath.normpath(path.raw)
    parts = tuple(part for part in normalized.split("/") if part not in {"", "."})
    return "", parts


def is_sensitive_path(value: str) -> bool:
    """Recognize protected system and credential paths independent of host OS."""

    path = classify_path(value)
    normalized = path.normalized.casefold()
    if path.syntax is PathSyntax.WINDOWS_ABSOLUTE:
        # Covers drive paths and UNC administrative shares such as
        # ``\\server\c$\Windows\System32``.
        if re.search(r"(?:^|\\)(?:[a-z]\$\\)?windows(?:\\|$)", normalized):
            return True
        if re.search(r"(?:^|\\)program files(?:\\|$)", normalized):
            return True
    elif path.syntax is PathSyntax.POSIX_ABSOLUTE:
        if re.search(r"^/(?:etc|usr/bin|bin|sbin|boot)(?:/|$)", normalized):
            return True

    return bool(re.search(r"(?:^|[\\/])\.(?:ssh|gnupg|aws)(?:[\\/]|$)", normalized))
