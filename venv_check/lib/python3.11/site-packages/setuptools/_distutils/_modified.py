"""Timestamp comparison of files and groups of files."""

from __future__ import annotations

import functools
import os.path
from collections.abc import Callable, Iterable
from typing import Literal, TypeVar

from jaraco.functools import splat

from .compat.py39 import zip_strict
from .errors import DistutilsFileError

_SourcesT = TypeVar(
    "_SourcesT", bound="str | bytes | os.PathLike[str] | os.PathLike[bytes]"
)
_TargetsT = TypeVar(
    "_TargetsT", bound="str | bytes | os.PathLike[str] | os.PathLike[bytes]"
)


def _newer(source, target):
    return not os.path.exists(target) or (
        os.path.getmtime(source) > os.path.getmtime(target)
    )


def newer(
    source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
) -> bool:
    """
    Is source modified more recently than target.

    Returns True if 'source' is modified more recently than
    'target' or if 'target' does not exist.

    Raises DistutilsFileError if 'source' does not exist.
    """
    if not os.path.exists(source):
        raise DistutilsFileError(f"file {os.path.abspath(source)!r} does not exist")

    return _newer(source, target)


def newer_pairwise(
    sources: Iterable[_SourcesT],
    targets: Iterable[_TargetsT],
    newer: Callable[[_SourcesT, _TargetsT], bool] = newer,
) -> tuple[list[_SourcesT], list[_TargetsT]]:
    """
    Filter filenames where sources are newer than targets.

    Walk two filename iterables in parallel, testing if each source is newer
    than its corresponding target.  Returns a pair of lists (sources,
    targets) where source is newer than target, according to the semantics
    of 'newer()'.
    """
    newer_pairs = filter(splat(newer), zip_strict(sources, targets))
    return tuple(map(list, zip(*newer_pairs))) or ([], [])


def newer_group(
    sources: Iterable[str | bytes | os.PathLike[str] | os.PathLike[bytes]],
    target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    missing: Literal["error", "ignore", "newer"] = "error",
) -> bool:
    """
    Is target out-of-date with respect to any file in sources.

    Return True if 'target' is out-of-date with respect to any file
    listed in 'sources'. In other words, if 'target' exists and is newer
    than every file in 'sources', return False; otherwise return True.
    ``missing`` controls how to handle a missing source file:

    - error (default): allow the ``stat()`` call to fail.
    - ignore: silently disregard any missing source files.
    - newer: treat missing source files as "target out of date". This
      mode is handy in "dry-run" mode: it will pretend to carry out
      commands that wouldn't work because inputs are missing, but
      that doesn't matter because dry-run won't run the commands.
    """

    def missing_as_newer(source):
        return missing == 'newer' and not os.path.exists(source)

    ignored = os.path.exists if missing == 'ignore' else None
    return not os.path.exists(target) or any(
        missing_as_newer(source) or _newer(source, target)
        for source in filter(ignored, sources)
    )


newer_pairwise_group = functools.partial(newer_pairwise, newer=newer_group)
