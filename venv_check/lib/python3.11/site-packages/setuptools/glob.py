"""
Filename globbing utility. Mostly a copy of `glob` from Python 3.5.

Changes include:
 * `yield from` and PEP3102 `*` removed.
 * Hidden files are not ignored.
"""

from __future__ import annotations

import fnmatch
import os
import re
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, AnyStr, overload

if TYPE_CHECKING:
    from _typeshed import BytesPath, StrOrBytesPath, StrPath

__all__ = ["glob", "iglob", "escape"]


def glob(pathname: AnyStr, recursive: bool = False) -> list[AnyStr]:
    """Return a list of paths matching a pathname pattern.

    The pattern may contain simple shell-style wildcards a la
    fnmatch. However, unlike fnmatch, filenames starting with a
    dot are special cases that are not matched by '*' and '?'
    patterns.

    If recursive is true, the pattern '**' will match any files and
    zero or more directories and subdirectories.
    """
    return list(iglob(pathname, recursive=recursive))


def iglob(pathname: AnyStr, recursive: bool = False) -> Iterator[AnyStr]:
    """Return an iterator which yields the paths matching a pathname pattern.

    The pattern may contain simple shell-style wildcards a la
    fnmatch. However, unlike fnmatch, filenames starting with a
    dot are special cases that are not matched by '*' and '?'
    patterns.

    If recursive is true, the pattern '**' will match any files and
    zero or more directories and subdirectories.
    """
    it = _iglob(pathname, recursive)
    if recursive and _isrecursive(pathname):
        s = next(it)  # skip empty string
        assert not s
    return it


def _iglob(pathname: AnyStr, recursive: bool) -> Iterator[AnyStr]:
    dirname, basename = os.path.split(pathname)
    glob_in_dir = glob2 if recursive and _isrecursive(basename) else glob1

    if not has_magic(pathname):
        if basename:
            if os.path.lexists(pathname):
                yield pathname
        else:
            # Patterns ending with a slash should match only directories
            if os.path.isdir(dirname):
                yield pathname
        return

    if not dirname:
        yield from glob_in_dir(dirname, basename)
        return
    # `os.path.split()` returns the argument itself as a dirname if it is a
    # drive or UNC path.  Prevent an infinite recursion if a drive or UNC path
    # contains magic characters (i.e. r'\\?\C:').
    if dirname != pathname and has_magic(dirname):
        dirs: Iterable[AnyStr] = _iglob(dirname, recursive)
    else:
        dirs = [dirname]
    if not has_magic(basename):
        glob_in_dir = glob0
    for dirname in dirs:
        for name in glob_in_dir(dirname, basename):
            yield os.path.join(dirname, name)


# These 2 helper functions non-recursively glob inside a literal directory.
# They return a list of basenames. `glob1` accepts a pattern while `glob0`
# takes a literal basename (so it only has to check for its existence).


@overload
def glob1(dirname: StrPath, pattern: str) -> list[str]: ...
@overload
def glob1(dirname: BytesPath, pattern: bytes) -> list[bytes]: ...
def glob1(dirname: StrOrBytesPath, pattern: str | bytes) -> list[str] | list[bytes]:
    if not dirname:
        if isinstance(pattern, bytes):
            dirname = os.curdir.encode('ASCII')
        else:
            dirname = os.curdir
    try:
        names = os.listdir(dirname)
    except OSError:
        return []
    # mypy false-positives: str or bytes type possibility is always kept in sync
    return fnmatch.filter(names, pattern)  # type: ignore[type-var, return-value]


def glob0(dirname, basename):
    if not basename:
        # `os.path.split()` returns an empty basename for paths ending with a
        # directory separator.  'q*x/' should match only directories.
        if os.path.isdir(dirname):
            return [basename]
    else:
        if os.path.lexists(os.path.join(dirname, basename)):
            return [basename]
    return []


# This helper function recursively yields relative pathnames inside a literal
# directory.


@overload
def glob2(dirname: StrPath, pattern: str) -> Iterator[str]: ...
@overload
def glob2(dirname: BytesPath, pattern: bytes) -> Iterator[bytes]: ...
def glob2(dirname: StrOrBytesPath, pattern: str | bytes) -> Iterator[str | bytes]:
    assert _isrecursive(pattern)
    yield pattern[:0]
    yield from _rlistdir(dirname)


# Recursively yields relative pathnames inside a literal directory.
@overload
def _rlistdir(dirname: StrPath) -> Iterator[str]: ...
@overload
def _rlistdir(dirname: BytesPath) -> Iterator[bytes]: ...
def _rlistdir(dirname: StrOrBytesPath) -> Iterator[str | bytes]:
    if not dirname:
        if isinstance(dirname, bytes):
            dirname = os.curdir.encode('ASCII')
        else:
            dirname = os.curdir
    try:
        names = os.listdir(dirname)
    except OSError:
        return
    for x in names:
        yield x
        # mypy false-positives: str or bytes type possibility is always kept in sync
        path = os.path.join(dirname, x) if dirname else x  # type: ignore[arg-type]
        for y in _rlistdir(path):
            yield os.path.join(x, y)  # type: ignore[arg-type]


magic_check = re.compile('([*?[])')
magic_check_bytes = re.compile(b'([*?[])')


def has_magic(s: str | bytes) -> bool:
    if isinstance(s, bytes):
        return magic_check_bytes.search(s) is not None
    else:
        return magic_check.search(s) is not None


def _isrecursive(pattern: str | bytes) -> bool:
    if isinstance(pattern, bytes):
        return pattern == b'**'
    else:
        return pattern == '**'


def escape(pathname):
    """Escape all special characters."""
    # Escaping is done by wrapping any of "*?[" between square brackets.
    # Metacharacters do not work in the drive part and shouldn't be escaped.
    drive, pathname = os.path.splitdrive(pathname)
    if isinstance(pathname, bytes):
        pathname = magic_check_bytes.sub(rb'[\1]', pathname)
    else:
        pathname = magic_check.sub(r'[\1]', pathname)
    return drive + pathname
