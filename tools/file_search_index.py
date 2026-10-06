"""Indexed filename search: answer a name query from a locate database instead of
walking the tree (#127861).

``rg --files`` re-walks the filesystem on every call, so its cost scales with the
tree (1.3 s for 3.3 M files on the reporting host, 19 s once ``--sortr=modified``
adds a stat per file). A locate index answers the same question out of a prebuilt
database in a constant ~0.4 s, whatever the tree size — so it is worth a query
where the walk cannot early-exit and never worth one where the walk is cheap.

Guards ride with that speed:

* Scope: a git work-tree is bounded and already ignore-filtered, so the walk there
  is milliseconds and stays the engine (:func:`inside_worktree`). Unfiltered shared
  trees — the ones that walk millions of entries — are where the index answers.
* Freshness: an index is a snapshot, so a database older than
  :func:`max_age_seconds` (nothing has refreshed it — no updatedb timer, say) is
  refused and the caller walks instead.
* Ignore semantics: locate has no notion of gitignore. The filter reproduces what
  the walk hides unconditionally — hidden names, symlinks — but not a nested
  work-tree's .gitignore, which it cannot read; an ignored directory that is not
  hidden may therefore appear (:func:`keep_indexed_path`), and the answer says so.

Files created since the last updatedb are still unseen, so the index is treated as
a positive cache only: the caller walks whenever the database cannot serve the
request (None) and whenever the query comes back empty.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
import time
from typing import Callable, List, Optional, Sequence

# First engine on PATH wins, and each has its own standard database.
ENGINES = ("plocate", "locate", "mlocate")
_DATABASES = {
    "plocate": "/var/lib/plocate/plocate.db",
    "locate": "/var/lib/mlocate/mlocate.db",
    "mlocate": "/var/lib/mlocate/mlocate.db",
}
# locate(1)'s "only report files that still exist"; mlocate has no equivalent.
_EXISTS_FLAG = {"plocate": "-e", "locate": "-e"}
# A week: the threshold is here to refuse a database nothing refreshes, not to
# race the update cadence the operator chose.
DEFAULT_MAX_AGE_SECONDS = 7 * 24 * 3600
# rg's -g reaches locate as a substring match with * ? [] wildcards. These rg
# brings and locate cannot, so requests using them keep to the walk.
_RG_ONLY_TOKENS = ("{", "}", "**", "!", "\\")
# How far up from a search root to look for a work-tree root before giving up.
_WORKTREE_CLIMB = 64
# Candidate slack over the caller's bound (see fetch_window).
_WINDOW_FACTOR = 10
_WINDOW_MINIMUM = 200
# Attached to every indexed answer: what a caller cannot tell from the paths alone.
SNAPSHOT_WARNING = (
    "Results came from the locate index: files created after its last updatedb run "
    "are not listed, and directories a nested .gitignore hides may appear.")


def max_age_seconds() -> int:
    """Freshness threshold in seconds (``HERMES_FILE_SEARCH_INDEX_MAX_AGE``)."""
    raw = os.environ.get("HERMES_FILE_SEARCH_INDEX_MAX_AGE", "").strip()
    return int(raw) if raw.isdigit() else DEFAULT_MAX_AGE_SECONDS


def disabled() -> bool:
    """``HERMES_FILE_SEARCH_ENGINE=rg`` pins filename search to the walk."""
    return os.environ.get("HERMES_FILE_SEARCH_ENGINE", "").strip().lower() == "rg"


def which_exists(name: str) -> bool:
    """Whether *name* is on this host's PATH. Bare by design: the caller runs this
    only for a LOCAL backend, where the controller's PATH *is* the execution host's,
    and a shell round trip would cost more than the query it gates."""
    return shutil.which(name) is not None


def locate_pattern(pattern: str) -> Optional[str]:
    """The rg ``-g`` glob as a locate pattern, None when locate cannot express it."""
    glob = pattern if "/" in pattern or pattern.startswith("*") else f"*{pattern}"
    return None if any(token in glob for token in _RG_ONLY_TOKENS) else glob


def db_path(engine: str) -> str:
    """The database *engine* reads (``HERMES_FILE_SEARCH_LOCATE_DB`` overrides it)."""
    return os.environ.get("HERMES_FILE_SEARCH_LOCATE_DB", "").strip() or _DATABASES[engine]


def locate_patterns(pattern: str, roots: Sequence[str]) -> Optional[List[str]]:
    """The rg ``-g`` glob as one locate pattern per root, None when locate cannot
    express it.

    The root is part of the pattern, not a post-filter: locate's own ``-l`` bound
    would otherwise be spent on matches outside the roots, and the caller would
    receive a silently short list. One pattern per root: locate ORs them, and every
    root keeps its own leading path literal.
    """
    glob = locate_pattern(pattern)
    if glob is None or not roots:
        return None
    scoped = dict.fromkeys(f"{root.rstrip('/')}/{glob}" for root in roots if root.startswith("/"))
    return list(scoped) or None


def inside_worktree(root: str) -> bool:
    """Whether *root* or an ancestor is a git work-tree root. There the walk is
    bounded and ignore-filtered — cheap, exact and fresh — so rg keeps the query."""
    current = os.path.abspath(root)
    for _ in range(_WORKTREE_CLIMB):
        if os.path.exists(os.path.join(current, ".git")):
            return True
        parent = os.path.dirname(current)
        if parent == current:
            return False
        current = parent
    return False


def indexed_argv(pattern: str, roots: Sequence[str], fetch_limit: int,
                 which: Optional[Callable[[str], bool]] = None,
                 now: Optional[float] = None) -> Optional[List[str]]:
    """argv for an indexed name query under *roots*, or None when the index cannot
    serve it.

    ``which(name)`` answers "is this command on the execution host's PATH" —
    :func:`which_exists` by default for a local backend, or the caller's own probe
    when the host is not this process.

    None means "walk this one": no locate binary, no database, a database older
    than :func:`max_age_seconds`, a glob or root outside locate's syntax, or the
    walk pinned by ``HERMES_FILE_SEARCH_ENGINE``.
    """
    if disabled():
        return None
    patterns = locate_patterns(pattern, roots)
    if patterns is None:
        return None
    engine = next((name for name in ENGINES if (which or which_exists)(name)), None)
    if engine is None:
        return None
    database = db_path(engine)
    try:
        age = (time.time() if now is None else now) - os.stat(database).st_mtime
    except OSError:
        return None
    if age > max_age_seconds():
        return None
    argv = [engine, "-d", database]
    if flag := _EXISTS_FLAG.get(engine):
        argv.append(flag)
    # ``--`` keeps a dash-prefixed pattern a pattern, ``-l`` stops the query at the
    # caller's bound (and, unlike piping into head, does not SIGPIPE plocate).
    return [*argv, "-l", str(fetch_limit), "--", *patterns]


def within_root(path: str, root: str) -> bool:
    """Path-boundary containment: ``/s/x`` is inside ``/s``, ``/s/xyz`` is not."""
    trimmed = root.rstrip("/")
    return bool(trimmed) and (path == trimmed or path.startswith(trimmed + "/"))


def fetch_window(fetch_limit: int) -> int:
    """How many indexed candidates to ask for when only *fetch_limit* are wanted.

    locate matches the whole path, so the query's ``-l`` bound is spent on
    candidates the filter then drops — hidden directories, ignored directories, and
    (because the query is root-prefixed) paths that only matched the root's own name.
    A window is a bounded number of extra path lines, not extra scan time, so paying
    several times the caller's bound keeps the answer full.
    """
    return max(fetch_limit * _WINDOW_FACTOR, _WINDOW_MINIMUM)


def relative_match(path: str, root: str, pattern: str) -> bool:
    """Whether *path* matches the caller's rg glob the way the walk matches it: rg
    applies ``-g`` to the path RELATIVE to the search root, so a pattern matching the
    root's own name (``*tortoise*`` under ``/s/tortoise``) must not match every file
    inside it. Patterns locate cannot express never reach here."""
    glob = locate_pattern(pattern)
    return bool(glob) and fnmatch.fnmatchcase(path[len(root.rstrip("/")):].lstrip("/"), glob)


def keep_indexed_path(path: str, roots: Sequence[str], pattern: str = "*") -> bool:
    """Whether an indexed path survives what the walk filters for free: a real file
    (not a symlink, and not inside one — the walk follows neither), inside one of
    *roots*, matching the query relative to its root, and not hidden.

    Deliberately no ignored-directory list: the walk hides ``node_modules`` and
    friends only through a nested work-tree's .gitignore, which locate cannot read,
    and guessing there would hide files the walk does report. Extra paths are noise
    the caller can see; missing ones are silent. Dot-named components of the ROOT
    itself (``~/.hermes/skills``) are the caller's own choice and stay."""
    if not path.startswith("/") or not any(within_root(path, root) for root in roots):
        return False
    root = max((root for root in roots if within_root(path, root)), key=len)
    relative = path[len(root.rstrip("/")):].strip("/")
    if any(part.startswith(".") for part in relative.split("/")):
        return False
    if not relative_match(path, root, pattern):
        return False
    if os.path.islink(path) or os.path.realpath(path) != path:
        return False
    return os.path.isfile(path)
