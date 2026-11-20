"""Refuse terminal commands that walk a whole tree rooted at a broad root.

Agent-issued ``find /``, ``du -sh ~`` or ``rg needle $HOME`` read the entire disk
or home directory per call and starve every other process on a small host. A
walk is refused when one of its absolute path arguments IS, or CONTAINS, a broad
root: the filesystem root, the user's home, or an operator-listed root
(``HERMES_SEARCH_BROAD_ROOTS``, ``os.pathsep``-separated). Shallow listings
(``find -maxdepth <= 3``, ``rg --max-depth <= 3``) stay allowed, and input that
does not tokenize is never blocked. Pure path arithmetic: nothing touches disk.
"""

import os
import shlex
from typing import Optional

ENV_VAR = "HERMES_SEARCH_BROAD_ROOTS"
_MAX_SHALLOW_DEPTH = 3
_WRAPPERS = {"sudo", "nice", "ionice", "timeout", "env", "command", "exec", "time", "nohup"}
_SEPARATORS = set(";&|()")


def _broad_roots() -> list[str]:
    listed = [r for r in os.environ.get(ENV_VAR, "").split(os.pathsep) if r.strip()]
    roots = ["/", os.path.expanduser("~"), *listed]
    return [os.path.normpath(os.path.expanduser(r.strip())) for r in roots]


def _broad_root_hit(path: str) -> Optional[str]:
    target = os.path.normpath(path)
    for root in _broad_roots():
        if os.path.commonpath([target, root]) == target:
            what = "it is a broad root" if target == root else f"it contains the broad root {root}"
            return (f"Refusing to walk {target}: {what}, and a recursive walk there reads GiBs and "
                    "starves every other process on the disk. Narrow the walk to the task or project "
                    f"directory (or use find -maxdepth {_MAX_SHALLOW_DEPTH} to list).")
    return None


def _depth(tokens: list[str], flags: tuple[str, ...]) -> Optional[int]:
    for i, tok in enumerate(tokens):
        name, _, val = tok.partition("=")
        if name in flags:
            val = val or (tokens[i + 1] if i + 1 < len(tokens) else "")
            return int(val) if val.isdigit() else None
    return None


def _walk_paths(tokens: list[str]) -> list[str]:
    """Absolute path args of a whole-tree walk (find/du/grep -r/rg); [] when bounded."""
    while tokens and (tokens[0] in _WRAPPERS or "=" in tokens[0] or tokens[0].startswith("-")
                      or tokens[0].isdigit() or tokens[0][:-1].isdigit()):
        tokens = tokens[1:]
    if not tokens:
        return []
    prog, args = os.path.basename(tokens[0]), tokens[1:]
    if prog == "find":
        depth = _depth(args, ("-maxdepth",))
    elif prog == "rg":
        depth = _depth(args, ("--max-depth", "-d", "--maxdepth"))
    elif prog in ("grep", "egrep", "fgrep"):
        recursive = any(a in ("--recursive", "--dereference-recursive")
                        or (a.startswith("-") and not a.startswith("--") and ("r" in a or "R" in a))
                        for a in args)
        depth = None if recursive else 0
    elif prog == "du":
        depth = None
    else:
        return []
    if depth is not None and depth <= _MAX_SHALLOW_DEPTH:
        return []
    paths = (os.path.expanduser(os.path.expandvars(a)) for a in args)
    return [p for p in paths if os.path.isabs(p)]


def broad_walk_error(command: str) -> Optional[str]:
    """Refusal message when a shell command walks a tree that is or contains a broad root."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None
    simple: list[str] = []
    for tok in [*tokens, ";"]:
        if tok and set(tok) <= _SEPARATORS:
            for path in _walk_paths(simple):
                err = _broad_root_hit(path)
                if err:
                    return err
            simple = []
        else:
            simple.append(tok)
    return None
