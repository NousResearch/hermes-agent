"""Re-read hint for terminal file views: ``cat``/``sed -n``/``head``/``tail``/``less`` of one path whose whole content
this task already saw through read_file, unchanged since, gets one line saying so. ``sed -n A,Bp`` windows through
the terminal bypass read_file's dedup, and repeat reads are most of a subagent's reading. Hint only: never blocks.
"""

from __future__ import annotations

import os
import re
import shlex
from typing import Callable, Optional

REREAD_HINT = "(already read this session with read_file; unchanged since)"
# Anything that makes the command more than a plain view of one file (pipes, redirection, chaining, substitution).
_SHELL_META = re.compile(r"[|;&<>`$(){}\n*?\[\]]")
_SED_PRINT = re.compile(r"^\d+(,\d+)?p$|^\$p$")
_COUNT = re.compile(r"^-\d+$|^-[nc]\d+$")


def _single_path(args: list[str]) -> Optional[str]:
    return args[0] if len(args) == 1 and not args[0].startswith("-") else None


def _head_tail_path(args: list[str]) -> Optional[str]:
    if len(args) >= 3 and args[0] in ("-n", "-c") and args[1].lstrip("+").isdigit():
        args = args[2:]
    elif args and _COUNT.match(args[0]):
        args = args[1:]
    return _single_path(args)


def _sed_path(args: list[str]) -> Optional[str]:
    if len(args) == 3 and args[0] == "-n" and _SED_PRINT.match(args[1]):
        return args[2]
    return None


# Viewer -> extracts the one viewed path from its arguments (None when the arguments are anything else).
_VIEWERS: dict[str, Callable[[list[str]], Optional[str]]] = {
    "cat": _single_path,
    "less": _single_path,
    "head": _head_tail_path,
    "tail": _head_tail_path,
    "sed": _sed_path,
}


def viewed_path(command: str) -> Optional[str]:
    """The single file a plain view command shows, else None."""
    if _SHELL_META.search(command):
        return None
    try:
        argv = shlex.split(command)
    except ValueError:
        return None
    parse = _VIEWERS.get(argv[0]) if argv else None
    return parse(argv[1:]) if parse else None


def reread_hint(command: str, cwd: Optional[str], task_id: str) -> Optional[str]:
    """``REREAD_HINT`` when *command* views a file whose whole current content this task already holds from
    read_file (its full-content baseline still matches the file's metadata), else None."""
    path = viewed_path(command)
    if path is None or not (cwd or os.path.isabs(path)):
        return None
    from tools.file_tools_read_tracking import _file_metadata, _read_tracker, _read_tracker_lock
    resolved = os.path.realpath(os.path.join(cwd or "", os.path.expanduser(path)))
    with _read_tracker_lock:
        baseline = ((_read_tracker.get(task_id) or {}).get("full_write_baselines") or {}).get(resolved)
    if baseline is None or _file_metadata(resolved) != baseline[:-1]:
        return None
    return REREAD_HINT
