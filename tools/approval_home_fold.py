"""Fold the resolved user home and HERMES_HOME in a command to ``~/`` / ``~/.hermes/``.

Detection-time normalization for :mod:`tools.approval_detection`: the dangerous-command patterns
name Hermes's own files through ``~/.hermes`` / ``$HERMES_HOME`` spellings, so every spelling a shell
uses for the resolved homes (native, Git Bash ``/c/…``, ``~``, ``$HOME``, ``$LOCALAPPDATA``, …) is
rewritten into those forms first.
"""

import functools
import os
import re

from hermes_platform.host import facts
from tools.approval_parser_budget import _scan_shell


# Shell metacharacters, quotes, and whitespace that terminate a path token.
_PATH_TOKEN_STOP = r"""\s'"`;|&<>()"""
_PATH_TAIL = r"(?P<tail>(?:[/\\][^/\\" + _PATH_TOKEN_STOP + r"]*)+)"


def _spell_component(component: str) -> str:
    """*component* as a regex, tolerating the optional escape Git Bash writes before a space.

    An unquoted path with a space is written ``/c/Users/John\\ Smith`` — the same directory as
    ``John Smith``. Escaping alone (``re.escape``) leaves the literal space, so a profile whose name
    contains one never folded and a ``sed -i`` of its config.yaml passed as safe.
    """
    return re.escape(component).replace(r"\ ", r"\\? ")


def _live_space_escapes(command: str, start: int = 0, end: int | None = None, found: set | None = None) -> set:
    """Indexes of each backslash in *command* the shell reads as escaping a space or tab: unquoted, or
    unquoted inside a ``$(...)`` / backtick body (a fresh quote context). Inside single or double
    quotes the backslash is a literal character."""
    found = set() if found is None else found
    for kind, i, j, quote in _scan_shell(command, start, end, subst="q"):
        if kind == "subst":
            _live_space_escapes(command, i + (2 if command.startswith("$(", i) else 1), j - 1, found)
        elif kind == "esc" and quote is None and command[i + 1] in " \t":
            found.add(i)
    return found


def _fold_matches(pattern, command: str, replace) -> str:
    """``pattern.sub(replace, command)``, keeping a match as written when an escaped space in it is
    not what the shell reads:

    - a quoted ``\\ `` is a literal backslash, so ``'/home/John\\ Smith'`` is another directory than
      the ``John Smith`` the optional escape in :func:`_spell_component` stands for;
    - an unquoted ``\\ `` right after the match continues the last name instead of separating a child:
      ``hermes\\ data\\ backup`` is a sibling of the home ``hermes data``, not inside it.
    """
    live = None

    def fold(match: re.Match) -> str:
        nonlocal live
        escapes = [i for i in range(match.start(), match.end()) if command.startswith(("\\ ", "\\\t"), i)]
        if escapes:
            live = _live_space_escapes(command) if live is None else live
            for i in escapes:
                # Inside the match the escape spells a component's space; at its end it follows the home.
                if (i in live) != (i + 1 < match.end()):
                    return match.group(0)
        return replace(match)

    return pattern.sub(fold, command)


@functools.lru_cache(maxsize=64)
def _home_prefix_fold_regex(path: str):
    """Compile a regex matching *path* as an absolute directory prefix.
    Components match with either separator so native Windows, forward-slash, and mixed forms all
    fold; the caller normalizes the tail's backslashes to ``/``. A non-empty tail is required, so a
    bare home is never folded. Returns ``None`` for an unset/degenerate path (no components, or a
    bare root/drive: ``/``, ``C:``, ``C:\\``, ``""``) so a stray HOME cannot rewrite unrelated
    prefixes.

    A depth-1 home is NOT degenerate: ``/root`` is root's own default HOME, and skipping it left
    ``/root/.ssh/authorized_keys`` unfolded — hence "safe" — while the identical command spelled
    under ``/home/alice`` folded to ``~/.ssh/authorized_keys`` and was denied. A single-component
    home must therefore match a mandatory leading separator, so a relative same-named path
    (``root/file``) is still left alone."""
    components = [c for c in re.split(r"[/\\]+", path) if c] if path else []
    if not components or (len(components) == 1 and re.fullmatch(r"[A-Za-z]:", components[0])):
        return None
    # Optional leading root separator; a Windows drive letter is a component. Mandatory (and anchored
    # at a path start: ``/data/root/x`` is not ``/data~/x``) for a single-component home, else ``root/file`` folds.
    # A POSIX single-component home splits only on ``/`` or ``\/``: ``/root\*`` is the literal name ``/root*``.
    sep = r"(?:\\?/)" if len(components) == 1 and facts.os_family() != "win32" and path.startswith("/") else r"[/\\]"
    lead = r"(?<![\w.~$}/\\-])" + sep + "+" if len(components) == 1 else sep + "*"
    tail = _PATH_TAIL if sep == r"[/\\]" else r"(?P<tail>(?:\\?/[^/\\" + _PATH_TOKEN_STOP + r"]*)+)"
    return re.compile(lead + (sep + "+").join(_spell_component(c) for c in components) + tail)


def _fold_home_prefixes(command: str, paths, replacement: str) -> str:
    """Fold each resolved home prefix in *command* to *replacement* (no trailing separator; the tail
    supplies it). Longest first so a deeper home folds before a shorter overlapping one that would clobber it."""
    for path in dict.fromkeys(sorted((p for p in paths if p), key=len, reverse=True)):
        pattern = _home_prefix_fold_regex(path)
        if pattern is not None:
            command = _fold_matches(
                pattern, command, lambda m: replacement + m.group("tail").replace("\\/", "/").replace("\\", "/"))
    return command


_MSYS_DRIVE_PATH_RE = re.compile(r"^([A-Za-z]):[/\\](.+)$")


def _msys_drive_spellings(path: str) -> list[str]:
    """``/c/Users/me`` (and the capital-drive form) for ``C:\\Users\\me``; empty without a drive."""
    match = _MSYS_DRIVE_PATH_RE.match(path) if path else None
    if match is None:
        return []
    letter, rest = match.group(1), match.group(2)
    return [f"/{letter.lower()}/{rest}", f"/{letter.upper()}/{rest}"]


def _path_below(root: str, path: str) -> str | None:
    """*path* relative to *root* when it sits underneath (either separator; Windows folds case),
    else ``None``. Only the comparison folds, so the caller gets the tail as it was written."""
    root_parts = [p for p in re.split(r"[/\\]+", root) if p]
    path_parts = [p for p in re.split(r"[/\\]+", path) if p]
    if not root_parts or len(path_parts) <= len(root_parts):
        return None
    head, wanted = path_parts[:len(root_parts)], root_parts
    if facts.os_family() == "win32":
        head, wanted = [p.lower() for p in head], [p.lower() for p in wanted]
    if head != wanted:
        return None
    return "/".join(path_parts[len(root_parts):])


def _home_spellings(paths, command: str) -> list[str]:
    """Every spelling a shell uses for one of *paths*: as written (native and Git Bash drive form)
    and the same location written through a home variable — ``~``, ``$HOME``, ``$LOCALAPPDATA`` /
    ``%LOCALAPPDATA%``, ``$USERPROFILE``. Windows's default HERMES_HOME nests under the profile,
    which Git Bash reaches as ``/c/Users/<me>/AppData/Local/hermes``; folding only the absolute
    native form left that spelling — and its ``~/…`` and ``$LOCALAPPDATA/…`` siblings — unmatched.
    The variable spellings are only derived when *command* writes one of ``~``, ``$``, ``%``, so a
    command with no variable syntax costs nothing beyond its own absolute prefixes."""
    absolute = list(dict.fromkeys(s for path in paths for s in (path, *_msys_drive_spellings(path))))
    if not any(marker in command for marker in ("~", "$", "%")):
        return absolute  # no variable syntax, so no variable spelling can match either
    # `~` is $HOME in Git Bash, so the variable wins over the native reading of it.
    home = os.environ.get("HOME") or os.path.expanduser("~")
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    roaming = os.environ.get("APPDATA", "")
    profile = os.environ.get("USERPROFILE", "")
    named = (("HOME", home), ("LOCALAPPDATA", local_appdata), ("APPDATA", roaming), ("USERPROFILE", profile))
    roots = [("~", home)] + [(spelled, root) for name, root in named
                             for spelled in (f"${name}", f"${{{name}}}", f"%{name}%")]
    relative = [f"{prefix}/{tail}"
                for prefix, root in roots
                for spelled_root in dict.fromkeys([root, *_msys_drive_spellings(root)])
                for candidate in absolute if (tail := _path_below(spelled_root, candidate))]
    return absolute + list(dict.fromkeys(relative))


def _rewrite_resolved_user_home(command: str) -> str:
    """User home (expanduser / realpath / $HOME) -> ``~/``; no-op when unset, degenerate, or unresolvable."""
    try:
        # expanduser, realpath, and an explicit HOME — Windows expanduser uses USERPROFILE, not HOME.
        home = os.path.expanduser("~")
        paths = [home, os.path.realpath(home), os.environ.get("HOME", "")]
        paths += [spelling for path in list(paths) if path for spelling in _msys_drive_spellings(path)]
    except Exception:
        return command
    return _fold_home_prefixes(command, paths, "~")


def _rewrite_resolved_hermes_home(command: str) -> str:
    """Resolved HERMES_HOME -> ``~/.hermes/`` so the _HERMES_CONFIG_PATH / _HERMES_ENV_PATH rules
    match Docker/gateway deployments and the platform default in every spelling a shell uses for it
    (native, Git Bash ``/c/…``, ``~/AppData/Local/hermes``, ``$LOCALAPPDATA/hermes``)."""
    try:
        from hermes_constants import get_hermes_home
        home = get_hermes_home().expanduser()
        paths = _home_spellings([str(home), str(home.resolve(strict=False))], command)
    except Exception:
        return command
    command = _fold_home_prefixes(command, paths, "~/.hermes")
    # The prefix fold needs a child tail; fold the bare home word too (`mv /opt/data /tmp/x`) so the
    # Hermes-home rules see it.
    for path in dict.fromkeys(sorted((p for p in paths if p), key=len, reverse=True)):
        pattern = _bare_home_fold_regex(path)
        if pattern is not None:
            command = _fold_matches(pattern, command, lambda _m: "~/.hermes")
    return command


@functools.lru_cache(maxsize=64)
def _bare_home_fold_regex(path: str):
    """*path* as a whole shell word (optional trailing separator); ``None`` when degenerate."""
    components = [c for c in re.split(r"[/\\]+", path) if c] if path else []
    if len(components) < 2:
        return None
    return re.compile(
        r"(?<![^\s'\"`=;|&<>(])[/\\]*" + r"[/\\]+".join(_spell_component(c) for c in components)
        + r"[/\\]?(?=[" + _PATH_TOKEN_STOP + r"*]|$)")
